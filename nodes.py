import re

from state import AgentState
from tools import (
    generate_conversation_summary,
    get_compatible_devices_mock,
    get_prepaid_offers_mock,
)


def save_user_message_node(state: AgentState):
    """Conserve le message utilisateur comme contexte interne de recommandation."""
    message = state.get("message", "").strip()

    if not message:
        return {}

    return {
        "history": [
            {
                "role": "user",
                "content": message,
            }
        ]
    }


def _extract_budget_amount(message: str) -> float | None:
    """Extrait un budget explicite comme 'moins de 20$', 'budget 30' ou 'forfait de 30$'."""
    constrained = re.search(
        r"(?:moins de|maximum|max|budget(?:\s+(?:de|est(?:\s+de)?))?|jusqu['’]?a|jusqu['’]?à)\s*"
        r"(?:\$|usd|dollars?)?\s*(\d+(?:[.,]\d+)?)",
        message,
        flags=re.IGNORECASE,
    )
    if constrained:
        return float(constrained.group(1).replace(",", "."))

    # Formes naturelles fréquentes : "forfait de 30$", "à 30 dollars", "$30".
    suffix = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:\$|usd|dollars?)",
        message,
        flags=re.IGNORECASE,
    )
    if suffix:
        return float(suffix.group(1).replace(",", "."))

    prefix = re.search(
        r"(?:\$|usd)\s*(\d+(?:[.,]\d+)?)",
        message,
        flags=re.IGNORECASE,
    )
    if prefix:
        return float(prefix.group(1).replace(",", "."))

    return None


def _explicit_budget_currency(message: str) -> str | None:
    """Normalise la devise EXPLICITEMENT accolée au montant dans cette demande.

    La sélection actuelle des forfaits ne gère que les montants USD ; aucun USD
    implicite n'est attribué à « budget 2 » sans symbole ni devise.
    """
    amount = r"\d+(?:[.,]\d+)?"
    if re.search(rf"(?:{amount}\s*(?:\$|USD\b|dollars?\b)|(?:\$|USD\b)\s*{amount})", message, re.I):
        return "USD"
    return None


def _price_match_mode(message: str, intent: str | None) -> str:
    """Précise la sémantique d'un montant, sans faire choisir l'offre au LLM.

    « forfait de 40$ » : prix demandé (exact).
    « budget maximum 40$ », « moins de 40$ » : plafond (maximum).
    Une recommandation avec une enveloppe chiffrée reste un plafond.
    Le mode n'a d'effet que lorsque le tour fournit un nouveau montant.
    """
    if re.search(
        r"(?:moins\s+de|en\s+dessous\s+de|maximum|max(?:imum)?\b|"
        r"budget\b|jusqu['’]?\s*[aà]|au\s+plus|pas\s+plus\s+de|"
        r"under\b|at\s+most\b|up\s+to\b|within\s+(?:my\s+)?budget)",
        message, re.I,
    ):
        return "maximum"
    return "exact" if intent == "BUY_PREPAID" else "maximum"


def _extract_requested_data_gb(message: str) -> float | None:
    """Extrait un besoin data explicite comme '20 Go' ou '10GB'."""
    match = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(?:go|gb)\b",
        message,
        flags=re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group(1).replace(",", "."))


def _filter_offers_by_data(offers, requested_data_gb):
    if requested_data_gb is None:
        return offers

    return [
        offer
        for offer in offers
        if float(offer.get("data_gb", 0)) >= float(requested_data_gb)
    ]


def extract_context_node(state: AgentState):
    """Extrait uniquement le contexte métier utile à la recommandation."""
    message = state.get("message", "")
    message_lower = message.lower()
    intent = state.get("intent")

    updates = {}

    if not state.get("initial_intent") and intent and intent not in {"unknown", "UNKNOWN"}:
        updates["initial_intent"] = intent

    if "flexi data max" in message_lower or "flexi data" in message_lower:
        # Un plan nommé explicitement prime sur un vieux budget hérité : la
        # personne peut changer d'offre au milieu d'une même conversation.
        updates["budget"] = None
        updates["budget_currency"] = None
        updates["price_match_mode"] = None
        updates["requested_data_gb"] = None
        updates["selected_offer_id"] = "PRE_001"
        updates["selected_offer_name"] = "Flexi Data Max"
    elif "eco prepaid" in message_lower:
        updates["budget"] = None
        updates["budget_currency"] = None
        updates["price_match_mode"] = None
        updates["requested_data_gb"] = None
        updates["selected_offer_id"] = "PRE_002"
        updates["selected_offer_name"] = "Eco Prepaid"

    if "routeur" in message_lower or "router" in message_lower:
        updates["category_preference"] = "4G_ROUTER"
    elif any(
        word in message_lower
        for word in ("smartphone", "téléphone", "telephone", "mobile")
    ):
        updates["category_preference"] = "SMARTPHONE"

    requested_data_gb = _extract_requested_data_gb(message)
    if requested_data_gb is not None:
        updates["requested_data_gb"] = requested_data_gb

    amount = _extract_budget_amount(message)
    if amount is not None:
        if intent == "RECOMMEND_DEVICE":
            updates["max_price"] = amount
        else:
            updates["budget"] = amount
            # Réinitialiser aussi l'unité si un nouveau budget remplace l'ancien.
            updates["budget_currency"] = _explicit_budget_currency(message)
            updates["price_match_mode"] = _price_match_mode(message, intent)

    return updates


def _select_offer_from_results(
    state: AgentState,
    offers: list[dict],
    *,
    is_recommendation: bool,
) -> tuple[dict | None, str]:
    """Une seule règle métier de sélection, utilisée par les deux branches offre.

    Un nouveau montant explicite prime sur une ancienne sélection. L'absence de
    nouveaux critères conserve la sélection précédente *seulement si elle reste
    éligible*. Le graph choisit, jamais le Response LLM.
    """
    if not offers:
        return None, "no_eligible_offer"

    message = state.get("message", "") or ""
    message_lower = message.lower()

    # Les noms officiels du catalogue sont des références produit, pas des
    # marqueurs linguistiques de conversation.
    named_offer = next(
        (
            offer
            for offer in offers
            if offer["name"].lower() in message_lower
        ),
        None,
    )
    if named_offer is not None:
        return named_offer, "explicit_offer_name"

    # Une nouvelle demande chiffrée ne doit pas garder une sélection héritée.
    mentioned_amount = _extract_budget_amount(message)
    if mentioned_amount is not None:
        matching_price = [
            offer for offer in offers
            if float(offer["price_monthly"]) == float(mentioned_amount)
        ]
        if matching_price:
            return max(matching_price, key=lambda o: o["data_gb"]), "exact_price"
        # Si le montant est un plafond sans correspondance exacte, privilégier
        # l'offre éligible offrant le plus de data (puis le prix le plus élevé).
        return max(
            offers,
            key=lambda o: (float(o["data_gb"]), float(o["price_monthly"])),
        ), "most_data_within_budget"

    previous_offer = next(
        (
            offer for offer in offers
            if offer["offer_id"] == state.get("selected_offer_id")
        ),
        None,
    )
    # Une demande explicite de data doit revalider l'offre précédente : si elle
    # est encore admissible, on la conserve, sinon on recalcule ci-dessous.
    if previous_offer is not None:
        return previous_offer, "previous_eligible_selection"

    if len(offers) == 1:
        return offers[0], "single_eligible_offer"

    if is_recommendation:
        return max(
            offers,
            key=lambda o: (float(o["data_gb"]), float(o["price_monthly"])),
        ), "most_data_among_eligible_offers"

    # BUY_PREPAID sans critère discriminant : liste d'offres, pas de sélection
    # artificielle. Le LLM ne pourra pas inventer une sélection dans le State.
    return None, "multiple_eligible_offers"


def _offer_result(state: AgentState, *, is_recommendation: bool) -> dict:
    # Consulter une fois le catalogue complet. Pour un prix *exact*, « 40$ »
    # ne signifie pas « n'importe quel forfait coûtant moins de 40$ ».
    # La liste complète reste un contexte informatif, jamais une sélection.
    full_catalogue = get_prepaid_offers_mock.invoke({"user_budget_max": None})
    available_offers = _filter_offers_by_data(
        full_catalogue, state.get("requested_data_gb")
    )
    amount = state.get("budget")
    if amount is None:
        offers = list(available_offers)
    elif state.get("price_match_mode") == "exact":
        offers = [
            offer for offer in available_offers
            if float(offer["price_monthly"]) == float(amount)
        ]
    else:
        offers = [
            offer for offer in available_offers
            if float(offer["price_monthly"]) <= float(amount)
        ]
    selected, basis = _select_offer_from_results(
        state, offers, is_recommendation=is_recommendation
    )

    # Un budget inférieur au catalogue ne constitue PAS une erreur technique.
    # Retourner le prix minimal métier comme fait vérifiable facultatif pour
    # que le verbaliseur puisse expliquer la limite sans rien inventer.
    lowest_available_offer = (
        min(available_offers, key=lambda offer: float(offer["price_monthly"]))
        if not offers and available_offers else None
    )

    return {
        "offers": offers,
        "available_offers": available_offers if not offers else [],
        "lowest_available_offer": lowest_available_offer,
        "selected_offer_id": selected["offer_id"] if selected else None,
        "selected_offer_name": selected["name"] if selected else None,
        "selection_basis": basis,
        # Ne pas exposer des appareils d'une recherche précédente.
        "devices": [],
        "closest_device": None,
        "device_budget_gap": None,
    }


def get_offers_node(state: AgentState):
    """BUY_PREPAID : cherche les offres et choisit seulement si les faits suffisent."""
    return _offer_result(state, is_recommendation=False)


def get_devices_node(state: AgentState):
    """Cherche d'abord dans le budget, puis calcule l'option compatible la plus proche."""
    offer_id = state.get("selected_offer_id")
    # Une ancienne absence d'offre n'est pas pertinente pour une recherche
    # d'appareil ultérieure.
    offer_updates = {"lowest_available_offer": None, "available_offers": []}

    # Une référence explicite à un forfait de N Go prime sur une sélection
    # antérieure : "pour mon forfait de 5 Go" désigne ici PRE_002. On ne
    # déduit une offre que si UNE SEULE ligne du catalogue correspond exactement.
    # Sans correspondance unique, on garde le comportement antérieur.
    mentioned_data_gb = _extract_requested_data_gb(state.get("message", "") or "")
    if mentioned_data_gb is not None:
        matching_plans = [
            offer
            for offer in get_prepaid_offers_mock.invoke({"user_budget_max": None})
            if float(offer.get("data_gb", -1)) == mentioned_data_gb
        ]
        if len(matching_plans) == 1:
            plan = matching_plans[0]
            offer_id = plan["offer_id"]
            offer_updates = {
                **offer_updates,
                "selected_offer_id": plan["offer_id"],
                "selected_offer_name": plan["name"],
                "selection_basis": "explicit_data_plan",
                "offers": [plan],
            }

    if not offer_id:
        return {
            **offer_updates,
            "devices": [],
            "closest_device": None,
            "device_budget_gap": None,
        }

    category = state.get("category_preference")
    max_price = state.get("max_price")

    devices = get_compatible_devices_mock.invoke(
        {
            "offer_id": offer_id,
            "category_preference": category,
            "max_price": max_price,
        }
    )

    if devices:
        return {
            **offer_updates,
            "devices": devices,
            "closest_device": None,
            "device_budget_gap": None,
        }

    # Aucun appareil ne respecte le budget. On récupère alors les appareils
    # compatibles sans plafond de prix uniquement pour proposer l'option la plus
    # proche, sans prétendre qu'elle respecte le budget utilisateur.
    if max_price is not None:
        compatible_devices = get_compatible_devices_mock.invoke(
            {
                "offer_id": offer_id,
                "category_preference": category,
                "max_price": None,
            }
        )

        priced_devices = [
            device
            for device in compatible_devices
            if device.get("price") is not None
        ]

        if priced_devices:
            closest_device = min(
                priced_devices,
                key=lambda device: float(device["price"]),
            )
            gap = max(
                0.0,
                float(closest_device["price"]) - float(max_price),
            )

            return {
                **offer_updates,
                "devices": [],
                "closest_device": closest_device,
                "device_budget_gap": gap,
            }

    return {
        **offer_updates,
        "devices": [],
        "closest_device": None,
        "device_budget_gap": None,
    }


def recommend_offer_node(state: AgentState):
    """ASK_RECOMMENDATION : sélection métier parmi les offres éligibles."""
    return _offer_result(state, is_recommendation=True)


def summary_node(state: AgentState):
    """Produit des données de synthèse; l'Agent Communication les verbalise."""
    summary = generate_conversation_summary.invoke(
        {
            "history": state.get("history", []),
            "initial_intent": state.get("initial_intent"),
            "budget": state.get("budget"),
            "selected_offer_name": state.get("selected_offer_name"),
            "devices": state.get("devices", []),
        }
    )

    return {"summary": summary}
