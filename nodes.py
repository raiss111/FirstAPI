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
        r"(?:moins de|maximum|max|budget(?: de)?|jusqu['’]?a|jusqu['’]?à)\s*"
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
        updates["selected_offer_id"] = "PRE_001"
        updates["selected_offer_name"] = "Flexi Data Max"
    elif "eco prepaid" in message_lower:
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

    return updates


def get_offers_node(state: AgentState):
    offers = get_prepaid_offers_mock.invoke(
        {
            "user_budget_max": state.get("budget"),
        }
    )

    offers = _filter_offers_by_data(
        offers,
        state.get("requested_data_gb"),
    )

    updates = {
        "offers": offers,
        # Une recherche d'offre ne doit pas laisser traîner les appareils
        # d'une ancienne recommandation dans le State visible.
        "devices": [],
    }

    if len(offers) == 1:
        updates["selected_offer_id"] = offers[0]["offer_id"]
        updates["selected_offer_name"] = offers[0]["name"]

    return updates


def get_devices_node(state: AgentState):
    offer_id = state.get("selected_offer_id")

    if not offer_id:
        return {"devices": []}

    devices = get_compatible_devices_mock.invoke(
        {
            "offer_id": offer_id,
            "category_preference": state.get("category_preference"),
            "max_price": state.get("max_price"),
        }
    )

    return {"devices": devices}


def recommend_offer_node(state: AgentState):
    """Choisit une offre parmi celles éligibles, sans recommander d'appareil."""
    offers = get_prepaid_offers_mock.invoke(
        {
            "user_budget_max": state.get("budget"),
        }
    )

    offers = _filter_offers_by_data(
        offers,
        state.get("requested_data_gb"),
    )

    if not offers:
        return {
            "offers": [],
            "selected_offer_id": None,
            "selected_offer_name": None,
            "devices": [],
        }

    selected_offer_id = state.get("selected_offer_id")
    selected_offer = next(
        (offer for offer in offers if offer["offer_id"] == selected_offer_id),
        None,
    )

    # MVP : à budget égal, privilégier l'offre avec le plus de data.
    if selected_offer is None:
        selected_offer = max(offers, key=lambda offer: offer["data_gb"])

    return {
        "offers": offers,
        "selected_offer_id": selected_offer["offer_id"],
        "selected_offer_name": selected_offer["name"],
        "devices": [],
    }


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
