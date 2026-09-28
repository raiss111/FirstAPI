"""Verbalisation des seuls résultats fournis par les composants métier.

Le générateur n'est ni un classificateur, ni un routeur, ni un service d'achat.
Une validation ciblée bloque les erreurs observées dans les tests MVP ; en cas de
sortie non conforme, on régénère une fois, puis on refuse de publier le texte.
"""

import json
import re
from typing import Any

from groq import Groq


COMMUNICATION_MODEL = "openai/gpt-oss-20b"


class ResponseValidationError(RuntimeError):
    """Le LLM a renvoyé une réponse contredisant les faits métier disponibles."""


LLM_RESPONSE_SYSTEM_PROMPT = """
You are TOBi's FINAL LANGUAGE GENERATOR. Your ONLY role is to turn the computed
BUSINESS_CONTEXT into a short, natural user-facing answer. You are NOT the
intent classifier, recommender, transaction processor or dialogue policy engine.

BUSINESS_CONTEXT is the SOLE authority for product details, prices, currencies,
selected offer, purchase status, beneficiary and successful actions. Current user
message and history may only inform response language and tone. Do not invent
anything from general telecom knowledge.

PURCHASE-STATE CONTRACT:
- If purchase_state.status == AWAITING_CONFIRMATION: summarize the selected
  offer and ask ONE SHORT question requesting confirmation of the MOCK purchase.
  Say it is simulated. Never imply the purchase has already occurred.
- If purchase_state.status == COMPLETED: use ONLY purchase_state.action_result.
  If status=SIMULATED_SUCCESS and real_transaction=false, say explicitly that
  the MOCK/SIMULATED purchase was completed. Never claim a real transaction.
- If CANCELLED: simply state that the simulated purchase was canceled.
- If OFFER_UNAVAILABLE: NO offer has been selected; no purchase can be confirmed.
  recommendation_state.price_match_mode has two distinct meanings:
  "exact" means the user asked for a plan AT THAT PRICE (e.g. « forfait de
  40$ »); say that NO plan exists at that exact price, NOT that all cheaper
  plans are unavailable. "maximum" means a budget CEILING (e.g. « budget
  maximum de 40$ »); say no plan fits that limit if offers=[]. The amount and
  its explicitly supplied currency are in budget/budget_currency. You may
  mention ONLY the facts in recommendation_state.available_offers or
  lowest_available_offer as clearly labelled OTHER available options, NEVER
  as the requested plan or a selected plan. Never reuse the previous selection.
  Never request confirmation, payment or phone details on an unavailable offer.
- Otherwise: only verbalize the recommendation / TOBi result actually provided.
- If the current message is ambiguous while confirmation is pending, ask only
  for clarification of confirmation; never execute anything yourself.

NEVER mention any purchase PROCEDURE, STEPS, payment channels, app, website,
store, account creation, confirmation SMS/email, activation process or the
unavailability of procedures. Never offer to guide somebody through a purchase.
A short confirmation question for a MOCK purchase is allowed only when
purchase_state.status == AWAITING_CONFIRMATION.

RECOMMENDATION CONTRACT:
- If current_request.intent == RECOMMEND_DEVICE, answer the DEVICE question, not
  the offer-selection question. Mention only devices in recommendation_state.devices
  or closest_device. If there are none, state only that none is listed in the mock.
  Do not invent a smartphone or say no smartphone is available when devices exist.
- selected_offer is the only offer selected by the Recommendation Agent.
- Preserve exact product names and numbers. Use only explicit currency; don't
  invent euros, change currencies, prices or purchase eligibility.
- Never make your own product choice, route an intent or decide an action.
- Do not reveal internal codes, JSON, session state, statuses or tool names.
- When current_request.scope=recommendation, ignore stale tobi_state; when
  scope=tobi, ignore old recommendation/purchase results unless explicitly
  marked relevant for this turn.

TOBi CONTRACT:
- Ask only about tobi_state.missing_slots if present; don't add missing fields.
- Describe a business action as executed only if the backend result proves it;
  distinguish simulations from any real transaction.

LANGUAGE: Prefer the current user's language. For short/ambiguous messages, use
recent conversation or CRM preferred language as fallback.

OUTPUT: ONE LINE of plain text (ideally 1-2 short sentences). No actual newlines,
no literal \n, Markdown, bold markers, underscores, vertical bars, separators,
bullets, numbered lists, decorative symbols, emoji, JSON or quotation wrappers.
Use normal spaces; provide ONLY the final human response text.
""".strip()


# Contrôles ciblés sur les anomalies observées ; le contrôle des faits en amont
# reste indispensable. Ces expressions ne servent PAS à classifier des intents.
# Ces garde-fous concernent UNIQUEMENT la sortie du verbaliseur, jamais
# l'extraction linguistique de l'Intent Agent.
_ANY_PURCHASE_PROCEDURE = (
    r"\b(?:proc[eé]dures?|purchase steps?|[eé]tapes? (?:pour |d['’])?(?:l['’])?achat|steps? (?:to |for )?"
    r"(?:buy|purchase|checkout)|purchase instructions|checkout|activation|"
    r"rendez[- ]vous (?:sur|dans)|visitez (?:le |notre )?(?:site|magasin)|"
    r"website|\b(?:application|app|oneapp|magasin|store|site web|point de vente)\b|"
    r"go to (?:the |our )?(?:app|website|store)|"
    r"connectez[- ]vous|log in|cr[eé]ez (?:votre |un )?compte|"
    r"SMS|e[- ]?mail de confirmation|confirmation email)\b",
)

_CURRENCY_MARKERS: dict[str, str] = {
    "USD": r"\$|\bUSD\b|\bdollars?\b",
    "EUR": r"€|\bEUR\b|\beuros?\b",
    "CDF": r"\bCDF\b|\bFC\b|\bfrancs? congolais\b",
    "GBP": r"£|\bGBP\b|\bpounds?\b",
}


def _numeric_facts(business_context: dict[str, Any]) -> set[float]:
    rec = business_context.get("recommendation_state") or {}
    values: set[float] = set()
    for key in ("budget", "max_price", "requested_data_gb", "device_budget_gap"):
        value = rec.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.add(float(value))
    products = list(rec.get("offers") or []) + list(rec.get("devices") or [])
    products.extend(rec.get("available_offers") or [])
    if rec.get("closest_device"):
        products.append(rec["closest_device"])
    if rec.get("lowest_available_offer"):
        products.append(rec["lowest_available_offer"])
    if (business_context.get("purchase_state") or {}).get("offer"):
        products.append(business_context["purchase_state"]["offer"])
    for product in products:
        for key in ("price_monthly", "price", "data_gb", "calls_min"):
            value = product.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.add(float(value))
    return values


def _validate_recommendation_response(
    response_text: str,
    business_context: dict[str, Any],
) -> list[str]:
    """Contrôle la sortie en fonction de l'état métier (dont l'achat simulé)."""
    issues: list[str] = []
    rec = business_context.get("recommendation_state") or {}
    purchase = business_context.get("purchase_state") or {}
    scope = (business_context.get("current_request") or {}).get("scope")

    # Pour une demande d'achat, aucune procédure (même présentée comme absente).
    is_purchase_turn = (business_context.get("current_request") or {}).get("intent") == "BUY_PREPAID" or bool(purchase.get("status"))
    if is_purchase_turn:
        for pattern in _ANY_PURCHASE_PROCEDURE:
            if re.search(pattern, response_text, re.I):
                issues.append("purchase_procedure_mentioned")
                break
        if re.search(r"\b(?:guider|guide you|help you buy|aider [aà] acheter)\b", response_text, re.I):
            issues.append("unsolicited_purchase_guidance")

    if scope == "recommendation" or purchase.get("status"):
        products = list(rec.get("offers") or []) + list(rec.get("devices") or [])
        products.extend(rec.get("available_offers") or [])
        if rec.get("closest_device"):
            products.append(rec["closest_device"])
        if rec.get("lowest_available_offer"):
            products.append(rec["lowest_available_offer"])
        if purchase.get("offer"):
            products.append(purchase["offer"])
        allowed_currencies = {
            str(item["currency"]).upper()
            for item in products if item.get("currency")
        }
        # Le budget du client est aussi un fait métier. Quand aucune offre ne
        # passe le filtre, ne pas rejeter son « 2 USD » comme une devise inventée.
        # En revanche, un ancien budget sans unité n'autorise aucune déduction.
        budget_currency = str(rec.get("budget_currency") or "").upper()
        if isinstance(rec.get("budget"), (int, float)) and budget_currency in _CURRENCY_MARKERS:
            allowed_currencies.add(budget_currency)

        # Les prix exprimés avec une devise doivent correspondre à une paire
        # (montant, devise) réelle, et non à un volume data ou un ancien budget.
        allowed_money = {
            (float(item[key]), str(item.get("currency", "")).upper())
            for item in products for key in ("price_monthly", "price")
            if isinstance(item.get(key), (int, float)) and item.get("currency")
        }
        if budget_currency in _CURRENCY_MARKERS and isinstance(rec.get("budget"), (int, float)):
            allowed_money.add((float(rec["budget"]), budget_currency))
        token_currency = {"$": "USD", "€": "EUR", "£": "GBP", "USD": "USD", "EUR": "EUR", "CDF": "CDF", "GBP": "GBP", "FC": "CDF"}
        amount = r"(\d+(?:[.,]\d+)?)"
        money_suffix = re.compile(rf"(?<!\w){amount}\s*(\$|€|£|USD\b|EUR\b|CDF\b|GBP\b|FC\b)", re.I)
        money_prefix = re.compile(rf"(?<!\w)(\$|€|£|USD\b|EUR\b|CDF\b|GBP\b|FC\b)\s*{amount}", re.I)
        for match in money_suffix.finditer(response_text):
            pair = (float(match.group(1).replace(",", ".")), token_currency[match.group(2).upper()])
            if pair not in allowed_money:
                issues.append("unsupported_price_currency_pair")
                break
        for match in money_prefix.finditer(response_text):
            pair = (float(match.group(2).replace(",", ".")), token_currency[match.group(1).upper()])
            if pair not in allowed_money:
                issues.append("unsupported_price_currency_pair")
                break

        for currency, pattern in _CURRENCY_MARKERS.items():
            if currency not in allowed_currencies and re.search(pattern, response_text, re.I):
                issues.append(f"unsupported_currency:{currency}")
        allowed_numbers = _numeric_facts(business_context)
        for match in re.finditer(
            r"(?<!\w)(\d+(?:[.,]\d+)?)\s*(?:\$|€|£|USD\b|EUR\b|CDF\b|GBP\b|Go\b|GB\b|"
            r"minutes?\b|min\b)", response_text, re.I,
        ):
            if float(match.group(1).replace(",", ".")) not in allowed_numbers:
                issues.append("unsupported_number_with_unit")
                break

        selected = business_context.get("selected_offer")
        if selected is None and re.search(
            r"\b(?:je (?:te|vous) recommande|i recommend|best (?:offer|plan))\b",
            response_text, re.I,
        ):
            issues.append("llm_made_a_product_selection")

    # Une demande d'appareil doit verbaliser le résultat Appareil du graphe,
    # sans réinterpréter le texte ou sélectionner un nouveau produit.
    if (business_context.get("current_request") or {}).get("intent") == "RECOMMEND_DEVICE":
        listed_devices = list(rec.get("devices") or [])
        if not listed_devices and rec.get("closest_device"):
            listed_devices = [rec["closest_device"]]
        if listed_devices and not any(
            str(device.get("model", "")).lower() in response_text.lower()
            for device in listed_devices if device.get("model")
        ):
            issues.append("device_result_not_mentioned")

        # La présence du Xiaomi correct ne doit pas suffire à autoriser l'ajout
        # d'un Samsung non renvoyé. Le catalogue de contrôle n'est pas visible
        # du LLM et n'autorise aucune sélection supplémentaire.
        allowed_ids = {device.get("device_id") for device in listed_devices}
        text_lower = response_text.casefold()
        for catalog_device in business_context.get("_validation_device_catalog") or []:
            if catalog_device.get("device_id") in allowed_ids:
                continue
            full_name = " ".join((
                str(catalog_device.get("brand", "")),
                str(catalog_device.get("model", "")),
            )).strip().casefold()
            model = str(catalog_device.get("model", "")).strip().casefold()
            if (full_name and full_name in text_lower) or (model and model in text_lower):
                issues.append("unlisted_device_mentioned")
                break

    status = purchase.get("status")
    if status == "OFFER_UNAVAILABLE":
        if re.search(
            r"\b(?:confirmez|confirm(?:ez|ation)?|validez|proceed|checkout)\b",
            response_text, re.I,
        ):
            issues.append("unavailable_offer_confirmation_not_allowed")
    if status != "COMPLETED" and re.search(
        r"\b(?:achat|purchase|commande|order)\b.{0,55}"
        r"\b(?:effectu[eé]|r[eé]alis[eé]|finalis[eé]|completed|successful|done)\b",
        response_text, re.I,
    ):
        issues.append("purchase_claim_without_execution")
    if status == "COMPLETED" and (purchase.get("action_result") or {}).get("mode") == "mock":
        if not re.search(r"\b(?:simul[eé]e?s?|simulated|simulation|mock|ficti(?:f|ve)|test)\b", response_text, re.I):
            issues.append("mock_disclosure_missing")
    if status == "AWAITING_CONFIRMATION":
        if "?" not in response_text:
            issues.append("confirmation_question_missing")
        if not re.search(r"\b(?:simul[eé]e?s?|simulation|simulated|mock|test)\b", response_text, re.I):
            issues.append("pending_mock_disclosure_missing")
    if (scope == "recommendation" or status) and re.search(
        r"\b(?:1\.|2\.|3\.|[eé]tape\s*\d+)\s+", response_text, re.I,
    ):
        issues.append("numbered_list_not_allowed")

    return issues


def _normalize_plain_text(text: str) -> str:
    """Retourne UNE SEULE LIGNE, sans ornements, sans modifier les faits métier."""
    replacements = {
        "\u00a0": " ", "\u202f": " ", "\u200b": "", "\ufeff": "",
        "\u2060": "", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u2011": "-",
        "\u2022": "; ", "\u2023": "; ",
    }
    for original, standard in replacements.items():
        text = text.replace(original, standard)
    # Literal backslash-n as well as actual line breaks never reach JSON response.
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = re.sub(r"[\r\n\t\f\v]+", " ", text)
    text = re.sub(r"(?:\*\*|__)(.*?)(?:\*\*|__)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\s*(?:\|+|_{2,}|[-=]{3,})\s*", " ", text)
    # Remove Markdown-only presentation characters, but preserve currency,
    # product names, numbers, punctuation and ordinary hyphens.
    text = re.sub(r"(?<!\w)[*#]+", "", text)
    text = text.replace("*", "").replace("_", "").replace("|", "")
    return re.sub(r"\s+", " ", text).strip()


def _no_offer_safe_fallback(business_context: dict[str, Any]) -> str | None:
    """Dernier recours de disponibilité : verbalise des FAITS déjà calculés.

    Strictement réservé à OFFER_UNAVAILABLE après deux sorties LLM invalides.
    Aucun choix de produit, nouvelle règle d'achat ou procédure n'est effectué.
    Ce traitement évite de transformer une absence normale de produit en HTTP 502.
    """
    purchase = business_context.get("purchase_state") or {}
    rec = business_context.get("recommendation_state") or {}
    if purchase.get("status") != "OFFER_UNAVAILABLE" or rec.get("selected_offer_id"):
        return None
    amount = rec.get("budget")
    currency = rec.get("budget_currency")
    shown_amount = (
        f"{amount:g}" if isinstance(amount, (float, int)) and not isinstance(amount, bool)
        else None
    )
    unit = f" {currency}" if currency in {"USD", "CDF"} else ""
    if rec.get("price_match_mode") == "exact" and shown_amount:
        lead = f"Aucun forfait au prix exact de {shown_amount}{unit} n'est disponible."
    elif shown_amount:
        lead = f"Aucun forfait ne correspond au budget de {shown_amount}{unit}."
    else:
        lead = "Aucun forfait ne correspond à cette demande."
    # Alternatives factuelles uniquement, jamais sélectionnées ni proposées à
    # la confirmation. Ne pas extrapoler si le catalogue compatible est vide.
    alternatives = rec.get("available_offers") or []
    if alternatives:
        details = ", ".join(
            f"{offer['name']} à {float(offer['price_monthly']):g} {offer['currency']}"
            for offer in alternatives
            if offer.get("name") and isinstance(offer.get("price_monthly"), (float, int))
            and offer.get("currency")
        )
        if details:
            lead += f" Offres existantes : {details}."
    return _normalize_plain_text(lead)


def generate_user_response(
    client: Groq,
    *,
    user_message: str,
    business_context: dict[str, Any],
    history: list[dict[str, str]],
    preferred_language: str | None,
    validation_device_catalog: list[dict[str, Any]] | None = None,
) -> str:
    """Génère un texte naturel à partir du contexte métier ; échoue en sécurité."""
    recent_history = history[-10:]
    prompt = (
        "LAST_USER_MESSAGE:\n"
        f"{user_message}\n\n"
        "RECENT_CONVERSATION (language/style only; not a source of new facts):\n"
        f"{json.dumps(recent_history, ensure_ascii=False, default=str)}\n\n"
        "CRM_PREFERRED_LANGUAGE_FALLBACK:\n"
        f"{preferred_language or 'not provided'}\n\n"
        "BUSINESS_CONTEXT (sole factual authority):\n"
        f"{json.dumps(business_context, ensure_ascii=False, default=str)}"
    )
    issues: list[str] = []

    for attempt in range(2):
        instruction = ""
        if attempt:
            instruction = (
                "\n\nTHE PREVIOUS CANDIDATE WAS REJECTED BY BACKEND FACT CHECKS: "
                + ", ".join(issues)
                + ". Regenerate a shorter, strictly grounded answer. "
                "Only verbalize the explicit business result, do not mention "
                "procedures, websites, applications or invented transactions. "
                "Return a single-line plain-text response."
            )

        response = client.chat.completions.create(
            model=COMMUNICATION_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_RESPONSE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt + instruction},
            ],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            issues = ["empty_response"]
            continue

        candidate = _normalize_plain_text(content.strip())
        validation_context = {
            **business_context,
            "_validation_device_catalog": validation_device_catalog or [],
        }
        issues = _validate_recommendation_response(candidate, validation_context)
        if not issues:
            return candidate

    # Une absence d'offre n'est pas une panne. Ne pas exposer une erreur serveur
    # si les deux candidats LLM ont échoué à verbaliser ce résultat ordinaire.
    # Ce recours se limite aux faits STRUCTURÉS déjà calculés par le graphe.
    fallback = _no_offer_safe_fallback(business_context)
    if fallback is not None and not _validate_recommendation_response(
        fallback, {**business_context, "_validation_device_catalog": validation_device_catalog or []}
    ):
        return fallback

    # Pour les autres états, on échoue en sécurité sans inventer de transaction.
    raise ResponseValidationError(
        "Le générateur n'a pas produit de réponse conforme aux faits métier."
    )
