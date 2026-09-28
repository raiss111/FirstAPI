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
- If OFFER_UNAVAILABLE: present the real available offers or the absence of a
  match, without inventing a selection. Do not ask for payment/phone details.
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
    if rec.get("closest_device"):
        products.append(rec["closest_device"])
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
        if rec.get("closest_device"):
            products.append(rec["closest_device"])
        if purchase.get("offer"):
            products.append(purchase["offer"])
        allowed_currencies = {
            str(item["currency"]).upper()
            for item in products if item.get("currency")
        }
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

    status = purchase.get("status")
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


def generate_user_response(
    client: Groq,
    *,
    user_message: str,
    business_context: dict[str, Any],
    history: list[dict[str, str]],
    preferred_language: str | None,
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
        issues = _validate_recommendation_response(candidate, business_context)
        if not issues:
            return candidate

    # Pas de texte métier écrit à la main en remplacement du LLM : on refuse
    # simplement de publier une réponse non conforme.
    raise ResponseValidationError(
        "Le générateur n'a pas produit de réponse conforme aux faits métier."
    )
