import json
import re
from typing import Any

from groq import Groq

MODEL = "openai/gpt-oss-20b"

# -----------------------------------------------------------------------------
# Catalogue d'intentions
# -----------------------------------------------------------------------------
TOBI_INTENTS = (
    "explain_tariff_plan",
    "topup_airtime",
    "check_airtime_balance",
    "handover_human",
)

RECOMMENDATION_INTENTS = (
    "BUY_PREPAID",
    "ASK_RECOMMENDATION",
    "RECOMMEND_DEVICE",
    "SUMMARIZE",
)

FALLBACK_INTENT = "unknown"

INTENTS = TOBI_INTENTS + RECOMMENDATION_INTENTS + (FALLBACK_INTENT,)


INTENT_SYSTEM_PROMPT = """
Tu es l'Agent Intent / NLU de TOBi.

Ton rôle est uniquement de comprendre la demande utilisateur et de retourner
UNE intention parmi la liste autorisée.

Tu dois comprendre la langue de l'utilisateur, quelle qu'elle soit, ainsi que
les formulations courtes ou contextuelles lorsque l'historique est fourni.

Intentions TOBi :
- explain_tariff_plan : expliquer le plan tarifaire actuel ou un plan nommé.
- topup_airtime : acheter/recharger du crédit d'appel (Airtime).
- check_airtime_balance : consulter le solde Airtime.
- handover_human : demander un conseiller humain par chat ou appel.

Intentions de recommandation :
- BUY_PREPAID : rechercher/consulter une offre prépayée OU exprimer une demande
  explicite d'achat de forfait prépayé. La distinction est portée par
  purchase_action et non par une nouvelle intention.
- ASK_RECOMMENDATION : demander quelle OFFRE/FORFAIT est recommandée, une
  précision ou une alternative à une offre discutée précédemment.
- RECOMMEND_DEVICE : demander un TÉLÉPHONE/SMARTPHONE/APPAREIL compatible ou
  adapté à une offre, même si la phrase dit simplement "tu recommandes quoi ?".
  Exemple : "Pour mon forfait de 5 Go, quel smartphone me recommandes-tu ?"
  => RECOMMEND_DEVICE, recommendation_target=device, JAMAIS ASK_RECOMMENDATION.
  À l'inverse, "quel forfait conseilles-tu pour mon smartphone ?"
  => ASK_RECOMMENDATION, recommendation_target=offer.
- SUMMARIZE : demander un résumé de la conversation ou des choix.

Fallback existant :
- unknown : aucune intention ci-dessus ne correspond suffisamment à la demande.

Règles importantes :
- Ne transforme pas un dépôt, retrait ou transfert M-Pesa générique en
  topup_airtime.
- Le message ACTUEL prime sur l'historique : une ancienne demande d'achat
  ne doit pas transformer une nouvelle question sur un téléphone en achat ou
  en recommandation de forfait.
- recommendation_target représente l'OBJET demandé au tour actuel :
  device si l'utilisateur veut un appareil/téléphone, offer s'il veut une
  offre/forfait, none si ni l'un ni l'autre. Comprends-le sémantiquement dans
  toutes les langues ; ne te limite pas à des mots-clés français.
- N'utilise pas unknown lorsqu'un message court peut être compris grâce à
  l'historique fourni.
- Si BUY_PREPAID correspond à une demande explicite d'acheter, mets
  purchase_action=request. Pour une recherche/consultation explicite sans
  volonté d'acheter, mets purchase_action=browse.
- Mets purchase_action=none seulement si la formulation n'indique ni nouvel
  achat, ni consultation, ni confirmation/annulation claire.
- Si PENDING_PURCHASE est actif et le message confirme clairement cet achat
  simulé, mets intent=BUY_PREPAID et purchase_action=confirm.
- Si l'utilisateur annule clairement l'achat en attente, mets
  intent=BUY_PREPAID et purchase_action=cancel.
- Une confirmation sans achat en attente ne doit jamais être transformée
  artificiellement en demande d'achat.
- beneficiary_name est UNIQUEMENT le prénom/nom du bénéficiaire de l'achat
  prépayé explicitement présent dans le message ; sinon null. N'invente rien.
- Ne génère aucune réponse utilisateur, ne recommande rien et n'exécute rien.
- Retourne uniquement l'objet JSON demandé.
""".strip()


ENTITY_SYSTEM_PROMPT = """
Tu es un extracteur d'entités métier pour TOBi.

Comprends la langue de l'utilisateur et le contexte conversationnel fourni.
N'invente aucune valeur qui n'est pas explicitement présente ou clairement
implicite dans le message/contexte.

Entités possibles :
- beneficiary_type : self | other
- target_msisdn : numéro de téléphone fourni par l'utilisateur
- amount : montant numérique
- currency : USD | CDF
- payment_method : mpesa | airtime | card
- preferred_channel : chat | call
- plan_name : nom du plan tarifaire mentionné

Règles :
- Si le message indique clairement que la recharge est pour un tiers, mets
  beneficiary_type=other.
- Si le message indique clairement qu'elle est pour l'utilisateur connecté,
  mets beneficiary_type=self.
- Si ce n'est pas déterminable, laisse beneficiary_type à null.
- Ne déduis pas target_msisdn si aucun numéro n'est fourni.
- Pour topup_airtime, payment_method sera complété par le backend avec mpesa si
  aucune méthode n'est fournie ; ne l'invente donc pas ici.
- Ne génère aucune réponse destinée à l'utilisateur.
""".strip()


# -----------------------------------------------------------------------------
# Helpers généraux
# -----------------------------------------------------------------------------
def _history_to_text(history: list[dict[str, Any]] | None, limit: int = 8) -> str:
    if not history:
        return "Aucun historique pertinent."

    lines: list[str] = []
    for item in history[-limit:]:
        role = str(item.get("role", "unknown"))
        content = str(item.get("content", ""))
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def get_intent_scope(intent: str) -> str:
    if intent in TOBI_INTENTS:
        return "tobi"
    if intent in RECOMMENDATION_INTENTS:
        return "recommendation"
    return "unknown"


# -----------------------------------------------------------------------------
# Classification d'intention via LLM
# -----------------------------------------------------------------------------
def classify_intent(
    client: Groq,
    user_message: str,
    history: list[dict[str, Any]] | None = None,
    pending_purchase: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Comprend l'intention et, pour BUY_PREPAID, le sens transactionnel.

    purchase_action est un signal NLU et ne déclenche jamais une action à lui seul.
    L'orchestrateur exige aussi un achat en attente avant toute confirmation.
    """
    message = (user_message or "").strip()
    if not message:
        return {"intent": FALLBACK_INTENT}

    history_text = _history_to_text(history)
    purchase_context = json.dumps(pending_purchase, ensure_ascii=False) if pending_purchase else "AUCUN"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Historique de conversation :\n"
                    f"{history_text}\n\n"
                    "PENDING_PURCHASE (achat simulé à confirmer, ou AUCUN) :\n"
                    f"{purchase_context}\n\n"
                    "Message actuel :\n"
                    f"{message}"
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "intent_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "intent": {"type": "string", "enum": list(INTENTS)},
                        "purchase_action": {
                            "type": "string",
                            "enum": ["none", "browse", "request", "confirm", "cancel"],
                        },
                        "beneficiary_name": {"type": ["string", "null"]},
                        "recommendation_target": {
                            "type": "string",
                            "enum": ["offer", "device", "none"],
                        },
                    },
                    "required": ["intent", "purchase_action", "beneficiary_name", "recommendation_target"],
                    "additionalProperties": False,
                },
            },
        },
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    intent = data.get("intent", FALLBACK_INTENT)
    if intent not in INTENTS:
        intent = FALLBACK_INTENT

    # Cohérence entre les deux signaux sémantiques renvoyés dans le MÊME appel
    # LLM : l'objet demandé prime sur le verbe générique "recommander". Il ne
    # s'agit pas d'une liste de mots-clés ni d'une nouvelle décision du LLM de
    # réponse. Les intents TOBi, SUMMARIZE et les achats explicites sont intacts.
    target = data.get("recommendation_target")
    if target == "device" and intent in {"ASK_RECOMMENDATION", "unknown"}:
        intent = "RECOMMEND_DEVICE"

    result: dict[str, Any] = {"intent": intent}
    if intent == "BUY_PREPAID":
        action = data.get("purchase_action", "none")
        if action not in {"none", "browse", "request", "confirm", "cancel"}:
            action = "none"
        # Une confirmation hors d'un achat en attente ne vaut jamais achat.
        if action in {"confirm", "cancel"} and not pending_purchase:
            action = "none"
        name = data.get("beneficiary_name")
        name = name.strip()[:100] if isinstance(name, str) and name.strip() else None
        result.update({"purchase_action": action, "beneficiary_name": name})
    return result


# -----------------------------------------------------------------------------
# Extraction / validation déterministe des valeurs structurées
# -----------------------------------------------------------------------------
def _normalize_msisdn(value: str) -> str | None:
    """Normalise un numéro Vodacom local 081/082 vers le format +243."""
    compact = re.sub(r"[\s\-()]", "", value)

    if re.fullmatch(r"0(?:81|82)\d{7}", compact):
        return "+243" + compact[1:]

    if re.fullmatch(r"\+243(?:81|82)\d{7}", compact):
        return compact

    return None


def _extract_target_msisdn(message: str) -> str | None:
    compact_message = re.sub(r"[\s\-()]", "", message)
    candidates = re.findall(
        r"(?:\+243(?:81|82)\d{7}|0(?:81|82)\d{7})",
        compact_message,
    )

    if not candidates:
        return None

    return _normalize_msisdn(candidates[0])


def _extract_amount_and_currency(
    message: str,
) -> tuple[float | int | None, str | None]:
    text = message.lower().replace(",", ".")

    patterns = (
        (r"(?P<amount>\d+(?:\.\d+)?)\s*\$", "USD"),
        (r"(?P<amount>\d+(?:\.\d+)?)\s*(?:usd|dollars?)\b", "USD"),
        (r"(?P<amount>\d+(?:\.\d+)?)\s*(?:cdf|fc|francs?)\b", "CDF"),
    )

    for pattern, currency in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            amount = float(match.group("amount"))
            if amount.is_integer():
                return int(amount), currency
            return amount, currency

    return None, None


def _resolve_beneficiary_type(
    extracted_beneficiary_type: str | None,
    target_msisdn: str | None,
) -> str | None:
    """
    Applique uniquement une règle métier vérifiable.

    La compréhension linguistique de self/other appartient au LLM.
    Si un numéro cible explicite est fourni, on sait en revanche qu'il s'agit
    d'un tiers pour ce workflow.
    """
    if target_msisdn:
        return "other"

    if extracted_beneficiary_type in {"self", "other"}:
        return extracted_beneficiary_type

    return None


# -----------------------------------------------------------------------------
# Extraction sémantique des entités via LLM
# -----------------------------------------------------------------------------
def _extract_entities_with_llm(
    client: Groq,
    intent: str,
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    history_text = _history_to_text(history)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ENTITY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Intent déjà identifié : {intent}\n\n"
                    "Historique de conversation :\n"
                    f"{history_text}\n\n"
                    "Message actuel :\n"
                    f"{message}\n\n"
                    "Retourne exactement un objet JSON avec ces clés :\n"
                    "beneficiary_type, target_msisdn, amount, currency, "
                    "payment_method, preferred_channel, plan_name.\n"
                    "Utilise null pour toute valeur absente ou indéterminable."
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)

    # On ne laisse passer que les clés attendues.
    return {
        "beneficiary_type": data.get("beneficiary_type"),
        "target_msisdn": data.get("target_msisdn"),
        "amount": data.get("amount"),
        "currency": data.get("currency"),
        "payment_method": data.get("payment_method"),
        "preferred_channel": data.get("preferred_channel"),
        "plan_name": data.get("plan_name"),
    }


def extract_tobi_entities(
    client: Groq,
    intent: str,
    message: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Extrait les entités TOBi avec le LLM puis valide/normalise les valeurs
    structurées avec des règles déterministes.
    """
    if intent not in TOBI_INTENTS:
        return {
            "entities": {},
            "missing_slots": [],
            "slots_complete": True,
        }

    llm_entities = _extract_entities_with_llm(
        client=client,
        intent=intent,
        message=message,
        history=history,
    )

    entities: dict[str, Any] = {}

    # Numéro : extraction/validation déterministe prioritaire.
    deterministic_msisdn = _extract_target_msisdn(message)
    llm_msisdn = llm_entities.get("target_msisdn")
    normalized_llm_msisdn = (
        _normalize_msisdn(str(llm_msisdn)) if llm_msisdn else None
    )
    target_msisdn = deterministic_msisdn or normalized_llm_msisdn

    # Montant + devise : extraction déterministe prioritaire si présents.
    deterministic_amount, deterministic_currency = _extract_amount_and_currency(message)

    amount = deterministic_amount
    currency = deterministic_currency

    if amount is None:
        llm_amount = llm_entities.get("amount")
        if isinstance(llm_amount, (int, float)):
            amount = llm_amount
        elif isinstance(llm_amount, str):
            try:
                amount = float(llm_amount.replace(",", "."))
                if amount.is_integer():
                    amount = int(amount)
            except ValueError:
                amount = None

    if currency is None:
        llm_currency = str(llm_entities.get("currency") or "").upper()
        if llm_currency in {"USD", "CDF"}:
            currency = llm_currency

    beneficiary_type = _resolve_beneficiary_type(
        extracted_beneficiary_type=llm_entities.get("beneficiary_type"),
        target_msisdn=target_msisdn,
    )

    payment_method = str(llm_entities.get("payment_method") or "").lower()
    if payment_method not in {"mpesa", "airtime", "card"}:
        payment_method = None

    preferred_channel = str(llm_entities.get("preferred_channel") or "").lower()
    if preferred_channel not in {"chat", "call"}:
        preferred_channel = None

    plan_name = llm_entities.get("plan_name")
    if isinstance(plan_name, str):
        plan_name = plan_name.strip() or None
    else:
        plan_name = None

    # Construction des entities pertinentes.
    if intent == "topup_airtime":
        if beneficiary_type is not None:
            entities["beneficiary_type"] = beneficiary_type
        if target_msisdn is not None:
            entities["target_msisdn"] = target_msisdn
        if amount is not None:
            entities["amount"] = amount
        if currency is not None:
            entities["currency"] = currency

        # Le cahier des charges prévoit mpesa comme valeur par défaut du Top Up.
        entities["payment_method"] = payment_method or "mpesa"

    elif intent == "handover_human":
        if preferred_channel is not None:
            entities["preferred_channel"] = preferred_channel

    elif intent == "explain_tariff_plan":
        if plan_name is not None:
            entities["plan_name"] = plan_name

    # check_airtime_balance n'a pas d'entité obligatoire.

    missing_slots = _get_missing_slots(intent, entities)

    return {
        "entities": entities,
        "missing_slots": missing_slots,
        "slots_complete": len(missing_slots) == 0,
    }


# -----------------------------------------------------------------------------
# Slot filling
# -----------------------------------------------------------------------------
def _get_missing_slots(intent: str, entities: dict[str, Any]) -> list[str]:
    if intent == "topup_airtime":
        required = [
            "beneficiary_type",
            "amount",
            "currency",
            "payment_method",
        ]

        missing = [name for name in required if entities.get(name) in {None, ""}]

        if entities.get("beneficiary_type") == "other" and not entities.get("target_msisdn"):
            missing.append("target_msisdn")

        return missing

    if intent == "handover_human":
        return [] if entities.get("preferred_channel") else ["preferred_channel"]

    # explain_tariff_plan : plan_name optionnel
    # check_airtime_balance : aucune entité requise
    return []


# -----------------------------------------------------------------------------
# Pipeline NLU normalisé
# -----------------------------------------------------------------------------
def analyze_nlu_request(
    client: Groq,
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    classification = classify_intent(
        client=client,
        user_message=query,
        history=history,
    )
    intent = classification["intent"]

    if intent in TOBI_INTENTS:
        extraction = extract_tobi_entities(
            client=client,
            intent=intent,
            message=query,
            history=history,
        )
        entities_dict = extraction["entities"]
        missing_slots = extraction["missing_slots"]
        slots_complete = extraction["slots_complete"]
    else:
        entities_dict = {}
        missing_slots = []
        slots_complete = True

    entities_list = [
        {"entity": name, "value": value}
        for name, value in entities_dict.items()
    ]

    return {
        "query": query,
        "intent": {"name": intent},
        "entities": entities_list,
        "missing_slots": missing_slots,
        "slots_complete": slots_complete,
    }
