import json

from groq import Groq

MODEL = "openai/gpt-oss-20b"

# Intents TOBi définis par le cahier des charges NLU.
# Ils remplacent les anciens intents banque / crédit.
TOBI_INTENTS = (
    "explain_tariff_plan",
    "topup_airtime",
    "check_airtime_balance",
    "handover_human",
)

# Seuls ces intents sont autorisés à entrer dans l'Agent de Recommandation.
RECOMMENDATION_INTENTS = (
    "BUY_PREPAID",
    "ASK_RECOMMENDATION",
    "RECOMMEND_DEVICE",
    "SUMMARIZE",
)

FALLBACK_INTENT = "unknown"

INTENTS = (
    *TOBI_INTENTS,
    *RECOMMENDATION_INTENTS,
    FALLBACK_INTENT,
)

SYSTEM_PROMPT = """
Tu es l'Agent Intent Recognition général d'un chatbot TOBi.

Ta seule mission est de classifier le message utilisateur dans exactement UNE
intention parmi les intentions autorisées ci-dessous.

INTENTIONS TOBi :
- explain_tariff_plan : l'utilisateur veut comprendre son plan tarifaire actuel,
  ses avantages ou ses options.
- topup_airtime : l'utilisateur veut acheter ou recharger du crédit d'appel
  (Airtime), pour lui-même ou pour un tiers.
- check_airtime_balance : l'utilisateur veut consulter son solde Airtime.
- handover_human : l'utilisateur veut être transféré ou parler à un conseiller
  humain, par chat ou par appel.

INTENTIONS DU MODULE DE RECOMMANDATION TÉLÉCOM :
- BUY_PREPAID : l'utilisateur veut voir, acheter ou rechercher une offre / un forfait prépayé.
- ASK_RECOMMENDATION : l'utilisateur demande une recommandation d'offre parmi les
  offres disponibles dans son contexte, ou demande pourquoi une recommandation précédente
  lui a été faite.
- RECOMMEND_DEVICE : l'utilisateur demande un téléphone, smartphone, routeur ou appareil
  compatible avec une offre, y compris lorsqu'il dit "avec ça".
- SUMMARIZE : l'utilisateur demande explicitement un résumé de la conversation, des choix
  ou des recommandations précédentes.

FALLBACK EXISTANT :
- unknown : aucune des intentions précédentes ne correspond clairement.

Règles de classification :
- Choisis exactement une seule intention parmi cette liste.
- N'invente jamais une autre intention.
- Ne réponds pas au besoin métier : classe uniquement l'intention.
- Les intentions TOBi et les intentions du module de recommandation doivent rester distinctes.
- Utilise topup_airtime uniquement si l'utilisateur exprime clairement l'achat ou la recharge de crédit d'appel / Airtime / unités téléphoniques.
- Un dépôt M-Pesa, un retrait, un transfert d'argent ou une demande générique de dépôt n'est PAS un topup Airtime.
- La simple mention de M-Pesa ne suffit jamais à choisir topup_airtime.
- Exemple : "comment faire un dépôt ?" doit rester unknown avec le fallback actuel.
- Si l'utilisateur demande son solde Airtime, utilise check_airtime_balance.
- Si l'utilisateur demande un conseiller humain, utilise handover_human.
- Si l'utilisateur demande une offre prépayée, utilise BUY_PREPAID.
- Si l'utilisateur demande "pourquoi cette recommandation ?", "pourquoi tu me recommandes ça ?"
  ou demande la raison d'une recommandation précédente, utilise ASK_RECOMMENDATION.
- Si l'utilisateur demande quel appareil est compatible avec une offre, utilise RECOMMEND_DEVICE.
- Le fallback avancé du cahier des charges n'est pas encore implémenté : conserve unknown
  pour les demandes non reconnues.
""".strip()

INTENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "intent_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": list(INTENTS),
                }
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
}


def get_intent_scope(intent: str) -> str:
    """Retourne le périmètre fonctionnel correspondant à l'intention."""
    if intent in RECOMMENDATION_INTENTS:
        return "recommendation"

    if intent in TOBI_INTENTS:
        return "tobi"

    return "unknown"



def _apply_intent_guardrails(intent: str, user_message: str) -> str:
    """
    Corrige uniquement les confusions métier évidentes entre Airtime et
    opérations M-Pesa génériques.

    Ce garde-fou ne met PAS en place le fallback avancé du cahier des charges :
    il réutilise simplement notre fallback existant ``unknown`` lorsqu'une
    requête ne correspond pas réellement à ``topup_airtime``.
    """
    if intent != "topup_airtime":
        return intent

    text = user_message.lower()

    airtime_markers = (
        "airtime",
        "crédit d'appel",
        "credit d'appel",
        "crédit appel",
        "credit appel",
        "unités",
        "unites",
        "recharge de crédit",
        "recharge du crédit",
        "recharger du crédit",
        "recharger le crédit",
        "acheter du crédit",
        "acheter des unités",
        "acheter les unités",
    )

    non_airtime_money_actions = (
        "dépôt",
        "depot",
        "retirer",
        "retrait",
        "transfert",
        "transférer",
        "transferer",
        "envoyer de l'argent",
        "envoyer argent",
    )

    has_airtime_marker = any(marker in text for marker in airtime_markers)
    has_non_airtime_action = any(
        marker in text for marker in non_airtime_money_actions
    )

    if has_non_airtime_action and not has_airtime_marker:
        return FALLBACK_INTENT

    return intent

def classify_intent(client: Groq, user_message: str) -> dict[str, str]:
    message = user_message.strip()

    if not message:
        return {"intent": FALLBACK_INTENT}

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
        response_format=INTENT_SCHEMA,
    )

    content = response.choices[0].message.content
    if not content:
        return {"intent": FALLBACK_INTENT}

    result = json.loads(content)
    intent = result.get("intent")

    if intent not in INTENTS:
        return {"intent": FALLBACK_INTENT}

    intent = _apply_intent_guardrails(intent, message)
    return {"intent": intent}

# ---------------------------------------------------------------------------
# Extraction d'entités TOBi - étape 2 du cahier des charges.
# Cette logique est volontairement séparée du fallback avancé et des scores de
# confiance, qui seront traités dans une étape dédiée.
# ---------------------------------------------------------------------------

import re
from typing import Any


TOBI_ENTITY_NAMES = (
    "beneficiary_type",
    "target_msisdn",
    "amount",
    "currency",
    "payment_method",
    "preferred_channel",
    "plan_name",
)


def _normalize_msisdn(value: str) -> str | None:
    """Normalise un numéro Vodacom local 081/082 vers le format +243."""
    compact = re.sub(r"[\s\-()]", "", value)

    if re.fullmatch(r"0(?:81|82)\d{7}", compact):
        return "+243" + compact[1:]

    if re.fullmatch(r"\+243(?:81|82)\d{7}", compact):
        return compact

    return None


def _extract_target_msisdn(message: str) -> str | None:
    candidates = re.findall(
        r"(?:\+243(?:81|82)\d{7}|0(?:81|82)\d{7})",
        re.sub(r"[\s\-()]", "", message),
    )

    if not candidates:
        return None

    return _normalize_msisdn(candidates[0])


def _extract_amount_and_currency(message: str) -> tuple[float | int | None, str | None]:
    text = message.lower().replace(",", ".")

    # Exemples couverts : 5$, 5 USD, 5000 CDF, 5000 francs.
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




def _extract_beneficiary_type(message: str, target_msisdn: str | None) -> str:
    """Déduit si la recharge est pour soi-même ou pour un tiers."""
    text = message.lower()

    other_markers = (
        "pour quelqu'un",
        "pour quelqu’un",
        "pour un tiers",
        "pour une autre personne",
        "pour un autre numéro",
        "pour un autre numero",
        "pour mon frère",
        "pour mon frere",
        "pour ma soeur",
        "pour ma mère",
        "pour ma mere",
        "pour mon père",
        "pour mon pere",
        "pour mon ami",
        "pour mon amie",
    )

    self_markers = (
        "pour moi",
        "pour moi-même",
        "pour moi meme",
        "sur mon numéro",
        "sur mon numero",
        "ma ligne",
    )

    if target_msisdn or any(marker in text for marker in other_markers):
        return "other"

    if any(marker in text for marker in self_markers):
        return "self"

    # Pour le Top Up, l'absence d'indication contraire signifie la ligne connectée.
    return "self"


def _extract_payment_method(message: str) -> str | None:
    text = message.lower()

    if "m-pesa" in text or "mpesa" in text or "m pesa" in text:
        return "mpesa"

    if "carte" in text or "card" in text:
        return "card"

    if "airtime" in text or "crédit d'appel" in text or "credit d'appel" in text:
        return "airtime"

    return None


def _extract_preferred_channel(message: str) -> str | None:
    text = message.lower()

    if re.search(r"\bchat\b", text):
        return "chat"

    if re.search(r"\b(?:call|appel|appeler|téléphone|telephone)\b", text):
        return "call"

    return None


def _extract_plan_name(message: str) -> str | None:
    """Extraction prudente d'un nom de plan explicite ; le champ reste optionnel."""
    patterns = (
        r"(?:plan|offre|forfait)\s+(?:tarifaire\s+)?[\"']?([A-Za-z0-9][A-Za-z0-9 _\-]{1,40})[\"']?",
        r"[\"']([^\"']{2,40})[\"']\s+(?:plan|offre|forfait)",
    )

    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .?!,;:")
            # Évite de prendre une phrase générique entière comme nom de plan.
            if 1 <= len(value.split()) <= 5:
                return value

    return None


def extract_tobi_entities(intent: str, user_message: str) -> dict[str, Any]:
    """
    Extrait les entités métier du cahier des charges pour une intention TOBi.

    Retourne :
    - entities : dictionnaire normalisé des valeurs trouvées/déduites ;
    - slots_complete : indique si les données nécessaires au workflow sont présentes.

    Cette étape ne gère volontairement ni les scores de confiance ni le fallback
    avancé du cahier des charges.
    """
    message = user_message.strip()
    entities: dict[str, Any] = {}

    if intent == "topup_airtime":
        target_msisdn = _extract_target_msisdn(message)
        amount, currency = _extract_amount_and_currency(message)
        payment_method = _extract_payment_method(message) or "mpesa"

        beneficiary_type = _extract_beneficiary_type(message, target_msisdn)

        entities["beneficiary_type"] = beneficiary_type

        if target_msisdn:
            entities["target_msisdn"] = target_msisdn

        if amount is not None:
            entities["amount"] = amount

        if currency is not None:
            entities["currency"] = currency

        entities["payment_method"] = payment_method

        required_complete = amount is not None and currency is not None
        if beneficiary_type == "other":
            required_complete = required_complete and target_msisdn is not None

        return {
            "entities": entities,
            "slots_complete": required_complete,
        }

    if intent == "handover_human":
        preferred_channel = _extract_preferred_channel(message)

        if preferred_channel:
            entities["preferred_channel"] = preferred_channel

        return {
            "entities": entities,
            "slots_complete": preferred_channel is not None,
        }

    if intent == "explain_tariff_plan":
        plan_name = _extract_plan_name(message)
        if plan_name:
            entities["plan_name"] = plan_name

        # plan_name est optionnel dans le cahier des charges.
        return {
            "entities": entities,
            "slots_complete": True,
        }

    if intent == "check_airtime_balance":
        # Aucune entité requise par le cahier des charges.
        return {
            "entities": {},
            "slots_complete": True,
        }

    # Les intents de recommandation et notre fallback existant ne sont pas
    # modifiés par cette étape.
    return {
        "entities": {},
        "slots_complete": True,
    }


def analyze_nlu_request(client: Groq, user_message: str) -> dict[str, Any]:
    """
    Exécute le pipeline NLU actuel :
    1. classification de l'intention ;
    2. extraction des entités TOBi ;
    3. calcul de slots_complete ;
    4. normalisation dans une structure JSON proche du contrat TOBi.

    Les scores de confiance et le fallback avancé ne sont volontairement
    pas encore gérés à cette étape.
    """
    query = user_message.strip()
    classification = classify_intent(client, query)
    intent = classification["intent"]

    extraction = extract_tobi_entities(intent, query)

    entities = [
        {
            "entity": entity_name,
            "value": value,
        }
        for entity_name, value in extraction["entities"].items()
    ]

    return {
        "query": query,
        "intent": {
            "name": intent,
        },
        "entities": entities,
        "slots_complete": extraction["slots_complete"],
    }
