import json

from groq import Groq

MODEL = "openai/gpt-oss-20b"

# Intents historiques de l'Agent Intent : ils restent disponibles et ne sont
# pas remplacés par le nouveau module de recommandation.
BANKING_INTENTS = (
    "credit_request",
    "credit_information",
    "credit_requirements",
    "credit_repayment",
    "account_information",
    "account_balance",
    "transaction_information",
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
    *BANKING_INTENTS,
    *RECOMMENDATION_INTENTS,
    FALLBACK_INTENT,
)

SYSTEM_PROMPT = """
Tu es l'Agent Intent Recognition général d'un chatbot.

Ta seule mission est de classifier le message utilisateur dans exactement UNE
intention parmi les intentions autorisées ci-dessous.

INTENTIONS BANQUE / CRÉDIT EXISTANTES :
- credit_request : l'utilisateur veut demander, obtenir ou souscrire un crédit.
- credit_information : l'utilisateur demande des informations générales sur le crédit.
- credit_requirements : l'utilisateur demande les conditions, critères, pièces ou
  exigences nécessaires pour obtenir un crédit.
- credit_repayment : l'utilisateur parle du remboursement, des échéances ou du paiement
  d'un crédit.
- account_information : l'utilisateur demande des informations générales sur son compte.
- account_balance : l'utilisateur demande son solde ou le montant disponible sur son compte.
- transaction_information : l'utilisateur demande des informations sur une transaction,
  un mouvement ou l'historique des opérations.

INTENTIONS DU MODULE DE RECOMMANDATION TÉLÉCOM :
- BUY_PREPAID : l'utilisateur veut voir, acheter ou rechercher une offre / un forfait prépayé.
- ASK_RECOMMENDATION : l'utilisateur demande une recommandation générale ou un pack
  combinant une offre prépayée et un appareil.
- RECOMMEND_DEVICE : l'utilisateur demande un téléphone, smartphone, routeur ou appareil
  compatible avec une offre, y compris lorsqu'il dit "avec ça".
- SUMMARIZE : l'utilisateur demande explicitement un résumé de la conversation, des choix
  ou des recommandations précédentes.

AUTRE :
- unknown : aucune des intentions précédentes ne correspond clairement.

Règles de classification :
- Choisis exactement une seule intention parmi cette liste.
- N'invente jamais une autre intention.
- Ne réponds pas au besoin métier : classe uniquement l'intention.
- Les intentions banque / crédit doivent être conservées et distinguées des intentions
  du module de recommandation.
- Si le message demande explicitement un crédit ou son remboursement, privilégie
  l'intention banque / crédit correspondante même si un téléphone est mentionné.
- Si le message demande un appareil ou un pack sans demande de crédit, utilise l'intention
  de recommandation correspondante.
- Un simple choix d'offre comme "Je choisis Flexi Data Max" peut être unknown ; l'Agent de
  Recommandation peut conserver ce contexte lorsqu'il est déjà dans une session de
  recommandation.
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

    if intent in BANKING_INTENTS:
        return "banking"

    return "unknown"


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

    return {"intent": intent}
