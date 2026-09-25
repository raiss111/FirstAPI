import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Identité de session et entrée métier
    session_id: str
    message: str
    intent: str
    initial_intent: str

    # Contexte métier de recommandation
    budget: float | None
    requested_data_gb: float | None
    selected_offer_id: str | None
    selected_offer_name: str | None
    category_preference: str | None
    max_price: float | None

    # Résultats structurés des tools
    offers: list[dict[str, Any]]
    devices: list[dict[str, Any]]

    # Mémoire interne utile au contexte de recommandation
    history: Annotated[list[dict[str, str]], operator.add]

    # Résultat de synthèse métier, jamais réponse utilisateur
    summary: str
