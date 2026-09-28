from langgraph.graph import END, START, StateGraph

from nodes import (
    extract_context_node,
    get_devices_node,
    get_offers_node,
    recommend_offer_node,
    save_user_message_node,
    summary_node,
)
from state import AgentState


def route_intent(state: AgentState):
    """Choisit uniquement le traitement métier de recommandation."""
    intent = state.get("intent")

    if intent == "BUY_PREPAID":
        return "get_offers"

    if intent == "RECOMMEND_DEVICE":
        return "get_devices"

    if intent == "ASK_RECOMMENDATION":
        return "recommend_offer"

    if intent == "SUMMARIZE":
        return "summary"

    return "end"


builder = StateGraph(AgentState)

builder.add_node("save_user_message", save_user_message_node)
builder.add_node("extract_context", extract_context_node)
builder.add_node("get_offers", get_offers_node)
builder.add_node("get_devices", get_devices_node)
builder.add_node("recommend_offer", recommend_offer_node)
builder.add_node("summary", summary_node)

builder.add_edge(START, "save_user_message")
builder.add_edge("save_user_message", "extract_context")

builder.add_conditional_edges(
    "extract_context",
    route_intent,
    {
        "get_offers": "get_offers",
        "get_devices": "get_devices",
        "recommend_offer": "recommend_offer",
        "summary": "summary",
        "end": END,
    },
)

# L'Agent de Recommandation produit uniquement des données métier.
builder.add_edge("get_offers", END)
builder.add_edge("get_devices", END)
builder.add_edge("recommend_offer", END)
builder.add_edge("summary", END)

# IMPORTANT MVP : pas de checkpointer LangGraph ici.
# La mémoire conversationnelle unique appartient à l'orchestrateur
# (communication_agent.py). Le graphe reçoit à chaque appel le contexte de
# recommandation déjà mémorisé par cet orchestrateur.
recommendation_graph = builder.compile()
