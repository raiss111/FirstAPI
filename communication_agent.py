from typing import Any

from groq import Groq

from crm_mock import get_customer_profile_mock
from intent_agent import classify_intent, get_intent_scope
from recommendation_graph import recommendation_graph


# Mémoire conversationnelle visible de l'Agent de Communication.
# Le Recommendation Agent possède sa propre mémoire LangGraph via thread_id.
_COMMUNICATION_HISTORY: dict[str, list[dict[str, str]]] = {}


def _get_history(session_id: str) -> list[dict[str, str]]:
    return _COMMUNICATION_HISTORY.setdefault(session_id, [])


def _safe_recommendation_data(result: dict[str, Any]) -> dict[str, Any]:
    """Garde uniquement les données métier utiles à la communication et aux tests."""
    return {
        "selected_offer_id": result.get("selected_offer_id"),
        "selected_offer_name": result.get("selected_offer_name"),
        "budget": result.get("budget"),
        "max_price": result.get("max_price"),
        "category_preference": result.get("category_preference"),
        "offers": result.get("offers", []),
        "devices": result.get("devices", []),
        "summary": result.get("summary"),
    }


def _build_recommendation_input(
    session_id: str,
    message: str,
    intent: str,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Prépare l'entrée du Recommendation Agent.

    Priorité du contexte :
    1. choix déjà mémorisés dans la session LangGraph ;
    2. informations explicites du nouveau message, extraites ensuite par le graphe ;
    3. CRM uniquement comme valeur par défaut lorsqu'aucun contexte de session n'existe.
    """
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    current_values: dict[str, Any] = {}

    try:
        snapshot = recommendation_graph.get_state(config)
        if snapshot is not None and snapshot.values:
            current_values = dict(snapshot.values)
    except Exception:
        # Aucune mémoire précédente est un cas normal au premier message.
        current_values = {}

    state: dict[str, Any] = {
        "session_id": session_id,
        "message": message,
        "intent": intent,
    }

    preferences = profile.get("preferences", {})

    # Le CRM sert seulement de valeur par défaut. Il ne remplace jamais
    # une valeur déjà mémorisée dans la conversation.
    if current_values.get("budget") is None:
        crm_budget = preferences.get("offer_budget_max")
        if crm_budget is not None:
            state["budget"] = crm_budget

    if current_values.get("max_price") is None:
        crm_max_price = preferences.get("device_budget_max")
        if crm_max_price is not None:
            state["max_price"] = crm_max_price

    if current_values.get("category_preference") is None:
        crm_category = preferences.get("device_category")
        if crm_category:
            state["category_preference"] = crm_category

    # Une offre CRM courante n'est utilisée que comme secours pour une demande
    # d'appareil lorsqu'aucune offre n'a encore été choisie dans la session.
    if (
        intent == "RECOMMEND_DEVICE"
        and current_values.get("selected_offer_id") is None
        and profile.get("current_offer_id")
    ):
        state["selected_offer_id"] = profile.get("current_offer_id")
        state["selected_offer_name"] = profile.get("current_offer_name")

    return state, config


def _find_selected_offer(
    recommendation_data: dict[str, Any],
) -> dict[str, Any] | None:
    selected_offer_id = recommendation_data.get("selected_offer_id")
    selected_offer_name = recommendation_data.get("selected_offer_name")
    offers = recommendation_data.get("offers", [])

    if selected_offer_id:
        offer = next(
            (
                item
                for item in offers
                if item.get("offer_id") == selected_offer_id
            ),
            None,
        )
        if offer is not None:
            return offer

    if selected_offer_name:
        offer = next(
            (
                item
                for item in offers
                if item.get("name") == selected_offer_name
            ),
            None,
        )
        if offer is not None:
            return offer

    return None


def _format_offer(offer: dict[str, Any]) -> str:
    return (
        f"{offer.get('name', 'Offre')} : "
        f"{offer.get('price_monthly', '?')}$ / mois, "
        f"{offer.get('data_gb', '?')} Go, "
        f"appels : {offer.get('calls_min', 'non précisé')}"
    )


def _format_device(device: dict[str, Any]) -> str:
    brand = device.get("brand", "")
    model = device.get("model", "Appareil")
    price = device.get("price")
    reason = device.get("reason")

    label = f"{brand} {model}".strip()

    if price is not None:
        label += f" ({price}$)"

    if reason:
        label += f" — {reason}"

    return label


def _render_user_response(
    *,
    intent: str,
    scope: str,
    recommendation_data: dict[str, Any],
) -> str:
    """
    Transforme les résultats des autres agents en réponse utilisateur.

    Cette fonction ne choisit aucune offre et aucun appareil : elle verbalise uniquement
    les données déjà produites par l'Intent Agent et le Recommendation Agent.
    """
    if scope == "banking":
        return (
            "Votre demande concerne le domaine banque/crédit. "
            "Elle a bien été reconnue par l'Agent Intent, mais le module bancaire "
            "n'est pas connecté à ce MVP de recommandation."
        )

    if scope == "unknown":
        return (
            "Je n'ai pas pu déterminer précisément votre demande. "
            "Pouvez-vous la reformuler ?"
        )

    if intent == "BUY_PREPAID":
        offers = recommendation_data.get("offers", [])

        if not offers:
            return "Aucune offre prépayée ne correspond à vos critères actuels."

        lines = ["Voici les offres prépayées disponibles :"]
        lines.extend(f"- {_format_offer(offer)}" for offer in offers)
        return "\n".join(lines)

    if intent == "ASK_RECOMMENDATION":
        offers = recommendation_data.get("offers", [])
        selected_offer = _find_selected_offer(recommendation_data)
        selected_offer_name = recommendation_data.get("selected_offer_name")

        if selected_offer is None and not selected_offer_name:
            return (
                "Je n'ai pas encore assez d'informations pour recommander une offre. "
                "Précisez votre budget ou vos besoins."
            )

        if selected_offer is not None:
            lines = [
                "Je vous recommande cette offre :",
                f"- {_format_offer(selected_offer)}",
            ]
        else:
            lines = [f"Je vous recommande {selected_offer_name}."]

        # L'alternative est uniquement une autre offre déjà fournie par le
        # Recommendation Agent. Aucun nouveau choix n'est calculé ici.
        selected_id = recommendation_data.get("selected_offer_id")
        alternative = next(
            (
                offer
                for offer in offers
                if offer.get("offer_id") != selected_id
            ),
            None,
        )

        if alternative is not None:
            lines.append(f"Alternative : {_format_offer(alternative)}")

        return "\n".join(lines)

    if intent == "RECOMMEND_DEVICE":
        offer_name = recommendation_data.get("selected_offer_name")
        devices = recommendation_data.get("devices", [])

        if not devices:
            if offer_name:
                return (
                    f"Aucun appareil compatible n'a été trouvé pour l'offre "
                    f"{offer_name} avec vos critères actuels."
                )

            return (
                "Je n'ai pas encore assez de contexte pour recommander un appareil. "
                "Précisez l'offre à utiliser ou choisissez d'abord un forfait."
            )

        # Si les tags existent, on privilégie RECOMMENDED puis ALTERNATIVE.
        primary = next(
            (
                device
                for device in devices
                if device.get("compatibility_tag") == "RECOMMENDED"
            ),
            devices[0],
        )

        alternative = next(
            (
                device
                for device in devices
                if device.get("compatibility_tag") == "ALTERNATIVE"
                and device.get("device_id") != primary.get("device_id")
            ),
            None,
        )

        header = (
            f"Pour l'offre {offer_name}, je vous propose :"
            if offer_name
            else "Voici l'appareil recommandé :"
        )

        lines = [header, f"- Recommandation : {_format_device(primary)}"]

        if alternative is not None:
            lines.append(f"- Alternative : {_format_device(alternative)}")

        return "\n".join(lines)

    if intent == "SUMMARIZE":
        summary = recommendation_data.get("summary")

        if summary:
            return summary

        return "Je n'ai pas encore assez d'éléments pour résumer cette conversation."

    return "Je ne peux pas traiter cette demande pour le moment."


def handle_user_message(
    client: Groq,
    *,
    session_id: str,
    customer_id: str,
    message: str,
) -> dict[str, Any]:
    """
    Point d'orchestration du troisième agent.

    Le Gateway appelle uniquement l'Agent de Communication. Celui-ci :
    1. lit le CRM Mock ;
    2. appelle l'Agent Intent ;
    3. appelle le Recommendation Agent seulement pour le scope recommendation ;
    4. transforme les données obtenues en message utilisateur.

    L'Agent Intent ne répond pas à l'utilisateur.
    Le Recommendation Agent ne répond pas à l'utilisateur.
    """
    profile = get_customer_profile_mock.invoke(
        {
            "customer_id": customer_id,
        }
    )

    history = _get_history(session_id)
    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    # Agent 1 : classification uniquement.
    intent_result = classify_intent(client, message)
    intent = intent_result["intent"]
    scope = get_intent_scope(intent)

    recommendation_called = False
    recommendation_data: dict[str, Any] = {}

    # Agent 2 : recommandation uniquement.
    if scope == "recommendation":
        recommendation_called = True

        state, config = _build_recommendation_input(
            session_id=session_id,
            message=message,
            intent=intent,
            profile=profile,
        )

        result = recommendation_graph.invoke(
            state,
            config=config,
        )

        recommendation_data = _safe_recommendation_data(result)

    # Agent 3 : communication uniquement.
    # La réponse est volontairement construite depuis les données structurées afin
    # d'éviter qu'un LLM réinterprète ASK_RECOMMENDATION comme RECOMMEND_DEVICE.
    response_text = _render_user_response(
        intent=intent,
        scope=scope,
        recommendation_data=recommendation_data,
    )

    history.append(
        {
            "role": "assistant",
            "content": response_text,
        }
    )

    return {
        "session_id": session_id,
        "customer_id": customer_id,
        "intent": intent,
        "scope": scope,
        "response": response_text,
        "data": {
            "communication_agent_called": True,
            "intent_agent_called": True,
            "recommendation_agent_called": recommendation_called,
            "crm_profile": profile,
            "recommendation": recommendation_data,
        },
    }
