from copy import deepcopy
from functools import wraps
from threading import RLock
from typing import Any

from groq import Groq

from crm_mock import get_customer_profile_mock
from intent_agent import classify_intent, get_intent_scope
from llm_response_generator import generate_user_response
from recommendation_graph import recommendation_graph
from tobi_session_manager import process_tobi_message
from tobi_services_mock import purchase_prepaid_mock


# Les routes FastAPI synchrones peuvent tourner dans plusieurs threads.
# On sérialise l'accès à l'unique état de conversation du MVP.
_CONVERSATION_LOCK = RLock()


def _serialized_turn(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        with _CONVERSATION_LOCK:
            return func(*args, **kwargs)
    return wrapped


class ConversationCustomerMismatchError(ValueError):
    """Un autre profil essaie d'entrer dans la conversation MVP déjà active."""


# -----------------------------------------------------------------------------
# UNE SEULE CONVERSATION ACTIVE POUR LE MVP
# -----------------------------------------------------------------------------
# Cet objet est créé au chargement de l'application et vit en RAM jusqu'au
# prochain redémarrage du processus FastAPI. Il constitue la seule source de
# vérité conversationnelle de l'orchestrateur.
_CONVERSATION_STATE: dict[str, Any] = {
    "customer_id": None,
    "profile": {},
    "history": [],
    "last_intent": None,
    "current_intent": None,
    "current_scope": "unknown",
    "recommendation": {},
    "tobi": {},
    "purchase": {},  # état d'achat simulé dans LA mémoire centrale
}


@_serialized_turn
def get_conversation_state() -> dict[str, Any]:
    """Retourne une copie du State central, utile au debug/tests du MVP."""
    return deepcopy(_CONVERSATION_STATE)


def _safe_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "customer_id": profile.get("customer_id"),
        "first_name": profile.get("first_name"),
        "segment": profile.get("segment"),
        "preferred_language": profile.get("preferred_language"),
        "current_offer_id": profile.get("current_offer_id"),
        "current_offer_name": profile.get("current_offer_name"),
        "category_device": profile.get("category_device"),
        "phone_number": profile.get("phone_number"),
        "profile_status": profile.get("profile_status"),
    }


def _safe_recommendation_data(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "initial_intent": result.get("initial_intent"),
        "selected_offer_id": result.get("selected_offer_id"),
        "selected_offer_name": result.get("selected_offer_name"),
        "selection_basis": result.get("selection_basis"),
        "budget": result.get("budget"),
        "requested_data_gb": result.get("requested_data_gb"),
        "max_price": result.get("max_price"),
        "category_preference": result.get("category_preference"),
        "offers": result.get("offers", []),
        "devices": result.get("devices", []),
        "closest_device": result.get("closest_device"),
        "device_budget_gap": result.get("device_budget_gap"),
        "summary": result.get("summary"),
    }


def _resolve_customer(customer_id: str | None) -> str | None:
    """Associe au maximum un profil client à la conversation active."""
    active_customer_id = _CONVERSATION_STATE.get("customer_id")

    if customer_id:
        if active_customer_id and active_customer_id != customer_id:
            raise ConversationCustomerMismatchError(
                "Cette conversation MVP est déjà associée à un autre client. "
                "Redémarre l'application pour commencer une nouvelle conversation."
            )

        if active_customer_id is None:
            _CONVERSATION_STATE["customer_id"] = customer_id
            active_customer_id = customer_id

    return active_customer_id


@_serialized_turn
def bind_current_conversation_to_customer(customer_id: str) -> None:
    """Lie une conversation anonyme au nouveau profil CRM sans perdre l'historique."""
    active_customer_id = _CONVERSATION_STATE.get("customer_id")

    if active_customer_id and active_customer_id != customer_id:
        raise ConversationCustomerMismatchError(
            "La conversation active est déjà associée à un autre client."
        )

    _CONVERSATION_STATE["customer_id"] = customer_id


def _build_recommendation_input(
    *,
    message: str,
    intent: str,
    profile: dict[str, Any],
    history: list[dict[str, str]],
) -> dict[str, Any]:
    """Construit l'entrée du Recommendation Agent depuis le State central."""
    previous = dict(_CONVERSATION_STATE.get("recommendation") or {})

    # Le graphe est volontairement sans mémoire propre. On lui redonne les faits
    # de recommandation déjà mémorisés par l'orchestrateur.
    state: dict[str, Any] = {
        **previous,
        "message": message,
        "intent": intent,
        "history": list(history),
    }

    if state.get("category_preference") is None and profile.get("category_device"):
        state["category_preference"] = profile["category_device"]

    if (
        intent == "RECOMMEND_DEVICE"
        and state.get("selected_offer_id") is None
        and profile.get("current_offer_id")
    ):
        state["selected_offer_id"] = profile.get("current_offer_id")
        state["selected_offer_name"] = profile.get("current_offer_name")

    return state


@_serialized_turn
def handle_user_message(
    client: Groq,
    *,
    message: str,
    customer_id: str | None = None,
) -> dict[str, Any]:
    """Orchestre UN tour dans l'unique conversation active du MVP.

    La fonction ne crée aucune session. La conversation existe tant que le
    processus FastAPI reste démarré et elle est réinitialisée au redémarrage.
    """
    active_customer_id = _resolve_customer(customer_id)
    profile = get_customer_profile_mock.invoke({"customer_id": active_customer_id})
    safe_profile = _safe_profile(profile)
    _CONVERSATION_STATE["profile"] = safe_profile

    history: list[dict[str, str]] = _CONVERSATION_STATE["history"]
    previous_history = list(history)

    pending = dict(_CONVERSATION_STATE.get("purchase") or {})
    pending_for_nlu = None
    if pending.get("status") == "AWAITING_CONFIRMATION":
        pending_for_nlu = {
            "offer": pending.get("offer"),
            "beneficiary_name": pending.get("beneficiary_name"),
        }

    intent_result = classify_intent(
        client=client,
        user_message=message,
        history=previous_history,
        pending_purchase=pending_for_nlu,
    )
    classified_intent = intent_result["intent"]
    scope = get_intent_scope(classified_intent)
    purchase_action = intent_result.get("purchase_action", "none")

    recommendation_called = False
    tobi_service_called = False
    effective_intent = classified_intent

    recommendation_data = dict(_CONVERSATION_STATE.get("recommendation") or {})
    tobi_data = dict(_CONVERSATION_STATE.get("tobi") or {})
    purchase_data = dict(_CONVERSATION_STATE.get("purchase") or {})
    purchase_touched = bool(pending_for_nlu)

    # Seule une confirmation reconnue par l'Intent Agent ET un achat en attente
    # autorisent l'exécution. On utilise l'offre figée lors de la demande.
    if pending_for_nlu and classified_intent == "BUY_PREPAID" and purchase_action == "confirm":
        frozen_offer = pending["offer"]
        result = purchase_prepaid_mock.invoke({
            "offer_id": frozen_offer["offer_id"],
            "confirmed": True,
            "beneficiary_name": pending.get("beneficiary_name"),
            "customer_id": active_customer_id,
        })
        purchase_data = {
            **pending,
            "status": "COMPLETED" if result.get("status") == "SIMULATED_SUCCESS" else "FAILED",
            "action_result": result,
        }
        _CONVERSATION_STATE["purchase"] = purchase_data
        tobi_service_called = True

    elif pending_for_nlu and classified_intent == "BUY_PREPAID" and purchase_action == "cancel":
        purchase_data = {**pending, "status": "CANCELLED", "action_result": None}
        _CONVERSATION_STATE["purchase"] = purchase_data

    elif pending_for_nlu and classified_intent in {"BUY_PREPAID", "unknown"} and purchase_action == "none":
        # Réponse incertaine : rien n'est acheté, l'attente reste ouverte.
        purchase_data = pending

    else:
        # Nouvelle demande : on ne peut pas confirmer un ancien achat ensuite.
        if pending_for_nlu:
            purchase_data = {**pending, "status": "SUPERSEDED", "action_result": None}
            _CONVERSATION_STATE["purchase"] = purchase_data

        if scope == "recommendation":
            recommendation_called = True
            recommendation_input = _build_recommendation_input(
                message=message,
                intent=classified_intent,
                profile=profile,
                history=previous_history,
            )
            result = recommendation_graph.invoke(recommendation_input)
            recommendation_data = _safe_recommendation_data(result)
            _CONVERSATION_STATE["recommendation"] = recommendation_data

            if classified_intent == "BUY_PREPAID" and purchase_action == "request":
                selected = next(
                    (offer for offer in recommendation_data.get("offers", [])
                     if offer.get("offer_id") == recommendation_data.get("selected_offer_id")),
                    None,
                )
                # Pas de sélection = aucune transaction possible. On ne demande
                # jamais au générateur LLM de choisir l'offre à la place du graphe.
                purchase_data = {
                    "status": "AWAITING_CONFIRMATION" if selected else "OFFER_UNAVAILABLE",
                    "offer": deepcopy(selected) if selected else None,
                    "beneficiary_name": intent_result.get("beneficiary_name"),
                    "action_result": None,
                }
                _CONVERSATION_STATE["purchase"] = purchase_data
                purchase_touched = True

        elif scope == "tobi":
            effective_intent, tobi_data = process_tobi_message(
                client,
                customer_id=active_customer_id,
                classified_intent=classified_intent,
                message=message,
                previous_state=tobi_data,
                history=previous_history,
            )
            _CONVERSATION_STATE["tobi"] = tobi_data
            tobi_service_called = tobi_data.get("action_result") is not None

    # Présentation d'un résultat déjà choisi par le Recommendation Agent :
    # l'orchestrateur ne sélectionne ni ne classe lui-même les produits.
    selected_offer = next(
        (
            offer
            for offer in recommendation_data.get("offers", [])
            if offer.get("offer_id") == recommendation_data.get("selected_offer_id")
        ),
        None,
    )

    # Ne pas répéter un ancien succès d'achat comme si c'était le résultat
    # du message courant. Le dernier achat demeure dans la mémoire centrale.
    purchase_for_response = (
        purchase_data if purchase_touched and purchase_data.get("status") in {
            "AWAITING_CONFIRMATION", "COMPLETED", "CANCELLED", "FAILED", "OFFER_UNAVAILABLE"
        } else {}
    )
    if purchase_for_response.get("offer") and purchase_for_response.get("status") in {
        "AWAITING_CONFIRMATION", "COMPLETED", "CANCELLED", "FAILED",
    }:
        selected_offer = purchase_for_response["offer"]

    business_context = {
        "current_request": {
            "intent": effective_intent,
            "scope": scope,
        },
        "capabilities": {
            "recommendation": {
                "can_browse_prepaid_offers": True,
                "can_recommend_offers": True,
                "can_recommend_devices": True,
                "can_execute_prepaid_purchase": False,
            },
            "tobi": {
                "can_execute_mock_actions": True,
                "can_execute_mock_prepaid_purchase_after_confirmation": True,
                "can_execute_real_purchase": False,
            },
        },
        "crm_profile": safe_profile,
        "recommendation_state": recommendation_data,
        "selected_offer": selected_offer,
        "purchase_state": purchase_for_response,
        "response_constraints": {
            "purchase_instructions_available": False,
            "do_not_mention_purchase_procedure": True,
            "do_not_request_checkout_details": True,
            "one_line_plain_text": True,
            "mock_only": True,
        },
        "tobi_state": tobi_data,
    }

    response_text = generate_user_response(
        client,
        user_message=message,
        business_context=business_context,
        history=previous_history,
        preferred_language=profile.get("preferred_language"),
    )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response_text})

    _CONVERSATION_STATE["last_intent"] = _CONVERSATION_STATE.get("current_intent")
    _CONVERSATION_STATE["current_intent"] = effective_intent
    _CONVERSATION_STATE["current_scope"] = scope

    return {
        "customer_id": safe_profile.get("customer_id"),
        "profile_status": safe_profile.get("profile_status", "UNKNOWN"),
        "intent": effective_intent,
        "scope": scope,
        "response": response_text,
        "data": {
            "communication_agent_called": True,
            "intent_agent_called": True,
            "recommendation_agent_called": recommendation_called,
            "tobi_service_called": tobi_service_called,
            "crm_profile": safe_profile,
            "recommendation": recommendation_data,
            "tobi": tobi_data,
            "purchase": purchase_data,
        },
    }
