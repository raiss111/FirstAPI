from typing import Any

from groq import Groq

from intent_agent import FALLBACK_INTENT, extract_tobi_entities
from tobi_services_mock import (
    check_airtime_balance_mock,
    explain_tariff_plan_mock,
    handover_human_mock,
    topup_airtime_mock,
)


def _missing_tobi_slots(intent: str, entities: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    if intent == "topup_airtime":
        if entities.get("amount") is None:
            missing.append("amount")
        if entities.get("currency") is None:
            missing.append("currency")
        if entities.get("beneficiary_type") is None:
            missing.append("beneficiary_type")
        if (
            entities.get("beneficiary_type") == "other"
            and not entities.get("target_msisdn")
        ):
            missing.append("target_msisdn")

    elif intent == "handover_human":
        if not entities.get("preferred_channel"):
            missing.append("preferred_channel")

    return missing


def _merge_tobi_entities(
    previous: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(previous)

    for key, value in incoming.items():
        if value is not None:
            merged[key] = value

    return merged


def _auth_required_result(intent: str) -> dict[str, Any]:
    return {
        "action": intent,
        "status": "AUTH_REQUIRED",
        "executed": False,
        "mode": "mock",
        "customer_id": None,
        "reason": "Aucun profil client authentifié n'est associé à la conversation.",
    }


def _execute_tobi_action(
    *,
    intent: str,
    customer_id: str | None,
    entities: dict[str, Any],
) -> dict[str, Any]:
    if customer_id is None:
        return _auth_required_result(intent)

    if intent == "topup_airtime":
        return topup_airtime_mock.invoke(
            {
                "customer_id": customer_id,
                "beneficiary_type": entities["beneficiary_type"],
                "target_msisdn": entities.get("target_msisdn"),
                "amount": entities["amount"],
                "currency": entities["currency"],
                "payment_method": entities.get("payment_method", "mpesa"),
            }
        )

    if intent == "check_airtime_balance":
        return check_airtime_balance_mock.invoke({"customer_id": customer_id})

    if intent == "explain_tariff_plan":
        return explain_tariff_plan_mock.invoke(
            {
                "customer_id": customer_id,
                "plan_name": entities.get("plan_name"),
            }
        )

    if intent == "handover_human":
        return handover_human_mock.invoke(
            {
                "customer_id": customer_id,
                "preferred_channel": entities["preferred_channel"],
            }
        )

    return {
        "action": intent,
        "status": "UNSUPPORTED_MOCK_ACTION",
        "mode": "mock",
    }


def process_tobi_message(
    client: Groq,
    *,
    customer_id: str | None,
    classified_intent: str,
    message: str,
    previous_state: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Traite un tour TOBi sans posséder de mémoire propre.

    L'état précédent est fourni par l'orchestrateur et le nouvel état lui est
    retourné. Il n'existe donc plus de dictionnaire global de sessions TOBi.
    """
    previous = dict(previous_state or {})
    pending_intent = None

    if previous and not previous.get("slots_complete", False):
        pending_intent = previous.get("intent")

    effective_intent = classified_intent

    if pending_intent and classified_intent in {FALLBACK_INTENT, pending_intent}:
        effective_intent = pending_intent

    continue_pending = (
        bool(previous)
        and not previous.get("slots_complete", False)
        and previous.get("intent") == effective_intent
    )

    extraction = extract_tobi_entities(
        client=client,
        intent=effective_intent,
        message=message,
        history=history,
    )
    incoming_entities = extraction.get("entities", {})

    if continue_pending:
        entities = _merge_tobi_entities(
            previous.get("entities", {}),
            incoming_entities,
        )
    else:
        entities = dict(incoming_entities)

    missing_slots = _missing_tobi_slots(effective_intent, entities)
    slots_complete = len(missing_slots) == 0

    state: dict[str, Any] = {
        "intent": effective_intent,
        "entities": entities,
        "missing_slots": missing_slots,
        "slots_complete": slots_complete,
        "action_result": None,
    }

    if slots_complete:
        state["action_result"] = _execute_tobi_action(
            intent=effective_intent,
            customer_id=customer_id,
            entities=entities,
        )

    return effective_intent, state
