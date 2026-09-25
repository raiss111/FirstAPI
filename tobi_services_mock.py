from typing import Any

from langchain_core.tools import tool


# Données purement simulées pour le MVP.
# Elles ne proviennent d'aucune API Vodacom/M-Pesa réelle.
TOBI_CUSTOMER_DATA: dict[str, dict[str, Any]] = {
    "CUST_001": {
        "airtime_balance": {
            "amount": 7.5,
            "currency": "USD",
        },
        "tariff_plan": {
            "plan_name": "Flexi Data Max",
            "price_monthly": 15,
            "currency": "USD",
            "data_gb": 20,
            "calls": "Illimité",
        },
    },
    "CUST_002": {
        "airtime_balance": {
            "amount": 3500,
            "currency": "CDF",
        },
        "tariff_plan": {
            "plan_name": "Eco Prepaid",
            "price_monthly": 8,
            "currency": "USD",
            "data_gb": 5,
            "calls": "100 minutes",
        },
    },
}


@tool
def topup_airtime_mock(
    customer_id: str,
    beneficiary_type: str,
    amount: float,
    currency: str,
    payment_method: str = "mpesa",
    target_msisdn: str | None = None,
) -> dict[str, Any]:
    """Simule la préparation d'une recharge Airtime sans exécuter de transaction réelle."""
    if beneficiary_type == "other" and not target_msisdn:
        return {
            "action": "topup_airtime",
            "status": "MISSING_TARGET_MSISDN",
            "executed": False,
            "mode": "mock",
        }

    return {
        "action": "topup_airtime",
        "status": "SIMULATED_SUCCESS",
        "executed": False,
        "mode": "mock",
        "customer_id": customer_id,
        "beneficiary_type": beneficiary_type,
        "target_msisdn": target_msisdn,
        "amount": amount,
        "currency": currency,
        "payment_method": payment_method,
    }


@tool
def check_airtime_balance_mock(customer_id: str) -> dict[str, Any]:
    """Retourne un solde Airtime simulé pour le MVP."""
    customer_data = TOBI_CUSTOMER_DATA.get(customer_id)

    if customer_data is None:
        return {
            "action": "check_airtime_balance",
            "status": "MOCK_PROFILE_NOT_FOUND",
            "mode": "mock",
            "customer_id": customer_id,
            "balance": None,
        }

    return {
        "action": "check_airtime_balance",
        "status": "SIMULATED_SUCCESS",
        "mode": "mock",
        "customer_id": customer_id,
        "balance": dict(customer_data["airtime_balance"]),
    }


@tool
def explain_tariff_plan_mock(
    customer_id: str,
    plan_name: str | None = None,
) -> dict[str, Any]:
    """Retourne les informations simulées du plan tarifaire du client."""
    customer_data = TOBI_CUSTOMER_DATA.get(customer_id)

    if customer_data is None:
        return {
            "action": "explain_tariff_plan",
            "status": "MOCK_PROFILE_NOT_FOUND",
            "mode": "mock",
            "customer_id": customer_id,
            "plan": None,
        }

    plan = dict(customer_data["tariff_plan"])

    # plan_name est optionnel dans le cahier des charges. Dans ce Mock,
    # on conserve le plan rattaché au profil et on expose le nom demandé
    # uniquement pour rendre le test observable.
    return {
        "action": "explain_tariff_plan",
        "status": "SIMULATED_SUCCESS",
        "mode": "mock",
        "customer_id": customer_id,
        "requested_plan_name": plan_name,
        "plan": plan,
    }


@tool
def handover_human_mock(
    customer_id: str,
    preferred_channel: str,
) -> dict[str, Any]:
    """Simule une demande de transfert vers un conseiller humain."""
    return {
        "action": "handover_human",
        "status": "SIMULATED_QUEUED",
        "executed": False,
        "mode": "mock",
        "customer_id": customer_id,
        "preferred_channel": preferred_channel,
    }
