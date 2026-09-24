from typing import Any

from langchain_core.tools import tool


CRM_PROFILES: dict[str, dict[str, Any]] = {
    "CUST_001": {
        "customer_id": "CUST_001",
        "first_name": "chers client",
        "segment": "PREPAID",
        "preferred_language": "fr",
        "current_offer_id": "PRE_001",
        "current_offer_name": "Flexi Data Max",
        "preferences": {
            "offer_budget_max": 20,
            "device_budget_max": 200,
            "device_category": "SMARTPHONE",
        },
    },
    "CUST_002": {
        "customer_id": "CUST_002",
        "first_name": "Patrick",
        "segment": "PREPAID",
        "preferred_language": "fr",
        "current_offer_id": "PRE_002",
        "current_offer_name": "Eco Prepaid",
        "preferences": {
            "offer_budget_max": 10,
            "device_budget_max": 160,
            "device_category": "SMARTPHONE",
        },
    },
}


@tool
def get_customer_profile_mock(customer_id: str) -> dict[str, Any]:
    """Retourne un profil client simulé depuis le CRM Mock."""
    profile = CRM_PROFILES.get(customer_id)

    if profile is not None:
        return profile

    # Profil minimal pour garder le MVP utilisable avec un identifiant inconnu.
    return {
        "customer_id": customer_id,
        "first_name": None,
        "segment": "UNKNOWN",
        "preferred_language": "fr",
        "current_offer_id": None,
        "current_offer_name": None,
        "preferences": {
            "offer_budget_max": None,
            "device_budget_max": None,
            "device_category": None,
        },
    }
