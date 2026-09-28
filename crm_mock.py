import re
from typing import Any

from langchain_core.tools import tool


CRM_PROFILES: dict[str, dict[str, Any]] = {
    "CUST_001": {
        "customer_id": "CUST_001",
        "first_name": "Amina",
        "segment": "PREPAID",
        "preferred_language": "fr",
        "current_offer_id": "PRE_001",
        "current_offer_name": "Flexi Data Max",
        "category_device": "SMARTPHONE",
        "phone_number": None,
        "profile_status": "REGISTERED",
    },
    "CUST_002": {
        "customer_id": "CUST_002",
        "first_name": "Patrick",
        "segment": "PREPAID",
        "preferred_language": "fr",
        "current_offer_id": "PRE_002",
        "current_offer_name": "Eco Prepaid",
        "category_device": "SMARTPHONE",
        "phone_number": "+243812345679",
        "profile_status": "REGISTERED",
    },
}


def _anonymous_profile() -> dict[str, Any]:
    """Profil temporaire : aucune identité réelle n'est inventée."""
    return {
        "customer_id": None,
        "first_name": None,
        "segment": "UNKNOWN",
        "preferred_language": None,
        "current_offer_id": None,
        "current_offer_name": None,
        "category_device": None,
        "phone_number": None,
        "profile_status": "ANONYMOUS",
    }


def _unknown_profile(customer_id: str) -> dict[str, Any]:
    """Profil minimal quand un identifiant fourni n'existe pas dans le Mock."""
    return {
        "customer_id": customer_id,
        "first_name": None,
        "segment": "UNKNOWN",
        "preferred_language": None,
        "current_offer_id": None,
        "current_offer_name": None,
        "category_device": None,
        "phone_number": None,
        "profile_status": "UNKNOWN",
    }


def _next_customer_id() -> str:
    numeric_ids: list[int] = []

    for customer_id in CRM_PROFILES:
        match = re.fullmatch(r"CUST_(\d+)", customer_id)
        if match:
            numeric_ids.append(int(match.group(1)))

    next_number = max(numeric_ids, default=0) + 1
    return f"CUST_{next_number:03d}"


def _normalize_phone_number(value: str | None) -> str | None:
    if value is None:
        return None

    compact = re.sub(r"[\s\-()]", "", value)

    if re.fullmatch(r"0(?:81|82)\d{7}", compact):
        return "+243" + compact[1:]

    if re.fullmatch(r"\+243(?:81|82)\d{7}", compact):
        return compact

    raise ValueError(
        "Le numéro doit être un numéro Vodacom au format 081/082... ou +24381/+24382..."
    )


@tool
def get_customer_profile_mock(customer_id: str | None = None) -> dict[str, Any]:
    """Retourne un profil CRM simulé sans attribuer de client par défaut."""
    if not customer_id:
        return _anonymous_profile()

    profile = CRM_PROFILES.get(customer_id)
    if profile is not None:
        return dict(profile)

    return _unknown_profile(customer_id)


@tool
def create_customer_profile_mock(
    first_name: str,
    segment: str = "PREPAID",
    preferred_language: str | None = None,
    current_offer_id: str | None = None,
    current_offer_name: str | None = None,
    category_device: str | None = None,
    phone_number: str | None = None,
) -> dict[str, Any]:
    """Crée explicitement un nouveau profil dans le CRM Mock en mémoire.

    Cette fonction n'est pas appelée automatiquement pendant un simple chat.
    La création d'un profil doit être une action explicite du backend / parcours
    d'inscription.
    """
    normalized_phone = _normalize_phone_number(phone_number)

    if normalized_phone is not None:
        for existing in CRM_PROFILES.values():
            if existing.get("phone_number") == normalized_phone:
                raise ValueError("Un profil existe déjà avec ce numéro de téléphone.")

    customer_id = _next_customer_id()

    profile = {
        "customer_id": customer_id,
        "first_name": first_name.strip(),
        "segment": segment.strip().upper() if segment else "PREPAID",
        "preferred_language": (
            preferred_language.strip().lower() if preferred_language else None
        ),
        "current_offer_id": current_offer_id,
        "current_offer_name": current_offer_name,
        "category_device": (
            category_device.strip().upper() if category_device else None
        ),
        "phone_number": normalized_phone,
        "profile_status": "REGISTERED",
    }

    CRM_PROFILES[customer_id] = profile
    return dict(profile)
