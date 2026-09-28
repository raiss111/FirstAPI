from typing import Any

from langchain_core.tools import tool


@tool
def get_prepaid_offers_mock(
    user_budget_max: float | None = None,
) -> list[dict[str, Any]]:
    """
    Retourne les offres prépayées disponibles.

    Si un budget maximal est fourni, retourne uniquement les offres dont
    le prix mensuel est inférieur ou égal à ce budget.
    """
    offers = [
        {
            "offer_id": "PRE_001",
            "name": "Flexi Data Max",
            "price_monthly": 15,
            "currency": "USD",
            "data_gb": 20,
            "calls_min": "Illimité",
        },
        {
            "offer_id": "PRE_002",
            "name": "Eco Prepaid",
            "price_monthly": 8,
            "currency": "USD",
            "data_gb": 5,
            "calls_min": 100,
        },
    ]

    if user_budget_max is not None:
        offers = [
            offer
            for offer in offers
            if offer["price_monthly"] <= user_budget_max
        ]

    return offers


@tool
def get_compatible_devices_mock(
    offer_id: str,
    category_preference: str | None = None,
    max_price: float | None = None,
) -> list[dict[str, Any]]:
    """
    Retourne les appareils compatibles avec une offre.

    Peut filtrer les appareils selon une catégorie préférée et un prix maximal.
    """
    devices = [
        {
            "device_id": "DEV_101",
            "brand": "Samsung",
            "model": "Galaxy A15 5G",
            "price": 180,
            "currency": "USD",
            "category": "SMARTPHONE",
            "compatible_offers": ["PRE_001"],
            "compatibility_tag": "RECOMMENDED",
            "reason": (
                "Meilleur rapport qualité/prix pour le forfait Flexi Data Max"
            ),
        },
        {
            "device_id": "DEV_102",
            "brand": "Xiaomi",
            "model": "Redmi Note 13",
            "price": 150,
            "currency": "USD",
            "category": "SMARTPHONE",
            # Extension de données MOCK pour permettre une recommandation
            # de smartphone avec Eco Prepaid (PRE_002) sans modifier le modèle,
            # le prix ni son référencement initial comme alternative PRE_001.
            # La compatibilité réelle reste à confirmer avant une intégration.
            "compatible_offers": ["PRE_001", "PRE_002"],
            "compatibility_tag": "ALTERNATIVE",
            "reason": "Option économique avec grand écran",
        },
    ]

    compatible_devices = [
        device
        for device in devices
        if offer_id in device["compatible_offers"]
    ]

    if offer_id == "PRE_002":
        # Unique smartphone référencé pour cette offre dans le catalogue mock :
        # statut recommandé propre à PRE_002, sans changer ALTERNATIVE pour PRE_001.
        compatible_devices = [
            {**device, "compatibility_tag": "RECOMMENDED"}
            if device["device_id"] == "DEV_102" else device
            for device in compatible_devices
        ]

    if category_preference is not None:
        compatible_devices = [
            device
            for device in compatible_devices
            if device["category"] == category_preference
        ]

    if max_price is not None:
        compatible_devices = [
            device
            for device in compatible_devices
            if device["price"] <= max_price
        ]

    return compatible_devices


@tool
def generate_conversation_summary(
    history: list[dict[str, str]],
    initial_intent: str | None = None,
    budget: float | None = None,
    selected_offer_name: str | None = None,
    devices: list[dict[str, Any]] | None = None,
) -> str:
    """Génère un résumé structuré de la conversation et des choix connus."""
    if not history:
        return "Aucune conversation à résumer."

    lines = ["Résumé de la conversation :"]

    if initial_intent:
        lines.append(f"- Intention initiale : {initial_intent}")

    if budget is not None:
        lines.append(f"- Budget retenu : {budget}$")

    if selected_offer_name:
        lines.append(f"- Offre retenue : {selected_offer_name}")

    if devices:
        recommended = next(
            (
                device
                for device in devices
                if device.get("compatibility_tag") == "RECOMMENDED"
            ),
            devices[0],
        )
        lines.append(
            "- Appareil recommandé : "
            f"{recommended['brand']} {recommended['model']} "
            f"({recommended['price']}$)"
        )

        alternative = next(
            (
                device
                for device in devices
                if device.get("compatibility_tag") == "ALTERNATIVE"
            ),
            None,
        )
        if alternative:
            lines.append(
                "- Alternative : "
                f"{alternative['brand']} {alternative['model']} "
                f"({alternative['price']}$)"
            )

    user_messages = [
        item.get("content", "")
        for item in history
        if item.get("role") == "user" and item.get("content")
    ]
    if user_messages:
        lines.append(f"- Nombre de demandes utilisateur : {len(user_messages)}")

    return "\n".join(lines)
