from typing import Any

from groq import Groq

from crm_mock import get_customer_profile_mock
from intent_agent import (
    FALLBACK_INTENT,
    classify_intent,
    extract_tobi_entities,
    get_intent_scope,
)
from recommendation_graph import recommendation_graph
from tobi_services_mock import (
    check_airtime_balance_mock,
    explain_tariff_plan_mock,
    handover_human_mock,
    topup_airtime_mock,
)


# Mémoire conversationnelle visible de l'Agent de Communication.
_COMMUNICATION_HISTORY: dict[str, list[dict[str, str]]] = {}

# Mémoire métier TOBi pour le slot filling multi-tour.
# Elle reste séparée de la mémoire LangGraph du Recommendation Agent.
_TOBI_SESSION_STATE: dict[str, dict[str, Any]] = {}


def _get_history(session_id: str) -> list[dict[str, str]]:
    return _COMMUNICATION_HISTORY.setdefault(session_id, [])


def _safe_recommendation_data(result: dict[str, Any]) -> dict[str, Any]:
    """Garde uniquement les données métier utiles à la communication et aux tests."""
    return {
        "selected_offer_id": result.get("selected_offer_id"),
        "selected_offer_name": result.get("selected_offer_name"),
        "budget": result.get("budget"),
        "requested_data_gb": result.get("requested_data_gb"),
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
        current_values = {}

    state: dict[str, Any] = {
        "session_id": session_id,
        "message": message,
        "intent": intent,
    }

    preferences = profile.get("preferences", {})

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


def _get_recommendation_snapshot(session_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = recommendation_graph.get_state(config)
        if snapshot is not None and snapshot.values:
            return _safe_recommendation_data(dict(snapshot.values))
    except Exception:
        pass

    return {}


def _is_recommendation_explanation_request(message: str) -> bool:
    text = message.lower().replace("’", "'")
    markers = (
        "pourquoi",
        "raison de cette recommandation",
        "raison de ta recommandation",
        "raison de votre recommandation",
        "pour quelle raison",
        "qu'est-ce qui justifie",
        "qu'est ce qui justifie",
    )
    return any(marker in text for marker in markers)


def _explain_current_recommendation(
    recommendation_data: dict[str, Any],
) -> str:
    devices = recommendation_data.get("devices", [])
    if devices:
        primary = next(
            (d for d in devices if d.get("compatibility_tag") == "RECOMMENDED"),
            devices[0],
        )
        reason = primary.get("reason")
        if reason:
            return (
                f"Je vous ai recommandé {_format_device(primary)} parce que {reason.lower()}."
            )
        return f"Je vous ai recommandé {_format_device(primary)} car il correspond aux critères enregistrés dans votre contexte."

    selected_offer = _find_selected_offer(recommendation_data)
    if selected_offer is None:
        return (
            "Je n'ai pas encore de recommandation active à expliquer dans cette conversation. "
            "Demandez-moi d'abord une recommandation d'offre ou d'appareil."
        )

    budget = recommendation_data.get("budget")
    requested_data_gb = recommendation_data.get("requested_data_gb")
    offers = recommendation_data.get("offers", [])

    reasons = []
    if budget is not None:
        reasons.append(
            f"son prix de {selected_offer.get('price_monthly')}$ respecte votre budget de {budget:g}$"
        )
    if requested_data_gb is not None:
        reasons.append(
            f"elle fournit {selected_offer.get('data_gb')} Go, ce qui couvre votre besoin d'au moins {requested_data_gb:g} Go"
        )
    if not reasons:
        reasons.append(
            f"parmi les offres éligibles, elle propose le plus de data ({selected_offer.get('data_gb')} Go)"
        )

    alternative = next(
        (o for o in offers if o.get("offer_id") != selected_offer.get("offer_id")),
        None,
    )

    text = (
        f"Je vous ai recommandé {selected_offer.get('name')} parce que "
        + " et ".join(reasons)
        + "."
    )

    if alternative is not None:
        text += (
            f" À titre de comparaison, {alternative.get('name')} coûte "
            f"{alternative.get('price_monthly')}$ et offre {alternative.get('data_gb')} Go."
        )

    return text


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


# ---------------------------------------------------------------------------
# Couche de coordination métier TOBi
# ---------------------------------------------------------------------------


def _missing_tobi_slots(intent: str, entities: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    if intent == "topup_airtime":
        if entities.get("amount") is None:
            missing.append("amount")
        if entities.get("currency") is None:
            missing.append("currency")
        if (
            entities.get("beneficiary_type") == "other"
            and not entities.get("target_msisdn")
        ):
            missing.append("target_msisdn")

    elif intent == "handover_human":
        if not entities.get("preferred_channel"):
            missing.append("preferred_channel")

    return missing


def _message_mentions_payment_method(message: str) -> bool:
    text = message.lower()
    return any(
        marker in text
        for marker in (
            "m-pesa",
            "mpesa",
            "m pesa",
            "carte",
            "card",
            "airtime",
            "crédit d'appel",
            "credit d'appel",
        )
    )


def _message_explicitly_selects_self(message: str) -> bool:
    text = message.lower()
    return any(
        marker in text
        for marker in (
            "pour moi",
            "pour moi-même",
            "pour moi meme",
            "ma ligne",
            "sur mon numéro",
            "sur mon numero",
        )
    )


def _merge_tobi_entities(
    previous: dict[str, Any],
    incoming: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    """Fusionne les slots d'un échange multi-tour sans écraser le contexte utile."""
    merged = dict(previous)

    for key, value in incoming.items():
        if value is None:
            continue

        # extract_tobi_entities met mpesa par défaut. Sur un follow-up comme
        # "5$", ce défaut ne doit pas écraser une méthode explicitement choisie avant.
        if (
            key == "payment_method"
            and merged.get("payment_method")
            and not _message_mentions_payment_method(message)
        ):
            continue

        # Même principe pour beneficiary_type : un simple follow-up de montant
        # ne doit pas transformer un tiers déjà choisi en "self".
        if (
            key == "beneficiary_type"
            and merged.get("beneficiary_type") == "other"
            and value == "self"
            and not _message_explicitly_selects_self(message)
        ):
            continue

        merged[key] = value

    return merged


def _execute_tobi_action(
    *,
    intent: str,
    customer_id: str,
    entities: dict[str, Any],
) -> dict[str, Any]:
    """Appelle uniquement le Mock métier correspondant à l'intention TOBi."""
    if intent == "topup_airtime":
        return topup_airtime_mock.invoke(
            {
                "customer_id": customer_id,
                "beneficiary_type": entities.get("beneficiary_type", "self"),
                "target_msisdn": entities.get("target_msisdn"),
                "amount": entities["amount"],
                "currency": entities["currency"],
                "payment_method": entities.get("payment_method", "mpesa"),
            }
        )

    if intent == "check_airtime_balance":
        return check_airtime_balance_mock.invoke(
            {
                "customer_id": customer_id,
            }
        )

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


def _process_tobi_message(
    *,
    session_id: str,
    customer_id: str,
    classified_intent: str,
    message: str,
) -> tuple[str, dict[str, Any]]:
    """
    Gère le slot filling multi-tour puis appelle le service Mock lorsque les slots
    obligatoires sont complets.

    Un follow-up très court comme "5$" peut être classé unknown isolément. Si une
    action TOBi incomplète est déjà en attente dans la même session, il est alors
    interprété comme une réponse de slot pour cette action.
    """
    previous = _TOBI_SESSION_STATE.get(session_id)
    pending_intent = None

    if previous and not previous.get("slots_complete", False):
        pending_intent = previous.get("intent")

    effective_intent = classified_intent

    if pending_intent and classified_intent in {FALLBACK_INTENT, pending_intent}:
        effective_intent = pending_intent

    # Si l'utilisateur exprime clairement une nouvelle intention TOBi alors qu'un
    # ancien workflow est terminé ou différent, on repart sur un nouvel état.
    continue_pending = (
        previous is not None
        and not previous.get("slots_complete", False)
        and previous.get("intent") == effective_intent
    )

    extraction = extract_tobi_entities(effective_intent, message)
    incoming_entities = extraction.get("entities", {})

    if continue_pending:
        entities = _merge_tobi_entities(
            previous.get("entities", {}),
            incoming_entities,
            message,
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

    _TOBI_SESSION_STATE[session_id] = state
    return effective_intent, state


def _render_missing_slots(tobi_data: dict[str, Any]) -> str:
    intent = tobi_data.get("intent")
    missing = set(tobi_data.get("missing_slots", []))

    if intent == "topup_airtime":
        if "target_msisdn" in missing and {"amount", "currency"} & missing:
            return (
                "Pour continuer la recharge, indiquez le montant avec la devise "
                "(USD ou CDF) ainsi que le numéro Vodacom du bénéficiaire."
            )

        if "target_msisdn" in missing:
            return (
                "Quel est le numéro Vodacom du bénéficiaire ? "
                "Vous pouvez utiliser le format 081..., 082... ou +243..."
            )

        if "amount" in missing or "currency" in missing:
            return (
                "Quel montant souhaitez-vous recharger et dans quelle devise "
                "(USD ou CDF) ?"
            )

    if intent == "handover_human" and "preferred_channel" in missing:
        return "Préférez-vous être mis en relation avec un conseiller par chat ou par appel ?"

    return "Il me manque encore certaines informations pour continuer cette opération."


def _render_tobi_action(tobi_data: dict[str, Any]) -> str:
    action_result = tobi_data.get("action_result") or {}
    action = action_result.get("action")
    status = action_result.get("status")

    if status == "MOCK_PROFILE_NOT_FOUND":
        return (
            "Je n'ai pas trouvé de données Mock pour ce profil client. "
            "Aucune opération réelle n'a été exécutée."
        )

    if action == "topup_airtime":
        amount = action_result.get("amount")
        currency = action_result.get("currency")
        payment_method = action_result.get("payment_method")
        beneficiary_type = action_result.get("beneficiary_type")
        target_msisdn = action_result.get("target_msisdn")

        beneficiary = (
            f"le numéro {target_msisdn}"
            if beneficiary_type == "other" and target_msisdn
            else "votre ligne"
        )

        return (
            f"Simulation TOBi : une recharge Airtime de {amount} {currency} pour "
            f"{beneficiary} via {payment_method} a été préparée avec succès. "
            "Aucune transaction réelle Vodacom/M-Pesa n'a été exécutée."
        )

    if action == "check_airtime_balance":
        balance = action_result.get("balance") or {}
        return (
            f"Solde Airtime simulé : {balance.get('amount')} "
            f"{balance.get('currency')}. Ces données proviennent du Mock du MVP."
        )

    if action == "explain_tariff_plan":
        plan = action_result.get("plan") or {}
        if not plan:
            return "Aucune information de plan tarifaire Mock n'est disponible pour ce profil."

        return (
            f"Plan tarifaire simulé : {plan.get('plan_name')}. "
            f"Prix : {plan.get('price_monthly')} {plan.get('currency')} / mois, "
            f"data : {plan.get('data_gb')} Go, appels : {plan.get('calls')}. "
            "Ces informations sont issues du Mock du MVP."
        )

    if action == "handover_human":
        channel = action_result.get("preferred_channel")
        return (
            f"Transfert simulé vers un conseiller par {channel}. "
            "Aucun conseiller réel n'a été contacté dans ce MVP."
        )

    return "Le workflow TOBi Mock a été traité."


def _render_user_response(
    *,
    message: str,
    intent: str,
    scope: str,
    recommendation_data: dict[str, Any],
    tobi_data: dict[str, Any],
) -> str:
    """Transforme les résultats structurés des agents/services en réponse utilisateur."""
    if scope == "tobi":
        if not tobi_data.get("slots_complete", False):
            return _render_missing_slots(tobi_data)
        return _render_tobi_action(tobi_data)

    if scope == "unknown":
        return (
            "Je n'ai pas pu déterminer précisément votre demande. "
            "Pouvez-vous la reformuler ?"
        )

    if intent == "BUY_PREPAID":
        offers = recommendation_data.get("offers", [])
        requested_data_gb = recommendation_data.get("requested_data_gb")

        if not offers:
            if requested_data_gb is not None:
                return (
                    f"Je n'ai trouvé aucune offre prépayée avec au moins "
                    f"{requested_data_gb:g} Go correspondant à vos critères actuels."
                )
            return "Aucune offre prépayée ne correspond à vos critères actuels."

        if requested_data_gb is not None and len(offers) == 1:
            offer = offers[0]
            return (
                f"Pour un besoin de {requested_data_gb:g} Go, l'offre qui correspond est "
                f"{offer.get('name')} à {offer.get('price_monthly')}$ par mois, "
                f"avec {offer.get('data_gb')} Go et appels : {offer.get('calls_min')}."
            )

        if requested_data_gb is not None:
            lines = [
                f"Voici les offres qui couvrent au moins {requested_data_gb:g} Go :"
            ]
        else:
            lines = ["Voici les offres prépayées disponibles :"]

        lines.extend(f"- {_format_offer(offer)}" for offer in offers)
        return "\n".join(lines)

    if intent == "ASK_RECOMMENDATION":
        if _is_recommendation_explanation_request(message):
            return _explain_current_recommendation(recommendation_data)

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
    Point d'orchestration de l'Agent de Communication.

    - Intent Agent : classification + extraction TOBi, sans réponse utilisateur.
    - Recommendation Agent : uniquement les recommandations.
    - Services TOBi Mock : uniquement les actions métier TOBi simulées.
    - Communication Agent : collecte les slots manquants et formule la réponse finale.
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

    intent_result = classify_intent(client, message)
    classified_intent = intent_result["intent"]
    scope = get_intent_scope(classified_intent)

    explanation_request = _is_recommendation_explanation_request(message)
    recommendation_snapshot = _get_recommendation_snapshot(session_id)

    # Une question comme "pourquoi cette recommandation ?" dépend du contexte
    # conversationnel. Si le classificateur isolé renvoie unknown mais qu'une
    # recommandation existe dans cette session, on la traite comme un follow-up
    # de recommandation sans ajouter un nouvel intent au catalogue.
    if (
        explanation_request
        and scope == "unknown"
        and recommendation_snapshot
        and (
            recommendation_snapshot.get("selected_offer_id")
            or recommendation_snapshot.get("devices")
        )
    ):
        classified_intent = "ASK_RECOMMENDATION"
        scope = "recommendation"

    # Si une action TOBi incomplète attend des slots, un follow-up très court peut
    # être classé unknown isolément. On conserve alors le workflow TOBi en attente.
    pending_tobi = _TOBI_SESSION_STATE.get(session_id)
    if (
        scope == "unknown"
        and pending_tobi
        and not pending_tobi.get("slots_complete", False)
    ):
        scope = "tobi"

    recommendation_called = False
    recommendation_data: dict[str, Any] = {}
    tobi_service_called = False
    tobi_data: dict[str, Any] = {}
    effective_intent = classified_intent

    if scope == "recommendation":
        # Un changement explicite de domaine annule un ancien slot filling TOBi en attente.
        _TOBI_SESSION_STATE.pop(session_id, None)

        if explanation_request:
            # Expliquer une recommandation déjà faite est une tâche de communication.
            # On lit le State existant sans refaire une nouvelle recommandation.
            recommendation_data = recommendation_snapshot
        else:
            recommendation_called = True
            state, config = _build_recommendation_input(
                session_id=session_id,
                message=message,
                intent=classified_intent,
                profile=profile,
            )

            result = recommendation_graph.invoke(
                state,
                config=config,
            )
            recommendation_data = _safe_recommendation_data(result)

    elif scope == "tobi":
        effective_intent, tobi_data = _process_tobi_message(
            session_id=session_id,
            customer_id=customer_id,
            classified_intent=classified_intent,
            message=message,
        )
        tobi_service_called = tobi_data.get("action_result") is not None

    response_text = _render_user_response(
        message=message,
        intent=effective_intent,
        scope=scope,
        recommendation_data=recommendation_data,
        tobi_data=tobi_data,
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
        "intent": effective_intent,
        "scope": scope,
        "response": response_text,
        "data": {
            "communication_agent_called": True,
            "intent_agent_called": True,
            "recommendation_agent_called": recommendation_called,
            "tobi_service_called": tobi_service_called,
            "crm_profile": profile,
            "recommendation": recommendation_data,
            "tobi": tobi_data,
        },
    }
