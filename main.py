import os
from typing import Any

import groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, ConfigDict, Field

from communication_agent import (
    ConversationCustomerMismatchError,
    bind_current_conversation_to_customer,
    handle_user_message,
)
from crm_mock import create_customer_profile_mock, get_customer_profile_mock
from intent_agent import analyze_nlu_request
from llm_response_generator import ResponseValidationError

load_dotenv()


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "Bonjour, je cherche un forfait prépayé."
                }
            ]
        }
    )

    # customer_id est facultatif. S'il est donné une première fois, l'orchestrateur
    # le mémorise pour toute la conversation active jusqu'au redémarrage.
    customer_id: str | None = Field(
        default=None,
        max_length=100,
        description=(
            "Profil CRM facultatif. Une fois associé à la conversation active, "
            "il reste mémorisé jusqu'au redémarrage de l'application."
        ),
    )
    message: str = Field(
        min_length=1,
        max_length=2000,
        description="Message courant de l'utilisateur.",
    )


class ChatResponse(BaseModel):
    customer_id: str | None
    profile_status: str
    intent: str
    scope: str
    response: str
    state: dict[str, Any]
    offers: list[dict[str, Any]]


class NLURequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class CreateProfileRequest(BaseModel):
    first_name: str = Field(min_length=1, max_length=100)
    segment: str = Field(default="PREPAID", min_length=1, max_length=50)
    preferred_language: str | None = Field(default=None, max_length=20)
    current_offer_id: str | None = Field(default=None, max_length=100)
    current_offer_name: str | None = Field(default=None, max_length=200)
    category_device: str | None = Field(default=None, max_length=100)
    phone_number: str | None = Field(default=None, max_length=30)


def create_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "La variable GROQ_API_KEY est absente. Vérifie ton fichier .env."
        )

    return Groq(api_key=api_key.strip())


app = FastAPI(
    title="Agentic AI Vodacom - 3 Agents",
    version="1.8.0",
)

client = create_client()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def front():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api")
def api_root():
    return {
        "message": "API Agentic AI Vodacom active",
        "conversation_mode": "single_active_conversation_in_memory",
        "architecture": (
            "Gateway -> Communication/Orchestrator -> Intent -> "
            "{TOBi | Recommendation} -> LLM"
        ),
    }       


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Utilise l'unique conversation active du MVP.

    Aucun session_id n'est demandé ou généré. Tous les appels /chat appartiennent
    à la même conversation jusqu'au redémarrage du processus FastAPI.
    """
    try:
        result = handle_user_message(
            client,
            customer_id=request.customer_id,
            message=request.message,
        )

        data = result.get("data", {})
        recommendation_state = data.get("recommendation", {})
        tobi_state = data.get("tobi", {})
        purchase_state = data.get("purchase", {})

        if result["scope"] == "recommendation":
            visible_state = {**recommendation_state, "purchase": purchase_state}
            offers = recommendation_state.get("offers", [])
        elif result["scope"] == "tobi":
            visible_state = tobi_state
            offers = []
        else:
            # Utile pour contrôler une confirmation en attente si le NLU retourne
            # unknown sur une réponse ambiguë : l'état n'est pas perdu.
            visible_state = (
                {"purchase": purchase_state}
                if purchase_state.get("status") == "AWAITING_CONFIRMATION"
                else {}
            )
            offers = []

        return {
            "customer_id": result.get("customer_id"),
            "profile_status": result.get("profile_status", "UNKNOWN"),
            "intent": result["intent"],
            "scope": result["scope"],
            "response": result["response"],
            "state": visible_state,
            "offers": offers,
        }

    except ConversationCustomerMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ResponseValidationError as error:
        raise HTTPException(
            status_code=502,
            detail="Réponse LLM non conforme.",
        ) from error
    except groq.APIError as error:
        raise HTTPException(
            status_code=502,
            detail="Erreur lors de la communication avec Groq.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Erreur interne du service.",
        ) from error


@app.post("/crm/profiles")
def create_profile(request: CreateProfileRequest):
    """Crée un profil CRM Mock.

    Si la conversation active est encore anonyme, le nouveau profil lui est
    associé sans perdre son historique.
    """
    try:
        profile = create_customer_profile_mock.invoke(request.model_dump())
        bind_current_conversation_to_customer(profile["customer_id"])
        return profile
    except ConversationCustomerMismatchError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/crm/profiles/{customer_id}")
def get_profile(customer_id: str):
    return get_customer_profile_mock.invoke({"customer_id": customer_id})


@app.post("/nlu")
def nlu(request: NLURequest):
    """Endpoint NLU de diagnostic, indépendant de la conversation /chat."""
    try:
        return analyze_nlu_request(client, request.query)
    except groq.APIError as error:
        raise HTTPException(
            status_code=502,
            detail="Erreur lors de la communication avec Groq.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne : {type(error).__name__}",
        ) from error
