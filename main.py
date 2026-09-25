import os
from typing import Any

import groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel, Field

from communication_agent import handle_user_message
from intent_agent import analyze_nlu_request

load_dotenv()


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    customer_id: str = Field(default="CUST_001", min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    scope: str
    response: str
    state: dict[str, Any]
    offers: list[dict[str, Any]]


class NLURequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


def create_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "La variable GROQ_API_KEY est absente. Vérifie ton fichier .env."
        )

    return Groq(api_key=api_key.strip())


app = FastAPI(
    title="Agentic AI Vodacom - 3 Agents",
    version="1.4.0",
)

client = create_client()


@app.get("/")
def root():
    return {
        "message": "API Agentic AI Vodacom active",
        "architecture": "Communication -> Intent -> {TOBi Services Mock | Recommendation}",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint conversationnel complet.

    Le Gateway HTTP dialogue avec l'Agent de Communication. Pour le MVP
    académique, la réponse expose aussi l'intent, le scope, le State métier du
    Recommendation Agent et les offres afin de rendre le fonctionnement observable.
    """
    try:
        result = handle_user_message(
            client,
            session_id=request.session_id,
            customer_id=request.customer_id,
            message=request.message,
        )

        data = result.get("data", {})
        recommendation_state = data.get("recommendation", {})
        tobi_state = data.get("tobi", {})

        if result["scope"] == "recommendation":
            visible_state = recommendation_state
            offers = recommendation_state.get("offers", [])
        elif result["scope"] == "tobi":
            visible_state = tobi_state
            offers = []
        else:
            visible_state = {}
            offers = []

        return {
            "session_id": result["session_id"],
            "intent": result["intent"],
            "scope": result["scope"],
            "response": result["response"],
            "state": visible_state,
            "offers": offers,
        }

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


@app.post("/nlu")
def nlu(request: NLURequest):
    """
    Endpoint de test direct de l'Agent Intent / NLU.

    Il permet d'observer séparément :
    - l'intention détectée ;
    - les entités TOBi extraites ;
    - l'état slots_complete.

    Les scores de confiance et le fallback avancé ne sont pas encore inclus.
    """
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
