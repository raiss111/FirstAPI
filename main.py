import os
from typing import Any

import groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq
from pydantic import BaseModel, Field

from communication_agent import handle_user_message

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


def create_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "La variable GROQ_API_KEY est absente. Vérifie ton fichier .env."
        )

    return Groq(api_key=api_key.strip())


app = FastAPI(
    title="Agentic AI Vodacom - 3 Agents",
    version="1.3.0",
)

client = create_client()


@app.get("/")
def root():
    return {
        "message": "API Agentic AI Vodacom active",
        "architecture": "Communication -> Intent -> Recommendation",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):
    """
    Le Gateway HTTP ne dialogue directement qu'avec l'Agent de Communication.

    Pour le MVP académique, la réponse expose aussi l'intent, le scope,
    le State métier du Recommendation Agent et les offres disponibles,
    afin de rendre le fonctionnement interne observable.
    """
    try:
        result = handle_user_message(
            client,
            session_id=request.session_id,
            customer_id=request.customer_id,
            message=request.message,
        )

        recommendation_state = result.get("data", {}).get("recommendation", {})
        offers = recommendation_state.get("offers", [])

        return {
            "session_id": result["session_id"],
            "intent": result["intent"],
            "scope": result["scope"],
            "response": result["response"],
            "state": recommendation_state,
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
