import os
from pathlib import Path
from typing import Any

import groq
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import Groq
from pydantic import BaseModel, Field

from communication_agent import handle_user_message
from intent_agent import analyze_nlu_request

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# ---------------------------------------------------------------------------
# Contrats Pydantic
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    customer_id: str = Field(default="CUST_001", min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    customer_id: str | None = None
    profile_status: str = "UNKNOWN"
    intent: str
    scope: str
    response: str
    state: dict[str, Any]
    offers: list[dict[str, Any]]


class NLURequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# Client Groq
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Static (front)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def front():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint conversationnel complet.

    Le Gateway HTTP dialogue avec l'Agent de Communication.
    La réponse expose aussi l'intent, le scope, le State métier et les offres
    afin de rendre le fonctionnement observable.
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
            "customer_id": result.get("customer_id"),
            "profile_status": result.get("profile_status", "UNKNOWN"),
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
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne : {type(error).__name__}",
        ) from error


# ---------------------------------------------------------------------------
# /nlu
# ---------------------------------------------------------------------------
@app.post("/nlu")
def nlu(request: NLURequest):
    """
    Endpoint de test direct de l'Agent Intent / NLU.

    Il permet d'observer séparément :
    - l'intention détectée ;
    - les entités TOBi extraites ;
    - l'état slots_complete.
    """
    try:
        return analyze_nlu_request(client, request.query)

    except groq.APIError as error:
        raise HTTPException(
            status_code=502,
            detail="Erreur lors de la communication avec Groq.",
        ) from error
    except Exception as error:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne : {type(error).__name__}",
        ) from error    