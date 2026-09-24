# Agentic AI Vodacom - MVP

Architecture implémentée :

Utilisateur -> FastAPI/Gateway -> Agent Intent Recognition -> Agent de Recommandation LangGraph -> Tools Mock -> Réponse

## Installation

```powershell
python -m pip install -r requirements.txt
```

Créer `.env` :

```env
GROQ_API_KEY=ta_cle_groq
```

## Test du graphe sans Agent Intent

```powershell
python test_graph.py
```

## Lancer l'API

```powershell
python -m fastapi dev main.py
```

Puis ouvrir :

- `http://127.0.0.1:8000/docs`
- tester `POST /chat`

Exemple :

```json
{
  "session_id": "demo-001",
  "message": "Je veux acheter un forfait prépayé à moins de 20 dollars"
}
```

Réutiliser exactement le même `session_id` pour les messages suivants afin de conserver le contexte.
