# api

FastAPI service — wraps the `gfix` merge-conflict engine and adds RAG-augmented resolution via pgvector.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`GET /health` → `{"status": "ok"}`. Full API built out in Phases 1-5.
