import os
import sys
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(REPO_ROOT, "CODE")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import config  # noqa: E402
from retrieval_service import TrendScoutBackend  # noqa: E402


app = FastAPI(title="TrendScout AI API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    return_context: bool = True


class QueryResponse(BaseModel):
    answer: str
    context: dict[str, Any] | None = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "python": sys.version.split()[0],
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    artifacts = {
        "articles_raw": os.path.exists(config.ARTICLES_FILE),
        "snippets": os.path.exists(config.SNIPPETS_FILE),
        "kpi_entities": os.path.exists(config.KPI_ENTITIES_FILE),
        "snippets_embeddings": os.path.exists(os.path.join(config.DATA_DIR, "snippets_embeddings.parquet")),
        "entity_map": os.path.exists(config.ENTITY_MAP_FILE),
    }

    env = {
        "neo4j_env_set": bool(config.NEO4J_URI and config.NEO4J_USERNAME and config.NEO4J_PASSWORD),
        "chroma_path": config.CHROMA_DB_PATH,
    }

    return {"artifacts": artifacts, "env": env}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        backend = TrendScoutBackend()
        result = backend.generate_answer(req.query, return_context=req.return_context)
        if isinstance(result, dict):
            return QueryResponse(answer=result.get("answer", ""), context=result)
        return QueryResponse(answer=str(result), context=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

