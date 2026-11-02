"""REST API around the agent.

    .venv/Scripts/python -m uvicorn app.api:app --reload
    → http://127.0.0.1:8000/docs   (interactive documentation, to try the API in a browser)

The model is the one of LLM_MODEL (qwen2.5:7b by default, or claude-haiku-5-5 with a key in .env).
The agent is the one of AGENT: "handwritten" (src/agent.py, the default) or "langgraph" (src/agent_langgraph.py,
which also remembers conversations: send back the conversation_id of an answer to ask a follow-up question).
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import llm
from src.agent import run_agent
from src.agent_langgraph import ask as langgraph_ask
from src.rag_tool import retrieve

AGENT = os.getenv("AGENT", "handwritten")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once when the server starts (before `yield`) and once when it stops (after)."""
    retrieve("warm-up")  # load the embedding model and the index now, so that no user waits for it
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Football AI agent",
    description="Answers questions on the 2022 World Cup and Euro 2024 statistics (SQL) "
                "and on the IFAB Laws of the Game (RAG). An LLM agent chooses the tools.",
    version="1.0",
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Is there offside from a throw-in?"])
    conversation_id: str | None = None  # LangGraph agent only: the id of a previous answer, to continue it


class Source(BaseModel):
    law_number: int
    law_title: str
    section: str


class Answer(BaseModel):
    answer: str
    tools_used: list[str]   # which tools the agent called
    sql: list[str]          # the SQL queries it ran
    sources: list[Source]   # the passages of the Laws it read
    model: str
    latency_s: float
    tokens: int
    conversation_id: str | None  # send it back with the next question to continue the conversation


@app.get("/health")
def health() -> dict:
    """Is the service up, and with which model?"""
    return {"status": "ok", "model": llm.MODEL, "agent": AGENT}


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:
    """Ask the agent a question."""
    try:
        if AGENT == "langgraph":  # with memory: a new conversation gets a new id, a known id continues it
            conversation_id = body.conversation_id or uuid.uuid4().hex
            result = langgraph_ask(body.question, thread_id=conversation_id)
        else:                     # the hand-written agent has no memory: each question is independent
            conversation_id = None
            result = run_agent(body.question)
    except Exception as e:  # e.g. Ollama not started, no internet for the API
        raise HTTPException(status_code=503, detail=f"The agent is unavailable: {e}")
    return Answer(
        answer=result["answer"],
        tools_used=result["tools_used"],
        sql=result["sql"],
        sources=[Source(law_number=p["law_number"], law_title=p["law_title"], section=p["section"])
                 for p in result["passages"]],
        model=llm.MODEL,
        latency_s=round(result["latency_s"], 1),
        tokens=result["tokens"],
        conversation_id=conversation_id,
    )
