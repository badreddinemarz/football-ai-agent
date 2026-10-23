"""REST API around the agent.

    .venv/Scripts/python -m uvicorn app.api:app --reload
    → http://127.0.0.1:8000/docs   (interactive documentation, to try the API in a browser)

The model is the one of LLM_MODEL (qwen2.5:7b by default, or claude-haiku-5-5 with a key in .env).
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import llm
from src.agent import run_agent

app = FastAPI(
    title="Football AI agent",
    description="Answers questions on the 2022 World Cup and Euro 2024 statistics (SQL) "
                "and on the IFAB Laws of the Game (RAG). An LLM agent chooses the tools.",
    version="1.0",
)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500, examples=["Is there offside from a throw-in?"])


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


@app.get("/health")
def health() -> dict:
    """Is the service up, and with which model?"""
    return {"status": "ok", "model": llm.MODEL}


@app.post("/ask", response_model=Answer)
def ask(body: Question) -> Answer:
    """Ask the agent a question."""
    try:
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
    )
