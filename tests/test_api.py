"""Tests of the REST API without any LLM: run_agent is replaced by a fake."""

from fastapi.testclient import TestClient

import app.api as api

client = TestClient(api.app)


def fake_agent(question):
    return {"answer": "No, there is no offside from a throw-in (Law 11).",
            "tools_used": ["search_laws_of_the_game"], "sql": [], "contexts": [],
            "passages": [{"law_number": 11, "law_title": "Offside", "section": "3. No offence", "text": "..."}],
            "tokens": 1200, "latency_s": 3.14159}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_the_answer_and_its_sources(monkeypatch):
    monkeypatch.setattr(api, "run_agent", fake_agent)
    response = client.post("/ask", json={"question": "Is there offside from a throw-in?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("No")
    assert body["tools_used"] == ["search_laws_of_the_game"]
    assert body["sources"] == [{"law_number": 11, "law_title": "Offside", "section": "3. No offence"}]
    assert body["latency_s"] == 3.1


def test_ask_rejects_an_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_reports_an_unavailable_agent(monkeypatch):
    def broken(question):
        raise ConnectionError("Ollama is not running")
    monkeypatch.setattr(api, "run_agent", broken)
    response = client.post("/ask", json={"question": "Messi goals?"})
    assert response.status_code == 503
