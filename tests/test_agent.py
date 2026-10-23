"""Tests of run_agent without Ollama: the LLM and the tools are replaced by fakes (monkeypatch)."""

import pytest

import src.agent as agent
from src.llm import LLMResponse


class FakeLLM:
    """Plays back prepared responses, one per call, and remembers what it received."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []  # (messages, tools) for each call

    def __call__(self, messages, tools=None, temperature=0.0, tools_allowed=True):
        self.calls.append(([dict(m) for m in messages], tools if tools_allowed else None))  # None: no tool usable
        return self.responses.pop(0)


def text(answer):
    return LLMResponse(text=answer, input_tokens=100, output_tokens=10, latency_s=1.0)


def tool_call(*calls):
    return LLMResponse(text="", tool_calls=[{"name": n, "arguments": a} for n, a in calls],
                       input_tokens=100, output_tokens=10, latency_s=1.0)


def fake_stats(question):
    return {"content": "goals\n    8", "sql": "SELECT 8 AS goals", "tokens": 50, "latency_s": 2.0}


def fake_laws(query):
    return {"content": "[Law 11] No offside from a throw-in.", "passages": [{"law_number": 11}],
            "tokens": 20, "latency_s": 0.5}


@pytest.fixture
def llm(monkeypatch):
    """Use it as llm(response1, response2, ...); the tools are always fakes."""
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "query_stats_database", fake_stats)
    monkeypatch.setitem(agent.TOOL_FUNCTIONS, "search_laws_of_the_game", fake_laws)

    def install(*responses):
        fake = FakeLLM(*responses)
        monkeypatch.setattr(agent, "chat", fake)
        return fake
    return install


def tool_messages(fake, call_index):
    """The tool results the LLM received on a given call."""
    messages, _ = fake.calls[call_index]
    return [m for m in messages if m["role"] == "tool"]


def test_answers_directly_without_tool(llm):
    fake = llm(text("I can only help with football statistics and rules."))
    result = agent.run_agent("What is the capital of Australia?")

    assert result["answer"] == "I can only help with football statistics and rules."
    assert result["tools_used"] == []
    messages, tools = fake.calls[0]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert tools == agent.TOOLS


def test_calls_a_tool_then_answers(llm):
    fake = llm(tool_call(("query_stats_database", {"question": "Mbappé goals 2022"})),
               text("Mbappé scored 8 goals."))
    result = agent.run_agent("How many goals did Mbappé score at the 2022 World Cup?")

    assert result["answer"] == "Mbappé scored 8 goals."
    assert result["tools_used"] == ["query_stats_database"]
    assert result["sql"] == ["SELECT 8 AS goals"]
    assert tool_messages(fake, 1)[0]["content"] == "goals\n    8"  # the LLM saw the tool result


def test_mixed_question_uses_both_tools(llm):
    llm(tool_call(("query_stats_database", {"question": "red cards Euro 2024"}),
                  ("search_laws_of_the_game", {"query": "sending-off offences"})),
        text("There were 2 red cards (Law 12)."))
    result = agent.run_agent("How many red cards at Euro 2024, and what justifies one?")

    assert result["tools_used"] == ["query_stats_database", "search_laws_of_the_game"]
    assert result["passages"] == [{"law_number": 11}]


def test_unknown_tool_is_reported_to_the_llm(llm):
    fake = llm(tool_call(("get_weather", {"city": "Doha"})), text("Sorry."))
    result = agent.run_agent("Weather in Doha?")

    assert result["answer"] == "Sorry."
    assert result["tools_used"] == []
    assert tool_messages(fake, 1)[0]["content"].startswith("Error")


def test_wrong_arguments_are_reported_to_the_llm(llm):
    fake = llm(tool_call(("query_stats_database", {"q": "Mbappé goals"})), text("Sorry."))
    result = agent.run_agent("How many goals did Mbappé score?")

    assert result["answer"] == "Sorry."
    assert tool_messages(fake, 1)[0]["content"].startswith("Error")


def test_stops_after_max_turns_and_forces_an_answer(llm):
    endless = [tool_call(("search_laws_of_the_game", {"query": "offside"})) for _ in range(agent.MAX_TURNS)]
    fake = llm(*endless, text("Final answer."))
    result = agent.run_agent("Offside?")

    assert result["answer"] == "Final answer."
    assert len(fake.calls) == agent.MAX_TURNS + 1
    assert fake.calls[-1][1] is None  # the last call has no tools


def test_adds_up_tokens_and_latency(llm):
    llm(tool_call(("query_stats_database", {"question": "q"})), text("8."))
    result = agent.run_agent("q")

    assert result["tokens"] == 110 + 50 + 110  # two LLM calls + the tool
    assert result["latency_s"] == pytest.approx(1.0 + 2.0 + 1.0)
