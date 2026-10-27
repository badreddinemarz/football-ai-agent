"""The LangGraph agent: its graph and its tools, without calling any LLM."""

import src.agent_langgraph as lg


def test_graph_nodes_and_edges():
    graph = lg.build_graph().get_graph()  # building it needs no LLM call
    assert set(graph.nodes) == {"__start__", "llm", "tools", "final", "__end__"}
    edges = {(e.source, e.target) for e in graph.edges}
    assert {("__start__", "llm"), ("llm", "tools"), ("llm", "__end__"),
            ("tools", "llm"), ("tools", "final"), ("final", "__end__")} <= edges


def test_tools_have_the_same_names_as_the_hand_written_agent():
    assert [t.name for t in lg.TOOLS] == ["query_stats_database", "search_laws_of_the_game"]
    assert list(lg.TOOLS[0].args) == ["question"] and list(lg.TOOLS[1].args) == ["query"]


# --- Conversation memory, without any LLM: the model is replaced by a fake that plays back prepared answers ---

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class FakeChatModel(GenericFakeChatModel):
    """LangChain's fake model, made to accept tools (the agent binds them)."""

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def fake_llm(monkeypatch):
    """Use it as fake_llm("answer 1", "answer 2", ...): the agent's next answers, one per LLM call."""
    monkeypatch.setattr(lg, "_memory_graphs", {})  # a fresh memory for each test

    def install(*answers):
        model = FakeChatModel(messages=iter([AIMessage(a) for a in answers]))
        monkeypatch.setattr(lg, "chat_model", lambda: model)
    return install


def saved_messages(thread_id):
    """What the checkpointer keeps for a conversation."""
    return lg.memory_graph().get_state({"configurable": {"thread_id": thread_id}}).values["messages"]


def test_a_conversation_keeps_the_previous_questions_and_answers(fake_llm):
    fake_llm("Messi scored 7 goals.", "Mbappé scored 8 goals.")
    lg.ask("How many goals did Messi score?", thread_id="t1")
    second = lg.ask("And Mbappe?", thread_id="t1")

    assert second["answer"] == "Mbappé scored 8 goals."
    kinds = [type(m).__name__ for m in saved_messages("t1")]
    assert kinds == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]


def test_the_instructions_are_sent_once_per_conversation(fake_llm):
    fake_llm("a", "b", "c")
    for question in ["q1", "q2", "q3"]:
        lg.ask(question, thread_id="t1")
    assert sum(isinstance(m, SystemMessage) for m in saved_messages("t1")) == 1


def test_conversations_are_kept_apart(fake_llm):
    fake_llm("Messi scored 7 goals.", "I need more context.")
    lg.ask("How many goals did Messi score?", thread_id="t1")
    lg.ask("And Mbappe?", thread_id="t2")

    assert [m.text for m in saved_messages("t2") if isinstance(m, HumanMessage)] == ["And Mbappe?"]


def test_messages_since_question_keeps_only_the_current_question():
    messages = [SystemMessage("rules"), HumanMessage("q1"), AIMessage("a1"), HumanMessage("q2"), AIMessage("a2")]
    assert [m.text for m in lg.messages_since_question(messages)] == ["q2", "a2"]


# --- What the LLM sees: compact_history (pure function, no LLM) ---

from langchain_core.messages import ToolMessage


def tool_turn(n):
    """One question answered with a tool: question, tool call (with Claude's thinking), result, answer."""
    call_id = f"call{n}"
    return [HumanMessage(f"question {n}"),
            AIMessage(content=[{"type": "thinking", "thinking": "...", "signature": "sig"},
                               {"type": "tool_use", "id": call_id, "name": "query_stats_database", "input": {}}],
                      tool_calls=[{"name": "query_stats_database", "args": {}, "id": call_id}]),
            ToolMessage(f"result {n}", tool_call_id=call_id),
            AIMessage(content=[{"type": "thinking", "thinking": "...", "signature": "sig"},
                               {"type": "text", "text": f"answer {n}"}])]


def conversation(n_turns):
    return [SystemMessage("rules")] + [m for n in range(1, n_turns + 1) for m in tool_turn(n)]


def has_thinking(message):
    return isinstance(message.content, list) and any(b.get("type") == "thinking" for b in message.content)


def test_old_turns_become_question_and_plain_answer():
    seen = lg.compact_history(conversation(4))
    assert [m.text for m in seen[:5]] == ["rules", "question 1", "answer 1", "question 2", "answer 2"]


def test_the_current_question_is_sent_unchanged():
    messages = conversation(4)
    assert lg.compact_history(messages)[-4:] == messages[-4:]


def test_no_tool_call_is_separated_from_its_result():
    seen = lg.compact_history(conversation(4))
    calls = {c["id"] for m in seen if isinstance(m, AIMessage) for c in m.tool_calls}
    results = {m.tool_call_id for m in seen if isinstance(m, ToolMessage)}
    assert calls == results == {"call3", "call4"}


def test_only_the_current_question_keeps_claude_thinking():
    seen = lg.compact_history(conversation(4))
    assert [has_thinking(m) for m in seen[:-4]] == [False] * len(seen[:-4])
    assert has_thinking(seen[-3]) and has_thinking(seen[-1])


def test_the_oldest_turns_are_forgotten_past_the_token_budget(monkeypatch):
    monkeypatch.setattr(lg, "OLD_TURNS_MAX_TOKENS", 40)
    seen = lg.compact_history(conversation(20))
    questions = [m.text for m in seen if isinstance(m, HumanMessage)]
    assert "question 1" not in questions and questions[-2:] == ["question 19", "question 20"]
    assert isinstance(seen[1], HumanMessage)  # after the rules, the window starts on a question


def test_a_single_question_is_not_changed():
    messages = conversation(1)
    assert lg.compact_history(messages) == messages
