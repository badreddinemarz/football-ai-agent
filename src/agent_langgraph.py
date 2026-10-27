"""The same agent as src/agent.py, rebuilt with LangGraph, to compare a hand-written loop with the framework.

Same tools (the functions of src/agent.py), same instructions, same limit of MAX_TURNS tool turns.
Only the loop changes: it becomes a graph of nodes, and LangGraph runs it.

    START → llm ──(tool calls?)──yes──▶ tools ──(turns left?)──yes──▶ llm
              │                                  └──no──▶ final (answer without tools) → END
              └──no──▶ END
"""

import time

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import src.agent as hand_written
import src.llm as llm


# --- The tools: LangChain wraps our functions. The docstring becomes the description the LLM reads,
# the type hints become the JSON schema. The text goes to the LLM, the "artifact" (sql, passages,
# cost) is kept for the evaluation without being shown to the LLM.

@tool(response_format="content_and_artifact")
def query_stats_database(question: str) -> tuple[str, dict]:
    """Statistics of the 2022 FIFA World Cup and UEFA Euro 2024: matches, scores, goals, shots, xG, cards,
    lineups, passes. Use it for any question about numbers, players, teams or matches of these two competitions.

    Args:
        question: The statistics question, complete and self-contained.
    """
    output = hand_written.query_stats_database(question)
    return output["content"], output


@tool(response_format="content_and_artifact")
def search_laws_of_the_game(query: str) -> tuple[str, dict]:
    """Search the official IFAB Laws of the Game (the rules of football), e.g. offside, fouls, cards,
    penalties, throw-ins.

    Args:
        query: The rules question to search for.
    """
    output = hand_written.search_laws_of_the_game(query)
    return output["content"], output


TOOLS = [query_stats_database, search_laws_of_the_game]


def chat_model():
    """The LangChain model for llm.MODEL: Claude through the API, or a local model through Ollama."""
    if llm.MODEL.startswith("claude-"):
        return ChatAnthropic(model=llm.MODEL, max_tokens=16000)
    return ChatOllama(model=llm.MODEL, temperature=0)


def messages_since_question(messages: list) -> list:
    """The messages of the current question: from the last HumanMessage to the end. With memory, the state
    also holds the previous questions, which must count neither as turns nor as tools used now."""
    last_question = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[last_question:]


# --- What the LLM sees. The checkpointer keeps the whole conversation; before each call we send a lighter
# version: old turns become "question + answer", and the oldest are dropped past a token budget.

KEEP_FULL_TURNS = 2       # the current question and the one before keep their tool calls and results
OLD_TURNS_MAX_TOKENS = 3000  # past that, the oldest compacted turns are forgotten (sliding window)


def split_turns(messages: list) -> tuple[list, list[list]]:
    """(system messages, turns): a turn starts at a HumanMessage and holds everything until the next one."""
    system = [m for m in messages if isinstance(m, SystemMessage)]
    turns = []
    for m in messages:
        if isinstance(m, HumanMessage):
            turns.append([m])
        elif turns and not isinstance(m, SystemMessage):
            turns[-1].append(m)
    return system, turns


def without_thinking(message):
    """The message without Claude's thinking blocks. They are signed against everything before them: once
    older turns are compacted, that prefix has changed and the API would refuse the old signature."""
    if not isinstance(message, AIMessage) or not isinstance(message.content, list):
        return message
    content = [b for b in message.content if not (isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"))]
    return message.model_copy(update={"content": content})


def compact_history(messages: list) -> list:
    """The conversation as the LLM will see it. The saved conversation itself is never changed."""
    system, turns = split_turns(messages)
    old, recent = turns[:-KEEP_FULL_TURNS], turns[-KEEP_FULL_TURNS:]

    compacted = []
    for turn in old:  # a whole turn at a time, so no tool call is ever separated from its result
        question, answers = turn[0], [m for m in turn if isinstance(m, AIMessage) and m.text]
        compacted.append(question)
        if answers:
            compacted.append(AIMessage(answers[-1].text))  # the final answer, as plain text
    # Sliding window on the old part: keep the most recent compacted turns that fit, starting on a question
    compacted = trim_messages(compacted, max_tokens=OLD_TURNS_MAX_TOKENS, token_counter=count_tokens_approximately,
                              strategy="last", start_on="human")

    *previous, current = recent if recent else [[]]
    previous = [without_thinking(m) for turn in previous for m in turn]
    return system + compacted + previous + current  # the current question stays exactly as it is


def build_graph(checkpointer=None):
    """checkpointer: where LangGraph saves the state after each step. With one (e.g. InMemorySaver), a
    conversation identified by its thread_id is resumed at the next call: that is the agent's memory."""
    model = chat_model()
    with_tools = model.bind_tools(TOOLS)
    # Last turn: the tools stay described (Claude ties its thinking to them) but cannot be called.
    no_tools = model.bind_tools(TOOLS, tool_choice={"type": "none"}) if llm.MODEL.startswith("claude-") else model

    def call_llm(state: MessagesState) -> dict:
        return {"messages": [with_tools.invoke(compact_history(state["messages"]))]}

    def final_answer(state: MessagesState) -> dict:
        return {"messages": [no_tools.invoke(compact_history(state["messages"]))]}

    def after_tools(state: MessagesState) -> str:
        # Only the turns of the current question: with memory, the previous questions' turns are in the state too
        turns = sum(isinstance(m, AIMessage) for m in messages_since_question(state["messages"]))
        return "llm" if turns < hand_written.MAX_TURNS else "final"

    graph = StateGraph(MessagesState)          # the state: the list of messages, shared by the nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(TOOLS))   # runs every tool call of the last message
    graph.add_node("final", final_answer)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)   # tool calls → "tools", none → END
    graph.add_conditional_edges("tools", after_tools, ["llm", "final"])  # turns left → "llm", else → "final"
    graph.add_edge("final", END)
    return graph.compile(checkpointer=checkpointer)


def run_agent(question: str) -> dict:
    """One question, no memory. Same output as src.agent.run_agent, so the evaluation can compare the two."""
    start = time.perf_counter()
    state = build_graph().invoke({"messages": [SystemMessage(hand_written.system_prompt()), HumanMessage(question)]})
    return summarize(state["messages"], time.perf_counter() - start)


# --- Conversation memory: one InMemorySaver keeps every conversation, each under its thread_id ---

_memory_graphs = {}  # model name → graph compiled with the shared checkpointer


def memory_graph():
    if llm.MODEL not in _memory_graphs:
        _memory_graphs[llm.MODEL] = build_graph(checkpointer=InMemorySaver())
    return _memory_graphs[llm.MODEL]


def ask(question: str, thread_id: str) -> dict:
    """A question inside a conversation: the agent sees the previous questions and answers of thread_id."""
    graph, config = memory_graph(), {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()
    is_new = not graph.get_state(config).values.get("messages")
    new_messages = ([SystemMessage(hand_written.system_prompt())] if is_new else []) + [HumanMessage(question)]
    state = graph.invoke({"messages": new_messages}, config)  # LangGraph adds them to the saved conversation
    return summarize(messages_since_question(state["messages"]), time.perf_counter() - start)


def summarize(messages: list, latency_s: float) -> dict:
    """The answer and what the evaluation needs, from the messages of one question."""
    # The forced last answer can come back with only Claude's thinking and no text: say so, never answer "".
    answer = messages[-1].text or hand_written.NO_ANSWER
    result = {"answer": answer, "tools_used": [], "sql": [], "passages": [],
              "contexts": [], "tokens": 0, "latency_s": latency_s}

    price_in, price_out = llm.PRICES.get(llm.MODEL, llm.DEFAULT_PRICE) if llm.MODEL.startswith("claude-") else (0, 0)
    for message in messages:
        if isinstance(message, AIMessage) and message.usage_metadata:
            tokens_in, tokens_out = message.usage_metadata["input_tokens"], message.usage_metadata["output_tokens"]
            result["tokens"] += tokens_in + tokens_out
            llm.spent_usd += (tokens_in * price_in + tokens_out * price_out) / 1_000_000  # for the budget check
        if isinstance(message, ToolMessage) and message.artifact:  # a tool that ran (not an error)
            output = message.artifact
            result["tools_used"].append(message.name)
            if "sql" in output:
                result["sql"].append(output["sql"])
            result["passages"] += output.get("passages", [])
            result["contexts"].append(output["content"])
            result["tokens"] += output["tokens"]
    return result
