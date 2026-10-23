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
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
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


def build_graph():
    model = chat_model()
    with_tools = model.bind_tools(TOOLS)
    # Last turn: the tools stay described (Claude ties its thinking to them) but cannot be called.
    no_tools = model.bind_tools(TOOLS, tool_choice={"type": "none"}) if llm.MODEL.startswith("claude-") else model

    def call_llm(state: MessagesState) -> dict:
        return {"messages": [with_tools.invoke(state["messages"])]}

    def final_answer(state: MessagesState) -> dict:
        return {"messages": [no_tools.invoke(state["messages"])]}

    def after_tools(state: MessagesState) -> str:
        turns = sum(isinstance(m, AIMessage) for m in state["messages"])
        return "llm" if turns < hand_written.MAX_TURNS else "final"

    graph = StateGraph(MessagesState)          # the state: the list of messages, shared by the nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(TOOLS))   # runs every tool call of the last message
    graph.add_node("final", final_answer)
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)   # tool calls → "tools", none → END
    graph.add_conditional_edges("tools", after_tools, ["llm", "final"])  # turns left → "llm", else → "final"
    graph.add_edge("final", END)
    return graph.compile()


def run_agent(question: str) -> dict:
    """Same output as src.agent.run_agent, so the evaluation can compare the two."""
    start = time.perf_counter()
    state = build_graph().invoke({"messages": [SystemMessage(hand_written.system_prompt()), HumanMessage(question)]})
    # The forced last answer can come back with only Claude's thinking and no text: say so, never answer "".
    answer = state["messages"][-1].text or hand_written.NO_ANSWER
    result = {"answer": answer, "tools_used": [], "sql": [], "passages": [],
              "contexts": [], "tokens": 0, "latency_s": time.perf_counter() - start}

    price_in, price_out = llm.PRICES.get(llm.MODEL, llm.DEFAULT_PRICE) if llm.MODEL.startswith("claude-") else (0, 0)
    for message in state["messages"]:
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
