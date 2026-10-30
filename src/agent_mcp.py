"""The LangGraph agent, with its tools taken from the MCP server (app/mcp_server.py) instead of Python imports.

The agent knows nothing about the database or the Laws of the Game: it connects to the server, asks for its tools,
gives them to the LLM, and the tools forward the LLM's calls to the server. The same server also serves Claude Code.

langchain-mcp-adapters does the client side for us:
    MultiServerMCPClient   knows how to start / reach each server (here one, "football", over stdio)
    client.session(...)    opens ONE connection: the server starts, and stops when the block ends
    load_mcp_tools(...)    lists the server's tools and turns each one into a LangChain tool that calls the server
"""

import asyncio
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import src.llm as llm
from src.agent_langgraph import chat_model

ROOT = Path(__file__).resolve().parents[1]
SERVERS = MultiServerMCPClient({
    "football": {"transport": "stdio", "command": sys.executable,
                 "args": [str(ROOT / "app" / "mcp_server.py")], "cwd": str(ROOT)},
})
MAX_STEPS = 12  # LangGraph's recursion_limit: llm and tools nodes together, so about 5 tool rounds

SYSTEM_PROMPT = (
    "You are a football assistant. Use the tools of the football server: read the database schema before writing "
    "SQL, and search the Laws of the Game for rules questions. Answer only from the tool results, in English, and "
    "cite the law for rules. If the question is not about football statistics or rules, say you cannot help."
)


def build_graph(model, tools):
    """The usual agent graph. The nodes are async: calling an MCP tool means waiting for the server."""
    with_tools = model.bind_tools(tools)

    async def call_llm(state: MessagesState) -> dict:
        return {"messages": [await with_tools.ainvoke(state["messages"])]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", call_llm)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition)
    graph.add_edge("tools", "llm")
    return graph.compile()


async def ask(question: str, model=None, session=None) -> dict:
    """Run the agent on the question. Without a session, open one to the football server for this question."""
    if session is None:
        async with SERVERS.session("football") as session:  # with stdio, this starts the server; leaving stops it
            return await ask(question, model, session)

    start = time.perf_counter()
    tools = await load_mcp_tools(session)  # tools/list, then one LangChain tool per MCP tool
    graph = build_graph(model or chat_model(), tools)
    state = await graph.ainvoke({"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]},
                                {"recursion_limit": MAX_STEPS})

    calls = [c for m in state["messages"] if isinstance(m, AIMessage) for c in m.tool_calls]
    tokens = sum(m.usage_metadata["total_tokens"] for m in state["messages"]
                 if isinstance(m, AIMessage) and m.usage_metadata)
    return {"answer": state["messages"][-1].text,
            "tools_used": [c["name"] for c in calls],
            "sql": [c["args"]["sql"] for c in calls if c["name"] == "query_database"],
            "tokens": tokens,
            "latency_s": time.perf_counter() - start}


def run_agent(question: str) -> dict:
    return asyncio.run(ask(question))


if __name__ == "__main__":
    llm.MODEL = "claude-haiku-5-5"
    result = run_agent(" ".join(sys.argv[1:]) or "How many goals did Kylian Mbappe score at the 2022 World Cup?")
    print(result["answer"], "\n\ntools:", result["tools_used"], "\nsql:", result["sql"],
          f"\n{result['tokens']} tokens, {result['latency_s']:.1f} s")
