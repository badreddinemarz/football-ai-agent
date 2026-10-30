"""The MCP agent, with a fake LLM and the real MCP server connected in-process: no API call, no subprocess."""

import asyncio

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

import src.agent_mcp as agent_mcp
from app import mcp_server


class FakeChatModel(GenericFakeChatModel):
    """LangChain's fake model, made to accept tools; it records the messages of each call."""
    seen: list = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def asks(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_the_mcp_tools_become_langchain_tools_with_the_server_schema():
    async def go():
        async with create_connected_server_and_client_session(mcp_server.mcp) as session:
            return await load_mcp_tools(session)
    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"get_database_schema", "query_database", "search_laws"}
    parameters = convert_to_openai_tool(tools["query_database"])["function"]["parameters"]
    assert list(parameters["properties"]) == ["sql"] and parameters["required"] == ["sql"]


def test_the_llm_tool_call_goes_to_the_server_and_its_answer_comes_back():
    model = FakeChatModel(messages=iter([asks("query_database", {"sql": "DROP TABLE matches"}),
                                         AIMessage("I cannot do that.")]), seen=[])

    async def go():
        async with create_connected_server_and_client_session(mcp_server.mcp) as session:
            return await agent_mcp.ask("Delete the matches table", model=model, session=session)
    result = asyncio.run(go())

    assert result["answer"] == "I cannot do that."
    assert result["tools_used"] == ["query_database"] and result["sql"] == ["DROP TABLE matches"]
    tool_result = [m for m in model.seen[1] if isinstance(m, ToolMessage)][0]
    assert "query refused" in tool_result.text  # refused by the server, not by the agent
