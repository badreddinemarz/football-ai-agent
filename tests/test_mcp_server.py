"""The MCP server, tested through a real MCP client connected in-process: no LLM, no subprocess."""

import asyncio

import pytest
from mcp import Client

from app import mcp_server
from src.sql_tool import DB_PATH

needs_db = pytest.mark.skipif(not DB_PATH.exists(), reason="data/football.duckdb is built locally, not in git")


def call(tool: str, arguments: dict) -> str:
    """Call a tool the way any MCP client would, and return its text."""
    async def go():
        async with Client(mcp_server.mcp) as client:
            result = await client.call_tool(tool, arguments)
            return result.content[0].text
    return asyncio.run(go())


def test_a_client_discovers_the_three_tools_with_their_descriptions():
    async def go():
        async with Client(mcp_server.mcp) as client:
            return (await client.list_tools()).tools
    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == {"get_database_schema", "query_database", "search_laws"}
    assert list(tools["query_database"].input_schema["properties"]) == ["sql"]
    assert "read-only" in tools["query_database"].description


def test_a_dangerous_query_is_refused_whatever_the_client():
    assert call("query_database", {"sql": "DROP TABLE matches"}).startswith("Error: query refused")
    assert call("query_database", {"sql": "SELECT 1; DELETE FROM shots"}).startswith("Error: query refused")


@needs_db
def test_a_query_returns_rows():
    result = call("query_database", {"sql": "SELECT COUNT(*) AS n FROM goal_events "
                                            "WHERE strip_accents(player_nickname) ILIKE '%mbappe%' "
                                            "AND competition = 'FIFA World Cup'"})
    assert result.split()[-1] == "8"


@needs_db
def test_a_wrong_column_comes_back_as_an_error_the_client_can_fix():
    assert call("query_database", {"sql": "SELECT goals FROM shots"}).startswith("Error:")


def test_search_laws_formats_the_passages(monkeypatch):
    import src.rag_tool
    monkeypatch.setattr(src.rag_tool, "retrieve", lambda query: [
        {"law_number": 11, "law_title": "Offside", "section": "Offence", "text": "No offside from a throw-in."}])
    assert call("search_laws", {"query": "offside throw-in"}) == "[Law 11 – Offside – Offence]\nNo offside from a throw-in."
