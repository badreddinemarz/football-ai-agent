"""MCP server: the football statistics database and the Laws of the Game, as tools any MCP client can use.

The tools are "low level": the client's own LLM (Claude Code, Claude Desktop, Cursor...) writes the SQL and the
search query, so no LLM runs here and no API credit is spent. Safety does not depend on the client: every query
goes through run_sql (validate_sql + read-only connection + LIMIT).

    .venv/Scripts/python app/mcp_server.py          # stdio: the MCP client starts this program and talks to it
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
from mcp.server.mcpserver import MCPServer

from src.sql_tool import UnsafeSQLError, run_sql, schema_description

mcp = MCPServer(
    name="football",
    instructions=(
        "Statistics of the 2022 FIFA World Cup and UEFA Euro 2024 (StatsBomb open data), and the IFAB Laws of the "
        "Game. Before writing SQL, call get_database_schema. Answer from the tool results, never from memory."
    ),
)


@mcp.tool()
def get_database_schema() -> str:
    """The tables, columns and rules of the football statistics database. Read it before writing SQL."""
    return schema_description()


@mcp.tool()
def query_database(sql: str) -> str:
    """Run ONE read-only DuckDB SELECT query on the 2022 World Cup and Euro 2024 statistics database and return the
    rows (at most 50). Call get_database_schema first. On an error, read the message, fix the query and try again."""
    try:
        rows = run_sql(sql)
    except UnsafeSQLError as e:
        return f"Error: query refused ({e}). Only one SELECT query is allowed."
    except duckdb.Error as e:
        return f"Error: {e}"
    return rows.to_string(index=False) if len(rows) else "No rows."


@mcp.tool()
def search_laws(query: str) -> str:
    """Search the IFAB Laws of the Game (2026/27, English) and return the 5 closest passages, each with its law
    number. Write the query in English with the official vocabulary (e.g. "goalkeeper", "penalty area")."""
    from src.rag_tool import retrieve  # imported here: loading the embedding model takes a few seconds

    return "\n\n".join(f"[Law {p['law_number']} – {p['law_title']} – {p['section']}]\n{p['text']}" for p in retrieve(query))


@mcp.resource("football://schema", name="database-schema", mime_type="text/plain")
def schema_resource() -> str:
    """The database schema, for clients that let the user attach a resource to the conversation."""
    return schema_description()


if __name__ == "__main__":
    mcp.run()  # stdio by default
