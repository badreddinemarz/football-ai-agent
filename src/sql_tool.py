"""SQL tool: the LLM writes a query, we check it is safe, then run it read-only on DuckDB."""

from pathlib import Path
import re

import duckdb
import pandas as pd

from src.llm import chat

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "football.duckdb"
MAX_ROWS = 50

# Rules the LLM cannot guess from column names alone.
SCHEMA_NOTES = """
- Competitions: 'FIFA World Cup' season '2022', 'UEFA Euro' season '2024'.
- A goal is a row of `shots` with is_goal = TRUE. Penalty-shootout kicks have is_shootout = TRUE
  and NEVER count as goals: always add `AND NOT is_shootout` when counting goals or shots.
  Own goals are not in `shots`. Match scores (matches.home_score/away_score) exclude shootouts.
- Players: filter on player_nickname with ILIKE, e.g. player_nickname ILIKE '%mbapp%'
  (nicknames are the common names, e.g. 'Kylian Mbappé', 'Lionel Messi').
- Join `shots`, `cards`, `lineups`, `player_match_stats` to `matches` on match_id to filter by competition.
- cards.card is one of 'Yellow Card', 'Second Yellow', 'Red Card'.
"""


class UnsafeSQLError(ValueError):
    """Raised when a generated query is not a single read-only SELECT."""


def validate_sql(query: str) -> str:
    """Return the cleaned query if it is safe to run, otherwise raise UnsafeSQLError.

    TODO (toi) — règles à implémenter, les tests dans tests/test_sql_tool.py te guident :
      1. enlever les espaces et un éventuel ';' final
      2. refuser plusieurs requêtes (un ';' au milieu)
      3. n'accepter que les requêtes qui commencent par SELECT ou WITH
      4. refuser les mots-clés dangereux (DROP, DELETE, INSERT, UPDATE, ALTER, CREATE,
         ATTACH, COPY, PRAGMA, INSTALL, LOAD...) — attention : un mot entier seulement,
         une colonne 'created_at' ne doit pas être refusée (indice : re et \\b)
    """
    raise NotImplementedError("validate_sql: à toi de jouer !")


def schema_description() -> str:
    """Tables and columns, read from the database itself, plus the hand-written notes."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    columns = con.execute(
        "SELECT table_name, column_name, data_type FROM information_schema.columns "
        "ORDER BY table_name, ordinal_position"
    ).fetchall()
    con.close()

    tables: dict[str, list[str]] = {}
    for table, column, dtype in columns:
        tables.setdefault(table, []).append(f"{column} {dtype}")
    lines = [f"{table}({', '.join(cols)})" for table, cols in tables.items()]
    return "Tables:\n" + "\n".join(lines) + "\n\nRules:" + SCHEMA_NOTES


def run_sql(query: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Validate then execute a query. The connection is read-only as a second safety net."""
    safe_query = validate_sql(query)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute(f"SELECT * FROM ({safe_query}) LIMIT {max_rows}").df()
    finally:
        con.close()


def question_to_sql(question: str) -> str:
    """Ask the LLM to translate a question into one DuckDB SELECT query."""
    system = (
        "You translate football questions into ONE DuckDB SQL query.\n"
        f"{schema_description()}\n"
        "Answer with the SQL query only, no explanation, no markdown."
    )
    response = chat([{"role": "system", "content": system}, {"role": "user", "content": question}])
    # Small models sometimes wrap the query in ```sql ... ``` despite the instruction
    return re.sub(r"^```(?:sql)?|```$", "", response.text.strip(), flags=re.IGNORECASE).strip()
