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
- A drawn knockout match (e.g. the 2022 final, 3-3) was decided on penalties: the winner is the team
  with more shots where is_goal AND is_shootout in that match.
- Players: filter on player_nickname with ILIKE, e.g. player_nickname ILIKE '%mbapp%'
  (nicknames are the common names, e.g. 'Kylian Mbappé', 'Lionel Messi').
- Teams: English names, filter with ILIKE, e.g. home_team ILIKE '%korea%' ('South Korea', 'Turkey').
- Join `shots`, `cards`, `lineups`, `player_match_stats` to `matches` on match_id to filter by competition.
- matches.stage is one of 'Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals',
  '3rd Place Final', 'Final'.
- shots.outcome is one of 'Goal', 'Saved', 'Saved to Post', 'Saved Off Target', 'Post', 'Off T',
  'Wayward', 'Blocked'. Shots on target: outcome IN ('Goal', 'Saved', 'Saved to Post').
- cards.card is one of 'Yellow Card', 'Second Yellow', 'Red Card'.
- lineups: played = TRUE means the player appeared in the match, started = TRUE means in the starting XI.
"""


# Whole words only (\b), so a column like 'created_at' is not rejected
FORBIDDEN_KEYWORDS = (
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|INSTALL|LOAD"
    r"|EXPORT|IMPORT|SET|CALL|TRUNCATE|GRANT)\b"
)


class UnsafeSQLError(ValueError):
    """Raised when a generated query is not a single read-only SELECT."""


def validate_sql(query: str) -> str:
    """Return the cleaned query if it is safe to run, otherwise raise UnsafeSQLError."""
    cleaned = query.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeSQLError("Empty query")
    if ";" in cleaned:
        raise UnsafeSQLError("Only one query is allowed")
    if not re.match(r"(SELECT|WITH)\b", cleaned, flags=re.IGNORECASE):
        raise UnsafeSQLError("Only SELECT or WITH queries are allowed")
    forbidden = re.search(FORBIDDEN_KEYWORDS, cleaned, flags=re.IGNORECASE)
    if forbidden:
        raise UnsafeSQLError(f"Forbidden keyword: {forbidden.group(0).upper()}")
    return cleaned


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


def clean_sql(text: str) -> str:
    """Small models sometimes wrap the query in ```sql ... ``` despite the instruction."""
    return re.sub(r"^```(?:sql)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()


def answer_with_sql(question: str, max_attempts: int = 2) -> dict:
    """The LLM writes a query and we run it. If it fails, the error goes back to the LLM for a new try.

    max_attempts=1 means no self-correction (used to measure what the second attempt brings).
    Returns {"sql", "rows" (DataFrame, or None if every attempt failed), "error", "attempts", "tokens", "latency_s"}.
    """
    system = (
        "You translate football questions into ONE DuckDB SQL query.\n"
        f"{schema_description()}\n"
        "Answer with the SQL query only, no explanation, no markdown."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    tokens, latency = 0, 0.0
    for attempt in range(1, max_attempts + 1):
        response = chat(messages)
        tokens += response.input_tokens + response.output_tokens
        latency += response.latency_s
        query = clean_sql(response.text)
        try:
            rows = run_sql(query)
            return {"sql": query, "rows": rows, "error": None, "attempts": attempt, "tokens": tokens, "latency_s": latency}
        except (UnsafeSQLError, duckdb.Error) as e:
            error = str(e)
            messages += [
                {"role": "assistant", "content": query},
                {"role": "user", "content": f"This query failed with the error:\n{error}\nWrite a corrected query."},
            ]
    return {"sql": query, "rows": None, "error": error, "attempts": max_attempts, "tokens": tokens, "latency_s": latency}
