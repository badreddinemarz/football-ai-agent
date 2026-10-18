"""SQL tool: the LLM writes a query, we check it is safe, then run it read-only on DuckDB."""

from pathlib import Path
import os
import re

import duckdb
import pandas as pd

from src.llm import chat

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "football.duckdb"
MAX_ROWS = 50
# The model that writes SQL can differ from the agent's (e.g. SQL_MODEL=qwen2.5-coder:7b). Empty = same model.
SQL_MODEL = os.getenv("SQL_MODEL") or None

# Rules the LLM cannot guess from column names alone.
SCHEMA_NOTES = """
- USE THE VIEWS FIRST. They already have competition and stage: NEVER JOIN them to matches.
  * goal_events: one row per real goal (no shoot-out kicks, no own goals): scorer player_nickname,
    assist_player_nickname. A player's goals or assists = COUNT(*).
  * team_matches: one row per team per match: team, opponent, goals_for, goals_against (own goals included),
    result 'win'/'draw'/'loss' (knockout draws decided by the shoot-out). Team goals, wins, winners: use it.
    Tournament winner: stage = 'Final' AND result = 'win'.
- competition: 'FIFA World Cup' (2022) or 'UEFA Euro' (2024). Base tables: JOIN matches USING (match_id).
- Filter names with ILIKE on a short part of the name: player_nickname ILIKE '%<name>%'; teams in English
  (e.g. 'South Korea', 'Turkey').
- On `shots`, shoot-out kicks (is_shootout = TRUE) are NEVER goals or shots: add AND NOT is_shootout.
  In-match penalties: shot_type = 'Penalty' (scored: AND is_goal). player_match_stats already excludes shoot-outs.
- "Most"/"top" questions: keep ties (WHERE x = (SELECT MAX(x) ...)), not LIMIT 1.
- stage: 'Group Stage', 'Round of 16', 'Quarter-finals', 'Semi-finals', '3rd Place Final', 'Final'.
- shots.outcome: 'Goal', 'Saved', 'Saved to Post', 'Saved Off Target', 'Post', 'Off T', 'Wayward', 'Blocked'.
  On target: 'Goal', 'Saved', 'Saved to Post'.
- cards.card: 'Yellow Card', 'Second Yellow', 'Red Card'. Red cards / sendings-off = 'Red Card' + 'Second Yellow'
  (a second yellow also has its first 'Yellow Card' row).
- lineups: played = TRUE appeared in the match, started = TRUE in the starting XI.
"""


# Whole words only (\b), so a column like 'created_at' is not rejected
FORBIDDEN_KEYWORDS = (
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|DETACH|COPY|PRAGMA|INSTALL|LOAD"
    r"|EXPORT|IMPORT|SET|CALL|TRUNCATE|GRANT)\b"
)


# Few-shot examples: they show the habits the notes alone did not teach qwen2.5:7b (only the tables
# needed, ties, the team_matches view, second yellows). Keep them different from the eval questions.
SQL_EXAMPLES = [
    ("How many shots on target did Harry Kane have at Euro 2024?",
     "SELECT COUNT(*) AS shots_on_target FROM shots JOIN matches USING (match_id) "
     "WHERE competition = 'UEFA Euro' AND player_nickname ILIKE '%kane%' AND NOT is_shootout "
     "AND outcome IN ('Goal', 'Saved', 'Saved to Post')"),
    ("Which players received the most yellow cards at the 2022 World Cup?",
     "WITH y AS (SELECT player_nickname, COUNT(*) AS yellow_cards FROM cards JOIN matches USING (match_id) "
     "WHERE competition = 'FIFA World Cup' AND card = 'Yellow Card' GROUP BY player_nickname) "
     "SELECT * FROM y WHERE yellow_cards = (SELECT MAX(yellow_cards) FROM y)"),
    ("How many goals did England concede at Euro 2024?",
     "SELECT SUM(goals_against) AS goals_conceded FROM team_matches "
     "WHERE competition = 'UEFA Euro' AND team ILIKE '%england%'"),
    ("How many red cards did Argentina get at the 2022 World Cup?",
     "SELECT COUNT(*) AS red_cards FROM cards JOIN matches USING (match_id) "
     "WHERE competition = 'FIFA World Cup' AND team = 'Argentina' AND card IN ('Red Card', 'Second Yellow')"),
]


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
        "Answer with the SQL query only, no explanation, no markdown. JOIN only the tables you need."
    )
    messages = [{"role": "system", "content": system}]
    for example_question, example_sql in SQL_EXAMPLES:
        messages += [{"role": "user", "content": example_question}, {"role": "assistant", "content": example_sql}]
    messages.append({"role": "user", "content": question})
    tokens, latency = 0, 0.0
    for attempt in range(1, max_attempts + 1):
        response = chat(messages, model=SQL_MODEL)
        tokens += response.input_tokens + response.output_tokens
        latency += response.latency_s
        query = clean_sql(response.text)
        try:
            rows = run_sql(query)
            return {"sql": query, "rows": rows, "error": None, "attempts": attempt, "tokens": tokens, "latency_s": latency}
        except (UnsafeSQLError, duckdb.Error) as e:
            error = str(e)
            messages += [
                {"role": "assistant", "content": response.text, "raw": response.raw},
                {"role": "user", "content": f"This query failed with the error:\n{error}\nWrite a corrected query."},
            ]
    return {"sql": query, "rows": None, "error": error, "attempts": max_attempts, "tokens": tokens, "latency_s": latency}
