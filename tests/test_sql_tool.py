import pytest

from src.sql_tool import UnsafeSQLError, validate_sql


@pytest.mark.parametrize("query", [
    "SELECT * FROM matches",
    "select count(*) from shots where is_goal",
    "  SELECT 1;  ",
    "WITH g AS (SELECT * FROM shots) SELECT count(*) FROM g",
    "SELECT created_at, updated_by FROM t",  # contains 'create'/'update' inside words: allowed
])
def test_accepts_read_only_queries(query):
    assert validate_sql(query).upper().lstrip().startswith(("SELECT", "WITH"))


def test_strips_trailing_semicolon():
    assert validate_sql("SELECT 1;") == "SELECT 1"


@pytest.mark.parametrize("query", [
    "DROP TABLE matches",
    "DELETE FROM shots",
    "SELECT 1; DROP TABLE matches",
    "UPDATE matches SET home_score = 10",
    "INSERT INTO cards VALUES (1)",
    "WITH x AS (SELECT 1) DELETE FROM shots",
    "ATTACH 'other.db'",
    "COPY matches TO 'out.csv'",
    "PRAGMA database_list",
    "",
])
def test_rejects_unsafe_queries(query):
    with pytest.raises(UnsafeSQLError):
        validate_sql(query)


def test_extract_sql_after_a_plan():
    from src.sql_tool import extract_sql
    text = "1. Data: goal_events\n2. Filters: ...\n3. Result: COUNT(*)\nSQL: SELECT COUNT(*) FROM goal_events"
    assert extract_sql(text) == "SELECT COUNT(*) FROM goal_events"


def test_extract_sql_without_a_plan():
    from src.sql_tool import extract_sql
    assert extract_sql("```sql\nSELECT 1 FROM matches\n```") == "SELECT 1 FROM matches"
