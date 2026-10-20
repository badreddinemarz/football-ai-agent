"""Build eval/questions.jsonl: the evaluation questions and their expected answers.

Numbers and names are never typed by hand: each statistics question has a gold SQL query, run on
data/football.duckdb, and its result becomes "must_contain" (what a correct answer has to contain).
Rules questions have a reference answer and the law(s) it comes from, checked in the 2026/27 PDF.

These questions are new: none of them was used to tune the prompts (eval/sql_dev_questions.jsonl,
eval/rag_questions.jsonl and the few-shot examples).

Run: .venv/Scripts/python eval/build_gold.py
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "src" can be imported

from src.sql_tool import run_sql

OUT = Path(__file__).resolve().parent / "questions.jsonl"
STATS_TOOL, LAWS_TOOL = "query_stats_database", "search_laws_of_the_game"

# (question, gold SQL)
STATS = [
    ("How many goals did Lionel Messi score at the 2022 World Cup?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'FIFA World Cup' AND player_nickname ILIKE '%messi%'"),
    ("How many goals did Harry Kane score at Euro 2024?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'UEFA Euro' AND player_nickname ILIKE '%kane%'"),
    ("How many yellow cards did Spain receive at Euro 2024?",
     "SELECT COUNT(*) FROM cards JOIN matches USING (match_id) WHERE competition = 'UEFA Euro' AND team = 'Spain' AND card = 'Yellow Card'"),
    ("How many matches did Morocco play at the 2022 World Cup?",
     "SELECT COUNT(*) FROM team_matches WHERE competition = 'FIFA World Cup' AND team = 'Morocco'"),
    ("How many goals did Olivier Giroud score at the 2022 World Cup?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'FIFA World Cup' AND player_nickname ILIKE '%giroud%'"),
    ("Which team did Spain play in the Euro 2024 semi-final?",
     "SELECT opponent FROM team_matches WHERE competition = 'UEFA Euro' AND team = 'Spain' AND stage = 'Semi-finals'"),
    ("How many matches at the 2022 World Cup were decided by a penalty shoot-out?",
     "SELECT COUNT(DISTINCT match_id) FROM team_matches WHERE competition = 'FIFA World Cup' AND decided_by_shootout"),
    ("How many headed goals were scored at the 2022 World Cup?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'FIFA World Cup' AND body_part = 'Head'"),
    ("How many headed goals were scored at Euro 2024?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'UEFA Euro' AND body_part = 'Head'"),
    ("Which team scored the most goals at Euro 2024, and how many?",
     "WITH t AS (SELECT team, SUM(goals_for) AS g FROM team_matches WHERE competition = 'UEFA Euro' GROUP BY team) "
     "SELECT * FROM t WHERE g = (SELECT MAX(g) FROM t)"),
    ("How many goals did Kylian Mbappé score in the 2022 World Cup final, not counting the shoot-out?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'FIFA World Cup' AND stage = 'Final' AND player_nickname ILIKE '%mbapp%'"),
    ("How many players were sent off at the 2022 World Cup?",
     "SELECT COUNT(*) FROM cards JOIN matches USING (match_id) WHERE competition = 'FIFA World Cup' AND card IN ('Red Card', 'Second Yellow')"),
    ("How many penalties were missed during matches at Euro 2024, not counting shoot-outs?",
     "SELECT COUNT(*) FROM shots JOIN matches USING (match_id) WHERE competition = 'UEFA Euro' AND shot_type = 'Penalty' AND NOT is_goal AND NOT is_shootout"),
    ("Which player took the most shots at the 2022 World Cup, and how many?",
     "WITH s AS (SELECT player_nickname, COUNT(*) AS n FROM shots JOIN matches USING (match_id) "
     "WHERE competition = 'FIFA World Cup' AND NOT is_shootout GROUP BY player_nickname) SELECT * FROM s WHERE n = (SELECT MAX(n) FROM s)"),
    ("In how many matches did Jude Bellingham play at Euro 2024?",
     "SELECT COUNT(*) FROM lineups JOIN matches USING (match_id) WHERE competition = 'UEFA Euro' AND player_nickname ILIKE '%bellingham%' AND played"),
    ("How many matches did Argentina lose at the 2022 World Cup?",
     "SELECT COUNT(*) FROM team_matches WHERE competition = 'FIFA World Cup' AND team = 'Argentina' AND result = 'loss'"),
    ("How many group-stage matches ended in a draw at Euro 2024?",
     "SELECT COUNT(*) FROM matches WHERE competition = 'UEFA Euro' AND stage = 'Group Stage' AND home_score = away_score"),
    ("How many goals did Cristiano Ronaldo score at the 2022 World Cup?",
     "SELECT COUNT(*) FROM goal_events WHERE competition = 'FIFA World Cup' AND player_nickname ILIKE '%ronaldo%'"),
    ("How many players scored at both the 2022 World Cup and Euro 2024?",
     "SELECT COUNT(*) FROM (SELECT player_id FROM goal_events WHERE competition = 'FIFA World Cup' "
     "INTERSECT SELECT player_id FROM goal_events WHERE competition = 'UEFA Euro')"),
    ("Which players scored the most headed goals at the 2022 World Cup, and how many?",
     "WITH h AS (SELECT player_nickname, COUNT(*) AS n FROM goal_events WHERE competition = 'FIFA World Cup' AND body_part = 'Head' "
     "GROUP BY player_nickname) SELECT * FROM h WHERE n = (SELECT MAX(n) FROM h)"),
    ("How many matches did the Netherlands win at Euro 2024?",
     "SELECT COUNT(*) FROM team_matches WHERE competition = 'UEFA Euro' AND team = 'Netherlands' AND result = 'win'"),
    ("How many yellow cards did Argentina receive in the 2022 World Cup final?",
     "SELECT COUNT(*) FROM cards JOIN matches USING (match_id) WHERE competition = 'FIFA World Cup' AND stage = 'Final' AND team = 'Argentina' AND card = 'Yellow Card'"),
    ("How many matches did England draw at Euro 2024?",
     "SELECT COUNT(*) FROM team_matches WHERE competition = 'UEFA Euro' AND team = 'England' AND result = 'draw'"),
    ("How many penalties were taken during matches at the 2022 World Cup, not counting shoot-outs?",
     "SELECT COUNT(*) FROM shots JOIN matches USING (match_id) WHERE competition = 'FIFA World Cup' AND shot_type = 'Penalty' AND NOT is_shootout"),
    ("In which stadium was the Euro 2024 final played?",
     "SELECT stadium FROM matches WHERE competition = 'UEFA Euro' AND stage = 'Final'"),
    ("Which player made the most assists at Euro 2024, and how many?",
     "WITH a AS (SELECT assist_player_nickname, COUNT(*) AS n FROM goal_events WHERE competition = 'UEFA Euro' "
     "AND assist_player_nickname IS NOT NULL GROUP BY 1) SELECT * FROM a WHERE n = (SELECT MAX(n) FROM a)"),
    ("Who was the top scorer of the 2022 World Cup, and how many goals did they score?",
     "WITH g AS (SELECT player_nickname, COUNT(*) AS n FROM goal_events WHERE competition = 'FIFA World Cup' GROUP BY 1) "
     "SELECT * FROM g WHERE n = (SELECT MAX(n) FROM g)"),
    ("How many goals were scored in total at the 2022 World Cup?",
     "SELECT SUM(goals_for) FROM team_matches WHERE competition = 'FIFA World Cup'"),
    ("Which team did Argentina beat in the 2022 World Cup semi-final?",
     "SELECT opponent FROM team_matches WHERE competition = 'FIFA World Cup' AND team = 'Argentina' AND stage = 'Semi-finals'"),
    ("How many matches did France win at the 2022 World Cup?",
     "SELECT COUNT(*) FROM team_matches WHERE competition = 'FIFA World Cup' AND team = 'France' AND result = 'win'"),
]

# (question, reference answer, laws where the answer is)
RULES = [
    ("Can a match continue if a team has only six players?",
     "No. A match may not start or continue if either team has fewer than seven players.", [3]),
    ("What happens if the goalkeeper handles the ball after a team-mate deliberately kicked it to them?",
     "An indirect free kick is awarded to the opponents.", [12]),
    ("Can a player be offside when receiving the ball directly from a corner kick?",
     "No. There is no offside offence when a player receives the ball directly from a corner kick.", [11]),
    ("How far apart are the goalposts?",
     "The distance between the inside of the posts is 7.32 m (8 yds).", [1]),
    ("What is the radius of the centre circle?",
     "9.15 m (10 yds).", [1]),
    ("How far from the goal is the penalty mark?",
     "11 m (12 yds) from the midpoint between the goalposts.", [1]),
    ("What happens if the player taking a free kick touches the ball again before anyone else has touched it?",
     "An indirect free kick is awarded to the opponents (a direct free kick if it is a handball offence).", [13]),
    ("Can a goal be scored directly from a throw-in into the opponents' goal?",
     "No. If the ball enters the opponents' goal directly from a throw-in, a goal kick is awarded.", [15]),
    ("How far must opponents stay from the corner arc at a corner kick?",
     "At least 9.15 m (10 yds) until the ball is in play.", [17]),
    ("Is a goal given if the ball touches the referee and goes directly into the goal?",
     "No. The ball is out of play and play restarts with a dropped ball.", [9]),
    ("Is the penalty taker allowed to feint after completing the run-up?",
     "No. Feinting once the run-up is completed is an offence and the referee cautions the kicker; feinting during the run-up is allowed.", [14]),
    ("From where is a goal kick taken?",
     "From any point within the goal area, by a player of the defending team.", [16]),
    ("Where must the ball be placed for a kick-off?",
     "The ball must be stationary on the centre mark.", [8]),
    ("What is awarded if a throw-in goes directly into the thrower's own goal?",
     "A corner kick is awarded to the opponents.", [15]),
    ("Does the team captain have special privileges?",
     "No. The captain has no special status or privileges but has some responsibility for the team's behaviour.", [3]),
    ("Can the referee change a restart decision after play has restarted?",
     "No. The referee may not change a restart decision once play has restarted.", [5]),
    ("Are a player's arms taken into account when judging offside?",
     "No. The hands and arms of all players, including goalkeepers, are not considered (the arm starts at the bottom of the armpit).", [11]),
    ("How long is extra time?",
     "Two equal periods not exceeding 15 minutes each.", [7, 10]),
    ("How many kicks does each team take at first in a penalty shoot-out?",
     "Five kicks each, taken alternately (then one each until there is a winner if still level).", [10]),
    ("During a penalty kick, the goalkeeper offends and the ball still goes into the goal. What happens?",
     "The goal is awarded.", [14]),
    ("What happens if a goal is scored while one of the scoring team's substitutes was on the field and interfered with play?",
     "The referee must disallow the goal (and restart with a direct free kick).", [3]),
    ("How is the winner of a match decided?",
     "The team scoring the greater number of goals wins; if both teams score the same number of goals, the match is drawn.", [10]),
    ("Can a goal be scored directly from an indirect free kick?",
     "No. If an indirect free kick goes directly into the opponents' goal, a goal kick is awarded.", [13]),
    ("Who can be cautioned or sent off during a penalty shoot-out?",
     "A player, substitute, substituted player or team official.", [10]),
    ("What colour must a player's undershirt be?",
     "A single colour that is the same as the main colour of the shirt sleeve, or a pattern/colours that exactly replicate the shirt sleeve.", [4]),
]

# (question, gold SQL for the statistics part, reference answer for the rules part, laws)
MIXED = [
    ("How many matches went to a penalty shoot-out at Euro 2024, and how many kicks does each team take at first in a shoot-out?",
     "SELECT COUNT(DISTINCT match_id) FROM team_matches WHERE competition = 'UEFA Euro' AND decided_by_shootout",
     "Each team first takes five kicks.", [10]),
    ("How many own goals were scored at Euro 2024, and what is awarded if a throw-in goes directly into the thrower's own goal?",
     "SELECT (SELECT SUM(goals_for) FROM team_matches WHERE competition = 'UEFA Euro') - (SELECT COUNT(*) FROM goal_events WHERE competition = 'UEFA Euro')",
     "A corner kick is awarded to the opponents.", [15]),
    ("How many players were sent off for a second yellow card at Euro 2024, and what is the sanction for a second caution in the same match?",
     "SELECT COUNT(*) FROM cards JOIN matches USING (match_id) WHERE competition = 'UEFA Euro' AND card = 'Second Yellow'",
     "A second caution in the same match is a sending-off offence: the player is sent off.", [12]),
    ("How many penalties were missed during matches at the 2022 World Cup, and what happens if the goalkeeper offends and the penalty is missed?",
     "SELECT COUNT(*) FROM shots JOIN matches USING (match_id) WHERE competition = 'FIFA World Cup' AND shot_type = 'Penalty' AND NOT is_goal AND NOT is_shootout",
     "The kick is only retaken if the goalkeeper's offence clearly impacted on the kicker.", [14]),
    ("How many matches did Argentina play at the 2022 World Cup, and how long is each period of extra time?",
     "SELECT COUNT(*) FROM team_matches WHERE competition = 'FIFA World Cup' AND team = 'Argentina'",
     "Extra time is two equal periods not exceeding 15 minutes each.", [7, 10]),
    ("Which team received the most yellow cards at Euro 2024, and can a team official be cautioned?",
     "WITH y AS (SELECT team, COUNT(*) AS n FROM cards JOIN matches USING (match_id) WHERE competition = 'UEFA Euro' "
     "AND card = 'Yellow Card' GROUP BY team) SELECT team FROM y WHERE n = (SELECT MAX(n) FROM y)",
     "Yes. Team officials can be cautioned or sent off.", [5, 10, 12]),
    ("What was Germany's goal difference in the Euro 2024 group stage, and how is the winner of a match decided?",
     "SELECT SUM(goals_for - goals_against) FROM team_matches WHERE competition = 'UEFA Euro' AND team = 'Germany' AND stage = 'Group Stage'",
     "The team scoring the greater number of goals wins; equal goals is a draw.", [10]),
    ("How many goals were scored in the knockout stage of the 2022 World Cup, and is a goal given if the ball touches the referee and goes straight into the goal?",
     "SELECT SUM(goals_for) FROM team_matches WHERE competition = 'FIFA World Cup' AND stage <> 'Group Stage'",
     "No. The ball is out of play and play restarts with a dropped ball.", [9]),
    ("Who refereed the Euro 2024 final, and can the referee change a restart decision after play has restarted?",
     "SELECT referee FROM matches WHERE competition = 'UEFA Euro' AND stage = 'Final'",
     "No. The referee may not change a restart decision once play has restarted.", [5]),
    ("How many matches were played at Euro 2024, and how long is the half-time interval?",
     "SELECT COUNT(*) FROM matches WHERE competition = 'UEFA Euro'",
     "The half-time interval must not exceed 15 minutes.", [7]),
]

OFF_TOPIC = [
    "What is the boiling point of water?",
    "Who won the 2018 World Cup?",  # football, but outside the data: the agent must not invent
    "Write me a short poem about the sea.",
    "What is the best recipe for pizza dough?",
    "How many goals did Pelé score in his career?",
]
DECLINE = "The assistant should say that it cannot answer: the question is outside its scope (2022 World Cup and Euro 2024 statistics, Laws of the Game)."

# Questions left out of the evaluation, with the reason. Ids stay the same (they are positions in the
# lists above), so the cached runs of the kept questions remain valid.
# Every kept statistics answer was recomputed independently from the raw StatsBomb JSON (no database,
# no views) and every kept rules reference was found word for word in the PDF, in the expected law.
DROPPED = {
    # ambiguous
    "stat20": "the data names the player 'Cho Kyu-Sung', his usual name is 'Cho Gue-sung': a right answer could fail",
    "stat22": "one of the 5 yellow cards was given during the shoot-out: is it 'in the final'?",
    "stat23": "England-Switzerland was 1-1 then won on penalties: officially a draw (3), a win in our view (2)",
    "mixed06": "card counts depend on how StatsBomb logs bench cards; the rules part needs three laws",
    # same kind of question as another one: not needed for a correct evaluation
    "stat03": "card count, like stat12", "stat05": "player goals, like stat01", "stat09": "headed goals, like stat08",
    "stat18": "player goals, like stat01", "stat24": "penalty count, like stat13", "stat29": "semi-final opponent, like stat06",
    "stat30": "team wins, like stat21", "mixed09": "same rule as rule16",
    "rule05": "Law 1 measure, like rule04", "rule06": "Law 1 measure, like rule04", "rule14": "same rule as mixed02",
    "rule15": "Law 3, like rule01", "rule18": "same rule as mixed05", "rule20": "Law 14, like rule11",
    "rule21": "Law 3, like rule01", "rule22": "same rule as mixed07", "rule23": "Law 13, like rule07",
    "rule24": "Law 10, like rule19",
}


def gold_values(sql: str) -> list[str]:
    """What a correct answer must contain: every value of the gold result (numbers without '.0')."""
    values = []
    for value in run_sql(sql).values.ravel():
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        values.append(str(value))
    return values


def main() -> None:
    rows = []
    for i, (question, sql) in enumerate(STATS, 1):
        must = gold_values(sql)
        rows.append({"id": f"stat{i:02d}", "category": "stats", "question": question,
                     "expected_answer": ", ".join(must), "must_contain": must,
                     "expected_tools": [STATS_TOOL], "expected_laws": [], "gold_sql": sql})
    for i, (question, reference, laws) in enumerate(RULES, 1):
        rows.append({"id": f"rule{i:02d}", "category": "rules", "question": question,
                     "expected_answer": reference, "must_contain": [],
                     "expected_tools": [LAWS_TOOL], "expected_laws": laws})
    for i, (question, sql, reference, laws) in enumerate(MIXED, 1):
        must = gold_values(sql)
        rows.append({"id": f"mixed{i:02d}", "category": "mixed", "question": question,
                     "expected_answer": f"{', '.join(must)}. {reference}", "must_contain": must,
                     "expected_tools": [STATS_TOOL, LAWS_TOOL], "expected_laws": laws, "gold_sql": sql})
    for i, question in enumerate(OFF_TOPIC, 1):
        rows.append({"id": f"off{i:02d}", "category": "off_topic", "question": question,
                     "expected_answer": DECLINE, "must_contain": [], "expected_tools": [], "expected_laws": []})

    rows = [row for row in rows if row["id"] not in DROPPED]
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} questions written to {OUT} ({len(DROPPED)} left out, see DROPPED)")


if __name__ == "__main__":
    main()
