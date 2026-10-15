"""Build data/football.duckdb from the raw StatsBomb JSON (run download_data.py first).

Tables (kept small and clean on purpose, so an LLM can write SQL against them):
  matches            one row per match
  lineups            one row per player listed in a match squad
  shots              one row per shot (goals = outcome 'Goal'), with xG and assister
  cards              one row per card shown
  player_match_stats one row per player per match: passes, dribbles, fouls...
"""

from collections import defaultdict
from pathlib import Path
import json

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "football.duckdb"

SHOOTOUT_PERIOD = 5  # StatsBomb period 5 = penalty shootout


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_matches(match_files):
    rows = []
    for f in match_files:
        for m in load(f):
            rows.append({
                "match_id": m["match_id"],
                "competition": m["competition"]["competition_name"],
                "season": m["season"]["season_name"],
                "match_date": m["match_date"],
                "stage": m["competition_stage"]["name"],
                "home_team": m["home_team"]["home_team_name"],
                "away_team": m["away_team"]["away_team_name"],
                "home_score": m["home_score"],
                "away_score": m["away_score"],
                "stadium": (m.get("stadium") or {}).get("name"),
                "referee": (m.get("referee") or {}).get("name"),
            })
    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


def build_lineups(match_ids):
    rows = []
    for match_id in match_ids:
        for team in load(RAW_DIR / "lineups" / f"{match_id}.json"):
            for p in team["lineup"]:
                positions = p.get("positions") or []
                rows.append({
                    "match_id": match_id,
                    "team": team["team_name"],
                    "player_id": p["player_id"],
                    "player_name": p["player_name"],
                    "player_nickname": p.get("player_nickname") or p["player_name"],
                    "jersey_number": p.get("jersey_number"),
                    "country": (p.get("country") or {}).get("name"),
                    "started": any(pos.get("start_reason") == "Starting XI" for pos in positions),
                    "played": len(positions) > 0,
                })
    return pd.DataFrame(rows)


def build_events_tables(match_ids, nicknames):
    shots, cards = [], []
    stats = defaultdict(lambda: defaultdict(int))  # (match_id, player_id, team) -> counters

    for match_id in match_ids:
        events = load(RAW_DIR / "events" / f"{match_id}.json")
        by_id = {e["id"]: e for e in events}

        for e in events:
            kind = e["type"]["name"]
            player = e.get("player")
            if player is None:
                continue
            key = (match_id, player["id"], e["team"]["name"])
            base = {
                "match_id": match_id,
                "period": e["period"],
                "minute": e["minute"],
                "team": e["team"]["name"],
                "player_id": player["id"],
                "player_name": player["name"],
                "player_nickname": nicknames.get(player["id"], player["name"]),
            }

            if kind == "Shot":
                s = e["shot"]
                is_goal = s["outcome"]["name"] == "Goal"
                key_pass = by_id.get(s.get("key_pass_id"))
                assister = key_pass["player"] if (is_goal and key_pass and key_pass.get("player")) else None
                shots.append({
                    **base,
                    "xg": s.get("statsbomb_xg"),
                    "outcome": s["outcome"]["name"],
                    "is_goal": is_goal,
                    "is_shootout": e["period"] == SHOOTOUT_PERIOD,
                    "shot_type": s["type"]["name"],          # Open Play, Penalty, Free Kick, Corner
                    "body_part": s["body_part"]["name"],
                    "assist_player_nickname": nicknames.get(assister["id"], assister["name"]) if assister else None,
                })
                if e["period"] != SHOOTOUT_PERIOD:
                    stats[key]["shots"] += 1
                    stats[key]["goals"] += is_goal
            elif kind == "Pass":
                stats[key]["passes"] += 1
                stats[key]["passes_completed"] += "outcome" not in e["pass"]
            elif kind == "Dribble":
                stats[key]["dribbles"] += 1
                stats[key]["dribbles_completed"] += e["dribble"]["outcome"]["name"] == "Complete"
            elif kind == "Interception":
                stats[key]["interceptions"] += 1
            elif kind == "Ball Recovery":
                stats[key]["ball_recoveries"] += 1

            if kind == "Foul Committed":
                stats[key]["fouls_committed"] += 1
            for field in ("foul_committed", "bad_behaviour"):
                card = (e.get(field) or {}).get("card")
                if card:
                    cards.append({**base, "card": card["name"], "reason": "Foul" if field == "foul_committed" else "Bad Behaviour"})

    stat_rows = []
    for (match_id, player_id, team), counters in stats.items():
        stat_rows.append({"match_id": match_id, "player_id": player_id, "team": team,
                          "player_nickname": nicknames.get(player_id), **counters})
    stat_cols = ["shots", "goals", "passes", "passes_completed", "dribbles", "dribbles_completed",
                 "interceptions", "ball_recoveries", "fouls_committed"]
    player_stats = pd.DataFrame(stat_rows)
    player_stats[stat_cols] = player_stats.reindex(columns=stat_cols).fillna(0).astype(int)

    return pd.DataFrame(shots), pd.DataFrame(cards), player_stats


def main() -> None:
    match_files = sorted((RAW_DIR / "matches").rglob("*.json"))
    if not match_files:
        raise SystemExit("No raw data found: run scripts/download_data.py first.")

    matches = build_matches(match_files)
    match_ids = matches["match_id"].tolist()
    lineups = build_lineups(match_ids)
    nicknames = dict(zip(lineups["player_id"], lineups["player_nickname"]))
    shots, cards, player_stats = build_events_tables(match_ids, nicknames)

    DB_PATH.unlink(missing_ok=True)
    con = duckdb.connect(str(DB_PATH))
    for name, df in [("matches", matches), ("lineups", lineups), ("shots", shots),
                     ("cards", cards), ("player_match_stats", player_stats)]:
        con.register("df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
        print(f"{name:20s} {len(df):>6} rows")
    con.close()
    print(f"Database written to {DB_PATH}")


if __name__ == "__main__":
    main()
