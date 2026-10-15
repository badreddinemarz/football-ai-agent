"""Download StatsBomb Open Data (raw JSON) for the competitions used in the project.

Data: StatsBomb Open Data — https://github.com/statsbomb/open-data
(free to use with attribution, see their license).

Files are written to data/raw/ and are never committed to git.
Re-running the script skips files that already exist.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# (competition_name, season_name) as written in StatsBomb's competitions.json
COMPETITIONS = [
    ("FIFA World Cup", "2022"),
    ("UEFA Euro", "2024"),
]


def fetch(relative_path: str) -> Path:
    """Download BASE_URL/relative_path to RAW_DIR/relative_path, unless it already exists."""
    target = RAW_DIR / relative_path
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{BASE_URL}/{relative_path}", timeout=60)
    response.raise_for_status()
    target.write_bytes(response.content)
    return target


def main() -> None:
    competitions = json.loads(fetch("competitions.json").read_text(encoding="utf-8"))

    match_ids = []
    for competition_name, season_name in COMPETITIONS:
        found = [
            c for c in competitions
            if c["competition_name"] == competition_name and c["season_name"] == season_name
        ]
        if not found:
            raise SystemExit(f"Not found in StatsBomb open data: {competition_name} {season_name}")
        c = found[0]
        matches_path = fetch(f"matches/{c['competition_id']}/{c['season_id']}.json")
        matches = json.loads(matches_path.read_text(encoding="utf-8"))
        print(f"{competition_name} {season_name}: {len(matches)} matches")
        match_ids += [m["match_id"] for m in matches]

    paths = [f"events/{m}.json" for m in match_ids] + [f"lineups/{m}.json" for m in match_ids]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, _ in enumerate(pool.map(fetch, paths), start=1):
            if i % 50 == 0 or i == len(paths):
                print(f"  {i}/{len(paths)} files")

    print(f"Done. Raw data in {RAW_DIR}")


if __name__ == "__main__":
    main()
