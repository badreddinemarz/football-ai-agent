"""Experiment: does agentic retrieval help? The agent, with and without the rule "if the passages do not
answer, search again with the official vocabulary", on 22 everyday-English rules questions
(eval/agentic_dev_questions.jsonl: a development set, none of them in eval/questions.jsonl).

    .venv/Scripts/python eval/experiments/agentic_search.py            (Claude Haiku, about 0.10 $)

Measured per question: did one of the searches find the expected section, how many searches, latency, cost.
Results are written to eval/experiments/agentic_search_results.json.
"""

from pathlib import Path
import json
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.agent as agent
import src.llm as llm
from src.rag_tool import laws_collection

MODEL = "claude-haiku-5-5"
QUESTIONS = ROOT / "eval" / "agentic_dev_questions.jsonl"
RESULTS = Path(__file__).resolve().parent / "agentic_search_results.json"


def check_gold(questions: list[dict]) -> None:
    """Every expected section must really hold the answer: its witness phrase is in one of its passages."""
    g = laws_collection().get(include=["documents", "metadatas"])
    for q in questions:
        assert any(m["law_number"] == q["law"] and m["section"] == q["section"] and q["witness"].lower() in d.lower()
                   for d, m in zip(g["documents"], g["metadatas"])), f"gold not found for {q['id']}"


def main() -> None:
    llm.MODEL = MODEL
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    check_gold(questions)
    results = {}
    for agentic in (False, True):
        agent.AGENTIC_SEARCH = agentic
        label = "agentic" if agentic else "baseline"
        rows = []
        for q in questions:
            spent_before = llm.spent_usd
            r = agent.run_agent(q["question"])
            found = any(p["law_number"] == q["law"] and p["section"] == q["section"] for p in r["passages"])
            rows.append({"id": q["id"], "found": found,
                         "searches": r["tools_used"].count("search_laws_of_the_game"),
                         "latency_s": round(r["latency_s"], 1), "cost_usd": llm.spent_usd - spent_before,
                         "answer": r["answer"]})
            print(f"{label:8s} {q['id']} found={found!s:5s} searches={rows[-1]['searches']} "
                  f"{r['latency_s']:.0f}s", flush=True)
        results[label] = rows
        print(f"== {label}: expected section found {sum(r['found'] for r in rows)}/{len(rows)} | "
              f"searches {sum(r['searches'] for r in rows)} | median latency "
              f"{statistics.median(r['latency_s'] for r in rows):.1f} s | cost {sum(r['cost_usd'] for r in rows):.3f} $",
              flush=True)
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
