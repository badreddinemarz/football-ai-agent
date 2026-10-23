"""Evaluate one model in 4 configurations on eval/questions.jsonl, then write eval/results.md.

    .venv/Scripts/python eval/evaluate.py --model claude-haiku-5-5
    .venv/Scripts/python eval/evaluate.py --model qwen2.5:7b        (hours on a CPU: run it at night)
    .venv/Scripts/python eval/evaluate.py --model claude-haiku-5-5 --sample 10   (10 questions, to measure the cost)
    .venv/Scripts/python eval/evaluate.py --report                  (only rewrite results.md)

Configurations: (A) the LLM alone, (B) RAG alone, (C) SQL alone, (D) the agent with both tools.
Every answer and every judgement is saved in eval/runs/<model>/<config>/<id>.json: an interrupted run
resumes where it stopped, and nothing is ever paid or computed twice.

Judges: correctness by JUDGE_MODEL (Claude Sonnet 5.5: it never grades its own answers), faithfulness by
the cheaper FAITHFULNESS_JUDGE_MODEL. The Claude API spend is added up in eval/runs/spent_usd.json and the
run stops by itself before going over --budget dollars.
"""

from pathlib import Path
import argparse
import json
import os
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.llm as llm
from src.agent import query_stats_database, run_agent
from src.llm import chat
from src.rag_tool import answer_from_laws

from metrics import contains_all, recall_at_k, right_tools

QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"
RUNS = Path(__file__).resolve().parent / "runs"
RESULTS = Path(__file__).resolve().parent / "results.md"
SPENT = RUNS / "spent_usd.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
FAITHFULNESS_JUDGE_MODEL = os.getenv("FAITHFULNESS_JUDGE_MODEL", "claude-haiku-5-5")


# --- The 4 configurations. Each returns the answer, what it was based on, and its cost. ---

LLM_ONLY_PROMPT = (
    "You are a football assistant for the 2022 FIFA World Cup, UEFA Euro 2024 and the Laws of the Game. "
    "Answer in English, in a few sentences. If the question is about something else, say you can only "
    "help with these topics."
)


def llm_only(question: str) -> dict:
    r = chat([{"role": "system", "content": LLM_ONLY_PROMPT}, {"role": "user", "content": question}])
    return {"answer": r.text, "tools_used": [], "passages": [], "context": "",
            "tokens": r.input_tokens + r.output_tokens, "latency_s": r.latency_s}


def rag_only(question: str) -> dict:
    r = answer_from_laws(question)
    context = "\n\n".join(f"[Law {p['law_number']} – {p['section']}]\n{p['text']}" for p in r["passages"])
    return {"answer": r["answer"], "tools_used": [], "passages": r["passages"], "context": context,
            "tokens": r["tokens"], "latency_s": r["latency_s"]}


def sql_only(question: str) -> dict:
    tool = query_stats_database(question)  # the SQL result as text, as the agent would see it
    r = chat([{"role": "system", "content": "Answer the question in English, in a few sentences, using ONLY this "
                                            "SQL result. If it does not answer the question, say you don't know."
                                            f"\n\nSQL result:\n{tool['content']}"},
              {"role": "user", "content": question}])
    return {"answer": r.text, "tools_used": [], "passages": [], "context": tool["content"], "sql": tool["sql"],
            "tokens": tool["tokens"] + r.input_tokens + r.output_tokens, "latency_s": tool["latency_s"] + r.latency_s}


def agent(question: str) -> dict:
    r = run_agent(question)
    return {"answer": r["answer"], "tools_used": r["tools_used"], "passages": r["passages"],
            "context": "\n\n".join(r["contexts"]), "sql": r["sql"], "tokens": r["tokens"], "latency_s": r["latency_s"]}


def langgraph_agent(question: str) -> dict:
    from src.agent_langgraph import run_agent as run_langgraph  # imported here: LangGraph is only needed for E
    r = run_langgraph(question)
    return {"answer": r["answer"], "tools_used": r["tools_used"], "passages": r["passages"],
            "context": "\n\n".join(r["contexts"]), "sql": r["sql"], "tokens": r["tokens"], "latency_s": r["latency_s"]}


CONFIGS = {"A_llm_only": llm_only, "B_rag_only": rag_only, "C_sql_only": sql_only, "D_agent": agent,
           "E_langgraph": langgraph_agent}


# --- The LLM judge ---

CORRECTNESS_PROMPT = """You grade an answer to a football question against a reference answer.

Question: {question}
Reference answer: {reference}
Answer to grade: {answer}

Is the answer correct, i.e. does it agree with the reference on every key fact? Extra details are fine;
a wrong, contradicting or missing key fact is not. Grade ONLY against the reference: do not use your
own football knowledge to judge details the reference does not mention. Reply with CORRECT or INCORRECT
on the first line, then one short sentence explaining why."""

FAITHFULNESS_PROMPT = """You check whether an answer invents facts that are not in the context the assistant received.

Question: {question}

Context the assistant received (database results and/or passages of the Laws of the Game):
{context}

Answer: {answer}

The context answers this question, so restating the question's own terms (competition, player, team)
is fine, and so is saying what the assistant covers (2022 World Cup and Euro 2024 statistics, Laws of the
Game). Check the facts the answer adds: numbers, names, rules, law numbers. Is each of them supported
by the context? Saying that the information is not available counts as supported. Reply with SUPPORTED
or UNSUPPORTED on the first line, then one short sentence explaining why."""


def judge(prompt: str, model: str) -> dict:
    """Ask a judge; True if its first word is CORRECT or SUPPORTED (not INCORRECT / UNSUPPORTED)."""
    r = chat([{"role": "user", "content": prompt}], model=model, effort="medium")
    first_word = re.search(r"[A-Za-z]+", r.text)
    verdict = first_word.group(0).upper() if first_word else ""
    return {"ok": verdict in ("CORRECT", "SUPPORTED"), "explanation": r.text.strip()}


def grade(q: dict, run: dict) -> dict:
    """All the metrics for one answer."""
    grades = {"recall_at_5": recall_at_k(run["passages"], q["expected_laws"])}
    if run["config"] in ("D_agent", "E_langgraph"):
        grades["right_tools"] = right_tools(run["tools_used"], q["expected_tools"])

    checks = []
    if q["must_contain"]:  # numbers and names: checked automatically
        grades["contains_gold"] = contains_all(run["answer"], q["must_contain"])
        checks.append(grades["contains_gold"])
    if q["category"] != "stats":  # rules, mixed, off topic: the judge compares with the reference
        grades["judge"] = judge(CORRECTNESS_PROMPT.format(
            question=q["question"], reference=q["expected_answer"], answer=run["answer"]), JUDGE_MODEL)
        checks.append(grades["judge"]["ok"])
    grades["correct"] = all(checks)

    if run["context"]:
        grades["faithfulness"] = judge(FAITHFULNESS_PROMPT.format(
            question=q["question"], context=run["context"], answer=run["answer"]), FAITHFULNESS_JUDGE_MODEL)
    return grades


# --- Run, with a cache on disk ---

def total_spent() -> float:
    """Claude API dollars spent by all the runs so far (previous ones are saved in SPENT)."""
    before = json.loads(SPENT.read_text())["usd"] if SPENT.exists() else 0.0
    return before + llm.spent_usd


def save_spent() -> None:
    SPENT.parent.mkdir(parents=True, exist_ok=True)
    SPENT.write_text(json.dumps({"usd": round(total_spent(), 4)}))
    llm.spent_usd = 0.0  # now counted in the file


def evaluate(model: str, configs: list[str], budget: float, sample: int | None, tag: str | None = None) -> None:
    llm.MODEL = model  # every chat() call without an explicit model now uses this one
    label = f"{model}@{tag}" if tag else model  # a tag keeps a new version's runs apart from the old ones
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    if sample:  # evenly spread, so every category is represented
        questions = questions[::max(1, len(questions) // sample)][:sample]
    errors_in_a_row = 0
    for config in configs:
        folder = RUNS / label.replace(":", "_") / config
        folder.mkdir(parents=True, exist_ok=True)
        for i, q in enumerate(questions, 1):
            if total_spent() >= budget:
                save_spent()
                print(f"Budget of {budget:.2f} $ reached: stopping. Raise --budget to continue (nothing is lost).")
                return
            path = folder / f"{q['id']}.json"
            run = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            # An error (no internet, Ollama stopped...) is never saved: it would count as a wrong answer
            # forever. The question is simply retried at the next run.
            try:
                if run is None:
                    run = CONFIGS[config](q["question"])
                    run.update(id=q["id"], category=q["category"], config=config, model=label)
                    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
                if "grades" not in run:
                    run["grades"] = grade(q, run)
                    path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
                errors_in_a_row = 0
            except Exception as e:
                save_spent()
                errors_in_a_row += 1
                print(f"{model} {config} {q['id']}: ERROR {e!r} (will be retried at the next run)", flush=True)
                if errors_in_a_row >= 3:
                    print("3 errors in a row (internet or Ollama down?): stopping. Run the same command again later.")
                    return
                continue
            save_spent()
            print(f"{label} {config} {i}/{len(questions)} {q['id']}: "
                  f"{'OK' if run['grades']['correct'] else 'KO'} ({run['latency_s']:.0f} s) "
                  f"| spent so far: {total_spent():.3f} $", flush=True)


# --- Report ---

def rate(values: list) -> str:
    values = [v for v in values if v is not None]
    return f"{sum(values)}/{len(values)} ({100 * sum(values) / len(values):.0f} %)" if values else "–"


def write_report() -> None:
    kept = {json.loads(line)["id"] for line in QUESTIONS.open(encoding="utf-8")}
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RUNS.glob("*/*/*.json"))]
    runs = [r for r in runs if "grades" in r and r["id"] in kept]  # questions left out are ignored
    lines = [
        "# Evaluation results", "",
        f"{len({r['id'] for r in runs})} questions (eval/questions.jsonl). Correctness judge: `{JUDGE_MODEL}`, "
        f"faithfulness judge: `{FAITHFULNESS_JUDGE_MODEL}`. Generated by `eval/evaluate.py` from the cached runs "
        "in `eval/runs/`.", "",
        "| Model | Configuration | Correct (all) | Stats | Rules | Mixed | Off topic | Right tools | Recall@5 | Faithful | Latency (median) | Tokens (mean) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for model in sorted({r["model"] for r in runs}):
        for config in CONFIGS:
            rs = [r for r in runs if r["model"] == model and r["config"] == config]
            if not rs:
                continue
            by_cat = lambda c: rate([r["grades"]["correct"] for r in rs if r["category"] == c])
            lines.append("| " + " | ".join([
                f"`{model}`", config, rate([r["grades"]["correct"] for r in rs]),
                by_cat("stats"), by_cat("rules"), by_cat("mixed"), by_cat("off_topic"),
                rate([r["grades"].get("right_tools") for r in rs]),
                rate([r["grades"]["recall_at_5"] for r in rs]),
                rate([r["grades"]["faithfulness"]["ok"] for r in rs if "faithfulness" in r["grades"]]),
                f"{statistics.median(r['latency_s'] for r in rs):.1f} s",
                f"{sum(r['tokens'] for r in rs) / len(rs):.0f}",
            ]) + " |")
    lines += ["", f"Claude API spend for the whole evaluation (answers + judges): {total_spent():.2f} $."]
    RESULTS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {RESULTS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="e.g. claude-haiku-5-5 or qwen2.5:7b")
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--budget", type=float, default=2.50, help="stop before spending more $ on the Claude API")
    parser.add_argument("--sample", type=int, help="only this many questions (spread over the categories)")
    parser.add_argument("--tag", help="name of a code version, e.g. agentic: its runs are kept apart")
    parser.add_argument("--report", action="store_true", help="only rewrite results.md")
    args = parser.parse_args()
    if not args.report:
        if not args.model:
            parser.error("--model is required (or use --report)")
        evaluate(args.model, args.configs, args.budget, args.sample, args.tag)
    write_report()


if __name__ == "__main__":
    main()
