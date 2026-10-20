"""Validate the LLM judge against a human: same answers, graded by both, then the agreement rate.

    .venv/Scripts/python eval/validate_judge.py --sample     (once: writes eval/judge_validation.md to fill in)
    .venv/Scripts/python eval/validate_judge.py              (after grading: agreement + disagreements)

The file shows the question, the reference answer and the answer, but NOT the judge's verdict, so the
human grade is not influenced by it.
"""

from pathlib import Path
import argparse
import json
import random
import re

EVAL = Path(__file__).resolve().parent
RUNS, QUESTIONS = EVAL / "runs", EVAL / "questions.jsonl"
SHEET, KEY = EVAL / "judge_validation.md", EVAL / "judge_validation_key.json"
SAMPLE_SIZE, SEED = 30, 42
GRADE_LINE = re.compile(r"\*\*Ta note \(C ou I\) :\*\* *([CIci]?)")


def sample() -> None:
    if SHEET.exists():
        raise SystemExit(f"{SHEET} already exists: delete it first if you really want a new sample.")
    questions = {json.loads(line)["id"]: json.loads(line) for line in QUESTIONS.open(encoding="utf-8")}
    candidates = []  # every answer graded by the correctness judge, for the kept questions
    for path in sorted(RUNS.glob("*/*/*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["id"] in questions and "judge" in run.get("grades", {}):
            candidates.append((path, run))
    picked = random.Random(SEED).sample(candidates, SAMPLE_SIZE)

    lines = ["# Validation du juge : note ces 30 réponses", "",
             "Même règle que le juge : une réponse est **correcte (C)** si elle est d'accord avec la réponse de",
             "référence sur **chaque fait important**. Les détails en plus sont acceptés s'ils ne la contredisent pas ;",
             "un fait faux, contradictoire ou manquant la rend **incorrecte (I)**. Pour les questions hors sujet,",
             "la bonne réponse est de refuser poliment. Écris C ou I après « Ta note », puis enregistre le fichier.", ""]
    for i, (path, run) in enumerate(picked, 1):
        q = questions[run["id"]]
        answer = "\n".join("> " + line for line in (run["answer"] or "(réponse vide)").splitlines())
        lines += [f"## {i}. {q['id']} ({q['category']})", "",
                  f"**Question :** {q['question']}", "",
                  f"**Réponse de référence :** {q['expected_answer']}", "",
                  "**Réponse à noter :**", "", answer, "",
                  "**Ta note (C ou I) :** ", "", "---", ""]
    SHEET.write_text("\n".join(lines), encoding="utf-8")
    KEY.write_text(json.dumps([str(p.relative_to(EVAL)) for p, _ in picked], indent=1), encoding="utf-8")
    print(f"{SAMPLE_SIZE} answers written to {SHEET}: fill in 'Ta note' with C or I, then run this script again.")


def agreement() -> None:
    human = [m.upper() for m in GRADE_LINE.findall(SHEET.read_text(encoding="utf-8"))]
    paths = json.loads(KEY.read_text(encoding="utf-8"))
    if "" in human:
        raise SystemExit(f"{human.count('')} answers are not graded yet in {SHEET}.")
    agree, disagreements = 0, []
    both = {"CC": 0, "CI": 0, "IC": 0, "II": 0}  # human grade + judge grade
    for i, (h, rel) in enumerate(zip(human, paths), 1):
        run = json.loads((EVAL / rel).read_text(encoding="utf-8"))
        j = "C" if run["grades"]["judge"]["ok"] else "I"
        both[h + j] += 1
        if h == j:
            agree += 1
        else:
            disagreements.append(f"- n°{i} {run['id']} ({run['model']}, {run['config']}): toi {h}, juge {j}. "
                                 f"Juge : {run['grades']['judge']['explanation'][:200]}")
    print(f"Agreement: {agree}/{len(human)} ({100 * agree / len(human):.0f} %)")
    print(f"Both correct {both['CC']}, both incorrect {both['II']}, "
          f"judge too harsh (you C, judge I) {both['CI']}, judge too lenient (you I, judge C) {both['IC']}")
    print("\n".join(disagreements))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="write the sheet to fill in")
    sample() if parser.parse_args().sample else agreement()
