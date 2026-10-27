# Error analysis

What went wrong, why, and what each failure taught me. Every number below was measured on this repository's
evaluation (`eval/questions.jsonl`, 48 questions, `eval/results.md`) or on a development set, as stated.

## 1. Where the agent fails

Final agent accuracy: **Claude Haiku 96 %** (46/48, with agentic search), **Qwen 2.5 7B 62 %** (30/48).

**The agent almost always picks the right tools: 96 % for both models** (46/48 each). Errors do not come from
choosing, they come from *executing* the tools.

Qwen 2.5 7B, 18 wrong answers, classified by cause:

| Cause | Count | Example |
|---|---|---|
| Wrong SQL (silent: the query runs and returns a plausible number) | 9 | 10 shoot-out matches instead of 5 |
| Wording of the answer, or a strict judge | 5 | answer says a goal *can* be scored from a throw-in, the passage says the opposite |
| Retrieval missed the right passage | 3 | ball touching the referee (Law 9) |
| Wrong tool | 1 | forgot the rules tool on a mixed question |

Claude Haiku, 4 wrong answers before agentic search: 3 retrieval failures, 1 correct answer marked wrong by the judge.

## 2. The SQL errors of the small model

All of them are *silent*: no exception, so the SQL self-correction loop never sees them.

| Pattern | What happened | Wrong → right |
|---|---|---|
| **Double counting on a 2-rows-per-match view** | `COUNT(*)` on `team_matches` counts each match once per team | 10 → 5, 28 → 14 |
| **Invented values** | `team = 'Own Goal'`, `shot_type = 'Goal'` match nothing | 0 → 10, 0 → 7 |
| **Forgotten filter** | no `NOT is_shootout` | 34 shots → 32 |
| **Filter added by mistake** | restricted to the final | 0 → 20 |
| **`LIMIT 1` instead of keeping ties** | one top scorer instead of the tied players | 1 → 6 |

These are errors of *knowledge of the schema*, not of reasoning. That is why a written plan does not help
(section 4) and why a semantic layer of SQL views did: SQL went from 3/12 (schema notes only) to 6/12 (+ 4
few-shot examples) to 7/12 (+ views) for Qwen, and 12/12 for Claude Haiku (development set, 12 questions).

## 3. Surprises found by reading the answers

1. **A tool bypass.** In the "SQL only" configuration, Claude Haiku answered rules questions with
   `SELECT 'the ball must be placed on the centre mark' AS answer`. The query passed the safety check (a
   read-only SELECT) *and* fooled the faithfulness judge (the answer matched its "SQL result"). Fix: a query
   must read a table (`FROM` is required).
2. **Silent truncation of the embedding model.** `multilingual-e5-base` reads 512 tokens and cuts the rest
   without a warning. English legal text is about 1.46 tokens per word, so my 400-word chunks went over for
   **19 of 103 chunks**. Fix: 300 words (119 chunks, longest 494 tokens) plus a guard that refuses to build a
   truncated index. Retrieval on the development set did not change measurably (English 35/35, French 21 → 23
   of 35): the real gain is that no text is lost.
3. **Accents.** "Mbappe" (no accent) returned 0 goals: the name filter was accent-sensitive. Found while
   preparing the API demo. Fix: `strip_accents` on both sides.
4. **An empty final answer.** After 3 searches, Claude's forced last answer sometimes contained only its
   thinking and no text. Both agents now answer "I could not find a reliable answer…" instead of nothing.
5. **A request refused by the API.** Removing the tools for the forced last answer made the Claude API refuse
   the request: thinking blocks are tied to the exact tool list. Fix: keep the tools described and forbid
   calling them (`tool_choice: none`).
6. **A network error cached as a wrong answer.** An internet cut during a night run was saved as a wrong
   answer and would have lowered the score for good. Errors are never cached any more, and a run stops
   cleanly after 3 errors in a row.
7. **A syntax error in the evaluation script** that no test could see, found only at the next long run. A test
   now imports the script.

## 4. Ideas that were tested and rejected

| Idea | Measured | Why it did not help |
|---|---|---|
| Chain of thought for SQL (a plan before the query), Qwen 7B | 7/12 → 5/12, latency 21 s → 42 s | the model copied the plan's structure (`JOIN matches` even on views that already hold the competition). A plan cannot add knowledge the model lacks. Claude Haiku reasons natively and scores 12/12 |
| Cross-encoder reranker | French questions 21 → 29/35 (+6 s), then nothing once questions are rewritten in English | it fixed a symptom; the cause was the language |
| Hybrid dense + BM25 (RRF) | 35 → 31/35 on English dev questions | the corpus repeats the same words everywhere ("referee", "penalty"), and my remaining failures were synonyms, which BM25 cannot handle |
| Local code model (qwen2.5-coder:7b) | 7/12, same as Qwen 2.5 7B | different errors, same score |
| LangGraph instead of my own loop | 92 % vs 96 %, same tools and prompt | equivalent within the noise: quality comes from the tools and prompts, not from the loop |

## 5. What worked

- **Rewriting the question in English with few-shot examples**: retrieval on French questions went from 60 %
  to 97 % (development set). Diagnosing first (the same questions in English scored 100 %) avoided adding a
  reranker.
- **Agentic search**: the agent checks that the passages answer the question and otherwise searches again
  with the official vocabulary. Development set: right passage found 20/22 → 22/22. Test set: 44/48 → 46/48,
  for +14 % tokens and +0.2 s median latency. The instruction is generic: it does not list the synonyms of
  the failing test questions.

## 6. How far to trust these numbers

- **48 questions**: one question is worth about 2 points. Only large gaps (21 % vs 62 %, 62 % vs 92 %) are
  conclusive; a 3-point difference proves nothing.
- **The judge is stricter than a human.** On 30 answers I graded blind, it agreed with me on 24 (80 %,
  Cohen's κ ≈ 0.6) and 5 of the 6 disagreements were the judge being stricter. Judged scores (rules, mixed,
  off-topic) are therefore, if anything, underestimated.
- **The faithfulness judge (Claude Haiku) was not validated by hand** and also grades its own answers.
- **Ground truth was checked three ways**: the 40 numeric answers were recomputed from the raw StatsBomb
  JSON by an independent script (40/40 identical), every rules reference was found word for word in the
  official PDF, and I checked a sample against external sources by hand.
- **Some results predate later fixes.** The first full runs were made before the 512-token and accent fixes;
  the Haiku agent runs with agentic search (`@v2-agentic`) come after both.
- The two errors that remain with Haiku are SQL: "how many penalties were *missed*" (1 vs 6: the model
  excluded saved penalties) and "goals in the knockout stage" (51 vs 52: an own goal is not in the goals
  view).
