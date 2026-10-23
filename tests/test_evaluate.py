"""The evaluation script must at least load: a syntax error there would only show up at the next long run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))


def test_evaluate_loads_with_all_configurations():
    import evaluate
    assert list(evaluate.CONFIGS) == ["A_llm_only", "B_rag_only", "C_sql_only", "D_agent", "E_langgraph"]
