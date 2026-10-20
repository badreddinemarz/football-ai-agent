"""Evaluation metrics: plain functions, no LLM, tested in tests/test_metrics.py."""

import re
import unicodedata

# Models often write small numbers in words ("Kane scored three goals")
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}


def normalize(text: str) -> str:
    """Lower case, without accents: 'Mbappé' -> 'mbappe'."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def numbers_in(text: str) -> set[float]:
    """Every number in the text, written in digits or (up to twenty) in words.

    Numbers are compared as numbers, not as text: '117 goals' does not contain 17.
    """
    text = normalize(text)
    numbers = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}
    numbers |= {float(NUMBER_WORDS[w]) for w in re.findall(r"[a-z]+", text) if w in NUMBER_WORDS}
    return numbers


def contains_value(answer: str, value: str) -> bool:
    """Is one expected value in the answer? A number must appear as a number; for a name, the last
    word is enough ('Lionel Messi' is found in 'Messi took 32 shots')."""
    try:
        return float(value) in numbers_in(answer)
    except ValueError:
        return normalize(value).split()[-1] in normalize(answer)


def contains_all(answer: str, must_contain: list[str]) -> bool:
    """Exactness of a statistics answer: every expected value is in it."""
    return all(contains_value(answer, value) for value in must_contain)


def recall_at_k(passages: list[dict], expected_laws: list[int], k: int = 5) -> bool | None:
    """Did the search find the right law in its first k passages? None when no law is expected."""
    if not expected_laws:
        return None
    return any(p["law_number"] in expected_laws for p in passages[:k])


def right_tools(tools_used: list[str], expected_tools: list[str]) -> bool:
    """Did the agent use exactly the expected tools (in any order, possibly several times)?"""
    return set(tools_used) == set(expected_tools)
