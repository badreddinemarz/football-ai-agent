import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import contains_all, contains_value, numbers_in, recall_at_k, right_tools


def test_numbers_in_digits_and_words():
    assert numbers_in("Kane scored three goals, Spain 15.") == {3.0, 15.0}


def test_number_is_not_matched_inside_another_number():
    assert not contains_value("A total of 117 goals were scored.", "17")
    assert contains_value("A total of 117 goals were scored.", "117")


def test_decimal_and_thousands():
    assert contains_value("Argentina had 13.94 xG", "13.94")
    assert contains_value("1,000 passes", "1000")


def test_name_by_last_word_without_accents():
    assert contains_value("Mbappe was the top scorer", "Kylian Mbappé")
    assert not contains_value("Messi was the top scorer", "Kylian Mbappé")


def test_contains_all_needs_every_value():
    must = ["Lionel Messi", "32"]
    assert contains_all("Lionel Messi took 32 shots.", must)
    assert not contains_all("Lionel Messi took the most shots.", must)
    assert not contains_all("Kylian Mbappé took 32 shots.", must)


def test_recall_at_k():
    passages = [{"law_number": 9}, {"law_number": 11}]
    assert recall_at_k(passages, [11]) is True
    assert recall_at_k(passages, [11], k=1) is False
    assert recall_at_k(passages, [14]) is False
    assert recall_at_k(passages, []) is None


def test_right_tools():
    assert right_tools(["query_stats_database"], ["query_stats_database"])
    assert right_tools(["search_laws_of_the_game", "query_stats_database", "query_stats_database"],
                       ["query_stats_database", "search_laws_of_the_game"])
    assert not right_tools(["query_stats_database"], ["query_stats_database", "search_laws_of_the_game"])
    assert right_tools([], [])
