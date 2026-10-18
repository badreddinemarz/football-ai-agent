from src.rag_tool import MAX_WORDS, OVERLAP_WORDS, chunk_laws

# Same layout as the text pypdf extracts from the real PDF: a "Law / number" title page,
# page headers (sometimes glued to the page number), bare page numbers, numbered sections.
SAMPLE = """Law
10
103Laws of the Game 2026/27   |  Law 10  |  Determining the Outcome of a Match
1. Goal scored
A goal is scored when the whole of the ball passes over the goal line.
2. Winning team
The team scoring the greater number of goals is the winner.
Law
11
109Laws of the Game 2026/27   |  Law 11  |  Offside
1. Offside position
It is not an offence to be in an offside position.
110
2. Offside offence
A player in an offside position is only penalised on becoming involved in active play.
112
Laws of the Game 2026/27   |  Law 11  |  Offside
3. No offence
There is no offside offence if a player receives the ball directly from:
• a goal kick
• a throw-in
• a corner kick
Law
15
141Laws of the Game 2026/27   |  Law 15  |  The Throw-in
A throw-in is awarded to the opponents of the player who last touched the ball.
1. Procedure
At the moment of delivering the ball, the thrower must face the field of play.
"""


def test_one_chunk_per_section():
    chunks = chunk_laws(SAMPLE)
    assert [(c["law_number"], c["section"]) for c in chunks] == [
        (10, "1. Goal scored"),
        (10, "2. Winning team"),
        (11, "1. Offside position"),
        (11, "2. Offside offence"),
        (11, "3. No offence"),
        (15, "Introduction"),
        (15, "1. Procedure"),
    ]


def test_chunk_fields():
    chunk = chunk_laws(SAMPLE)[2]
    assert set(chunk) == {"law_number", "law_title", "section", "text"}
    assert chunk["law_number"] == 11
    assert chunk["law_title"] == "Offside"


def test_keeps_the_section_text():
    no_offence = chunk_laws(SAMPLE)[4]
    assert "a throw-in" in no_offence["text"]
    assert "3. No offence" not in no_offence["text"]  # the title is in "section", not in "text"


def test_removes_page_headers_and_numbers():
    for chunk in chunk_laws(SAMPLE):
        assert "Laws of the Game 2026/27" not in chunk["text"]
        assert not any(line.strip().isdigit() for line in chunk["text"].splitlines())


def test_long_section_is_split_with_overlap():
    words = [f"w{i}" for i in range(1000)]
    chunks = chunk_laws("Law\n12\n1. Direct free kick\n" + " ".join(words))
    pieces = [c["text"].split() for c in chunks]

    assert len(pieces) > 1
    assert all(c["section"] == "1. Direct free kick" for c in chunks)
    assert all(len(p) <= MAX_WORDS for p in pieces)
    assert pieces[0][0] == "w0" and pieces[-1][-1] == "w999"  # nothing lost at either end
    for previous, current in zip(pieces, pieces[1:]):
        assert previous[-OVERLAP_WORDS:] == current[:OVERLAP_WORDS]


def test_empty_text():
    assert chunk_laws("") == []
