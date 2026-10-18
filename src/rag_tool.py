"""RAG tool: search the IFAB Laws of the Game and answer only from the retrieved passages."""

from functools import lru_cache
from pathlib import Path
import logging
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from src.llm import LLMResponse, chat

ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = ROOT / "data" / "laws_of_the_game.pdf"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "laws"
# Multilingual: a French question finds the English passage. Chosen over paraphrase-multilingual-MiniLM
# (5/10), multilingual-e5-small (7/10) and bge-m3 (7/10, much slower): 8/10 right passages in the top 5.
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
LAW_PAGES = (44, 150)  # Laws 1 to 17 in the English 2026/27 PDF (page numbers as printed, inclusive)
MAX_WORDS = 400
OVERLAP_WORDS = 50
# A page header (sometimes glued to the page number) or a line holding only a page number
PAGE_NOISE = r"^\d*Laws of the Game \d{4}/\d{2}\s*\|.*$|^\s*\d+\s*$"

LAW_TITLES = {
    1: "The Field of Play",
    2: "The Ball",
    3: "The Players",
    4: "The Players’ Equipment",
    5: "The Referee",
    6: "The Other Match Officials",
    7: "The Duration of the Match",
    8: "The Start and Restart of Play",
    9: "The Ball in and out of Play",
    10: "Determining the Outcome of a Match",
    11: "Offside",
    12: "Fouls and Misconduct",
    13: "Free Kicks",
    14: "The Penalty Kick",
    15: "The Throw-in",
    16: "The Goal Kick",
    17: "The Corner Kick",
}


def extract_laws_text(pdf_path: Path = PDF_PATH) -> str:
    """Raw text of the pages holding Laws 1 to 17, one page after the other."""
    logging.getLogger("pypdf").setLevel(logging.ERROR)  # hide harmless font warnings
    reader = PdfReader(pdf_path)
    first, last = LAW_PAGES
    return "\n".join(reader.pages[i].extract_text() or "" for i in range(first - 1, last))


def chunk_laws(text: str) -> list[dict]:
    """Split the Laws text into chunks: one per section, long sections cut into overlapping pieces.

    Each chunk is {"law_number": 11, "law_title": "Offside", "section": "1. Offside position",
                   "text": "It is not an offence to be in an offside position. ..."}
    """
    chunks = []
    # 1. Each law starts with a title page "Law\n11". The (group) keeps the number:
    #    parts = [text before Law 1, "1", text of Law 1, "2", text of Law 2, ...]
    parts = re.split(r"^Law\s*\n(\d{1,2})\s*$", text, flags=re.MULTILINE)
    for number, law_text in zip(parts[1::2], parts[2::2]):
        law_number = int(number)
        # 2. Remove page headers ("109Laws of the Game 2026/27 | Law 11 | Offside") and bare page numbers
        law_text = re.sub(PAGE_NOISE, "", law_text, flags=re.MULTILINE)
        # 3. Sections start with a line like "1. Offside position"; text before the first one is the intro
        pieces = re.split(r"^(\d{1,2}\.\s+\S.*)$", law_text, flags=re.MULTILINE)
        sections = [("Introduction", pieces[0])] + list(zip(pieces[1::2], pieces[2::2]))
        for section, content in sections:
            # 4. Long sections are cut into overlapping pieces
            for words in split_words(content.split()):
                chunks.append({
                    "law_number": law_number,
                    "law_title": LAW_TITLES[law_number],
                    "section": section.strip(),
                    "text": " ".join(words),
                })
    return chunks


def split_words(words: list[str]) -> list[list[str]]:
    """Cut a list of words into pieces of at most MAX_WORDS, each one starting OVERLAP_WORDS
    before the end of the previous one. A short list gives a single piece, an empty one none."""
    pieces = []
    for start in range(0, len(words), MAX_WORDS - OVERLAP_WORDS):
        pieces.append(words[start:start + MAX_WORDS])
        if start + MAX_WORDS >= len(words):  # this piece already reaches the end
            break
    return pieces


@lru_cache(maxsize=1)
def embedding_model() -> SentenceTransformer:
    """Loaded once (it takes a few seconds), then reused."""
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str], kind: str) -> list[list[float]]:
    """One vector per text. Normalised, so cosine similarity compares meaning, not length.

    e5 models were trained with a prefix telling questions ("query") from documents ("passage").
    """
    return embedding_model().encode([f"{kind}: {t}" for t in texts], normalize_embeddings=True).tolist()


def text_to_embed(chunk: dict) -> str:
    """The law and section titles are added to the text: they carry a lot of the meaning."""
    return f"Law {chunk['law_number']} – {chunk['law_title']} – {chunk['section']}\n{chunk['text']}"


def laws_collection(reset: bool = False):
    """The Chroma collection of chunks. reset=True deletes it first (e.g. after changing the model)."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=chromadb.Settings(anonymized_telemetry=False))
    if reset and COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None)


# The Laws are in English and the multilingual embeddings miss French football words
# ("touche" = throw-in, "six mètres" = goal kick): recall@5 on eval/rag_questions.jsonl is 21/35
# with the French question, 34/35 once the LLM translates it. A plain instruction was not enough
# for qwen2.5:7b (half the questions stayed in French): the examples below make it work.
TRANSLATE_SYSTEM = (
    "Rewrite the user's football rules question as one question IN ENGLISH, using the official vocabulary of the "
    "IFAB Laws of the Game (e.g. throw-in, goal kick, corner kick, kick-off, dropped ball, penalty kick, "
    "penalties (penalty shoot-out), sending-off, caution, handball, defective ball, goal area, penalty area). "
    "Output only the rewritten question."
)
TRANSLATE_EXAMPLES = [
    ("Quand l'arbitre doit-il sortir un carton jaune ?", "When must the referee caution a player (yellow card)?"),
    ("Combien y a-t-il d'arbitres assistants ?", "How many assistant referees are there?"),
    ("Un remplaçant peut-il entrer sur le terrain sans autorisation ?",
     "Can a substitute enter the field of play without the referee's permission?"),
]


def to_search_query(question: str) -> LLMResponse:
    """The question rewritten in English with the words of the Laws, in .text (one short LLM call, ~4 s)."""
    messages = [{"role": "system", "content": TRANSLATE_SYSTEM}]
    for french, english in TRANSLATE_EXAMPLES:
        messages += [{"role": "user", "content": f"Translate: {french}"}, {"role": "assistant", "content": english}]
    messages.append({"role": "user", "content": f"Translate: {question}"})
    response = chat(messages)
    response.text = response.text.strip() or question
    return response


def retrieve(question: str, k: int = 5) -> list[dict]:
    """The k chunks closest in meaning to the question, closest first."""
    result = laws_collection().query(query_embeddings=embed([question], "query"), n_results=k)
    return [
        {**metadata, "text": text, "distance": round(distance, 3)}
        for metadata, text, distance in zip(result["metadatas"][0], result["documents"][0], result["distances"][0])
    ]


def answer_from_laws(question: str, k: int = 5) -> dict:
    """Answer only from the retrieved passages and cite the law; otherwise say it is not in them."""
    translation = to_search_query(question)
    passages = retrieve(translation.text, k)
    context = "\n\n".join(f"[Law {p['law_number']} – {p['law_title']} – {p['section']}]\n{p['text']}" for p in passages)
    system = (
        "You answer questions about the IFAB Laws of the Game.\n"
        "Use ONLY the passages below. Cite the law you used, e.g. (Law 11).\n"
        "If the passages do not contain the answer, say that you don't know.\n"
        "Answer in English, in a few sentences.\n\n"
        f"Passages:\n{context}"
    )
    response = chat([{"role": "system", "content": system}, {"role": "user", "content": question}])
    return {
        "answer": response.text,
        "search_query": translation.text,
        "passages": passages,
        "tokens": sum(r.input_tokens + r.output_tokens for r in (translation, response)),
        "latency_s": round(translation.latency_s + response.latency_s, 1),
    }
