"""Build the search index of the Laws of the Game: PDF → chunks → embeddings → Chroma (data/chroma/).

Needs data/laws_of_the_game.pdf (English, 2026/27, from theifab.com).
Re-running the script rebuilds the index from scratch.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "src" can be imported

from src.rag_tool import PDF_PATH, chunk_laws, embed, extract_laws_text, laws_collection, text_to_embed


def main() -> None:
    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found: {PDF_PATH}\nDownload the Laws of the Game (English) from theifab.com.")

    chunks = chunk_laws(extract_laws_text())
    print(f"{len(chunks)} chunks from {len({c['law_number'] for c in chunks})} laws")

    collection = laws_collection(reset=True)  # start from an empty index

    print("Computing embeddings (the model is downloaded the first time)...")
    collection.add(
        ids=[f"law{c['law_number']}-{i}" for i, c in enumerate(chunks)],
        documents=[c["text"] for c in chunks],
        embeddings=embed([text_to_embed(c) for c in chunks], "passage"),
        metadatas=[{k: c[k] for k in ("law_number", "law_title", "section")} for c in chunks],
    )
    print(f"Index written to data/chroma/ ({collection.count()} chunks)")


if __name__ == "__main__":
    main()
