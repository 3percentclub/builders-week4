"""Level 1: fix the orphan-chunk bug.

Dog-Ear Books, a late-night Brooklyn bookstore, wants a clerk bot that answers
questions about its October table. Before indexing, every book card is sliced
into chunks. Right now it cuts the text every `max_words` words, so the title
only lands in the FIRST chunk. Every later chunk is an orphan: it knows what
happened, but not which book it happened in.

Fix chunk_book() so that:
  1. EVERY chunk starts with the book's header, e.g. "Shelf B01: The Haunting of Hill House by Shirley Jackson (1959)."
  2. No chunk is longer than max_words words, header included.
  3. Every word of the body appears exactly once, in order, across the chunks.
  4. If max_words is too small to fit the header plus at least one body word,
     raise ValueError.

Run the tests: python -m unittest -v
"""

import json
from pathlib import Path

SHELF = json.loads((Path(__file__).parent / "books.json").read_text())


def header(book: dict) -> str:
    return f"Shelf {book['shelf']}: {book['title']} by {book['author']} ({book['year']})."


def body(book: dict) -> str:
    return (
        f"Section: {book['section']}. Setting: {book['setting']}. "
        f"Summary: {book['summary']} How it ends: {book['ending']} Bookseller's note: {book['note']}"
    )


def chunk_book(book: dict, max_words: int = 60) -> list[str]:
    # BUG: only the first chunk gets the header.
    words = f"{header(book)} {body(book)}".split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


if __name__ == "__main__":
    hill_house = next(b for b in SHELF if b["title"] == "The Haunting of Hill House")
    for i, chunk in enumerate(chunk_book(hill_house, max_words=40), 1):
        print(f"--- chunk {i}\n{chunk}\n")
