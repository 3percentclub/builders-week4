"""Level 1: fix the orphan-chunk bug.

The Midnight Rental clerk bot slices every tape into chunks before indexing.
Right now it cuts the text every `max_words` words, so the title only lands in
the FIRST chunk. Every later chunk is an orphan: it knows what happened, but not
which movie it happened in.

Fix chunk_tape() so that:
  1. EVERY chunk starts with the tape's header, e.g. "Tape #1031: Halloween (1978)."
  2. No chunk is longer than max_words words, header included.
  3. Every word of the body appears exactly once, in order, across the chunks.
  4. If max_words is too small to fit the header plus at least one body word,
     raise ValueError.

Run the tests: python -m unittest -v
"""

import json
from pathlib import Path

SHELF = json.loads((Path(__file__).parent / "horror_movies.json").read_text())


def header(movie: dict) -> str:
    return f"Tape #{movie['tape']}: {movie['title']} ({movie['year']})."


def body(movie: dict) -> str:
    return (
        f"Directed by {movie['director']}. Starring {movie['cast']}. "
        f"Shelf: {movie['subgenre']}. Setting: {movie['setting']}. "
        f"Plot: {movie['plot']} Ending: {movie['ending']} Clerk's note: {movie['note']}"
    )


def chunk_tape(movie: dict, max_words: int = 60) -> list[str]:
    # BUG: only the first chunk gets the header.
    words = f"{header(movie)} {body(movie)}".split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


if __name__ == "__main__":
    halloween = next(m for m in SHELF if m["title"] == "Halloween")
    for i, chunk in enumerate(chunk_tape(halloween, max_words=40), 1):
        print(f"--- chunk {i}\n{chunk}\n")
