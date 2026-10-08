"""Pure, dependency-free helpers for the shelf: loading tapes, building text, fingerprints.

Kept separate from shelf.py so tests and CI run without LlamaIndex, Chroma, or an API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PART2 = Path(__file__).resolve().parent
ROOT = PART2.parent
DATA_PATHS = (ROOT / "horror_movies.json", PART2 / "extra_tapes.json")

REQUIRED_FIELDS = ("tape", "title", "year", "director", "cast", "subgenre", "setting", "plot", "ending", "note")


def load_movies(paths: tuple[Path, ...] = DATA_PATHS) -> list[dict]:
    """Load every tape. Fails loudly on duplicate tape numbers or missing fields."""
    movies: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        for movie in json.loads(path.read_text()):
            missing = [f for f in REQUIRED_FIELDS if f not in movie]
            if missing:
                raise ValueError(f"{path.name}: tape {movie.get('tape')} is missing {missing}")
            if movie["tape"] in seen:
                raise ValueError(f"{path.name}: duplicate tape #{movie['tape']}")
            seen.add(movie["tape"])
            movies.append(movie)
    return movies


def header(movie: dict) -> str:
    return f"Tape #{movie['tape']}: {movie['title']} ({movie['year']})."


def document_text(movie: dict) -> str:
    """Full searchable text, spoilers included. Retrieval sees everything; callers may not."""
    return (
        f"{header(movie)} Directed by {movie['director']}. Starring {movie['cast']}. "
        f"Shelf: {movie['subgenre']}. Setting: {movie['setting']}.\n"
        f"Plot: {movie['plot']}\n"
        f"Ending (spoilers): {movie['ending']}\n"
        f"Clerk's note: {movie['note']}"
    )


def with_header(chunk: str, movie: dict) -> str:
    """Prefix a chunk with its tape header so an orphaned ending chunk still says which movie it is.

    Overlap cannot do this: a 50-token overlap never reaches a title 400 tokens back.
    """
    h = header(movie)
    return chunk if chunk.startswith(h) else f"{h} {chunk}"


def safe_summary(movie: dict) -> str:
    """What a caller sees by default: everything except the ending."""
    return (
        f"{header(movie)} Directed by {movie['director']}. Starring {movie['cast']}. "
        f"Shelf: {movie['subgenre']}. Setting: {movie['setting']}. "
        f"Plot: {movie['plot']} Clerk's note: {movie['note']}"
    )


def public_record(movie: dict, include_spoilers: bool = False) -> dict:
    return dict(movie) if include_spoilers else {k: v for k, v in movie.items() if k != "ending"}


def fingerprint(movies: list[dict], chunk_size: int, chunk_overlap: int, embed_model: str) -> str:
    """Changes whenever the data, chunking, or embedding model changes, so a stale index is never reused."""
    payload = json.dumps(
        {"movies": movies, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "embed_model": embed_model},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


_CITATION = re.compile(r"(?:#|\btape\s*#?\s*)(\d{4})\b", re.IGNORECASE)


def cited_tapes(answer: str) -> set[str]:
    """Tape numbers the answer cites, written as "#1031", "Tape 1031", or "tape #1031"."""
    return set(_CITATION.findall(answer))
