import json
import tempfile
import unittest
from pathlib import Path

from records import (
    DATA_PATHS,
    cited_tapes,
    document_text,
    fingerprint,
    load_movies,
    public_record,
    safe_summary,
    with_header,
)


class LoadMovies(unittest.TestCase):
    def test_real_data_loads_with_unique_tapes(self):
        movies = load_movies()
        tapes = [m["tape"] for m in movies]
        self.assertEqual(len(tapes), len(set(tapes)))
        self.assertGreaterEqual(len(movies), 40)

    def test_duplicate_tape_fails_loudly(self):
        movie = load_movies(DATA_PATHS[:1])[0]
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a.json"), Path(d, "b.json")
            a.write_text(json.dumps([movie]))
            b.write_text(json.dumps([movie]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_movies((a, b))

    def test_missing_field_fails_loudly(self):
        movie = dict(load_movies(DATA_PATHS[:1])[0])
        del movie["plot"]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "a.json")
            p.write_text(json.dumps([movie]))
            with self.assertRaisesRegex(ValueError, "missing"):
                load_movies((p,))


class Text(unittest.TestCase):
    movie = load_movies()[0]

    def test_header_added_to_orphan_chunk_once(self):
        orphan = "Ending (spoilers): everyone dies."
        out = with_header(orphan, self.movie)
        self.assertIn(self.movie["title"], out)
        self.assertEqual(with_header(out, self.movie), out)

    def test_spoilers_hidden_by_default(self):
        self.assertIn(self.movie["ending"], document_text(self.movie))
        self.assertNotIn(self.movie["ending"], safe_summary(self.movie))
        self.assertNotIn("ending", public_record(self.movie))
        self.assertIn("ending", public_record(self.movie, include_spoilers=True))


class Fingerprint(unittest.TestCase):
    def test_changes_with_data_chunking_or_model(self):
        movies = load_movies()
        base = fingerprint(movies, 512, 50, "m")
        edited = [dict(movies[0], plot=movies[0]["plot"] + " edited")] + movies[1:]
        self.assertNotEqual(base, fingerprint(edited, 512, 50, "m"))
        self.assertNotEqual(base, fingerprint(movies, 256, 50, "m"))
        self.assertNotEqual(base, fingerprint(movies, 512, 0, "m"))
        self.assertNotEqual(base, fingerprint(movies, 512, 50, "other"))
        self.assertEqual(base, fingerprint(movies, 512, 50, "m"))


class Citations(unittest.TestCase):
    def test_all_citation_styles(self):
        text = "Try Tape #1031: Halloween, tape 2048, or #6104. Not 19999 or 1978 alone."
        self.assertEqual(cited_tapes(text), {"1031", "2048", "6104"})


if __name__ == "__main__":
    unittest.main()
