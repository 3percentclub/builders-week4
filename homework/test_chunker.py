import unittest

from chunker import SHELF, body, chunk_tape, header


class TestChunkTape(unittest.TestCase):
    def test_1_every_chunk_starts_with_the_title(self):
        for movie in SHELF:
            for chunk in chunk_tape(movie, max_words=40):
                self.assertTrue(
                    chunk.startswith(header(movie)),
                    f"Orphan chunk in {movie['title']}: {chunk[:60]}...",
                )

    def test_2_no_chunk_is_too_long(self):
        for size in (25, 40, 60):
            for movie in SHELF:
                for chunk in chunk_tape(movie, max_words=size):
                    self.assertLessEqual(len(chunk.split()), size)

    def test_3_body_is_kept_whole_and_in_order(self):
        for movie in SHELF:
            head_len = len(header(movie).split())
            rebuilt = []
            for chunk in chunk_tape(movie, max_words=40):
                rebuilt += chunk.split()[head_len:]
            self.assertEqual(rebuilt, body(movie).split(), f"Body changed for {movie['title']}")

    def test_4_big_window_means_one_chunk(self):
        for movie in SHELF:
            self.assertEqual(len(chunk_tape(movie, max_words=1000)), 1)

    def test_5_too_small_raises(self):
        movie = SHELF[0]
        with self.assertRaises(ValueError):
            chunk_tape(movie, max_words=len(header(movie).split()))


if __name__ == "__main__":
    unittest.main()
