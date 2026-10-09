import unittest

from chunker import SHELF, body, chunk_book, header


class TestChunkBook(unittest.TestCase):
    def test_1_every_chunk_starts_with_the_title(self):
        for book in SHELF:
            for chunk in chunk_book(book, max_words=40):
                self.assertTrue(
                    chunk.startswith(header(book)),
                    f"Orphan chunk in {book['title']}: {chunk[:60]}...",
                )

    def test_2_no_chunk_is_too_long(self):
        for size in (25, 40, 60):
            for book in SHELF:
                for chunk in chunk_book(book, max_words=size):
                    self.assertLessEqual(len(chunk.split()), size)

    def test_3_body_is_kept_whole_and_in_order(self):
        for book in SHELF:
            head_len = len(header(book).split())
            rebuilt = []
            for chunk in chunk_book(book, max_words=40):
                rebuilt += chunk.split()[head_len:]
            self.assertEqual(rebuilt, body(book).split(), f"Body changed for {book['title']}")

    def test_4_big_window_means_one_chunk(self):
        for book in SHELF:
            self.assertEqual(len(chunk_book(book, max_words=1000)), 1)

    def test_5_too_small_raises(self):
        book = SHELF[0]
        with self.assertRaises(ValueError):
            chunk_book(book, max_words=len(header(book).split()))


if __name__ == "__main__":
    unittest.main()
