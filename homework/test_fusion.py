import unittest

import fusion


@unittest.skipUnless(fusion.ATTEMPTING, "Level 2 is optional: set ATTEMPTING = True in fusion.py")
class TestRRF(unittest.TestCase):
    def test_1_docs_in_both_lists_win(self):
        self.assertEqual(fusion.rrf([["a", "b", "c"], ["c", "a", "d"]]), ["a", "c", "b", "d"])

    def test_2_bookstore_example(self):
        vector = ["B05", "B02", "B11"]
        bm25 = ["B11", "B05", "B14"]
        self.assertEqual(fusion.rrf([vector, bm25]), ["B05", "B11", "B02", "B14"])

    def test_3_k_changes_the_math(self):
        self.assertEqual(fusion.rrf([["x", "y"], ["y", "z"]], k=0), ["y", "x", "z"])

    def test_4_ties_keep_first_seen_order(self):
        self.assertEqual(fusion.rrf([["a"], ["b"]]), ["a", "b"])

    def test_5_no_duplicates(self):
        result = fusion.rrf([["a", "b"], ["b", "a"], ["a"]])
        self.assertEqual(sorted(result), ["a", "b"])
        self.assertEqual(result[0], "a")

    def test_6_empty_input(self):
        self.assertEqual(fusion.rrf([]), [])
        self.assertEqual(fusion.rrf([[], []]), [])


if __name__ == "__main__":
    unittest.main()
