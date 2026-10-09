import unittest

import parent_child
from chunker import SHELF, header


@unittest.skipUnless(parent_child.ATTEMPTING, "Level 3 is optional: set ATTEMPTING = True in parent_child.py")
class TestParentChild(unittest.TestCase):
    def test_1_child_ids_become_parent_ids(self):
        self.assertEqual(parent_child.parents_for(["B06#3"]), ["B06"])

    def test_2_best_child_decides_the_order(self):
        hits = ["B06#7", "B02#1", "B06#2", "B13#0"]
        self.assertEqual(parent_child.parents_for(hits), ["B06", "B02", "B13"])

    def test_3_no_duplicate_parents(self):
        hits = ["B08#1", "B08#4", "B08#9", "B17#2"]
        self.assertEqual(parent_child.parents_for(hits), ["B08", "B17"])

    def test_4_top_limits_the_parents(self):
        hits = ["B01#0", "B02#0", "B03#0", "B04#0", "B05#0"]
        self.assertEqual(parent_child.parents_for(hits, top=2), ["B01", "B02"])

    def test_5_empty_input(self):
        self.assertEqual(parent_child.parents_for([]), [])

    def test_6_small_match_big_answer(self):
        girl = next(b for b in SHELF if b["title"] == "The Girl on the Train")
        best_child = parent_child.child_ranker.rank("Who ends up with a corkscrew in his neck?", top=1)[0]
        self.assertNotIn(girl["title"], parent_child.CHILDREN[best_child], "The child chunk alone has no title")
        card = parent_child.small_to_big("Who ends up with a corkscrew in his neck?", top=1)[0]
        self.assertTrue(card.startswith(header(girl)))
        self.assertIn("corkscrew", card)
        self.assertIn(girl["summary"], card)


if __name__ == "__main__":
    unittest.main()
