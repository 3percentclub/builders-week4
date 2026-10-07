"""Level 2 (challenge): write Reciprocal Rank Fusion yourself.

In class, LlamaIndex's QueryFusionRetriever merged the vector list and the BM25
list for you. Now write the merge by hand.

    score(doc) = sum over every list the doc appears in of  1 / (k + rank)

rank starts at 1. A doc that's missing from a list gets nothing from that list.
Return doc ids sorted by score, highest first. If two docs tie, keep the one
that showed up first (reading list 1 top to bottom, then list 2, and so on).

Step 1: change ATTEMPTING to True. The fusion tests will start running.
Step 2: replace the NotImplementedError with your code.
"""

ATTEMPTING = False


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    raise NotImplementedError("Write Reciprocal Rank Fusion here")


if __name__ == "__main__":
    vector = ["2016", "2011", "2023"]  # Train to Busan, Cabin in the Woods, Barbarian
    bm25 = ["2023", "2016", "1999"]  # Barbarian, Train to Busan, Blair Witch
    print(rrf([vector, bm25]))
