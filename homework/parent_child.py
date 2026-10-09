"""Level 3 (challenge): parent-child retrieval, a.k.a. "search small, answer big".

The trade-off from class: small chunks MATCH well (one precise idea each), but
big chunks ANSWER well (the model gets the whole story). Parent-child gives you
both. You search tiny "child" chunks, then hand the model the full "parent" card
each child came from.

This file already:
  - cuts every book's body into small children of 12 words, with ids like "B06#3"
    (book B06, child number 3), and
  - ranks those children for a query with a keyword ranker.

You write the step in the middle: turn the ranked child ids into parent ids.

Rules for parents_for(child_hits, top):
  1. A child id looks like "B06#3". The parent id is the part before "#".
  2. Keep parents in the order of their BEST (earliest) child.
  3. No duplicates: if three children of B06 match, B06 shows up once.
  4. Return at most `top` parents.

Step 1: change ATTEMPTING to True. The parent-child tests will start running.
Step 2: replace the NotImplementedError with your code.
Step 3: run `python parent_child.py` and compare the child chunk to the parent card.
"""

from chunker import SHELF, body, header
from search_tools import KeywordRanker

ATTEMPTING = False
CHILD_WORDS = 12


def make_children(book: dict, size: int = CHILD_WORDS) -> dict[str, str]:
    words = body(book).split()
    return {f"{book['shelf']}#{n}": " ".join(words[i : i + size]) for n, i in enumerate(range(0, len(words), size))}


CHILDREN = {cid: text for book in SHELF for cid, text in make_children(book).items()}
PARENTS = {book["shelf"]: f"{header(book)} {body(book)}" for book in SHELF}
child_ranker = KeywordRanker(CHILDREN)


def parents_for(child_hits: list[str], top: int = 3) -> list[str]:
    raise NotImplementedError("Turn child ids into parent ids here")


def small_to_big(query: str, top: int = 3) -> list[str]:
    """Search the small children, return the full parent cards."""
    return [PARENTS[shelf] for shelf in parents_for(child_ranker.rank(query, top=20), top)]


if __name__ == "__main__":
    question = "Who ends up with a corkscrew in his neck?"
    best_child = child_ranker.rank(question, top=1)[0]
    print(f"Question: {question}\n")
    print(f"Best CHILD ({best_child}), what plain small-chunk search would hand the model:\n  {CHILDREN[best_child]}\n")
    print(f"PARENT card, what parent-child hands the model:\n  {small_to_big(question, top=1)[0]}")
