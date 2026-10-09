"""Shared helpers for Levels 3 and 4. You don't need to edit this file."""

import math
import re
from collections import Counter

STOP = set(
    "a an and are as at be but by for from has he her his in into is it its of on or she that the their "
    "they this to was were with you your who what which does did do how".split()
)


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


class KeywordRanker:
    """Tiny TF-IDF ranker, a stand-in for BM25. Rare words that match count more."""

    def __init__(self, texts: dict[str, str]):
        self.docs = {doc_id: Counter(words(t)) for doc_id, t in texts.items()}
        df = Counter(w for counts in self.docs.values() for w in counts)
        n = len(self.docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    def rank(self, query: str, top: int = 10) -> list[str]:
        q = words(query)
        scores = {doc_id: sum(counts[w] * self.idf.get(w, 0) for w in q) for doc_id, counts in self.docs.items()}
        hits = sorted((d for d, s in scores.items() if s > 0), key=lambda d: -scores[d])
        return hits[:top]
