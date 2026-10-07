"""Level 3 (bonus, API key): rewrite before you retrieve.

A two-turn chat with the Midnight Rental clerk:
  Turn 1: "Recommend a found-footage movie for tonight."
  Turn 2: "Anything like that but set on a train?"

The script searches the shelf two ways for turn 2:
  A. the raw chat history pasted in as the search query (the bug from class)
  B. a standalone question that an LLM rewrites first (the fix)

Search = two keyword rankers (one over title/cast/director, one over the plot),
merged with YOUR rrf() from fusion.py. So finish Level 2 first.

Standard library only. Works with any OpenAI-compatible provider.
Set LLM_API_KEY, LLM_BASE_URL, and MODEL (see README).
"""

import json
import math
import os
import re
import urllib.request
from collections import Counter

from chunker import SHELF
from fusion import rrf

STOP = set("a an and are as at be but by for from has he her his in into is it its of on or she that the their they this to was were with you your".split())


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP]


class KeywordRanker:
    """Tiny TF-IDF ranker. Rare words that match count more."""

    def __init__(self, texts: dict[str, str]):
        self.docs = {tape: Counter(words(t)) for tape, t in texts.items()}
        df = Counter(w for counts in self.docs.values() for w in counts)
        n = len(self.docs)
        self.idf = {w: math.log((n + 1) / (c + 0.5)) for w, c in df.items()}

    def rank(self, query: str, top: int = 10) -> list[str]:
        q = words(query)
        scores = {tape: sum(counts[w] * self.idf.get(w, 0) for w in q) for tape, counts in self.docs.items()}
        hits = sorted((t for t, s in scores.items() if s > 0), key=lambda t: -scores[t])
        return hits[:top]


TITLES = {m["tape"]: m["title"] for m in SHELF}
names = KeywordRanker({m["tape"]: f"{m['title']} {m['director']} {m['cast']}" for m in SHELF})
story = KeywordRanker({m["tape"]: f"{m['subgenre']} {m['setting']} {m['plot']} {m['note']}" for m in SHELF})


def search(query: str, top: int = 3) -> list[str]:
    return rrf([names.rank(query), story.rank(query)])[:top]


def llm(messages: list[dict]) -> str:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({"model": os.environ["MODEL"], "messages": messages, "temperature": 0}).encode(),
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


CONDENSE_PROMPT = (
    "Rewrite the user's LAST message as one standalone search query for a horror movie shelf. "
    "Keep only what the user wants now. Reply with the query only."
)


def main():
    history = [
        {"role": "user", "content": "Recommend a found-footage movie for tonight."},
        {
            "role": "assistant",
            "content": "The Blair Witch Project and Paranormal Activity are the classics. Both are shot on shaky "
            "handheld cameras, one lost in the woods and one in a house at night.",
        },
        {"role": "user", "content": "Anything like that but set on a train?"},
    ]
    raw = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    print("A. Raw history as the query:")
    print("  ", [TITLES[t] for t in search(raw)])

    condensed = llm([{"role": "system", "content": CONDENSE_PROMPT}, {"role": "user", "content": raw}])
    print(f"\nB. Rewritten query: {condensed!r}")
    hits = search(condensed)
    print("  ", [TITLES[t] for t in hits])

    shelf_text = "\n\n".join(
        f"Tape #{m['tape']}: {m['title']} ({m['year']}). {m['plot']}" for m in SHELF if m["tape"] in hits
    )
    answer = llm(
        [
            {"role": "system", "content": "You are the clerk at Midnight Rental. Recommend only tapes listed below, with the tape number.\n\n" + shelf_text},
            *history,
        ]
    )
    print(f"\nClerk: {answer}")


if __name__ == "__main__":
    main()
