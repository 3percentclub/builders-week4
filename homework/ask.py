"""Level 4 (bonus, API key): rewrite before you retrieve.

A two-turn chat with the Dog-Ear Books clerk:
  Turn 1: "Recommend a book told in letters or diaries."
  Turn 2: "Anything like that but set on a train?"

The script searches the shelf two ways for turn 2:
  A. the raw chat history pasted in as the search query (the bug from class)
  B. a standalone question that an LLM rewrites first (the fix)

Search = two keyword rankers (one over title/author, one over the story),
merged with YOUR rrf() from fusion.py. So finish Level 2 first.

Standard library only. Works with any OpenAI-compatible provider.
Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL (see README), as env vars or in a .env file.
"""

import json
import os
import urllib.request
from pathlib import Path

from chunker import SHELF
from fusion import rrf
from search_tools import KeywordRanker


def load_dotenv() -> None:
    """Read KEY=value lines from the nearest .env (this folder or any parent). Real env vars win."""
    for folder in [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent, *Path(__file__).resolve().parents]:
        path = folder / ".env"
        if path.is_file():
            for line in path.read_text().splitlines():
                name, sep, value = line.partition("=")
                value = value.strip().strip("'\"")
                if sep and value and not name.strip().startswith("#"):
                    os.environ.setdefault(name.strip(), value)
            return


load_dotenv()

TITLES = {b["shelf"]: b["title"] for b in SHELF}
names = KeywordRanker({b["shelf"]: f"{b['title']} {b['author']}" for b in SHELF})
story = KeywordRanker({b["shelf"]: f"{b['section']} {b['setting']} {b['summary']} {b['note']}" for b in SHELF})


def search(query: str, top: int = 3) -> list[str]:
    return rrf([names.rank(query), story.rank(query)])[:top]


def llm(messages: list[dict]) -> str:
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({"model": os.environ.get("LLM_MODEL") or os.environ["MODEL"], "messages": messages, "temperature": 0}).encode(),
        headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


CONDENSE_PROMPT = (
    "Rewrite the user's LAST message as one standalone search query for a bookstore's shelf. "
    "Keep only what the user wants now. Reply with the query only."
)

HISTORY = [
    {"role": "user", "content": "Recommend a book told in letters or diaries."},
    {
        "role": "assistant",
        "content": "Dracula and Frankenstein are the classics. Both are told through letters and diary entries, "
        "one from a castle in Transylvania and one from a ship stuck in the Arctic ice.",
    },
    {"role": "user", "content": "Anything like that but set on a train?"},
]


def main():
    raw = "\n".join(f"{m['role']}: {m['content']}" for m in HISTORY)

    print("A. Raw history as the query:")
    print("  ", [TITLES[s] for s in search(raw)])

    condensed = llm([{"role": "system", "content": CONDENSE_PROMPT}, {"role": "user", "content": raw}])
    print(f"\nB. Rewritten query: {condensed!r}")
    hits = search(condensed)
    print("  ", [TITLES[s] for s in hits])

    shelf_text = "\n\n".join(
        f"Shelf {b['shelf']}: {b['title']} by {b['author']} ({b['year']}). {b['summary']}" for b in SHELF if b["shelf"] in hits
    )
    answer = llm(
        [
            {"role": "system", "content": "You are the clerk at Dog-Ear Books. Recommend only books listed below, with the shelf number.\n\n" + shelf_text},
            *HISTORY,
        ]
    )
    print(f"\nClerk: {answer}")


if __name__ == "__main__":
    main()
