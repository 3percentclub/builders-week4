"""Agentic RAG: the model decides when to search, what to search for, and when to stop.

    python production/agent.py "that movie where the guy is stuck in a sunken place"
    python production/agent.py      # interactive, keeps conversation memory

Guardrails:
  - Retrieval budget: at most MAX_SEARCHES tool calls per question.
  - Every recommendation must cite a tape number that the search actually returned.
  - If nothing fits, the clerk says it's not on the shelf instead of guessing.
"""

from __future__ import annotations

import json
import os
import re
import sys

from openai import OpenAI

from shelf import get_shelf

MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
MAX_SEARCHES = 3

SYSTEM = f"""You are the late-night clerk at Midnight Rental, a horror video store.
Customers half-remember movies. Find the right tape using the search_shelf tool.

Rules:
- Search before answering. Rewrite vague or multi-part requests into focused searches.
- You have a budget of {MAX_SEARCHES} searches per question. Stop as soon as you're confident.
- Only recommend tapes that appeared in your search results. Cite them as "Tape #1234: Title".
- If no result fits, say "That one's not on the shelf." Never invent a movie.
- Keep it to two or three sentences, in a dry, spooky clerk voice. No spoilers unless asked."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_shelf",
            "description": "Hybrid search (meaning + exact words, re-ranked) over the store's horror tapes.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A focused search query."}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


class Clerk:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.shelf = get_shelf()
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM}]

    def ask(self, question: str, verbose: bool = True) -> str:
        self.messages.append({"role": "user", "content": question})
        seen_tapes: set[str] = set()

        for searches in range(MAX_SEARCHES + 1):
            budget_left = searches < MAX_SEARCHES
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=TOOLS,
                tool_choice="auto" if budget_left else "none",
                temperature=0,
            )
            message = response.choices[0].message
            self.messages.append(message.model_dump(exclude_none=True))

            if not message.tool_calls:
                return self._check_citations(message.content or "", seen_tapes)

            for call in message.tool_calls:
                query = json.loads(call.function.arguments)["query"]
                hits = self.shelf.search(query, k=3)
                seen_tapes.update(h.tape for h in hits)
                if verbose:
                    print(f"  [search] {query!r} -> {[f'#{h.tape} {h.title}' for h in hits]}", file=sys.stderr)
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": json.dumps([h.as_dict() for h in hits])}
                )

        return "That one's not on the shelf."

    @staticmethod
    def _check_citations(answer: str, seen_tapes: set[str]) -> str:
        cited = set(re.findall(r"#(\d{4})", answer))
        invented = cited - seen_tapes
        if invented:
            return f"{answer}\n\n(warning: cited tapes not returned by search: {sorted(invented)})"
        return answer


def main() -> None:
    clerk = Clerk()
    if len(sys.argv) > 1:
        print(clerk.ask(" ".join(sys.argv[1:])))
        return
    print("Midnight Rental is open. Describe a movie (Ctrl+C to leave).")
    while True:
        try:
            question = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nWe close at dawn.")
            return
        if question:
            print(f"clerk > {clerk.ask(question)}")


if __name__ == "__main__":
    main()
