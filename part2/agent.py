"""Agentic RAG: the model decides when to search, what to search for, and when to stop.

    python part2/agent.py "the one where the guard drives below the bottom floor of a garage"
    python part2/agent.py          # interactive, keeps conversation memory

Works with any OpenAI-compatible chat model that supports tool calling (see providers.py).

Guardrails, each enforced in code, not just in the prompt:
  1. Budget: at most MAX_TOOL_CALLS tool calls per question, counted per call, not per model turn.
  2. Grounding: the answer may only cite tapes a tool returned during this conversation.
     One corrective retry; if the model still cites an unseen tape, the answer is withheld.
  3. Spoilers: tools return plot summaries without endings unless the clerk was started with spoilers on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from records import cited_tapes

log = logging.getLogger("clerk")

MAX_TOOL_CALLS = 4
MAX_QUERY_CHARS = 300
NOT_ON_SHELF = "That one's not on the shelf."

SYSTEM = f"""You are the late-night clerk at Midnight Rental, a horror video store.
Customers half-remember movies. Find the right tape with your tools.

Rules:
- Search before answering. Rewrite vague or multi-part requests into short, focused searches.
- You have {MAX_TOOL_CALLS} tool calls per question. Stop as soon as you are confident.
- Only recommend tapes your tools returned. Cite each one as "Tape #1234: Title".
- If nothing returned actually matches the description, reply exactly: "{NOT_ON_SHELF}" Do not guess.
- Two or three sentences, dry spooky clerk voice."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_shelf",
            "description": "Hybrid search (meaning + exact words, re-ranked) over the store's horror tapes. Returns the top matches.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A short, focused search query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tape",
            "description": "Look up one tape by its 4-digit number, e.g. after the customer says 'tell me more about that one'.",
            "parameters": {
                "type": "object",
                "properties": {"tape": {"type": "string", "description": "4-digit tape number"}},
                "required": ["tape"],
            },
        },
    },
]


class ShelfLike(Protocol):
    def search(self, query: str, k: int = 3, mode: str = "rerank", include_spoilers: bool = False) -> list: ...
    def get_tape(self, tape: str, include_spoilers: bool = False) -> dict | None: ...


@dataclass
class Turn:
    answer: str
    cited: set[str]
    tool_calls: int
    grounded: bool
    searches: list[str] = field(default_factory=list)


class Clerk:
    def __init__(self, client: Any, model: str, shelf: ShelfLike, spoilers: bool = False,
                 on_tool: Callable[[str, str, list[str]], None] | None = None) -> None:
        self.client = client
        self.model = model
        self.shelf = shelf
        self.spoilers = spoilers
        self.on_tool = on_tool
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM}]
        # Persists across turns, so "tell me more about that one" can cite a tape found two questions ago.
        self.seen: set[str] = set()

    def ask(self, question: str) -> Turn:
        self.messages.append({"role": "user", "content": question})
        calls_used = 0
        searches: list[str] = []

        while True:
            budget_left = MAX_TOOL_CALLS - calls_used
            message = self._complete(tools_allowed=budget_left > 0)
            self.messages.append(_as_dict(message))
            if not message.tool_calls:
                break
            for call in message.tool_calls:
                if calls_used >= MAX_TOOL_CALLS:
                    # Every tool_call_id must get a reply, or the next request is rejected.
                    content = json.dumps({"error": "tool budget exhausted; answer with what you have"})
                else:
                    calls_used += 1
                    content = self._run_tool(call, searches)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": content})

        answer = (message.content or "").strip()
        unseen = cited_tapes(answer) - self.seen
        if unseen:
            log.warning("cited unseen tapes %s; retrying once", sorted(unseen))
            self.messages.append({
                "role": "user",
                "content": f"(system check) You cited {sorted(unseen)}, which no tool returned. "
                           f"Answer again citing only tapes your tools returned, or say \"{NOT_ON_SHELF}\"",
            })
            message = self._complete(tools_allowed=False)
            self.messages.append(_as_dict(message))
            answer = (message.content or "").strip()
            unseen = cited_tapes(answer) - self.seen

        grounded = not unseen
        if not grounded:
            answer = f"{NOT_ON_SHELF} (Withheld an answer that cited tapes search never returned: {sorted(unseen)}.)"
        return Turn(answer=answer, cited=cited_tapes(answer) & self.seen, tool_calls=calls_used,
                    grounded=grounded, searches=searches)

    def _complete(self, tools_allowed: bool):
        kwargs: dict[str, Any] = {"model": self.model, "messages": self.messages, "temperature": 0}
        if tools_allowed:
            kwargs["tools"] = TOOLS
        # Omitting tools (instead of tool_choice="none") is the most portable way to force an answer:
        # some OpenAI-compatible servers ignore tool_choice.
        return self.client.chat.completions.create(**kwargs).choices[0].message

    def _run_tool(self, call, searches: list[str]) -> str:
        name = call.function.name
        try:
            args = json.loads(call.function.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return json.dumps({"error": f"bad arguments: {exc}"})

        if name == "search_shelf":
            query = str(args.get("query", "")).strip()[:MAX_QUERY_CHARS]
            if not query:
                return json.dumps({"error": "query is required"})
            hits = self.shelf.search(query, k=3, include_spoilers=self.spoilers)
            tapes = [h.tape for h in hits]
            searches.append(query)
            self.seen.update(tapes)
            self._notify(name, query, tapes)
            return json.dumps([h.as_dict() for h in hits])

        if name == "get_tape":
            tape = str(args.get("tape", "")).strip().lstrip("#")
            record = self.shelf.get_tape(tape, include_spoilers=self.spoilers)
            if record is None:
                return json.dumps({"error": f"no tape #{tape} on the shelf"})
            self.seen.add(record["tape"])
            self._notify(name, tape, [record["tape"]])
            return json.dumps(record)

        return json.dumps({"error": f"unknown tool {name!r}"})

    def _notify(self, tool: str, arg: str, tapes: list[str]) -> None:
        if self.on_tool:
            self.on_tool(tool, arg, tapes)


def _as_dict(message) -> dict:
    return message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else dict(message)


def build_clerk(spoilers: bool = False, verbose: bool = True) -> Clerk:
    from providers import chat_client
    from shelf import get_shelf

    client, cfg = chat_client()
    log.info("using %s / %s", cfg.provider, cfg.model)

    def show(tool: str, arg: str, tapes: list[str]) -> None:
        if verbose:
            print(f"  [{tool}] {arg!r} -> {tapes}", file=sys.stderr)

    return Clerk(client, cfg.model, get_shelf(), spoilers=spoilers, on_tool=show)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    parser.add_argument("--spoilers", action="store_true", help="let tools return endings")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    from providers import ConfigError

    try:
        clerk = build_clerk(spoilers=args.spoilers)
    except ConfigError as exc:
        sys.exit(f"config error: {exc}")

    import openai

    def answer(question: str) -> str:
        try:
            return clerk.ask(question).answer
        except openai.AuthenticationError:
            sys.exit("auth failed: your provider rejected the key. Check LLM_PROVIDER and its key (part2/README.md).")
        except openai.APIConnectionError:
            sys.exit("can't reach the model server. If you're using Ollama, is `ollama serve` running?")
        except openai.NotFoundError as exc:
            sys.exit(f"model not found: {exc.message}. Set LLM_MODEL to a model your provider has.")

    if args.question:
        print(answer(" ".join(args.question)))
        return
    print("Midnight Rental is open. Describe a movie (Ctrl+C to leave).")
    while True:
        try:
            question = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nWe close at dawn.")
            return
        if question:
            print(f"clerk > {answer(question)}")


if __name__ == "__main__":
    main()
