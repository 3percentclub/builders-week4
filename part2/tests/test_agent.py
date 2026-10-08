"""Guardrail tests with a scripted fake model and a fake shelf. No network, no key, no index."""

import json
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from agent import MAX_TOOL_CALLS, NOT_ON_SHELF, Clerk


@dataclass
class FakeHit:
    tape: str
    title: str

    def as_dict(self):
        return {"tape": self.tape, "title": self.title}


class FakeShelf:
    def __init__(self):
        self.queries = []

    def search(self, query, k=3, mode="rerank", include_spoilers=False):
        self.queries.append(query)
        return [FakeHit("1031", "Halloween"), FakeHit("4410", "Scream")]

    def get_tape(self, tape, include_spoilers=False):
        return {"tape": tape, "title": "Halloween"} if tape == "1031" else None


def tool_call(i, name="search_shelf", args='{"query": "masked killer"}'):
    return SimpleNamespace(id=f"call_{i}", type="function", function=SimpleNamespace(name=name, arguments=args))


def reply(content=None, calls=None):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": c.id, "type": "function",
                              "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
    return SimpleNamespace(content=content, tool_calls=calls, model_dump=lambda exclude_none=True: msg)


class ScriptedClient:
    """Returns pre-written model messages in order and records every request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


class Budget(unittest.TestCase):
    def test_parallel_calls_count_individually(self):
        # One model turn asks for 3 searches, the next for 3 more. Only MAX_TOOL_CALLS may run.
        client = ScriptedClient([
            reply(calls=[tool_call(i) for i in range(3)]),
            reply(calls=[tool_call(i) for i in range(3, 6)]),
            reply("Tape #1031: Halloween."),
        ])
        shelf = FakeShelf()
        turn = Clerk(client, "m", shelf).ask("masked killer")
        self.assertEqual(len(shelf.queries), MAX_TOOL_CALLS)
        self.assertEqual(turn.tool_calls, MAX_TOOL_CALLS)
        # Every tool_call_id got a reply, including the over-budget ones.
        tool_ids = [m["tool_call_id"] for m in client.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertEqual(tool_ids, [f"call_{i}" for i in range(6)])
        # Once the budget is spent, tools are no longer offered.
        self.assertNotIn("tools", client.requests[-1])

    def test_malformed_arguments_do_not_crash(self):
        client = ScriptedClient([
            reply(calls=[tool_call(0, args="{not json"), tool_call(1, args='"a string"'), tool_call(2, args="{}")]),
            reply(NOT_ON_SHELF),
        ])
        turn = Clerk(client, "m", FakeShelf()).ask("???")
        errors = [json.loads(m["content"]) for m in client.requests[-1]["messages"] if m["role"] == "tool"]
        self.assertTrue(all("error" in e for e in errors))
        self.assertEqual(turn.answer, NOT_ON_SHELF)


class Grounding(unittest.TestCase):
    def test_follow_up_can_cite_a_tape_from_an_earlier_turn(self):
        client = ScriptedClient([
            reply(calls=[tool_call(0)]),
            reply("Tape #1031: Halloween."),
            reply("Tape 1031 again: still Halloween, still a classic."),  # no new search
        ])
        clerk = Clerk(client, "m", FakeShelf())
        clerk.ask("masked killer on Halloween")
        turn = clerk.ask("tell me more about that one")
        self.assertTrue(turn.grounded)
        self.assertEqual(turn.cited, {"1031"})

    def test_invented_tape_is_retried_then_withheld(self):
        client = ScriptedClient([
            reply(calls=[tool_call(0)]),
            reply("Try Tape #9999: The Fake One."),
            reply("Fine. Tape #9999 then."),
        ])
        turn = Clerk(client, "m", FakeShelf()).ask("masked killer")
        self.assertFalse(turn.grounded)
        self.assertTrue(turn.answer.startswith(NOT_ON_SHELF))
        self.assertEqual(turn.cited, set())

    def test_retry_that_fixes_the_citation_passes(self):
        client = ScriptedClient([
            reply(calls=[tool_call(0)]),
            reply("Try tape 9999."),
            reply("Tape #1031: Halloween."),
        ])
        turn = Clerk(client, "m", FakeShelf()).ask("masked killer")
        self.assertTrue(turn.grounded)
        self.assertEqual(turn.cited, {"1031"})


if __name__ == "__main__":
    unittest.main()
