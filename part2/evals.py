"""Measure the shelf. Retrieval evals need no API key; agent evals need any chat model (see providers.py).

    python part2/evals.py                    # retrieval: hit@1, hit@3, MRR per mode and per question kind
    python part2/evals.py --min-hit3 0.85    # exit 1 if rerank hit@3 drops below 0.85 (used in CI)
    python part2/evals.py --agent            # end-to-end: correct tape cited, grounded, abstains on absent movies

Question kinds in evals.json:
  paraphrase  describes the movie in words the data does not use
  exact       names, tape numbers, places: the words ARE in the data
  note        only the clerk's note answers it (proves the answer came from the shelf, not model memory)
  absent      not on the shelf; the right answer is to say so
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALS = json.loads((HERE / "evals.json").read_text())


def rank_of(expect: list[str], tapes: list[str]) -> int | None:
    """1-based rank of the first acceptable tape, or None."""
    for i, tape in enumerate(tapes, start=1):
        if tape in expect:
            return i
    return None


def score(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def retrieval(modes: list[str]) -> dict:
    from shelf import get_shelf

    shelf = get_shelf()
    positives = [e for e in EVALS if e["expect"]]
    results: dict = {"n": len(positives), "modes": {}}
    for mode in modes:
        by_kind: dict[str, list] = defaultdict(list)
        misses = []
        for e in positives:
            tapes = [h.tape for h in shelf.search(e["q"], k=10, mode=mode)]
            r = rank_of(e["expect"], tapes)
            by_kind[e["kind"]].append(r)
            if not r or r > 3:
                misses.append({"id": e["id"], "q": e["q"], "expect": e["expect"], "got": tapes[:3]})
        all_ranks = [r for ranks in by_kind.values() for r in ranks]
        results["modes"][mode] = {
            "all": score(all_ranks),
            "by_kind": {k: score(v) for k, v in sorted(by_kind.items())},
            "misses": misses,
        }
    return results


def print_retrieval(results: dict) -> None:
    kinds = sorted({k for m in results["modes"].values() for k in m["by_kind"]})
    print(f"\nRetrieval on {results['n']} answerable questions (hit@3 per kind)\n")
    print(f"{'mode':<8} {'hit@1':>6} {'hit@3':>6} {'mrr':>6}   " + "  ".join(f"{k:>10}" for k in kinds))
    for mode, m in results["modes"].items():
        a = m["all"]
        per_kind = "  ".join(f"{m['by_kind'][k]['hit@3']:>10.0%}" for k in kinds)
        print(f"{mode:<8} {a['hit@1']:>6.0%} {a['hit@3']:>6.0%} {a['mrr']:>6.2f}   {per_kind}")
    worst = results["modes"].get("rerank", next(iter(results["modes"].values())))
    if worst["misses"]:
        print("\nrerank misses (outside top 3):")
        for miss in worst["misses"]:
            print(f"  {miss['id']} {miss['q']!r} expected {miss['expect']} got {miss['got']}")


def agent_evals() -> dict:
    from agent import NOT_ON_SHELF, build_clerk

    rows = []
    for e in EVALS:
        clerk = build_clerk(verbose=False)  # fresh memory per question, so questions can't leak into each other
        turn = clerk.ask(e["q"])
        if e["expect"]:
            ok = bool(turn.cited & set(e["expect"]))
        else:
            ok = not turn.cited and NOT_ON_SHELF.lower().rstrip(".") in turn.answer.lower()
        rows.append({"id": e["id"], "kind": e["kind"], "ok": ok, "grounded": turn.grounded,
                     "tool_calls": turn.tool_calls, "answer": turn.answer})
        print(f"  {'PASS' if ok else 'FAIL'} {e['id']:<4} calls={turn.tool_calls} {turn.answer[:90]!r}", file=sys.stderr)

    by_kind: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["ok"])
    summary = {
        "n": len(rows),
        "accuracy": sum(r["ok"] for r in rows) / len(rows),
        "grounded": sum(r["grounded"] for r in rows) / len(rows),
        "avg_tool_calls": sum(r["tool_calls"] for r in rows) / len(rows),
        "by_kind": {k: sum(v) / len(v) for k, v in sorted(by_kind.items())},
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="vector,bm25,hybrid,rerank")
    parser.add_argument("--min-hit3", type=float, default=None, help="fail if rerank hit@3 is below this")
    parser.add_argument("--agent", action="store_true", help="run end-to-end agent evals (needs a chat model)")
    parser.add_argument("--out", type=Path, default=HERE / "eval_results.json")
    args = parser.parse_args()

    if args.agent:
        from providers import ConfigError

        try:
            results = agent_evals()
        except ConfigError as exc:
            sys.exit(f"config error: {exc}")
        s = results["summary"]
        print(f"\nAgent on {s['n']} questions: accuracy {s['accuracy']:.0%}, grounded {s['grounded']:.0%}, "
              f"avg tool calls {s['avg_tool_calls']:.1f}")
        print("by kind: " + ", ".join(f"{k} {v:.0%}" for k, v in s["by_kind"].items()))
        args.out.with_name("agent_results.json").write_text(json.dumps(results, indent=2))
        return

    results = retrieval([m.strip() for m in args.modes.split(",") if m.strip()])
    print_retrieval(results)
    args.out.write_text(json.dumps(results, indent=2))
    if args.min_hit3 is not None:
        got = results["modes"]["rerank"]["all"]["hit@3"]
        if got < args.min_hit3:
            sys.exit(f"\nFAIL: rerank hit@3 {got:.0%} < required {args.min_hit3:.0%}")
        print(f"\nOK: rerank hit@3 {got:.0%} >= {args.min_hit3:.0%}")


if __name__ == "__main__":
    main()
