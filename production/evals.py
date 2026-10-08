"""Score every retrieval mode against a labeled question set.

    python production/evals.py            # table for all modes
    python production/evals.py --misses   # also list the questions each mode got wrong

hit@3 = share of questions whose right tape is in the top 3.
MRR   = mean of 1/rank of the right tape (1.0 = always #1).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shelf import MODES, get_shelf

EVALS_PATH = Path(__file__).with_name("evals.json")
DEPTH = 10


def rank_of(expect: str, tapes: list[str]) -> int | None:
    return tapes.index(expect) + 1 if expect in tapes else None


def score(mode: str, cases: list[dict]) -> dict:
    shelf = get_shelf()
    ranks, misses = [], []
    for case in cases:
        tapes = [h.tape for h in shelf.search(case["query"], k=DEPTH, mode=mode)]
        rank = rank_of(case["expect"], tapes)
        ranks.append(rank)
        if rank is None or rank > 3:
            misses.append(f'{case["query"]!r} -> rank {rank or "not found"}')
    n = len(cases)
    return {
        "mode": mode,
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@3": sum(1 for r in ranks if r and r <= 3) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--misses", action="store_true", help="list failing questions per mode")
    args = parser.parse_args()

    cases = json.loads(EVALS_PATH.read_text())
    print(f"{len(cases)} labeled questions\n")
    print(f"{'mode':<8} {'hit@1':>6} {'hit@3':>6} {'MRR':>6}")
    results = [score(mode, cases) for mode in MODES]
    for r in results:
        print(f"{r['mode']:<8} {r['hit@1']:>6.0%} {r['hit@3']:>6.0%} {r['mrr']:>6.2f}")

    if args.misses:
        for r in results:
            if r["misses"]:
                print(f"\n{r['mode']} misses:")
                for m in r["misses"]:
                    print(f"  {m}")


if __name__ == "__main__":
    main()
