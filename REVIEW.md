# Code review guide: Builders Week 4

Hi Carson, thanks for taking this. You don't need to know RAG. What I need is a software engineer's eye on code quality: is it correct, readable, and would a beginner copying it pick up good habits?

**Audience:** fellows in a free AI class. Most can write some Python. Many run the lab in Google Colab, and some have never used a terminal.

## What's in the repo

| Path | What it is | Needs an API key? |
|---|---|---|
| `lab.ipynb` | In-class notebook (about 75 min). Builds a search pipeline over 38 horror movies, then runs 3 experiments. | Yes, OpenAI (whole run costs under $0.05) |
| `lab_as_script.py` | **Review branch only.** The notebook's code cells exported to plain Python so you can leave line comments in the PR. Comment here, and I'll apply the fixes to the notebook. | n/a |
| `horror_movies.json` | Dataset: 38 movie summaries written for the lab. | No |
| `homework/chunker.py` + `test_chunker.py` | Level 1: students fix a deliberately buggy chunker. | No |
| `homework/fusion.py` + `test_fusion.py` | Level 2: students write Reciprocal Rank Fusion from scratch. | No |
| `homework/ask.py` | Level 3 bonus: stdlib-only script that calls any OpenAI-compatible API. | Yes, any provider |
| `.github/workflows/tests.yml` | Runs the homework tests on every push. | No |
| `INSTRUCTOR-NOTES.md` | Expected results for the teacher. | No |

## RAG in 60 seconds (only what you need to review)

- **RAG** = search your own documents first, then paste the best matches into the LLM prompt so it answers from them.
- **Chunking** = splitting each document into smaller pieces before indexing. Pieces that are too small lose context: a chunk with a movie's ending but no title is an "orphan."
- **Vector search** finds text with similar *meaning* (embeddings). **BM25** is classic keyword search, good for exact names and numbers. **Hybrid** runs both.
- **Reciprocal Rank Fusion (RRF)** merges the two ranked lists: `score = sum(1 / (k + rank))`.
- **Re-ranking** = a small second model re-scores the top results for accuracy.
- **Condensing** = rewriting a follow-up ("anything like that but on a train?") into a standalone question before searching.

## How to test

**Homework (2 minutes, no key):**

```bash
git clone https://github.com/3percentclub/builders-week4 && cd builders-week4/homework
python -m unittest -v
```

Expected on `main`: **3 failures and 6 skipped.** That's the starting state students see, not a bug. The chunker bug is intentional, and the fusion tests are skipped until a student sets `ATTEMPTING = True`. I checked locally that a correct solution passes all 11 tests. Please try writing a quick fix yourself to see if the tests are fair and the instructions are clear.

**Lab (about 10 minutes, OpenAI key):**

```bash
cd builders-week4
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
jupyter notebook lab.ipynb    # Run All
```

You can also open it in Colab (steps in `README.md`). If you don't have a key, ask Maria and we can pair on it.

## What I'd love you to check

1. **Correctness:** anything wrong, fragile, or likely to break on a fresh Colab runtime?
2. **Slop check:** dead code, pointless abstractions, comments that state the obvious, inconsistent naming, copy-paste repetition.
3. **Beginner habits:** would you be OK with a junior copying this pattern into a real project?
4. **Tests:** do the homework tests check the right behavior? Is anything missing or over-specified?
5. **Dependencies:** `requirements.txt` pins major ranges. Too loose, too tight?
6. **Instructions:** do `README.md` and `homework/README.md` match what the code actually does?

Out of scope: the RAG technique choices themselves (Maria owns the curriculum), and the dataset wording.

## Known decisions (so you don't have to guess)

- Results can shift a little between runs because embeddings aren't perfectly deterministic. The notebook prints ranks rather than asserting them for that reason.
- `ask.py` uses only `urllib` on purpose, so it works with any provider and needs no installs.
- The notebook re-builds indexes in each experiment instead of sharing one helper so each cell reads top to bottom on its own. If you think a helper would be clearer, say so.

## How to give feedback

- **Comments:** leave line comments on the review PR (it shows every file as new). "Request changes" or "Approve" when you're done.
- **Small fixes you want to make yourself:** branch off `main`, commit, and open a PR into `main`. Please don't merge the review PR itself, since it only exists so every line is commentable.
- Then text Maria a one-line verdict: ship it, ship with fixes, or needs work.
