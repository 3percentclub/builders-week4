# Builders Week 4 Homework: No Orphans at Dog-Ear Books

**You need:** a free GitHub account and a browser. You don't need to install or pay for anything.

In class you tuned a horror video store's search engine. For homework you move to a new shelf: **Dog-Ear Books**, a late-night Brooklyn bookstore that wants a clerk bot for its October table. Same RAG ideas, new data, and this time you write the key pieces yourself. Everyone does Level 1. Levels 2, 3, and 4 are optional.

| Level | What you do | Time | API key? |
|---|---|---|---|
| **1. Required** | Fix the orphan-chunk bug + answer the Three C's | 15–20 min | No |
| **2. Challenge** | Write Reciprocal Rank Fusion yourself | 20–30 min | No |
| **3. Challenge** | Build parent-child retrieval: search small, answer big | 15–20 min | No |
| **4. Bonus** | Rewrite a follow-up question with an LLM before searching | 20–30 min | Yes, any provider |

## Setup (all on github.com)

1. On [github.com/3percentclub/builders-week4](https://github.com/3percentclub/builders-week4), click **Use this template** → **Create a new repository**. Make it public. (This homework lives in the `homework/` folder of the Week 4 repo.)
2. In your copy, open the **Actions** tab and click **Enable workflows** if GitHub asks. A red X is expected at first because 3 tests fail. Click into the run to see which ones.
3. To edit a file in `homework/`, open it, click the **pencil icon**, then click **Commit changes**. The tests re-run automatically in **Actions**.

## Level 1 (required): No orphan chunks

`chunker.py` slices each book card into chunks of `max_words` words. Only the first chunk gets the title, so a chunk like this one ends up in the index:

> Speeding down the long drive, she wonders why no one is stopping her, then turns the wheel and crashes into the great oak tree at the bend.

Which book is that? A reader might know it's *The Haunting of Hill House*, but the model has no way to know.

1. In `chunker.py`, fix `chunk_book()` so **every** chunk starts with the header (`Shelf B01: The Haunting of Hill House by Shirley Jackson (1959).`), no chunk goes over `max_words` (header included), and the body stays whole and in order.
2. Keep committing until **Actions** shows a green checkmark.
3. Answer the Level 1 questions in `decision.md`.

**Hint:** work out how many words the header takes up first. Each chunk then has room for `max_words - header_words` body words. You can use AI, but you should be able to explain your fix in one sentence.

## Level 2 (challenge, no key): Write RRF

Hybrid search runs two searches (vector and BM25) and merges the two ranked lists with **Reciprocal Rank Fusion**:

```
score(doc) = sum over each list of 1 / (k + rank)      # rank starts at 1, k = 60
```

1. In `fusion.py`, change `ATTEMPTING = False` to `ATTEMPTING = True`. The fusion tests will start running (and failing).
2. Write `rrf()` until all tests pass. A dictionary and `sorted()` are enough.
3. Answer the Level 2 questions in `decision.md`.

## Level 3 (challenge, no key): Search small, answer big

Small chunks match a question precisely, but they leave the model with half a story. **Parent-child retrieval** searches tiny "child" chunks, then hands the model the full "parent" card each child came from. It's the other way to fix orphan chunks, and the one most teams use on long documents.

`parent_child.py` already cuts every book into 12-word children (ids like `B06#3`) and ranks them. You write the step in the middle.

1. In `parent_child.py`, change `ATTEMPTING = False` to `ATTEMPTING = True`.
2. Write `parents_for()`: turn ranked child ids into parent ids, best first, no duplicates, at most `top`. The rules are in the file.
3. Run `python parent_child.py` (or read the test output) to see the child chunk next to the full parent card.
4. Answer the Level 3 questions in `decision.md`.

## Level 4 (bonus, API key): Rewrite before you retrieve

`ask.py` sets up the follow-up bug from class with new data: *"Recommend a book told in letters or diaries"* → *"Anything like that but set on a train?"* It searches the shelf with the raw chat history, then has an LLM rewrite the follow-up into a standalone question and searches again. Search uses **your** `rrf()`, so finish Level 2 first.

**Bring any API key.** It isn't tied to one company, and it uses only Python's standard library. Set three values, either in a `.env` file (copy [`.env.example`](../.env.example) from the repo root) or as environment variables / Colab secrets:

| Name | What it is |
|---|---|
| `LLM_API_KEY` | Your key |
| `LLM_BASE_URL` | Your provider's OpenAI-compatible URL (table below) |
| `LLM_MODEL` | A small, cheap model from that provider (the old name `MODEL` still works) |

| Provider | `LLM_BASE_URL` |
|---|---|
| OpenAI | `https://api.openai.com/v1` |
| Anthropic (Claude) | `https://api.anthropic.com/v1` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` |
| DeepSeek | `https://api.deepseek.com` |
| OpenRouter (many models, one key) | `https://openrouter.ai/api/v1` |
| Groq | `https://api.groq.com/openai/v1` |

**Run it in Google Colab (no terminal needed):** add all three values under **Secrets** (key icon on the left), then run:

```python
!git clone https://github.com/YOUR-USERNAME/YOUR-REPO
%cd YOUR-REPO/homework
import os
from google.colab import userdata
for name in ["LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"]:
    os.environ[name] = userdata.get(name)
!python ask.py
```

**Or on your own computer:** `export` the three values, then run `python ask.py`.

Answer the Level 4 questions in `decision.md`. **Never commit your API key.**

## What to submit (Google Classroom)

Submit a link to your repo with a green checkmark in **Actions**. If you did Level 2, 3, or 4, mention it in your submission.

## Stuck?

Post in Discord **#builders** with a screenshot of the red X in Actions.
