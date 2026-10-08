# Builders Week 4 Homework: No Orphans at Midnight Rental

**You need:** a free GitHub account and a browser. You don't need to install or pay for anything.

In the lab, small chunks cut each horror movie's ending away from its title, so the AI couldn't say which movie it was. This week you fix that by hand, then write the rank-fusion math that hybrid search uses. Everyone does Level 1. Levels 2 and 3 are optional.

| Level | What you do | Time | API key? |
|---|---|---|---|
| **1. Required** | Fix the orphan-chunk bug + answer the Three C's | 15–20 min | No |
| **2. Challenge** | Write Reciprocal Rank Fusion yourself | 20–30 min | No |
| **3. Bonus** | Rewrite a follow-up question with an LLM before searching | 20–30 min | Yes, any provider |

## Setup (all on github.com)

1. On [github.com/3percentclub/builders-week4](https://github.com/3percentclub/builders-week4), click **Use this template** → **Create a new repository**. Make it public. (This homework lives in the `homework/` folder of the Week 4 repo.)
2. In your copy, open the **Actions** tab and click **Enable workflows** if GitHub asks. A red X is expected at first because 3 tests fail. Click into the run to see which ones.
3. To edit a file in `homework/`, open it, click the **pencil icon**, then click **Commit changes**. The tests re-run automatically in **Actions**.

## Level 1 (required): No orphan chunks

`chunker.py` slices each tape into chunks of `max_words` words. Only the first chunk gets the title, so a chunk like this one ends up in the index:

> Loomis arrives and shoots him six times, knocking him off the balcony. When Loomis looks down at the lawn, the body is gone.

Which movie is that? You know it's Halloween, but the model has no way to know.

1. In `chunker.py`, fix `chunk_tape()` so **every** chunk starts with the header (`Tape #1031: Halloween (1978).`), no chunk goes over `max_words` (header included), and the body stays whole and in order.
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

## Level 3 (bonus, API key): Rewrite before you retrieve

`ask.py` replays the follow-up bug from class: *"Recommend a found-footage movie"* → *"Anything like that but set on a train?"* It searches the shelf with the raw chat history, then has an LLM rewrite the follow-up into a standalone question and searches again. Search uses **your** `rrf()`, so finish Level 2 first.

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

Answer the Level 3 questions in `decision.md`. **Never commit your API key.**

## What to submit (Google Classroom)

Submit a link to your repo with a green checkmark in **Actions**. If you did Level 2 or 3, mention it in your submission.

## Stuck?

Post in Discord **#builders** with a screenshot of the red X in Actions.
