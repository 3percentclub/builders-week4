# Midnight Rental

**3percentclub · AI Builders · Week 4 lab: Production RAG, Vector Search & Context Memory**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/3percentclub/builders-week4/blob/main/lab.ipynb)


It's 11:58 PM at the last video store in Brooklyn. A customer half-remembers a horror movie, and the clerk's AI has to find the right tape out of 38. You'll build a two-stage hybrid retrieval pipeline over the shelf, then tune it with three experiments:

1. **Chunk size:** 512 vs 128 tokens. Watch small chunks cut a movie's ending off from its title.
2. **Retrieval mode:** vector vs BM25 vs hybrid. Watch pure vector search lose an exact tape number.
3. **Re-ranking:** hybrid with vs without FlashRank. Watch the right tape move to #1.

Plus the multi-turn memory bug, and the `CondensePlusContextChatEngine` fix.

**Stack:** LlamaIndex · ChromaDB · BM25 (`bm25s`) · Reciprocal Rank Fusion · FlashRank · local embeddings (`all-MiniLM-L6-v2`)

---

## Start here (2 clicks, no API key)

1. Click **[Open in Colab](https://colab.research.google.com/github/3percentclub/builders-week4/blob/main/lab.ipynb)**.
2. Click **Runtime → Run all**.

Link to share: **https://colab.research.google.com/github/3percentclub/builders-week4/blob/main/lab.ipynb**

That's it. Nothing asks for a key, and it's free. (To keep your edits: **File → Save a copy in Drive**.)

> **Opened it before and it asked for an OpenAI key?** That was an old copy. Click the link above for the new one.

### Optional: have an LLM key?

Only the last two cells use a chat model. Without a key, they still run and show the search results instead.
If you have a key from **anywhere** (OpenRouter, Groq, Gemini, DeepSeek, OpenAI, Anthropic...), paste it into the **⚙️ SETTINGS** cell at the top of the notebook:

```python
LLM_API_KEY = "sk-or-..."   # any provider; it figures out which one
```

The same `LLM_API_KEY` works everywhere in this repo: Colab (settings cell or a Colab Secret), and locally in a `.env` file (copy `.env.example`). `LLM_MODEL` and `LLM_BASE_URL` are there too if you need a specific model or a local server like Ollama.

## Run it locally (Jupyter or VS Code)

```bash
git clone https://github.com/3percentclub/builders-week4
cd builders-week4
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # optional: add any LLM key. Windows: copy .env.example .env
jupyter notebook lab.ipynb
```

Locally, the `%pip install` cell is optional, and running it does no harm.

## What's in the notebook

| Block | What happens |
|---|---|
| 1. Setup | Install the stack, load local embeddings, optionally connect any LLM key |
| 2. Data | Load `horror_movies.json`: 38 tapes, one `Document` each (title, cast, plot, ending, clerk's note) |
| 3. Pipeline | `SentenceSplitter` → ChromaDB vector index + BM25 index → `QueryFusionRetriever` (RRF) → `FlashRankRerank(top_n=3)` |
| Experiment 1 | Rebuild at `chunk_size=128`; count orphan chunks and chunks that have title + ending |
| Experiment 2 | Rank of the right tape for a tape number (`1313`) and an actor surname (`Pleasence`), per retriever |
| Experiment 3 | Where Halloween lands for a vague "babysitter slasher" description, before and after FlashRank |
| 4. Memory | Raw chat history as the query vs a standalone question; then `CondensePlusContextChatEngine` |
| Results matrix | Fill in what your run printed |

Embeddings can shift a little between runs, so your numbers may differ slightly from your neighbor's. Record what you see.

## Troubleshooting

| Problem | Fix |
|---|---|
| "LLM check failed" in Block 1 | The key is wrong, expired, or out of credit. The lab keeps going without an LLM; only the last two cells show retrieved chunks instead of answers. Re-run Block 1 with a different key if you want them. |
| "Couldn't tell which provider that key is for" | Set `LLM_BASE_URL` and `LLM_MODEL` as well (any OpenAI-compatible endpoint). |
| The model name doesn't exist on your provider | Set `LLM_MODEL` to one it offers, then re-run Block 1. |
| Red "dependency conflict" text after `%pip install` | That's Colab's preinstalled packages complaining. If the cell finished, keep going. If imports fail, use **Runtime → Restart session** and run from the top. |
| `NameError: name 'docs' is not defined` (or similar) | You skipped a cell. Run everything from the top (**Runtime → Run all**). |
| FlashRank or embedding download fails | They download once (~22 MB and ~80 MB). Re-run the cell. |

---

## Part 2: Ship it (agentic RAG + MCP + evals)

The notebook teaches the pipeline. `part2/` turns it into something you'd ship. **Retrieval, evals, and the MCP server need no API key and no account**: embeddings and re-ranking run locally on your laptop. Only the chat agent needs a model, and it works with any provider (or none, via Ollama).

| File | What it does |
|---|---|
| `part2/shelf.py` | The pipeline as a module. Index persists to `.shelf_index/` and rebuilds by itself when the data, chunking, or embedding model changes. `search(query, k, mode)` for every caller. |
| `part2/evals.py` | Scores `vector`, `bm25`, `hybrid`, `rerank` on 40+ labeled questions (paraphrases, exact names, clerk's-note facts, tapes the models have never seen). `--agent` also scores the agent on "not on the shelf" questions. |
| `part2/agent.py` | Agentic RAG with guardrails: a hard budget of 3 searches per question, required tape citations, citations checked against what was actually retrieved, and "not on the shelf" instead of guessing. Spoiler-free unless you pass `--spoilers`. |
| `part2/mcp_server.py` | The shelf as an MCP server (`search_shelf`, `get_tape`). Week 3 gave your agent hands; this gives it a library card. |
| `part2/providers.py` | One place that picks your model provider. |
| `part2/tests/` | 18 tests for the guardrails, using a fake model. No network. |

```bash
pip install -r requirements.txt
cd part2

python evals.py                       # the scoreboard, no key needed
python -m unittest discover -s tests  # guardrail tests, no key needed
python agent.py "the guard drives below the bottom floor of a garage"
python agent.py                       # interactive, remembers the conversation
```

What our run printed (local embeddings, no key):

```
mode      hit@1  hit@3    MRR
vector      63%    76%   0.70
bm25        90%    95%   0.93
hybrid      80%    95%   0.88
rerank      95%    98%   0.96
```

Vector-only struggles with exact names and tape numbers. Re-ranking fixes most of it. One question still misses, and that's on purpose: go find out why.

### Pick a model (any one of these)

The agent uses whichever key it finds. Or set `LLM_PROVIDER` yourself.

| Provider | Set this | Notes |
|---|---|---|
| Ollama (free, local) | nothing | Install Ollama, run `ollama pull llama3.1`. The default when no key is set. |
| OpenAI | `OPENAI_API_KEY` | |
| Vercel AI Gateway | `AI_GATEWAY_API_KEY` | One key, hundreds of models. |
| OpenRouter | `OPENROUTER_API_KEY` | |
| Groq | `GROQ_API_KEY` | Free tier, very fast. |
| DeepSeek | `DEEPSEEK_API_KEY` | Its keys start with `sk-` like OpenAI's, so use this name (or `LLM_PROVIDER=deepseek`). |
| Anthropic / Gemini | `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | |
| Anything OpenAI-compatible | `LLM_PROVIDER=custom LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...` | LM Studio, vLLM, Together, your company's proxy. |

Change the model with `LLM_MODEL=...`. If you don't, the agent asks your provider for its model list and picks a small chat model. Want hosted embeddings instead of local? `EMBED_PROVIDER=openai` (the index rebuilds itself).

**Use it from Claude Code** (no API key needed, your Claude login is enough). From the repo folder:

```bash
claude mcp add midnight-rental -- "$PWD/.venv/bin/python" "$PWD/part2/mcp_server.py"
```

Then start `claude` and ask: "search the Midnight Rental shelf for monks and a clock tower." Note: a Claude Pro/Max login is **not** an API key, so it won't work as `LLM_API_KEY`. Use it through Claude Code like this instead.

**Plug it into Claude Desktop or Cursor** (absolute paths, no key needed):

```json
{
  "mcpServers": {
    "midnight-rental": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/builders-week4/part2/mcp_server.py"]
    }
  }
}
```

Restart, then ask: "what's on the Midnight Rental shelf about monks and a clock tower?"

**Make it yours (pick one):** write 10 eval questions the pipeline fails, then fix them · swap Chroma for pgvector or LanceDB and re-run the evals · point `shelf.py` at your own documents.

**Resume bullet (put YOUR numbers in):**
> Built an agentic RAG search tool exposed over MCP, with hybrid retrieval (BM25 + vectors) and cross-encoder re-ranking; wrote a 40-question eval harness that raised hit@1 from X% to Y%, with guardrail tests for citation grounding.

---

Made by [3percentclub](https://3percentclub.org) for the AI Builders fellowship. Movie summaries were written for this lab. **Homework:** see the [homework/](homework/) folder in this repo
