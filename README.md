# Midnight Rental

**3percentclub · AI Builders · Week 4 lab: Production RAG, Vector Search & Context Memory**

It's 11:58 PM at the last video store in Brooklyn. A customer half-remembers a horror movie, and the clerk's AI has to find the right tape out of 38. You'll build a two-stage hybrid retrieval pipeline over the shelf, then tune it with three experiments:

1. **Chunk size:** 512 vs 128 tokens. Watch small chunks cut a movie's ending off from its title.
2. **Retrieval mode:** vector vs BM25 vs hybrid. Watch pure vector search lose an exact tape number.
3. **Re-ranking:** hybrid with vs without FlashRank. Watch the right tape move to #1.

Plus the multi-turn memory bug, and the `CondensePlusContextChatEngine` fix.

**Stack:** LlamaIndex · ChromaDB · BM25 (`bm25s`) · Reciprocal Rank Fusion · FlashRank · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)

---

## Run it in Google Colab (recommended, nothing to install)

1. Go to [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub** tab.
2. Paste this repo's URL and pick `lab.ipynb`.
3. **File → Save a copy in Drive** so your edits are saved to your own copy.
4. Add your key: click the **key icon** in the left sidebar → **Add new secret** → name it `OPENAI_API_KEY`, paste the key, and turn on **Notebook access**. Or skip this and paste the key when the notebook asks.
5. Run cells top to bottom with `Shift + Enter`.

The whole lab costs well under $0.05 in OpenAI usage.

## Run it locally (Jupyter or VS Code)

```bash
git clone https://github.com/3percentclub/builders-week4
cd builders-week4
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...       # Windows PowerShell: $env:OPENAI_API_KEY="sk-..."
jupyter notebook lab.ipynb
```

Locally, the `%pip install` cell is optional, and running it does no harm.

## What's in the notebook

| Block | What happens |
|---|---|
| 1. Setup | Install the stack, load your key, check it works |
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
| `AuthenticationError` / 401 at the key check | Wrong or expired key. Re-run the key cell. In Colab, make sure the secret's **Notebook access** toggle is on. |
| `RateLimitError` / `insufficient_quota` | Your OpenAI account has no credit. Add a few dollars of credit, or pair with a neighbor. |
| Red "dependency conflict" text after `%pip install` | That's Colab's preinstalled packages complaining. If the cell finished, keep going. If imports fail, use **Runtime → Restart session** and run from the top. |
| `NameError: name 'docs' is not defined` (or similar) | You skipped a cell. Run everything from the top (**Runtime → Run all**). |
| FlashRank download fails | It downloads a ~22 MB model once. Re-run the Block 3 cell. |

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
| Anything OpenAI-compatible | `LLM_PROVIDER=custom LLM_BASE_URL=... LLM_API_KEY=... LLM_MODEL=...` | LM Studio, vLLM, Together, your company's proxy. |

Change the model with `LLM_MODEL=...`. Want hosted embeddings instead of local? `EMBED_PROVIDER=openai` (the index rebuilds itself).

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
