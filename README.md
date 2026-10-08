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

The notebook teaches the pipeline. `production/` turns it into something you'd actually ship and put on a resume. Same shelf, same retrieval stack, four small files:

| File | What it does |
|---|---|
| `production/shelf.py` | The pipeline as a reusable module. Chroma index persists to `.shelf_index/`, so you pay for embeddings once. One `search(query, k, mode)` for every caller. |
| `production/evals.py` | Scores `vector`, `bm25`, `hybrid`, and `rerank` against 15 labeled questions in `evals.json`. Prints hit@1, hit@3, and MRR. |
| `production/agent.py` | Agentic RAG. The model decides what to search and when to stop, with a 3-search budget, required tape citations, a check for invented tape numbers, and "not on the shelf" instead of guessing. |
| `production/mcp_server.py` | The shelf as an MCP server (`search_shelf`, `get_tape`). Week 3 gave your agent hands; this gives it a library card. |

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
cd production

python evals.py --misses                 # the scoreboard
python agent.py "the guy hypnotized with a teacup who sinks into the floor"
python agent.py                          # interactive, with memory
```

What our run printed (yours may shift slightly):

```
mode      hit@1  hit@3    MRR
vector      80%    87%   0.84
bm25        93%   100%   0.97
hybrid      87%   100%   0.92
rerank     100%   100%   1.00
```

Vector-only misses `tape 1313` and `Pleasence`, the exact-match problem from Experiment 2, now measured.

**Plug it into Claude Desktop or Cursor:** add this to your MCP config (use absolute paths), restart, and ask "what's on the Midnight Rental shelf about a haunted mirror?"

```json
{
  "mcpServers": {
    "midnight-rental": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/builders-week4/production/mcp_server.py"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

**Make it yours (pick one):** add 10 eval questions the pipeline fails, then fix them · swap Chroma for pgvector or LanceDB and re-run the evals · point `shelf.py` at your own documents.

**Resume bullet you've earned:**
> Built an agentic RAG search tool exposed over MCP, with hybrid retrieval (BM25 + vectors, RRF) and cross-encoder re-ranking; wrote an eval harness that raised hit@1 from 80% to 100%.

Using an OpenAI-compatible gateway instead of OpenAI? Set `OPENAI_BASE_URL` and `OPENAI_API_BASE` to its URL, plus `EMBED_MODEL` and `CHAT_MODEL`.

---

Made by [3percentclub](https://3percentclub.org) for the AI Builders fellowship. Movie summaries were written for this lab. **Homework:** see the [homework/](homework/) folder in this repo
