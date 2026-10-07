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
git clone https://github.com/3percentclub/builders-week4-midnight-rental
cd builders-week4-midnight-rental
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

Made by [3percentclub](https://3percentclub.org) for the AI Builders fellowship. Movie summaries were written for this lab. **Homework:** [builders-week4-homework](https://github.com/3percentclub/builders-week4-homework)
