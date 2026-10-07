# AUTO-EXPORTED from lab.ipynb for code review only. Edit the notebook, not this file.

# %% [markdown]
# # Midnight Rental: Tuning a RAG Pipeline on a Haunted Video Store
# 
# **3percentclub · AI Builders · Week 4 lab · 35 minutes**
# 
# It's 11:58 PM at **Midnight Rental**, the last video store in Brooklyn. There are 38 horror tapes on the shelf, and the clerk's AI assistant has to find the right one from a half-remembered description. You'll build the two-stage retrieval stack from the slides over the shelf, then break it and fix it three ways:
# 
# | Block | What you do |
# |---|---|
# | 1. Setup | Install, paste your OpenAI key |
# | 2. Data | Load the 38 tapes (title, cast, plot, ending, clerk's note) |
# | 3. Pipeline | Chunk → ChromaDB vector index + BM25 index → RRF fusion → FlashRank re-ranker |
# | Experiment 1 | Chunk size: 512 vs 128 tokens |
# | Experiment 2 | Exact tape numbers and names: vector vs BM25 vs hybrid |
# | Experiment 3 | Re-ranking: hybrid with vs without FlashRank |
# | 4. Memory | The raw-history bug, then `CondensePlusContextChatEngine` |
# 
# **Do now:** `File → Save a copy in Drive`, then run Block 1. Run cells top to bottom with `Shift + Enter`.
# 
# > Spoiler warning: every tape includes its ending. That's on purpose (Experiment 1 needs it).

# %% [markdown]
# ## Block 1 · Setup (about 2 minutes)
# 
# Installs the stack. Ignore any red "dependency conflict" lines from Colab's preinstalled packages. If the cell finishes, you're fine.

# %% [cell 1]
%pip install -q "llama-index-core>=0.14,<0.15" "llama-index-embeddings-openai>=0.7,<0.8" "llama-index-llms-openai>=0.8,<0.9" "llama-index-vector-stores-chroma>=0.6,<0.7" "llama-index-retrievers-bm25>=0.8,<0.9" "llama-index-postprocessor-flashrank-rerank>=0.3,<0.4" "chromadb>=1.5,<2"

# %% [markdown]
# **Your OpenAI key.** Best option: click the key icon in Colab's left sidebar, add a secret named `OPENAI_API_KEY`, and turn on notebook access. Otherwise the cell below asks you to paste it, and it stays hidden.
# 
# The whole lab costs well under $0.05 (`text-embedding-3-small` + `gpt-4o-mini`).

# %% [cell 2]
import os
import getpass

try:
    from google.colab import userdata

    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
except Exception:
    pass  # Not in Colab, or no secret saved. We'll ask below.

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI API key: ")

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

Settings.embed_model.get_text_embedding("key check")  # fails fast here if the key is wrong
print("Ready. Embeddings: text-embedding-3-small · LLM: gpt-4o-mini")

# %% [markdown]
# ## Block 2 · Load the shelf
# 
# Each tape becomes one `Document`: tape number, title, year, director, cast, shelf, setting, plot, ending, and the clerk's note. The tape number and title only appear **once, at the top**. Remember that for Experiment 1.

# %% [cell 3]
import json
import os

import requests
from llama_index.core import Document

DATA_URL = "https://raw.githubusercontent.com/3percentclub/builders-week4/main/horror_movies.json"

if os.path.exists("horror_movies.json"):
    with open("horror_movies.json") as f:
        movies = json.load(f)
else:
    movies = requests.get(DATA_URL, timeout=30).json()


def to_doc(m: dict) -> Document:
    text = (
        f"Tape #{m['tape']}: {m['title']} ({m['year']}). Directed by {m['director']}. "
        f"Starring {m['cast']}. Shelf: {m['subgenre']}. Setting: {m['setting']}.\n"
        f"Plot: {m['plot']}\n"
        f"Ending (spoilers): {m['ending']}\n"
        f"Clerk's note: {m['note']}"
    )
    # The tape number rides along as metadata for scoring, but stays out of the
    # embedding and the prompt, so each chunk only "knows" what its text says.
    return Document(
        text=text,
        metadata={"tape": m["tape"]},
        excluded_embed_metadata_keys=["tape"],
        excluded_llm_metadata_keys=["tape"],
    )


docs = [to_doc(m) for m in movies]
print(f"{len(docs)} tapes on the shelf:\n")
for m in movies:
    print(f"  #{m['tape']}  {m['title']} ({m['year']})")
print("\nOne full tape:\n")
print(docs[0].text)

# %% [markdown]
# ## Block 3 · Build the two-stage pipeline
# 
# This is the architecture slide, in code:
# 
# `Raw docs → chunk (512 tokens, 50 overlap) → ChromaDB vector index + BM25 keyword index → RRF merge (top 10) → FlashRank re-rank (top 3) → LLM`
# 
# `build_pipeline()` takes the chunk size as an argument so Experiment 1 can rebuild it. The first run downloads FlashRank's ~22 MB cross-encoder model.

# %% [cell 4]
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import QueryBundle
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

chroma = chromadb.EphemeralClient()  # in-memory; nothing to install or clean up


def build_pipeline(chunk_size: int = 512, chunk_overlap: int = 50):
    # 1. Ingest: slice each tape into chunks.
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(docs)

    # 2. Index: dense vectors in ChromaDB (fresh collection every rebuild).
    name = f"tapes_{chunk_size}"
    try:
        chroma.delete_collection(name)
    except Exception:
        pass
    store = ChromaVectorStore(chroma_collection=chroma.create_collection(name))
    index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store))

    # 3. Retrieve: dense (meaning) + sparse BM25 (exact words), merged with Reciprocal Rank Fusion.
    vector = index.as_retriever(similarity_top_k=10)
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    hybrid = QueryFusionRetriever(
        [vector, bm25],
        similarity_top_k=10,
        num_queries=1,  # 1 = no LLM query expansion; just fuse the two lists
        mode="reciprocal_rerank",
        use_async=False,
    )
    print(f"chunk_size={chunk_size}: {len(nodes)} chunks indexed")
    return nodes, vector, bm25, hybrid


def show(results, n: int = 5):
    for i, r in enumerate(results[:n], 1):
        text = " ".join(r.node.get_content().split())
        print(f"  #{i} {text[:110]}")


nodes, vector_retriever, bm25_retriever, hybrid_retriever = build_pipeline(chunk_size=512)

# 4. Re-rank: a cross-encoder reads query + chunk together and keeps the best 3.
reranker = FlashRankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=3)

print("\nSmoke test, hybrid top 3 for 'cursed videotape':")
show(hybrid_retriever.retrieve("cursed videotape"), 3)

# %% [markdown]
# ## Experiment 1 · Chunk size: 512 vs 128 tokens
# 
# **Task:** rebuild with `chunk_size=128` and compare.
# **Watch:** at 128 tokens, a tape's ending gets sliced away from its title. Those **orphan chunks** say *what happened* but not *which movie*, so the LLM can't name it.

# %% [cell 5]
nodes_128, vector_128, bm25_128, hybrid_128 = build_pipeline(chunk_size=128, chunk_overlap=20)


def orphans(chunks):
    return [c for c in chunks if "Tape #" not in c.get_content()]


print(f"\nOrphan chunks (no title) at 512: {len(orphans(nodes))} of {len(nodes)}")
print(f"Orphan chunks (no title) at 128: {len(orphans(nodes_128))} of {len(nodes_128)}")
print("\nAn orphan chunk at 128 tokens. Which movie is this?\n")
print(orphans(nodes_128)[0].get_content() if orphans(nodes_128) else "(none this run)")

# %% [cell 6]
def full_story(hit) -> bool:
    """True if this one chunk has the title AND the ending."""
    chunk = hit.node.get_content()
    return "Tape #" in chunk and "Ending" in chunk


q1 = "How does the movie end where the killer's body vanishes from the lawn?"
for label, retriever in [("512", vector_retriever), ("128", vector_128)]:
    hits = retriever.retrieve(q1)[:5]
    print(f"\nchunk_size={label}: {sum(full_story(h) for h in hits)} of the top 5 chunks have title + ending")
    show(hits)

print("\nThe #1 chunk at 128 tokens, in full. Which movie is it?\n")
print(vector_128.retrieve(q1)[0].node.get_content())

# %% [markdown]
# ## Experiment 2 · Exact tape numbers and names: vector vs BM25 vs hybrid
# 
# Embeddings capture *meaning*. A tape number like `1313` has no meaning, and a rare surname like `Pleasence` is mostly just letters to the embedding. BM25 matches the exact token.
# 
# **Task:** run the three retrievers on an exact tape number and an exact actor surname.
# **Watch:** the rank where the right tape appears. `None` means it isn't in the top 10 at all.

# %% [cell 7]
tests = [
    ("1313", "Tape #1313"),  # Oculus
    ("Pleasence", "Pleasence"),  # Donald Pleasence, Halloween
]

for query, needle in tests:
    print(f"\nQuery: {query!r}")
    for label, retriever in [("vector", vector_retriever), ("bm25", bm25_retriever), ("hybrid", hybrid_retriever)]:
        results = retriever.retrieve(query)
        rank = next((i for i, r in enumerate(results, 1) if needle in r.node.get_content()), None)
        print(f"  {label:7} right tape at rank: {rank}")

# %% [markdown]
# **Try your own:** a tape number from the Block 2 list, a director, or an actor's last name.

# %% [cell 8]
my_query = "5114"
print("vector:"); show(vector_retriever.retrieve(my_query), 3)
print("bm25:"); show(bm25_retriever.retrieve(my_query), 3)
print("hybrid:"); show(hybrid_retriever.retrieve(my_query), 3)

# %% [markdown]
# ## Experiment 3 · Re-ranking: hybrid with vs without FlashRank
# 
# A customer says: *"It's a slasher where the final girl's friends die while she's babysitting."* That's **Halloween** (tape #1031). But slashers all sound alike: masks, teens, final girls.
# 
# RRF merges two lists by *rank*, so the top 10 is a good shortlist but a loose order. The cross-encoder deep-reads query + chunk together and re-orders the shortlist.
# 
# **Task:** find where Halloween lands with vector only, hybrid only, and hybrid + FlashRank.
# **Watch:** whether FlashRank pushes the right tape to #1.

# %% [cell 9]
q3 = "A slasher where the final girl's friends die while she babysits"
answer = "1031"  # Halloween


def rank_of(results, tape=answer):
    return next((i for i, r in enumerate(results, 1) if r.node.metadata["tape"] == tape), None)


vector_hits = vector_retriever.retrieve(q3)
shortlist = hybrid_retriever.retrieve(q3)
reranked = reranker.postprocess_nodes(shortlist, QueryBundle(q3))

print(f"Query: {q3!r}\n")
print(f"Vector only:       Halloween at rank {rank_of(vector_hits)}"); show(vector_hits, 3)
print(f"\nHybrid only:       Halloween at rank {rank_of(shortlist)}"); show(shortlist, 3)
print(f"\nHybrid + FlashRank: Halloween at rank {rank_of(reranked)}"); show(reranked, 3)

# %% [markdown]
# **The full engine:** hybrid retrieval + FlashRank feeding the LLM. Delete the `node_postprocessors` argument to compare answers.

# %% [cell 10]
from llama_index.core.query_engine import RetrieverQueryEngine

engine = RetrieverQueryEngine.from_args(hybrid_retriever, node_postprocessors=[reranker])
print(engine.query(q3))

# %% [markdown]
# ## Block 4 · Memory: rewrite before you retrieve
# 
# **The bug:** paste the raw chat history into the search and the vector DB searches for the *assistant's* words too. Here the customer asked about found footage, then asked for "anything like that but on a train." The right tape is **Train to Busan** (#2016).

# %% [cell 11]
history = (
    "User: Recommend a found-footage movie for tonight.\n"
    "Assistant: The Blair Witch Project and Paranormal Activity are the classics. Both are shot on "
    "shaky handheld cameras, one lost in the woods and one in a house at night.\n"
    "User: Anything like that but set on a train?"
)
clean = "Horror movie set on a train"

for label, query in [("raw history", history), ("standalone question", clean)]:
    hits = vector_retriever.retrieve(query)[:3]
    found = any(h.node.metadata["tape"] == "2016" for h in hits)
    titles = [h.node.get_content().split(":")[1].split("(")[0].strip() for h in hits]
    print(f"{label:20} -> Train to Busan in top 3: {found}   {titles}")

# %% [markdown]
# **The fix:** `CondensePlusContextChatEngine` has a small LLM call rewrite each follow-up into a standalone question, then retrieves with *only* that. `verbose=True` prints the rewritten question so you can see it.

# %% [cell 12]
from llama_index.core.chat_engine import CondensePlusContextChatEngine

chat = CondensePlusContextChatEngine.from_defaults(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    system_prompt="You are the clerk at Midnight Rental. Recommend tapes from the shelf only, and give the tape number.",
    verbose=True,
)

print(chat.chat("Recommend a found-footage movie for tonight."), "\n")
print(chat.chat("Anything like that but set on a train?"))

# %% [markdown]
# ## Results matrix (fill this in)
# 
# Double-click this cell to edit it. Write what *your* run printed.
# 
# | Experiment | Setup | Test query | What I saw |
# |---|---|---|---|
# | 1. Chunk size | 512 vs 128 tokens | "...body vanishes from the lawn?" | Orphans 512: __ · 128: __ · Title + ending in top 5: 512 __/5 · 128 __/5 |
# | 2. Retrieval mode | Vector vs BM25 vs hybrid | "1313" + "Pleasence" | Rank: vector __ · BM25 __ · hybrid __ |
# | 3. Re-ranking | Hybrid vs + FlashRank | Babysitter slasher | Halloween rank: hybrid __ · FlashRank __ |
# | 4. Memory | Raw history vs standalone | "Anything like that but set on a train?" | Train to Busan in top 3: raw __ · clean __ |
# 
# ## Claim what you built
# 
# Only if you ran every block above, add this to your resume:
# 
# > Built a hybrid-search RAG pipeline using LlamaIndex, ChromaDB, and BM25 sparse retrieval; implemented Reciprocal Rank Fusion (RRF), cross-encoder re-ranking (FlashRank), and condensed-query memory for multi-turn chat, and measured how chunk size, retrieval mode, and re-ranking change retrieval quality.
# 
# **Homework:** see the [homework/](homework/) folder in this repo
