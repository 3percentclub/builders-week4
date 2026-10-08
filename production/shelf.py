"""The Midnight Rental shelf as a reusable search module.

Same pipeline as the notebook (chunk -> vector + BM25 -> RRF -> FlashRank),
packaged so an agent, an MCP server, and an eval script can all share it.
The Chroma index is persisted to disk, so embeddings are paid for once.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "horror_movies.json"
INDEX_DIR = ROOT / ".shelf_index"
COLLECTION = "tapes"
CANDIDATES = 10

Mode = Literal["vector", "bm25", "hybrid", "rerank"]
MODES: tuple[Mode, ...] = ("vector", "bm25", "hybrid", "rerank")


@dataclass(frozen=True)
class Hit:
    tape: str
    title: str
    score: float
    text: str

    def as_dict(self) -> dict:
        return {"tape": self.tape, "title": self.title, "score": round(self.score, 4), "text": self.text}


@lru_cache(maxsize=1)
def load_movies() -> tuple[dict, ...]:
    return tuple(json.loads(DATA_PATH.read_text()))


def _to_document(movie: dict) -> Document:
    text = (
        f"Tape #{movie['tape']}: {movie['title']} ({movie['year']}). Directed by {movie['director']}. "
        f"Starring {movie['cast']}. Shelf: {movie['subgenre']}. Setting: {movie['setting']}.\n"
        f"Plot: {movie['plot']}\n"
        f"Ending (spoilers): {movie['ending']}\n"
        f"Clerk's note: {movie['note']}"
    )
    hidden = ["tape", "title"]
    return Document(
        text=text,
        metadata={"tape": movie["tape"], "title": movie["title"]},
        excluded_embed_metadata_keys=hidden,
        excluded_llm_metadata_keys=hidden,
    )


def _configure_models() -> None:
    # model_name (not model) skips LlamaIndex's enum check, so OpenAI-compatible gateways work too.
    Settings.embed_model = OpenAIEmbedding(model_name=os.environ.get("EMBED_MODEL", "text-embedding-3-small"))
    # QueryFusionRetriever asks for an LLM even with num_queries=1; it never calls it.
    Settings.llm = MockLLM()


class Shelf:
    """Builds every retriever once and exposes one search() for all callers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        _configure_models()
        documents = [_to_document(m) for m in load_movies()]
        nodes = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).get_nodes_from_documents(documents)

        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        collection = client.get_or_create_collection(COLLECTION)
        store = ChromaVectorStore(chroma_collection=collection)
        if collection.count() == len(nodes):
            index = VectorStoreIndex.from_vector_store(store)
        else:
            client.delete_collection(COLLECTION)
            store = ChromaVectorStore(chroma_collection=client.create_collection(COLLECTION))
            index = VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store))

        self._vector = index.as_retriever(similarity_top_k=CANDIDATES)
        self._bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=CANDIDATES)
        self._hybrid = QueryFusionRetriever(
            [self._vector, self._bm25],
            similarity_top_k=CANDIDATES,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )
        self._reranker = FlashRankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=CANDIDATES)
        self._by_tape = {m["tape"]: m for m in load_movies()}

    def search(self, query: str, k: int = 3, mode: Mode = "rerank") -> list[Hit]:
        """Return the top-k distinct tapes for a query (one hit per tape)."""
        if mode == "vector":
            results = self._vector.retrieve(query)
        elif mode == "bm25":
            results = self._bm25.retrieve(query)
        else:
            results = self._hybrid.retrieve(query)
            if mode == "rerank":
                results = self._reranker.postprocess_nodes(results, query_bundle=QueryBundle(query))
        return _dedupe_by_tape(results)[:k]

    def get_tape(self, tape: str) -> dict | None:
        return self._by_tape.get(tape.lstrip("#").strip())


def _dedupe_by_tape(results: list[NodeWithScore]) -> list[Hit]:
    seen: set[str] = set()
    hits: list[Hit] = []
    for r in results:
        tape = r.node.metadata["tape"]
        if tape in seen:
            continue
        seen.add(tape)
        text = " ".join(r.node.get_content().split())
        # FlashRank returns numpy float32, which json.dumps can't serialize.
        hits.append(Hit(tape=tape, title=r.node.metadata["title"], score=float(r.score or 0.0), text=text))
    return hits


@lru_cache(maxsize=1)
def get_shelf() -> Shelf:
    return Shelf()
