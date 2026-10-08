"""The Midnight Rental shelf as a reusable search module.

Same pipeline as the notebook (chunk -> vector + BM25 -> RRF -> FlashRank), packaged so an
agent, an MCP server, and an eval script share one search().

    python part2/shelf.py --warm     # build the index + download the re-ranker BEFORE class

The Chroma collection name is a fingerprint of the data, chunk settings, and embedding model,
so editing a plot or switching models builds a fresh index instead of silently reusing a stale one.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from llama_index.postprocessor.flashrank_rerank import FlashRankRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from providers import embed_model_name, make_embed_model
from records import ROOT, document_text, fingerprint, load_movies, public_record, safe_summary, with_header

log = logging.getLogger("shelf")

INDEX_DIR = Path(os.environ.get("SHELF_INDEX_DIR", ROOT / ".shelf_index"))
CANDIDATES = 10
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"

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


def _configure_models() -> None:
    Settings.embed_model = make_embed_model()
    # QueryFusionRetriever asks for an LLM even with num_queries=1; it never calls it.
    Settings.llm = MockLLM()


def _build_nodes(movies: list[dict], chunk_size: int, chunk_overlap: int) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes: list[TextNode] = []
    for movie in movies:
        doc = Document(text=document_text(movie), metadata={"tape": movie["tape"]}, excluded_embed_metadata_keys=["tape"], excluded_llm_metadata_keys=["tape"])
        for i, node in enumerate(splitter.get_nodes_from_documents([doc])):
            node.text = with_header(node.text, movie)
            # Deterministic IDs: rebuilding the same fingerprint writes the same rows, never duplicates.
            node.id_ = f"{movie['tape']}-{i}"
            nodes.append(node)
    return nodes


class Shelf:
    """Builds every retriever once and exposes one search() for all callers."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        _configure_models()
        movies = load_movies()
        self._by_tape = {m["tape"]: m for m in movies}
        nodes = _build_nodes(movies, chunk_size, chunk_overlap)

        name = f"tapes_{fingerprint(movies, chunk_size, chunk_overlap, embed_model_name())}"
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        collection = client.get_or_create_collection(name)
        store = ChromaVectorStore(chroma_collection=collection)
        if collection.count() == len(nodes):
            log.info("reusing index %s (%d chunks)", name, len(nodes))
            index = VectorStoreIndex.from_vector_store(store)
        else:
            log.info("building index %s (%d chunks)", name, len(nodes))
            # upsert by deterministic ID, so a half-finished or concurrent build converges instead of duplicating.
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
        self._reranker = FlashRankRerank(model=RERANK_MODEL, top_n=CANDIDATES)

    def search(self, query: str, k: int = 3, mode: Mode = "rerank", include_spoilers: bool = False) -> list[Hit]:
        """Return the top-k distinct tapes for a query. Endings are hidden unless include_spoilers."""
        query = query.strip()
        if not query:
            return []
        if mode == "vector":
            results = self._vector.retrieve(query)
        elif mode == "bm25":
            results = self._bm25.retrieve(query)
        elif mode in ("hybrid", "rerank"):
            results = self._hybrid.retrieve(query)
            if mode == "rerank":
                results = self._reranker.postprocess_nodes(results, query_bundle=QueryBundle(query))
        else:
            raise ValueError(f"unknown mode {mode!r}; pick one of {MODES}")
        return self._to_hits(results, include_spoilers)[:k]

    def get_tape(self, tape: str, include_spoilers: bool = False) -> dict | None:
        movie = self._by_tape.get(tape.strip().lstrip("#").strip())
        return public_record(movie, include_spoilers) if movie else None

    def _to_hits(self, results: list[NodeWithScore], include_spoilers: bool) -> list[Hit]:
        seen: set[str] = set()
        hits: list[Hit] = []
        for r in results:
            tape = r.node.metadata["tape"]
            if tape in seen:
                continue
            seen.add(tape)
            movie = self._by_tape[tape]
            text = " ".join(r.node.get_content().split()) if include_spoilers else safe_summary(movie)
            # FlashRank returns numpy float32, which json.dumps can't serialize.
            hits.append(Hit(tape=tape, title=movie["title"], score=float(r.score or 0.0), text=text))
        return hits


@lru_cache(maxsize=1)
def get_shelf() -> Shelf:
    return Shelf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", action="store_true", help="build the index and download the re-ranker now")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    shelf = get_shelf()
    if args.warm:
        hits = shelf.search("babysitter stalked by a masked killer")
        print(f"Warm. Top hit: #{hits[0].tape} {hits[0].title}")
