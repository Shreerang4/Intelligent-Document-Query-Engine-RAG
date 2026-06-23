"""BM25 lexical retriever over a List[ChunkRecord].

No dependency on main.py. Takes the same chunk list produced by
pipeline_adapter.ingest_document() so both retrievers run over identical text.
"""
from typing import Dict, List

from rank_bm25 import BM25Okapi


def bm25_search(question: str, chunks: List[Dict], k: int = 10) -> List[Dict]:
    """Return up to k chunks ranked by BM25Okapi score (descending).

    Tokenises by whitespace after lowercasing — the same level of normalisation
    the embedding model sees, keeping comparisons fair.
    """
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question.lower().split())
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked_indices[:k]]
