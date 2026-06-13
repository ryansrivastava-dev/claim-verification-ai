"""Hybrid retrieval that combines BM25 and dense retrieval."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from retrieve_bm25 import BM25Retriever
    from retrieve_dense import DenseRetriever
except ImportError:  # pragma: no cover
    from src.retrieve_bm25 import BM25Retriever
    from src.retrieve_dense import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "data" / "processed" / "evidence_passages.csv"


def min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to 0-1. If all scores are equal, return zeros."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score == min_score:
        return np.zeros_like(scores, dtype=float)
    return (scores - min_score) / (max_score - min_score)


def combine_scores(bm25_scores: np.ndarray, dense_scores: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Combine normalized BM25 and dense scores.

    hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    bm25_norm = min_max_normalize(np.asarray(bm25_scores, dtype=float))
    dense_norm = min_max_normalize(np.asarray(dense_scores, dtype=float))
    if bm25_norm.shape != dense_norm.shape:
        raise ValueError("BM25 and dense score arrays must have the same shape")
    return alpha * bm25_norm + (1.0 - alpha) * dense_norm


class HybridRetriever:
    """Retrieve passages using a weighted BM25 + dense score."""

    def __init__(self, passages: pd.DataFrame, alpha: float = 0.5) -> None:
        self.passages = passages.reset_index(drop=True).copy()
        self.alpha = alpha
        self.bm25 = BM25Retriever(self.passages)
        self.dense = DenseRetriever(self.passages)

    @classmethod
    def from_csv(cls, path: Path = PASSAGES_PATH, alpha: float = 0.5) -> "HybridRetriever":
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/load_data.py first.")
        return cls(pd.read_csv(path), alpha=alpha)

    def retrieve(self, claim: str, top_k: int = 5) -> list[dict[str, Any]]:
        bm25_scores = self.bm25.scores(claim)
        dense_scores = self.dense.scores(claim)
        hybrid_scores = combine_scores(bm25_scores, dense_scores, alpha=self.alpha)
        top_indices = hybrid_scores.argsort()[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for idx in top_indices:
            row = self.passages.iloc[int(idx)]
            results.append(
                {
                    "passage_id": str(row.get("passage_id", idx)),
                    "doc_id": str(row.get("doc_id", "")),
                    "score": float(hybrid_scores[idx]),
                    "bm25_score": float(bm25_scores[idx]),
                    "dense_score": float(dense_scores[idx]),
                    "text": str(row.get("text", "")),
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve evidence with hybrid BM25 + dense retrieval.")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    retriever = HybridRetriever.from_csv(alpha=args.alpha)
    for result in retriever.retrieve(args.claim, args.top_k):
        print(result)


if __name__ == "__main__":
    main()
