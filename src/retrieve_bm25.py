"""BM25 keyword retrieval baseline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

try:
    from preprocess import clean_text
except ImportError:  # pragma: no cover
    from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "data" / "processed" / "evidence_passages.csv"


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return re.findall(r"[A-Za-z0-9]+", clean_text(text, lowercase=True))


class BM25Retriever:
    """Retrieve passages using BM25."""

    def __init__(self, passages: pd.DataFrame, text_col: str = "text") -> None:
        if BM25Okapi is None:
            raise ImportError("rank-bm25 is not installed. Run pip install -r requirements.txt")
        if passages.empty:
            raise ValueError("Passage dataframe is empty.")
        self.passages = passages.reset_index(drop=True).copy()
        self.text_col = text_col
        self.tokenized_corpus = [tokenize(text) for text in self.passages[self.text_col].fillna("")]
        self.model = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_csv(cls, path: Path = PASSAGES_PATH) -> "BM25Retriever":
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/load_data.py first.")
        return cls(pd.read_csv(path))

    def scores(self, claim: str) -> np.ndarray:
        return np.asarray(self.model.get_scores(tokenize(claim)), dtype=float)

    def retrieve(self, claim: str, top_k: int = 5) -> list[dict[str, Any]]:
        scores = self.scores(claim)
        top_indices = scores.argsort()[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for idx in top_indices:
            row = self.passages.iloc[int(idx)]
            results.append(
                {
                    "passage_id": str(row.get("passage_id", idx)),
                    "doc_id": str(row.get("doc_id", "")),
                    "score": float(scores[idx]),
                    "text": str(row.get(self.text_col, "")),
                }
            )
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve evidence with BM25.")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = BM25Retriever.from_csv()
    for result in retriever.retrieve(args.claim, args.top_k):
        print(result)


if __name__ == "__main__":
    main()
