"""TF-IDF retrieval baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from preprocess import clean_text
except ImportError:  # pragma: no cover
    from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "data" / "processed" / "evidence_passages.csv"


class TFIDFRetriever:
    """Retrieve passages using TF-IDF cosine similarity."""

    def __init__(self, passages: pd.DataFrame, text_col: str = "text") -> None:
        if passages.empty:
            raise ValueError("Passage dataframe is empty.")
        self.passages = passages.reset_index(drop=True).copy()
        self.text_col = text_col
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.passages[self.text_col].fillna("").map(clean_text))

    @classmethod
    def from_csv(cls, path: Path = PASSAGES_PATH) -> "TFIDFRetriever":
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/load_data.py first.")
        return cls(pd.read_csv(path))

    def retrieve(self, claim: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_vector = self.vectorizer.transform([clean_text(claim)])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
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
    parser = argparse.ArgumentParser(description="Retrieve evidence with TF-IDF.")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    retriever = TFIDFRetriever.from_csv()
    for result in retriever.retrieve(args.claim, args.top_k):
        print(result)


if __name__ == "__main__":
    main()
