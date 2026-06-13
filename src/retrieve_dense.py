"""Dense retrieval with Sentence-Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from preprocess import clean_text
except ImportError:  # pragma: no cover
    from src.preprocess import clean_text

ROOT = Path(__file__).resolve().parents[1]
PASSAGES_PATH = ROOT / "data" / "processed" / "evidence_passages.csv"
MODEL_DIR = ROOT / "models"


class DenseRetriever:
    """Retrieve passages using sentence embeddings and cosine similarity."""

    def __init__(
        self,
        passages: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_path: Path | None = None,
        text_col: str = "text",
    ) -> None:
        if passages.empty:
            raise ValueError("Passage dataframe is empty.")
        self.passages = passages.reset_index(drop=True).copy()
        self.text_col = text_col
        self.model_name = model_name
        self.cache_path = cache_path or MODEL_DIR / "dense_passage_embeddings.npy"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is not installed. Run pip install -r requirements.txt") from exc

        self.model = SentenceTransformer(model_name)
        self.embeddings = self._load_or_encode_passages()

    @classmethod
    def from_csv(cls, path: Path = PASSAGES_PATH, **kwargs: Any) -> "DenseRetriever":
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python src/load_data.py first.")
        return cls(pd.read_csv(path), **kwargs)

    def _load_or_encode_passages(self) -> np.ndarray:
        if self.cache_path.exists():
            embeddings = np.load(self.cache_path)
            if embeddings.shape[0] == len(self.passages):
                return embeddings
        texts = self.passages[self.text_col].fillna("").map(clean_text).tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
        np.save(self.cache_path, embeddings)
        return embeddings

    def scores(self, claim: str) -> np.ndarray:
        query_embedding = self.model.encode([clean_text(claim)], convert_to_numpy=True, normalize_embeddings=True)
        return cosine_similarity(query_embedding, self.embeddings).ravel()

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
    parser = argparse.ArgumentParser(description="Retrieve evidence with dense sentence embeddings.")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    retriever = DenseRetriever.from_csv(model_name=args.model_name)
    for result in retriever.retrieve(args.claim, args.top_k):
        print(result)


if __name__ == "__main__":
    main()
