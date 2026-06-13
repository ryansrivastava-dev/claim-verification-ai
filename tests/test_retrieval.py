import numpy as np
import pandas as pd

from src.retrieve_hybrid import combine_scores
from src.retrieve_tfidf import TFIDFRetriever


def sample_passages():
    return pd.DataFrame(
        [
            {"passage_id": "p1", "doc_id": "d1", "text": "High blood pressure increases stroke risk."},
            {"passage_id": "p2", "doc_id": "d2", "text": "Photosynthesis occurs in plants."},
            {"passage_id": "p3", "doc_id": "d3", "text": "Exercise can improve cardiovascular health."},
        ]
    )


def test_tfidf_retrieval_returns_top_k():
    retriever = TFIDFRetriever(sample_passages())
    results = retriever.retrieve("blood pressure stroke", top_k=2)
    assert len(results) == 2
    assert results[0]["passage_id"] == "p1"
    assert "score" in results[0]


def test_hybrid_score_combines_normalized_scores():
    bm25 = np.array([0.0, 5.0, 10.0])
    dense = np.array([10.0, 5.0, 0.0])
    combined = combine_scores(bm25, dense, alpha=0.5)
    assert np.allclose(combined, np.array([0.5, 0.5, 0.5]))


def test_hybrid_alpha_validation():
    try:
        combine_scores(np.array([1]), np.array([1]), alpha=1.5)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError for invalid alpha"
