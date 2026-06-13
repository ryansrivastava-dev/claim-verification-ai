"""End-to-end claim verification pipeline.

Example:
python src/pipeline.py --claim "High blood pressure increases risk of stroke." --retriever hybrid --top_k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd

try:
    from retrieve_tfidf import TFIDFRetriever
    from retrieve_bm25 import BM25Retriever
    from retrieve_dense import DenseRetriever
    from retrieve_hybrid import HybridRetriever
except ImportError:  # pragma: no cover
    from src.retrieve_tfidf import TFIDFRetriever
    from src.retrieve_bm25 import BM25Retriever
    from src.retrieve_dense import DenseRetriever
    from src.retrieve_hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


class RetrieverLike(Protocol):
    def retrieve(self, claim: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...


def build_retriever(name: str, passages: pd.DataFrame, alpha: float = 0.5) -> RetrieverLike:
    name = name.lower()
    if name == "tfidf":
        return TFIDFRetriever(passages)
    if name == "bm25":
        return BM25Retriever(passages)
    if name == "dense":
        return DenseRetriever(passages)
    if name == "hybrid":
        return HybridRetriever(passages, alpha=alpha)
    raise ValueError(f"Unknown retriever: {name}")


def load_classifier(classifier_name: str):
    if classifier_name == "tfidf_logistic_regression":
        path = MODEL_DIR / "tfidf_logreg.joblib"
    elif classifier_name == "embedding_random_forest":
        path = MODEL_DIR / "embedding_random_forest.joblib"
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    if not path.exists():
        return None
    return joblib.load(path)


def predict_label(claim: str, evidence: list[dict[str, Any]], classifier: Any, classifier_name: str) -> tuple[str, float | None]:
    """Predict the claim label using the top retrieved evidence.

    If no saved classifier exists, the function returns NOT_ENOUGH_INFO instead
    of pretending to know the answer.
    """
    evidence_text = " ".join(item.get("text", "") for item in evidence)
    input_text = f"{claim} [EVIDENCE] {evidence_text}".strip()

    if classifier is None:
        return "NOT_ENOUGH_INFO", None

    if classifier_name == "embedding_random_forest":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(classifier["encoder_name"])
        embedding = encoder.encode([input_text], convert_to_numpy=True, normalize_embeddings=True)
        pred = classifier["classifier"].predict(embedding)[0]
        confidence = None
        if hasattr(classifier["classifier"], "predict_proba"):
            confidence = float(classifier["classifier"].predict_proba(embedding).max())
        return str(pred), confidence

    pred = classifier.predict([input_text])[0]
    confidence = None
    if hasattr(classifier, "predict_proba"):
        confidence = float(classifier.predict_proba([input_text]).max())
    return str(pred), confidence


def run_pipeline(
    claim: str,
    retriever: RetrieverLike,
    classifier: Any = None,
    retriever_name: str = "custom",
    classifier_name: str = "untrained_fallback",
    top_k: int = 5,
) -> dict[str, Any]:
    evidence = retriever.retrieve(claim, top_k=top_k)
    predicted_label, confidence = predict_label(claim, evidence, classifier, classifier_name)
    return {
        "claim": claim,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "top_evidence": evidence,
        "retrieval_method": retriever_name,
        "classifier": classifier_name if classifier is not None else "untrained_fallback",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end scientific claim verification.")
    parser.add_argument("--claim", required=True)
    parser.add_argument("--retriever", choices=["tfidf", "bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--classifier", choices=["tfidf_logistic_regression", "embedding_random_forest"], default="tfidf_logistic_regression")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    passages_path = PROCESSED_DIR / "evidence_passages.csv"
    if not passages_path.exists():
        raise FileNotFoundError("Missing data/processed/evidence_passages.csv. Run python src/load_data.py first.")

    passages = pd.read_csv(passages_path)
    retriever = build_retriever(args.retriever, passages, alpha=args.alpha)
    classifier = load_classifier(args.classifier)
    result = run_pipeline(
        claim=args.claim,
        retriever=retriever,
        classifier=classifier,
        retriever_name=args.retriever,
        classifier_name=args.classifier,
        top_k=args.top_k,
    )
    if classifier is None:
        result["warning"] = "No trained classifier was found. Run python src/train_classifier.py for real predictions."
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
