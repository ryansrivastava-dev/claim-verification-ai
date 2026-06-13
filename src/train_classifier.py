"""Train lightweight scientific claim verification classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from preprocess import clean_text, normalize_label
except ImportError:  # pragma: no cover
    from src.preprocess import clean_text, normalize_label

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def load_training_data(path: Path = PROCESSED_DIR / "claim_evidence_pairs.csv") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError("Missing claim_evidence_pairs.csv. Run python src/load_data.py first.")
    df = pd.read_csv(path).fillna("")
    if "label" not in df.columns:
        raise ValueError("Training data must contain a label column.")
    df["label"] = df["label"].apply(normalize_label)
    df["input_text"] = (df["claim"].map(clean_text) + " [EVIDENCE] " + df.get("evidence_text", "").map(clean_text)).str.strip()
    # Keep labels that have at least two examples to avoid train/test split errors.
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= 2].index
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
    if df.empty or df["label"].nunique() < 2:
        raise ValueError("Not enough labeled examples to train a classifier.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    return train_test_split(
        df["input_text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def train_tfidf_logreg(X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=50000)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_embedding_classifier(X_train: pd.Series, y_train: pd.Series, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict[str, Any]:
    """Train a sentence-embedding + RandomForest classifier."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for embedding classifier.") from exc

    encoder = SentenceTransformer(model_name)
    X_embeddings = encoder.encode(X_train.tolist(), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_embeddings, y_train)
    return {"encoder_name": model_name, "classifier": clf}


def evaluate_model(model: Any, X_test: pd.Series, y_test: pd.Series, embedding: bool = False) -> dict[str, Any]:
    if embedding:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model["encoder_name"])
        X_embeddings = encoder.encode(X_test.tolist(), convert_to_numpy=True, normalize_embeddings=True)
        preds = model["classifier"].predict(X_embeddings)
    else:
        preds = model.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train claim verification classifiers.")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip sentence embedding classifier for faster CPU-only runs.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_training_data()
    X_train, X_test, y_train, y_test = split_data(df)

    metrics: dict[str, Any] = {}

    tfidf_model = train_tfidf_logreg(X_train, y_train)
    joblib.dump(tfidf_model, MODEL_DIR / "tfidf_logreg.joblib")
    metrics["tfidf_logistic_regression"] = evaluate_model(tfidf_model, X_test, y_test)

    if not args.skip_embeddings:
        try:
            embedding_model = train_embedding_classifier(X_train, y_train)
            joblib.dump(embedding_model, MODEL_DIR / "embedding_random_forest.joblib")
            metrics["embedding_random_forest"] = evaluate_model(embedding_model, X_test, y_test, embedding=True)
        except Exception as exc:
            metrics["embedding_random_forest_error"] = f"{type(exc).__name__}: {exc}"
            print(f"Embedding classifier skipped: {metrics['embedding_random_forest_error']}")

    (PROCESSED_DIR / "classifier_training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
