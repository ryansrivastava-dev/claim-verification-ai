"""Evaluate saved classifiers with accuracy, macro F1, per-class F1, and confusion matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from train_classifier import load_training_data, split_data
except ImportError:  # pragma: no cover
    from src.train_classifier import load_training_data, split_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"


def predict_with_model(model_name: str, model: Any, X_test: pd.Series):
    if model_name == "embedding_random_forest":
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(model["encoder_name"])
        X_embeddings = encoder.encode(X_test.tolist(), convert_to_numpy=True, normalize_embeddings=True)
        return model["classifier"].predict(X_embeddings)
    return model.predict(X_test)


def evaluate_saved_model(model_name: str, model_path: Path, X_test: pd.Series, y_test: pd.Series) -> dict[str, Any]:
    model = joblib.load(model_path)
    preds = predict_with_model(model_name, model, X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    labels = sorted(set(y_test) | set(preds))
    cm = confusion_matrix(y_test, preds, labels=labels)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(values_format="d")
    plt.title(f"Confusion matrix: {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=200)
    plt.close()

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro")),
        "per_class": {label: report.get(label, {}) for label in labels},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved classifiers.")
    parser.add_argument("--train_if_missing", action="store_true", help="Run train_classifier.py manually first unless this is set.")
    args = parser.parse_args()

    df = load_training_data()
    _, X_test, _, y_test = split_data(df)

    model_paths = {
        "tfidf_logistic_regression": MODEL_DIR / "tfidf_logreg.joblib",
        "embedding_random_forest": MODEL_DIR / "embedding_random_forest.joblib",
    }

    if args.train_if_missing and not model_paths["tfidf_logistic_regression"].exists():
        import subprocess, sys

        subprocess.check_call([sys.executable, str(ROOT / "src" / "train_classifier.py")])

    results: list[dict[str, Any]] = []
    notes: list[str] = []
    for name, path in model_paths.items():
        if not path.exists():
            notes.append(f"{name} skipped because {path} does not exist.")
            continue
        try:
            results.append(evaluate_saved_model(name, path, X_test, y_test))
        except Exception as exc:
            notes.append(f"{name} failed: {type(exc).__name__}: {exc}")

    results_df = pd.DataFrame([{k: v for k, v in row.items() if k != "per_class"} for row in results])
    results_df.to_csv(PROCESSED_DIR / "classifier_results.csv", index=False)
    (PROCESSED_DIR / "classifier_results_full.json").write_text(json.dumps({"results": results, "notes": notes}, indent=2), encoding="utf-8")
    print(results_df.to_string(index=False))
    if results_df.empty:
        print("No classifier results were produced. Run python src/train_classifier.py first.")
    elif "macro_f1" in results_df:
        best = results_df.sort_values("macro_f1", ascending=False).iloc[0]
        print(f"Best saved model by macro F1: {best['model']} ({best['macro_f1']:.4f})")
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
