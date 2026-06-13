"""Evaluate retrieval methods using gold evidence document IDs when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

try:
    from preprocess import _as_list
    from retrieve_tfidf import TFIDFRetriever
    from retrieve_bm25 import BM25Retriever
    from retrieve_dense import DenseRetriever
    from retrieve_hybrid import HybridRetriever
except ImportError:  # pragma: no cover
    from src.preprocess import _as_list
    from src.retrieve_tfidf import TFIDFRetriever
    from src.retrieve_bm25 import BM25Retriever
    from src.retrieve_dense import DenseRetriever
    from src.retrieve_hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"


def parse_doc_ids(value: Any) -> set[str]:
    return {str(item) for item in _as_list(value) if str(item).strip()}


def retrieval_metrics_for_claim(gold_doc_ids: set[str], retrieved: list[dict[str, Any]], ks: list[int]) -> dict[str, float]:
    """Compute per-claim retrieval metrics."""
    retrieved_doc_ids = [str(item.get("doc_id", "")) for item in retrieved]
    metrics: dict[str, float] = {}
    if not gold_doc_ids:
        return metrics

    for k in ks:
        top_docs = retrieved_doc_ids[:k]
        hits = sum(1 for doc_id in top_docs if doc_id in gold_doc_ids)
        metrics[f"recall@{k}"] = 1.0 if hits > 0 else 0.0
        metrics[f"precision@{k}"] = hits / max(k, 1)

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr"] = reciprocal_rank
    return metrics


def evaluate_retriever(name: str, retriever: Any, claims_df: pd.DataFrame, top_k: int = 10) -> dict[str, float]:
    ks = [1, 3, 5, 10]
    rows: list[dict[str, float]] = []

    for _, row in tqdm(claims_df.iterrows(), total=len(claims_df), desc=name):
        gold_doc_ids = parse_doc_ids(row.get("evidence_doc_ids", []))
        if not gold_doc_ids:
            continue
        retrieved = retriever.retrieve(str(row["claim"]), top_k=top_k)
        claim_metrics = retrieval_metrics_for_claim(gold_doc_ids, retrieved, ks)
        if claim_metrics:
            rows.append(claim_metrics)

    if not rows:
        return {"method": name, "evaluated_claims": 0}

    df = pd.DataFrame(rows)
    output = {"method": name, "evaluated_claims": int(len(df))}
    for col in df.columns:
        output[col] = float(df[col].mean())
    return output


def plot_results(results_df: pd.DataFrame, output_path: Path) -> None:
    metric = "recall@5" if "recall@5" in results_df.columns else None
    if metric is None or results_df.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["method"], results_df[metric])
    plt.ylabel(metric)
    plt.xlabel("Retrieval method")
    plt.title("Retrieval method comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval methods.")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_claims", type=int, default=0, help="Optional limit for quick experiments; 0 means all claims.")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    claims_path = PROCESSED_DIR / "claims.csv"
    passages_path = PROCESSED_DIR / "evidence_passages.csv"
    if not claims_path.exists() or not passages_path.exists():
        raise FileNotFoundError("Run python src/load_data.py first.")

    claims_df = pd.read_csv(claims_path)
    passages_df = pd.read_csv(passages_path)
    if args.max_claims:
        claims_df = claims_df.head(args.max_claims)

    retrievers: list[tuple[str, Callable[[], Any]]] = [
        ("tfidf", lambda: TFIDFRetriever(passages_df)),
        ("bm25", lambda: BM25Retriever(passages_df)),
        ("dense", lambda: DenseRetriever(passages_df)),
        ("hybrid_alpha_0.50", lambda: HybridRetriever(passages_df, alpha=0.50)),
    ]

    results: list[dict[str, Any]] = []
    notes: list[str] = []
    for name, builder in retrievers:
        try:
            retriever = builder()
            results.append(evaluate_retriever(name, retriever, claims_df, top_k=args.top_k))
        except Exception as exc:
            notes.append(f"{name} skipped: {type(exc).__name__}: {exc}")
            print(notes[-1])

    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DIR / "retrieval_results.csv", index=False)
    (PROCESSED_DIR / "retrieval_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    plot_results(results_df, FIGURES_DIR / "retrieval_comparison.png")
    print(results_df.to_string(index=False))
    if notes:
        print("Notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
