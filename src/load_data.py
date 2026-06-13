"""Load SciFact and save normalized CSV files.

The script tries several known Hugging Face loading patterns because public
mirrors can differ. It records actual dataset information from the loaded data
and never hard-codes or invents dataset counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import DatasetDict, load_dataset

try:
    from preprocess import clean_text, extract_evidence_doc_ids, normalize_label, make_passages, create_claim_evidence_pairs
except ImportError:  # pragma: no cover
    from src.preprocess import clean_text, extract_evidence_doc_ids, normalize_label, make_passages, create_claim_evidence_pairs

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _dataset_to_df(dataset: Any, split_name: str) -> pd.DataFrame:
    df = pd.DataFrame(dataset)
    df["split"] = split_name
    return df


def _combine_splits(dataset: Any) -> pd.DataFrame:
    if isinstance(dataset, DatasetDict):
        frames = [_dataset_to_df(split_ds, split_name) for split_name, split_ds in dataset.items()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return _dataset_to_df(dataset, "unknown")


def _try_load_scifact() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Try to load SciFact claims and corpus from Hugging Face Datasets."""
    errors: list[str] = []

    # Common pattern: one dataset script with separate configs.
    try:
        claims_ds = load_dataset("allenai/scifact", "claims")
        corpus_ds = load_dataset("allenai/scifact", "corpus")
        return _combine_splits(claims_ds), _combine_splits(corpus_ds), errors
    except Exception as exc:
        errors.append(f"allenai/scifact with configs claims/corpus failed: {type(exc).__name__}: {exc}")

    # Some mirrors expose everything without configs.
    try:
        ds = load_dataset("allenai/scifact")
        if isinstance(ds, DatasetDict):
            keys = set(ds.keys())
            claim_keys = [key for key in keys if "claim" in key.lower()]
            corpus_keys = [key for key in keys if "corpus" in key.lower() or "doc" in key.lower()]
            if claim_keys and corpus_keys:
                claims_df = pd.concat([_dataset_to_df(ds[key], key) for key in claim_keys], ignore_index=True)
                corpus_df = pd.concat([_dataset_to_df(ds[key], key) for key in corpus_keys], ignore_index=True)
                return claims_df, corpus_df, errors
        errors.append("allenai/scifact loaded, but claim/corpus splits were not identifiable.")
    except Exception as exc:
        errors.append(f"allenai/scifact without configs failed: {type(exc).__name__}: {exc}")

    # Community mirror fallback. This may or may not exist in a future environment.
    for dataset_name in ["scifact", "BeIR/scifact"]:
        try:
            ds = load_dataset(dataset_name)
            df = _combine_splits(ds)
            columns = set(df.columns)
            if "claim" in columns:
                claims_df = df
                corpus_df = pd.DataFrame()
                errors.append(f"Loaded {dataset_name}, but no separate corpus was found.")
                return claims_df, corpus_df, errors
        except Exception as exc:
            errors.append(f"{dataset_name} failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("Could not load SciFact from known Hugging Face sources.\n" + "\n".join(errors))


def normalize_claims(claims_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize claims into columns used by the rest of the project."""
    if claims_df.empty:
        return claims_df

    output = pd.DataFrame()
    output["claim_id"] = claims_df.get("claim_id", claims_df.get("id", pd.Series(range(len(claims_df))))).astype(str)
    output["claim"] = claims_df.get("claim", claims_df.get("text", "")).apply(clean_text)

    if "label" in claims_df.columns:
        label_source = claims_df["label"]
    elif "evidence_label" in claims_df.columns:
        label_source = claims_df["evidence_label"]
    else:
        label_source = pd.Series(["NOT_ENOUGH_INFO"] * len(claims_df))
    output["label"] = label_source.apply(normalize_label)

    output["evidence_doc_ids"] = claims_df.apply(lambda row: json.dumps(extract_evidence_doc_ids(row)), axis=1)
    output["split"] = claims_df.get("split", "unknown")
    return output.drop_duplicates(subset=["claim_id", "claim", "label", "split"]).reset_index(drop=True)


def normalize_corpus(corpus_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize corpus documents into doc_id, title, abstract."""
    if corpus_df.empty:
        return corpus_df

    output = pd.DataFrame()
    output["doc_id"] = corpus_df.get("doc_id", corpus_df.get("id", corpus_df.get("document_id", pd.Series(range(len(corpus_df)))))).astype(str)
    output["title"] = corpus_df.get("title", "")

    if "abstract" in corpus_df.columns:
        output["abstract"] = corpus_df["abstract"].apply(lambda value: " ".join(value) if isinstance(value, list) else clean_text(value))
    elif "text" in corpus_df.columns:
        output["abstract"] = corpus_df["text"].apply(clean_text)
    else:
        text_columns = [col for col in corpus_df.columns if col not in {"doc_id", "id", "document_id", "title", "split"}]
        output["abstract"] = corpus_df[text_columns].astype(str).agg(" ".join, axis=1) if text_columns else ""

    output["split"] = corpus_df.get("split", "corpus")
    return output.drop_duplicates(subset=["doc_id"]).reset_index(drop=True)


def write_failure_report(errors: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    report = f"""# SciFact loading failed

`src/load_data.py` tried to load SciFact from known Hugging Face Datasets sources, but loading failed.

## Exact error log

```text
{errors}
```

## Fallback plan

1. Check your internet connection.
2. Upgrade the Datasets package:

```bash
pip install --upgrade datasets
```

3. Try running again:

```bash
python src/load_data.py
```

4. If the Hugging Face dataset name or configuration changed, update `_try_load_scifact()` in `src/load_data.py` with the new source.
5. Do not report dataset counts, metrics, or conclusions until the data loads successfully and the evaluation scripts are run.
"""
    (PROCESSED_DIR / "LOAD_FAILED.md").write_text(report, encoding="utf-8")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    try:
        raw_claims, raw_corpus, warnings = _try_load_scifact()
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        write_failure_report(message)
        print("SciFact loading failed. See data/processed/LOAD_FAILED.md for details.")
        return

    claims_df = normalize_claims(raw_claims)
    corpus_df = normalize_corpus(raw_corpus)
    passages_df = make_passages(corpus_df) if not corpus_df.empty else pd.DataFrame()
    pairs_df = create_claim_evidence_pairs(claims_df, passages_df) if not passages_df.empty else pd.DataFrame()

    raw_claims.to_csv(RAW_DIR / "scifact_claims_raw.csv", index=False)
    raw_corpus.to_csv(RAW_DIR / "scifact_corpus_raw.csv", index=False)
    claims_df.to_csv(PROCESSED_DIR / "claims.csv", index=False)
    corpus_df.to_csv(PROCESSED_DIR / "corpus.csv", index=False)
    passages_df.to_csv(PROCESSED_DIR / "evidence_passages.csv", index=False)
    pairs_df.to_csv(PROCESSED_DIR / "claim_evidence_pairs.csv", index=False)

    summary = {
        "num_claim_rows": int(len(claims_df)),
        "num_corpus_documents": int(len(corpus_df)),
        "num_evidence_passages": int(len(passages_df)),
        "num_claim_evidence_pairs": int(len(pairs_df)),
        "claim_splits": claims_df["split"].value_counts(dropna=False).to_dict() if "split" in claims_df else {},
        "label_distribution": claims_df["label"].value_counts(dropna=False).to_dict() if "label" in claims_df else {},
        "warnings": warnings,
    }
    (PROCESSED_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("SciFact data loaded and processed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
