"""Utilities for turning source documents into retrievable evidence passages."""

from __future__ import annotations

import re
from typing import Iterable


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Normalize text while keeping meaning unchanged."""
    text = "" if text is None else str(text)
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_sentences(text: object) -> list[str]:
    """Split text into simple sentence-like chunks.

    This intentionally uses a lightweight regex instead of a large NLP model so
    the project stays laptop-friendly and deploys cleanly on Streamlit.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_RE.split(cleaned) if sentence.strip()]


def split_into_passages(text: object, max_chars: int = 750, min_chars: int = 80) -> list[str]:
    """Group sentences into passages suitable for retrieval."""
    sentences = split_sentences(text)
    passages: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > max_chars:
            candidate = clean_text(" ".join(current))
            if len(candidate) >= min_chars:
                passages.append(candidate)
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1

    if current:
        candidate = clean_text(" ".join(current))
        if len(candidate) >= min_chars:
            passages.append(candidate)

    if not passages and clean_text(text):
        cleaned = clean_text(text)
        passages = [cleaned[:max_chars]]

    return passages


def passages_from_documents(documents: Iterable[dict], max_chars: int = 750) -> list[dict]:
    """Create passage dictionaries from source document dictionaries."""
    rows: list[dict] = []
    for document in documents:
        doc_id = str(document.get("doc_id", document.get("page_id", "")))
        title = str(document.get("title", "Untitled source"))
        source_url = str(document.get("url", ""))
        source_name = str(document.get("source", "Unknown source"))
        trust_score = float(document.get("trust_score", 0.5))
        text = document.get("text", "")

        for index, passage in enumerate(split_into_passages(text, max_chars=max_chars), start=1):
            rows.append(
                {
                    "passage_id": f"{doc_id}_p{index}",
                    "doc_id": doc_id,
                    "title": title,
                    "source": source_name,
                    "url": source_url,
                    "trust_score": trust_score,
                    "text": passage,
                }
            )
    return rows
