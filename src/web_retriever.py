"""Live evidence retrieval for web-grounded claim verification.

This module does not use a toy corpus or a local sample dataset. At runtime it
queries public sources, extracts evidence text, splits that text into passages,
and ranks the passages with TF-IDF, BM25, or a hybrid score.

Sources included without API keys:
- Wikipedia / MediaWiki API for broad public facts.
- OpenAlex API for scholarly and scientific evidence.
- DuckDuckGo Search for broader web
  search snippets and reachable pages.

The system is evidence-grounded, not omniscient: if sources are unavailable or
insufficient, the correct behavior is to return limited evidence rather than
inventing an answer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - lightweight fallback for tests
    class BM25Okapi:  # type: ignore[no-redef]
        """Small BM25-like fallback used when rank-bm25 is unavailable."""

        def __init__(self, tokenized_corpus: list[list[str]]) -> None:
            self.tokenized_corpus = tokenized_corpus

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            query = set(query_tokens)
            scores = []
            for document_tokens in self.tokenized_corpus:
                if not document_tokens:
                    scores.append(0.0)
                    continue
                overlap = sum(1 for token in document_tokens if token in query)
                scores.append(overlap / max(1, len(document_tokens)))
            return scores

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from evidence_extractor import clean_text, passages_from_documents
    from source_ranker import blend_relevance_and_trust, source_quality_report, source_trust_score
except ImportError:  # pragma: no cover
    from src.evidence_extractor import clean_text, passages_from_documents
    from src.source_ranker import blend_relevance_and_trust, source_quality_report, source_trust_score


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
OPENALEX_API_URL = "https://api.openalex.org/works"
USER_AGENT = "claim-verification-ai/1.0 (educational evidence verifier; no data collection)"


@dataclass(frozen=True)
class SearchDocument:
    """A retrieved source document before passage ranking."""

    doc_id: str
    title: str
    url: str
    source: str
    text: str
    trust_score: float


def tokenize(text: str) -> list[str]:
    """Tokenize text for keyword retrieval."""
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def normalize_scores(scores: Iterable[float]) -> np.ndarray:
    """Min-max normalize scores to the 0-1 range."""
    scores = np.asarray(list(scores), dtype=float)
    if scores.size == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if max_score == min_score:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def _stable_id(value: str) -> str:
    """Create a short stable ID from text or URLs."""
    return hashlib.md5(value.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Convert OpenAlex abstract_inverted_index into readable text."""
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, word_positions in inverted_index.items():
        for position in word_positions:
            positions[int(position)] = word
    return " ".join(positions[index] for index in sorted(positions))


def _safe_domain(url: str) -> str:
    parsed = urlparse(url or "")
    return parsed.netloc.lower().replace("www.", "")


class WikipediaEvidenceRetriever:
    """Search Wikipedia and return source documents with page extracts."""

    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def search(self, claim: str, max_pages: int = 6) -> list[SearchDocument]:
        page_ids = self._search_page_ids(claim, max_pages=max_pages)
        if not page_ids:
            return []
        return self._fetch_page_extracts(page_ids)

    def _search_page_ids(self, query: str, max_pages: int) -> list[str]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": max_pages,
            "utf8": 1,
        }
        response = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        return [str(item["pageid"]) for item in search_results if "pageid" in item]

    def _fetch_page_extracts(self, page_ids: list[str]) -> list[SearchDocument]:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|info",
            "explaintext": 1,
            "inprop": "url",
            "pageids": "|".join(page_ids),
            "redirects": 1,
            "utf8": 1,
        }
        response = self.session.get(WIKIPEDIA_API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        documents: list[SearchDocument] = []
        for page_id, page in pages.items():
            if "missing" in page:
                continue
            title = str(page.get("title", "Untitled"))
            url = str(page.get("fullurl", f"https://en.wikipedia.org/?curid={page_id}"))
            text = clean_text(page.get("extract", ""))
            if len(text) < 80:
                continue
            documents.append(
                SearchDocument(
                    doc_id=f"wiki_{page_id}",
                    title=title,
                    url=url,
                    source="Wikipedia",
                    text=text,
                    trust_score=source_trust_score(url, "Wikipedia"),
                )
            )
        return documents


class OpenAlexEvidenceRetriever:
    """Search OpenAlex for scholarly works relevant to scientific claims."""

    def __init__(self, timeout: int = 12) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def search(self, claim: str, max_results: int = 6) -> list[SearchDocument]:
        params = {
            "search": claim,
            "per-page": max_results,
            "select": "id,display_name,publication_year,doi,abstract_inverted_index,authorships,primary_location",
        }
        response = self.session.get(OPENALEX_API_URL, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        works = data.get("results", [])

        documents: list[SearchDocument] = []
        for work in works:
            title = clean_text(work.get("display_name", "Untitled scholarly work"))
            abstract = clean_text(_reconstruct_openalex_abstract(work.get("abstract_inverted_index")))
            year = work.get("publication_year")
            doi = work.get("doi") or ""
            primary_location = work.get("primary_location") or {}
            landing_page = primary_location.get("landing_page_url") or ""
            url = doi or landing_page or work.get("id", "")
            if not title or len(abstract) < 80:
                continue
            text = clean_text(f"{title}. Published {year}. {abstract}") if year else clean_text(f"{title}. {abstract}")
            documents.append(
                SearchDocument(
                    doc_id=f"openalex_{_stable_id(url or title)}",
                    title=title,
                    url=url or "https://openalex.org",
                    source="OpenAlex",
                    text=text,
                    trust_score=source_trust_score(url or "https://openalex.org", "OpenAlex"),
                )
            )
        return documents


class GeneralWebEvidenceRetriever:
    """Search the public web with DuckDuckGo when the dependency is available."""

    def __init__(self, timeout: int = 12, fetch_pages: bool = True) -> None:
        self.timeout = timeout
        self.fetch_pages = fetch_pages
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def search(self, claim: str, max_results: int = 6) -> list[SearchDocument]:
        try:
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:  # pragma: no cover
                from ddgs import DDGS  # type: ignore
        except ImportError:
            return []

        documents: list[SearchDocument] = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(claim, max_results=max_results))
        except TypeError:  # older versions do not support context manager in the same way
            ddgs = DDGS()
            results = list(ddgs.text(claim, max_results=max_results))
        except Exception:
            return []

        for result in results:
            title = clean_text(result.get("title", "Untitled web source"))
            url = str(result.get("href") or result.get("url") or "")
            snippet = clean_text(result.get("body", ""))
            if not url:
                continue

            page_text = ""
            if self.fetch_pages:
                page_text = self._fetch_page_text(url)

            text = clean_text(f"{title}. {snippet}. {page_text}")
            if len(text) < 80:
                text = clean_text(f"{title}. {snippet}")
            if len(text) < 50:
                continue

            domain = _safe_domain(url)
            documents.append(
                SearchDocument(
                    doc_id=f"web_{_stable_id(url)}",
                    title=title or domain or "Web source",
                    url=url,
                    source=domain or "Web",
                    text=text,
                    trust_score=source_trust_score(url, domain),
                )
            )
        return documents

    def _fetch_page_text(self, url: str) -> str:
        """Fetch readable paragraph text from a page. Fail closed."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            content_type = response.headers.get("content-type", "").lower()
            if response.status_code >= 400 or "text/html" not in content_type:
                return ""
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            paragraphs = [clean_text(p.get_text(" ")) for p in soup.find_all("p")]
            paragraphs = [p for p in paragraphs if len(p) >= 40]
            return clean_text(" ".join(paragraphs[:10]))[:5000]
        except Exception:
            return ""


class MultiSourceEvidenceRetriever:
    """Retrieve and rank evidence across multiple public sources."""

    def __init__(self, timeout: int = 12, use_web: bool = True, use_wikipedia: bool = True, use_openalex: bool = True) -> None:
        self.timeout = timeout
        self.use_web = use_web
        self.use_wikipedia = use_wikipedia
        self.use_openalex = use_openalex
        self.web = GeneralWebEvidenceRetriever(timeout=timeout, fetch_pages=True)
        self.wikipedia = WikipediaEvidenceRetriever(timeout=timeout)
        self.openalex = OpenAlexEvidenceRetriever(timeout=timeout)

    def search_documents(self, claim: str, max_sources: int = 6) -> list[SearchDocument]:
        documents: list[SearchDocument] = []

        if self.use_web:
            documents.extend(self.web.search(claim, max_results=max_sources))
        if self.use_wikipedia:
            documents.extend(self.wikipedia.search(claim, max_pages=max_sources))
        if self.use_openalex:
            documents.extend(self.openalex.search(claim, max_results=max(3, max_sources // 2)))

        return self._dedupe_documents(documents)

    @staticmethod
    def _dedupe_documents(documents: list[SearchDocument]) -> list[SearchDocument]:
        seen: set[str] = set()
        unique: list[SearchDocument] = []
        for doc in documents:
            key = doc.url.lower().rstrip("/") or doc.title.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
        return unique

    def retrieve(
        self,
        claim: str,
        method: str = "Hybrid",
        top_k: int = 5,
        alpha: float = 0.5,
        max_pages: int = 6,
        trust_weight: float = 0.15,
    ) -> list[dict[str, Any]]:
        """Retrieve and rank top evidence passages."""
        documents = [document.__dict__ for document in self.search_documents(claim, max_sources=max_pages)]
        passages = passages_from_documents(documents)
        if not passages:
            return []

        passage_df = pd.DataFrame(passages).fillna("")
        texts = passage_df["text"].astype(str).tolist()

        tfidf = TfidfVectorizer(stop_words="english", max_features=30000, ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(texts)
        query_vector = tfidf.transform([claim])
        tfidf_scores = cosine_similarity(query_vector, tfidf_matrix).ravel()

        bm25 = BM25Okapi([tokenize(text) for text in texts])
        bm25_scores = np.asarray(bm25.get_scores(tokenize(claim)), dtype=float)

        tfidf_norm = normalize_scores(tfidf_scores)
        bm25_norm = normalize_scores(bm25_scores)

        method_normalized = method.strip().lower()
        if method_normalized in {"tf-idf", "tfidf"}:
            relevance_scores = tfidf_norm
        elif method_normalized == "bm25":
            relevance_scores = bm25_norm
        else:
            alpha = max(0.0, min(1.0, float(alpha)))
            relevance_scores = alpha * bm25_norm + (1.0 - alpha) * tfidf_norm

        trust_scores = passage_df["trust_score"].astype(float).to_numpy()
        final_scores = np.array(
            [
                blend_relevance_and_trust(rel, trust, trust_weight=trust_weight)
                for rel, trust in zip(relevance_scores, trust_scores)
            ]
        )

        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results: list[dict[str, Any]] = []
        for idx in top_indices:
            row = passage_df.iloc[int(idx)]
            quality = source_quality_report(
                url=str(row["url"]),
                source_name=str(row["source"]),
                relevance=float(relevance_scores[idx]),
                final_score=float(final_scores[idx]),
            )
            results.append(
                {
                    "passage_id": str(row["passage_id"]),
                    "doc_id": str(row["doc_id"]),
                    "title": str(row["title"]),
                    "source": str(row["source"]),
                    "url": str(row["url"]),
                    "text": str(row["text"]),
                    "score": float(final_scores[idx]),
                    "relevance_score": float(relevance_scores[idx]),
                    "tfidf_score": float(tfidf_norm[idx]),
                    "bm25_score": float(bm25_norm[idx]),
                    "trust_score": float(quality["trust_score"]),
                    "source_category": str(quality["source_category"]),
                    "trust_label": str(quality["trust_label"]),
                    "evidence_strength_label": str(quality["evidence_strength_label"]),
                }
            )
        return results


# Backward-compatible alias used by earlier app versions and tests.
WikipediaEvidenceRetrieverAlias = WikipediaEvidenceRetriever
