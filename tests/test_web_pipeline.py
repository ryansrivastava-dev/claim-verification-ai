from src.claim_type_classifier import detect_claim_type
from src.evidence_extractor import split_into_passages
from src.web_pipeline import predict_from_evidence, run_web_fact_check
from src.web_retriever import normalize_scores


class FakeRetriever:
    def retrieve(
        self,
        claim,
        method="Hybrid",
        top_k=5,
        alpha=0.5,
        max_pages=6,
    ):
        return [
            {
                "passage_id": "fake_1",
                "doc_id": "fake",
                "title": "Paris",
                "source": "TestSource",
                "url": "https://example.com/paris",
                "text": "Paris is the capital and largest city of France.",
                "score": 0.85,
                "relevance_score": 0.75,
                "tfidf_score": 0.70,
                "bm25_score": 0.80,
                "trust_score": 0.60,
            }
        ]


def test_split_into_passages_returns_text():
    text = "Paris is the capital of France. It is a major European city. It has many museums."
    passages = split_into_passages(text, max_chars=120, min_chars=20)
    assert passages
    assert "Paris" in passages[0]


def test_claim_type_detects_current_claim():
    profile = detect_claim_type("Who is the current CEO of Apple?")
    assert profile.needs_current_source is True


def test_web_pipeline_returns_required_fields_with_fake_retriever():
    result = run_web_fact_check(
        "The capital of France is Paris.",
        retriever=FakeRetriever(),
    )
    assert result["predicted_label"] in {"Likely Supported", "Likely Refuted", "Not Enough Evidence"}
    assert result["top_evidence"]
    assert result["top_evidence"][0]["url"] == "https://example.com/paris"
    assert "claim_profile" in result
    assert result["source_mode"] == "Live public evidence retrieval"


def test_normalize_scores_handles_constant_values():
    scores = normalize_scores([5.0, 5.0, 5.0])
    assert list(scores) == [0.0, 0.0, 0.0]


def test_current_role_claim_does_not_treat_old_biography_as_current_support():
    profile = detect_claim_type("Joe Biden is the current president")
    evidence = [
        {
            "title": "Joe Biden",
            "source": "Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Joe_Biden",
            "text": "Joe Biden is an American politician who served as the 46th president of the United States from 2021 to 2025.",
            "score": 0.80,
            "relevance_score": 0.70,
            "trust_score": 0.82,
        }
    ]
    label, confidence, explanation = predict_from_evidence("Joe Biden is the current president", evidence, profile)
    assert label == "Likely Refuted"
    assert confidence > 0.5
    assert "past" in explanation.lower() or "former" in explanation.lower()
