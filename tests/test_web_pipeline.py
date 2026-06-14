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


def test_web_pipeline_includes_modular_workflow_outputs():
    result = run_web_fact_check(
        "The capital of France is Paris.",
        retriever=FakeRetriever(),
    )
    assert "search_plan" in result
    assert "workflow_steps" in result
    assert "synthesis" in result
    assert "fact_check_report" in result
    assert "Evidence-Based Fact Verification Report" in result["fact_check_report"]
    assert result["top_evidence"][0]["summary"]


def test_search_plan_rewrites_current_role_claim():
    from src.query_planner import plan_search_queries

    profile = detect_claim_type("Joe Biden is the current president")
    plan = plan_search_queries("Joe Biden is the current president", profile=profile)
    assert "current president" in plan.primary_query.lower()
    assert "official" in plan.primary_query.lower()


def test_source_quality_labels_official_source():
    from src.source_ranker import source_category, source_trust_score, trust_label

    category = source_category("https://www.whitehouse.gov/administration/", "White House")
    score = source_trust_score("https://www.whitehouse.gov/administration/", "White House")
    assert category == "Official / government / institutional"
    assert score >= 0.90
    assert "trust" in trust_label(score).lower()


def test_report_builder_can_make_citation_list_and_pdf_bytes():
    from src.report_builder import build_citation_list, build_pdf_report_bytes

    result = {
        "claim": "The capital of France is Paris.",
        "predicted_label": "Likely Supported",
        "confidence": 0.87,
        "explanation": "Evidence matches the claim.",
        "claim_profile": {"category": "general knowledge", "needs_current_source": False},
        "search_plan": {"primary_query": "capital of France", "generated_queries": []},
        "synthesis": {"evidence_strength": "Strong", "analysis": "Evidence supports the claim."},
        "top_evidence": [
            {
                "title": "Paris",
                "url": "https://en.wikipedia.org/wiki/Paris",
                "source": "Wikipedia",
                "summary": "Paris is the capital of France.",
                "score": 0.85,
                "trust_score": 0.82,
                "source_category": "Reference",
                "trust_label": "Strong trust signal",
                "evidence_strength_label": "Strong evidence candidate",
            }
        ],
    }
    citations = build_citation_list(result)
    assert "https://en.wikipedia.org/wiki/Paris" in citations
    pdf_bytes = build_pdf_report_bytes(result)
    assert pdf_bytes.startswith(b"%PDF")
