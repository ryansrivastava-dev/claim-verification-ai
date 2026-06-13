from src.pipeline import run_pipeline


class DummyRetriever:
    def retrieve(self, claim, top_k=5):
        return [
            {
                "passage_id": "p1",
                "doc_id": "d1",
                "score": 1.0,
                "text": "High blood pressure increases risk of stroke.",
            }
        ][:top_k]


def test_pipeline_returns_required_fields():
    result = run_pipeline(
        claim="High blood pressure increases risk of stroke.",
        retriever=DummyRetriever(),
        classifier=None,
        retriever_name="dummy",
        top_k=1,
    )
    required = {"claim", "predicted_label", "confidence", "top_evidence", "retrieval_method", "classifier"}
    assert required.issubset(result.keys())
    assert result["predicted_label"] == "NOT_ENOUGH_INFO"
    assert len(result["top_evidence"]) == 1
