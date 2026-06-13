import pandas as pd

from src.preprocess import clean_text, make_passages, normalize_label, split_into_sentences


def test_clean_text_normalizes_whitespace():
    assert clean_text("  High\n blood\tpressure  ") == "High blood pressure"


def test_clean_text_lowercase_optional():
    assert clean_text("ABC Def", lowercase=True) == "abc def"


def test_split_into_sentences():
    sentences = split_into_sentences("First sentence. Second sentence? Third!")
    assert sentences == ["First sentence.", "Second sentence?", "Third!"]


def test_normalize_label():
    assert normalize_label("supports") == "SUPPORTED"
    assert normalize_label("REFUTES") == "REFUTED"
    assert normalize_label(None) == "NOT_ENOUGH_INFO"


def test_make_passages_from_corpus():
    corpus = pd.DataFrame(
        [{"doc_id": "D1", "title": "Blood Pressure", "abstract": "High blood pressure is linked to stroke. It is treatable."}]
    )
    passages = make_passages(corpus)
    assert len(passages) == 2
    assert set(["passage_id", "doc_id", "sent_id", "title", "text"]).issubset(passages.columns)
