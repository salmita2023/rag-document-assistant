import pytest

from app.ingestion import split_text
from evaluation.evaluate import first_correct_rank


def test_split_text_respects_size_and_overlap():
    text = "".join(str(i % 10) for i in range(2500))
    chunks = split_text(text, chunk_size=1000, overlap=200)
    assert len(chunks) == 3
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0][-200:] == chunks[1][:200]  # overlap is preserved


def test_split_text_short_text_gives_one_chunk():
    assert split_text("hello world", chunk_size=1000, overlap=200) == ["hello world"]


def test_split_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, overlap=100)


def test_first_correct_rank():
    results = [{"source": "a.pdf", "page": 1}, {"source": "a.pdf", "page": 4}]
    assert first_correct_rank(results, "a.pdf", [4]) == 2
    assert first_correct_rank(results, "b.pdf", [4]) is None
