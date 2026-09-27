# tests/test_chunker.py
from app.ingestion.chunker import chunk_text


def test_chunk_text_returns_empty_list_for_empty_text():
    result = chunk_text("")
    assert result == []


def test_chunk_text_single_chunk_when_text_is_short():
    text = "una dos tres cuatro cinco"
    result = chunk_text(text, chunk_size=10, overlap=2)
    assert result == ["una dos tres cuatro cinco"]


def test_chunk_text_splits_into_multiple_chunks():
    text = " ".join(f"palabra{i}" for i in range(10))
    result = chunk_text(text, chunk_size=4, overlap=0)
    assert len(result) == 3
    assert result[0] == "palabra0 palabra1 palabra2 palabra3"
    assert result[1] == "palabra4 palabra5 palabra6 palabra7"
    assert result[2] == "palabra8 palabra9"


def test_chunk_text_overlap_shares_words_between_chunks():
    text = " ".join(f"palabra{i}" for i in range(10))
    result = chunk_text(text, chunk_size=4, overlap=1)

    last_word_of_first_chunk = result[0].split()[-1]
    first_word_of_second_chunk = result[1].split()[0]

    assert last_word_of_first_chunk == first_word_of_second_chunk