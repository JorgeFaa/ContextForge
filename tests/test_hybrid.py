#tests/test_hybrid.py
from app.db.models import Chunk
from app.retrieval.hybrid import fuse_results


def make_chunk(chunk_id: int) -> Chunk:
    return Chunk(id = chunk_id, document_id=2, text=f"texto {chunk_id}", embedding=[])


def test_fuse_results_favor_chunk_present_in_both_lists():
    vector_results = [make_chunk(1), make_chunk(2), make_chunk(3)]
    keyword_results = [make_chunk(3), make_chunk(4), make_chunk(5)]

    result = fuse_results(vector_results, keyword_results, top_k=5)

    assert result[0].id == 3


def test_fuse_results_respects_top_k():
    vector_results = [make_chunk(i) for i in range(10)]
    keyword_results = [make_chunk(i) for i in range(10, 20)]

    result = fuse_results(vector_results, keyword_results, top_k=3)

    assert len(result) == 3


def test_fuse_results_deduplicates_shared_chunks():
    vector_results = [make_chunk(1), make_chunk(2)]
    keyword_results = [make_chunk(2), make_chunk(2)]

    result = fuse_results(vector_results, keyword_results, top_k=10)

    result_ids = [chunk.id for chunk in result]
    assert len(result_ids) == len(set(result_ids))


def test_fuse_results_empty_lists_return_empty():
    result = fuse_results([],[], top_k=5)
    assert result == []