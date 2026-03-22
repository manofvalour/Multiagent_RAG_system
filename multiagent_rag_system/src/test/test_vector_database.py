"""
Unit tests for VectorStore using a real in-process Qdrant instance.
No network required — uses local mode (QdrantClient(path=...)).
"""
from __future__ import annotations

import uuid
import numpy as np
import pytest

def _norm(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid divide-by-zero
    return (arr / norms).astype("float32")


def _rand_embeddings(n: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return _norm(rng.standard_normal((n, dim)).astype("float32"))


def _make_chunk(
    idx: int,
    doc_id: str = "doc-001",
    source: str = "test.txt",
    dim: int = 8,
) -> DocumentChunk:
    """
    Build a deterministic DocumentChunk with a valid UUID id.
    idx must be in [0, 99] to keep the UUID pattern valid.
    """
    from ..models.models import ContentType, DocumentChunk

    return DocumentChunk(
        id=f"aaaaaaaa-0000-0000-0000-{str(idx).zfill(12)}",
        content=f"chunk content number {idx}",
        source=source,
        chunk_index=idx,
        doc_id=doc_id,          # top-level field — hoisted into Qdrant payload
        metadata={"test": True},
    )



# Fixtures
DIM = 8   # small dimension for fast in-process tests

@pytest.fixture()
def isolated_settings(monkeypatch, tmp_path):
    """
    Patch the shared Settings singleton so every test gets:
      - its own on-disk Qdrant directory  (prevents cross-test data bleed)
      - its own collection name           (prevents schema conflicts)
      - local mode (qdrant_url = "")      (no Docker required)

    We patch the attributes on the already-cached Settings instance because
    `frozen=True` is set — we temporarily lift that by patching __setattr__.
    """
    from ..utils.config_loader import get_settings
    s = get_settings().vector_store
    emb = get_settings().embeddings

    collection = f"test_{uuid.uuid4().hex[:8]}"
    qdrant_path = str(tmp_path / "qdrant")

    monkeypatch.setattr(s, "url", "", raising=False)
    monkeypatch.setattr(s, "local_path", qdrant_path, raising=False)
    monkeypatch.setattr(s, "collection_name", collection, raising=False)
    monkeypatch.setattr(s, "hnsw_m", 8, raising=False)
    monkeypatch.setattr(s, "hnsw_ef_construct", 40, raising=False)
    monkeypatch.setattr(s, "hnsw_ef", 64, raising=False)
    monkeypatch.setattr(emb, "embedding_dim", DIM, raising=False)

    return s


@pytest.fixture()
async def store(isolated_settings):
    """
    A fully connected VectorStore backed by an isolated in-process Qdrant.
    Yielded so callers can use it directly without re-connecting.
    """
    from ..database.vector_store import VectorStore
    vs = VectorStore(dim=DIM)
    await vs.connect()
    yield vs


# Connection & lifecycle
@pytest.mark.asyncio
async def test_connect_creates_empty_collection(isolated_settings):
    """connect() must create the collection and report zero points."""
    from ..database.vector_store import VectorStore

    vs = VectorStore(dim=DIM)
    await vs.connect()

    assert await vs.count() == 0


@pytest.mark.asyncio
async def test_connect_is_idempotent(isolated_settings):
    """Calling connect() twice on the same collection must not raise."""
    from ..database.vector_store import VectorStore

    vs = VectorStore(dim=DIM)
    await vs.connect()
    await vs.connect()   # second call — _ensure_collection must be a no-op

    assert await vs.count() == 0


@pytest.mark.asyncio
async def test_collection_info_returns_expected_keys(store):
    """collection_info() must include the keys consumed by the health endpoint."""
    info = await store.collection_info()

    required_keys = {"name", "points_count", "status", "hnsw_m", "hnsw_ef_construct"}  #vectors_count
    assert required_keys.issubset(info.keys()), f"Missing keys: {required_keys - info.keys()}"
    assert info["status"] != "disconnected"


# Upsert / count
@pytest.mark.asyncio
async def test_add_chunks_increases_count(store):
    """add_chunks() must persist all supplied points."""
    n = 3
    chunks = [_make_chunk(i) for i in range(n)]
    embeddings = _rand_embeddings(n, DIM)

    await store.add_chunks(chunks, embeddings)

    assert await store.count() == n


@pytest.mark.asyncio
async def test_add_empty_list_is_noop(store):
    """add_chunks([]) must not raise and must leave the collection empty."""
    await store.add_chunks([], np.empty((0, DIM), dtype="float32"))
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_upsert_is_idempotent(store):
    """Upserting the same chunk twice must not duplicate it."""
    chunks = [_make_chunk(0)]
    embeddings = _rand_embeddings(1, DIM)

    await store.add_chunks(chunks, embeddings)
    await store.add_chunks(chunks, embeddings)   # identical upsert

    assert await store.count() == 1


@pytest.mark.asyncio
async def test_upsert_updates_existing_vector(store):
    """Upserting a chunk with the same id but a new vector must overwrite it."""
    chunk = _make_chunk(0)
    emb1 = _rand_embeddings(1, DIM)
    emb2 = _rand_embeddings(1, DIM)
    emb2[0] = -emb1[0]   # make it clearly different

    await store.add_chunks([chunk], emb1)
    await store.add_chunks([chunk], emb2)   # overwrite

    assert await store.count() == 1  # still one point, not two


# Search
@pytest.mark.asyncio
async def test_search_returns_nearest_chunk(store):
    """Searching with an exact stored vector must return that chunk as top hit."""
    chunks = [_make_chunk(i) for i in range(3)]
    embeddings = _rand_embeddings(3, DIM)
    await store.add_chunks(chunks, embeddings)

    results = await store.search(embeddings[1], top_k=3, threshold=0.0)

    assert len(results) >= 1
    assert results[0].chunk.id == chunks[1].id


@pytest.mark.asyncio
async def test_search_respects_top_k(store):
    """top_k must cap the number of returned results."""
    n = 5
    chunks = [_make_chunk(i) for i in range(n)]
    embeddings = _rand_embeddings(n, DIM)
    await store.add_chunks(chunks, embeddings)

    results = await store.search(embeddings[0], top_k=2, threshold=0.0)

    assert len(results) <= 2


@pytest.mark.asyncio
async def test_search_results_sorted_by_score_descending(store):
    """Results must come back in descending similarity order."""
    n = 4
    chunks = [_make_chunk(i) for i in range(n)]
    embeddings = _rand_embeddings(n, DIM)
    await store.add_chunks(chunks, embeddings)

    results = await store.search(embeddings[0], top_k=n, threshold=0.0)

    scores = [r.vector_score for r in results]
    assert scores == sorted(scores, reverse=True), "Results are not sorted descending"


@pytest.mark.asyncio
async def test_search_threshold_filters_low_scores(store):
    """A high threshold must exclude low-similarity results."""
    chunks = [_make_chunk(i) for i in range(3)]
    embeddings = _rand_embeddings(3, DIM)
    await store.add_chunks(chunks, embeddings)

    # Use a threshold of 1.0 — nothing can be this similar except itself
    results = await store.search(embeddings[0], top_k=3, threshold=0.99)

    for r in results:
        assert r.vector_score >= 0.99


@pytest.mark.asyncio
async def test_search_empty_collection_returns_empty_list(store):
    """search() on an empty collection must return [] without raising."""
    query = _rand_embeddings(1, DIM)[0]
    results = await store.search(query, top_k=5, threshold=0.0)

    assert results == []


@pytest.mark.asyncio
async def test_search_with_payload_filter(store):
    """Payload filter must restrict results to matching source field."""
    from ..models.models import ContentType, DocumentChunk

    chunks_a = [_make_chunk(i, source="alpha.txt") for i in range(3)]
    chunks_b = [_make_chunk(i + 10, source="beta.txt") for i in range(3)]
    all_chunks = chunks_a + chunks_b
    embeddings = _rand_embeddings(6, DIM)
    await store.add_chunks(all_chunks, embeddings)

    filt = {"must": [{"key": "source", "match": {"value": "alpha.txt"}}]}
    results = await store.search(embeddings[0], top_k=10, threshold=0.0, filters=filt)

    assert len(results) > 0
    for r in results:
        assert r.chunk.source == "alpha.txt", (
            f"Expected source='alpha.txt', got {r.chunk.source!r}"
        )


@pytest.mark.asyncio
async def test_retrieved_chunk_has_correct_fields(store):
    """RetrievedChunk must expose chunk and similarity_score."""
    chunk = _make_chunk(0, doc_id="doc-xyz")
    embedding = _rand_embeddings(1, DIM)
    await store.add_chunks([chunk], embedding)

    results = await store.search(embedding[0], top_k=1, threshold=0.0)

    assert len(results) == 1
    r = results[0]
    assert r.chunk.id == chunk.id
    assert r.chunk.content == chunk.content
    assert r.chunk.source == chunk.source
    assert 0.0 <= r.vector_score <= 1.0 + 1e-6   # cosine on L2-normalised vecs


# Delete
@pytest.mark.asyncio
async def test_delete_document_removes_correct_points(store):
    """delete_document() must remove only the points belonging to that doc_id."""
    # 2 chunks for doc-A, 2 for doc-B
    chunks = [_make_chunk(i, doc_id="doc-A") for i in range(2)] + \
             [_make_chunk(i + 10, doc_id="doc-B") for i in range(2)]
    embeddings = _rand_embeddings(4, DIM)
    await store.add_chunks(chunks, embeddings)

    removed = await store.delete_document("doc-A")

    assert removed == 2
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_delete_document_returns_correct_count(store):
    """Return value must equal the number of points actually deleted."""
    n = 5
    chunks = [_make_chunk(i, doc_id="doc-only") for i in range(n)]
    embeddings = _rand_embeddings(n, DIM)
    await store.add_chunks(chunks, embeddings)

    removed = await store.delete_document("doc-only")

    assert removed == n
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_delete_nonexistent_doc_returns_zero(store):
    """delete_document() on an unknown doc_id must return 0, not raise."""
    removed = await store.delete_document("ghost-doc-does-not-exist")
    assert removed == 0


@pytest.mark.asyncio
async def test_delete_is_idempotent(store):
    """Deleting the same doc twice must not raise on the second call."""
    chunk = _make_chunk(0, doc_id="doc-once")
    embedding = _rand_embeddings(1, DIM)
    await store.add_chunks([chunk], embedding)

    first = await store.delete_document("doc-once")
    second = await store.delete_document("doc-once")   # already gone

    assert first == 1
    assert second == 0


@pytest.mark.asyncio
async def test_delete_does_not_affect_other_documents(store):
    """Deleting one doc must leave sibling docs untouched and searchable."""
    chunks_a = [_make_chunk(i, doc_id="doc-A") for i in range(2)]
    chunks_b = [_make_chunk(i + 10, doc_id="doc-B") for i in range(3)]
    embeddings = _rand_embeddings(5, DIM)
    await store.add_chunks(chunks_a + chunks_b, embeddings)

    await store.delete_document("doc-A")

    # doc-B chunks must still be searchable
    results = await store.search(embeddings[2], top_k=10, threshold=0.0)
    surviving_ids = {r.chunk.id for r in results}
    for chunk in chunks_b:
        assert chunk.id in surviving_ids, f"doc-B chunk {chunk.id} was incorrectly deleted"


# get_vector_store singleton
@pytest.mark.asyncio
async def test_get_vector_store_returns_singleton(isolated_settings):
    """get_vector_store() must return the same instance on repeated calls."""
    from ..database.vector_store import VectorStore as vs_module

    # Reset module-level singleton so this test is self-contained
    vs_module._store = None

    from ..database.vector_store import get_vector_store
    store_a = await get_vector_store()
    store_b = await get_vector_store()

    assert store_a is store_b

    # Clean up so other tests start fresh
    vs_module._store = None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])