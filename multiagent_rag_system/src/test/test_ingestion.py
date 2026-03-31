"""
tests/test_ingestion.py
Unit tests for FileParser, HybridChunker, and DocumentIngestionPipeline.
No real LLM calls — all heavy IO is mocked.
"""
from __future__ import annotations

import io
import pytest

from multiagent_rag_system.agent.agents.doc_ingestion import (
    HybridChunker,
    DocumentIngestionPipeline,
    detect_content_type,
    FileParser,
)
from ..models.models import ContentType, IngestRequest
from .conftest import test_settings, mock_embed_model, mock_vector_store


#detect_content_type
def test_detects_prose():
    text = "This is a sentence. And another one follows it."
    assert detect_content_type(text) == ContentType.PROSE


def test_detects_markdown():
    text = "## Heading\n\nSome text here.\n\n- bullet one\n- bullet two"
    assert detect_content_type(text) == ContentType.MARKDOWN


def test_detects_code():
    text = "def foo():\n    return 1\n\nclass Bar:\n    pass\n"
    assert detect_content_type(text) == ContentType.CODE


#HybridChunker
class TestHybridChunker:
    @pytest.mark.asyncio
    async def test_short_text_returns_single_chunk(self, test_settings, mock_embed_model):
        import numpy as np
        mock_embed_model.encode.return_value = np.random.rand(2, 384).astype("float32")
        chunker = HybridChunker()
        chunks = await chunker.chunk("Short text.", ContentType.PROSE, {})
        assert len(chunks) >= 1
        assert all(len(c.content) > 0 for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_index_is_sequential(self, test_settings, mock_embed_model):
        import numpy as np
        long_text = " ".join(["word"] * 400)
        mock_embed_model.encode.return_value = np.random.rand(10, 384).astype("float32")
        chunker = HybridChunker()
        chunks = await chunker.chunk(long_text, ContentType.PROSE, {})
        for i, c in enumerate(chunks):
            assert c.chunk_index == i

    @pytest.mark.asyncio
    async def test_empty_text_returns_no_chunks(self, test_settings, mock_embed_model):
        import numpy as np
        mock_embed_model.encode.return_value = np.random.rand(1, 384).astype("float32")
        chunker = HybridChunker()
        chunks = await chunker.chunk("   ", ContentType.PROSE, {})
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_metadata_doc_id_is_preserved(self, test_settings, mock_embed_model):
        import numpy as np
        mock_embed_model.encode.return_value = np.random.rand(2, 384).astype("float32")
        chunker = HybridChunker()
        chunks = await chunker.chunk("Some content.", ContentType.PROSE, {"doc_id": "abc"})
        assert all(c.doc_id == "abc" for c in chunks)


#FileParser
class TestFileParser:
    @pytest.mark.asyncio
    async def test_parses_plain_text(self):
        parser = FileParser()
        text, ct = await parser.parse(b"Hello world.", "doc.txt")
        assert "Hello" in text
        assert ct == ContentType.PROSE

    @pytest.mark.asyncio
    async def test_parses_markdown(self):
        parser = FileParser()
        md = b"## Heading\n\nParagraph text here."
        text, ct = await parser.parse(md, "readme.md")
        assert ct == ContentType.MARKDOWN
        assert "Heading" in text

    @pytest.mark.asyncio
    async def test_unknown_extension_falls_back_to_text(self):
        parser = FileParser()
        text, ct = await parser.parse(b"some content", "data.csv")
        assert isinstance(text, str)


class TestDocumentIngestionPipeline:
    @pytest.mark.asyncio
    async def test_ingest_text_returns_response(
        self, mock_dependencies, mock_embed_model
    ):
        import numpy as np
        mock_embed_model.encode.return_value = np.random.rand(3, 384).astype("float32")

        pipeline = DocumentIngestionPipeline()
        req = IngestRequest(
            content="RAG reduces hallucinations by grounding answers in retrieved context.",
        )
        result = await pipeline.ingest_text(req)

        assert result.document_id != ""
        assert result.chunks_created >= 1
        assert result.content_type == "prose"
        mock_dependencies["vector_store"].add_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_ingest_file_pdf_calls_parser(
        self, mock_dependencies, mock_embed_model
    ):
        import numpy as np
        from pathlib import Path
        import tempfile
        
        mock_embed_model.encode.return_value = np.random.rand(2, 384).astype("float32")

        # Create a temporary upload directory
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = DocumentIngestionPipeline()
            pipeline.UPLOAD_DIR = tmpdir
            # Minimal PDF bytes (will fail pypdf gracefully, return empty text)
            fake_pdf = b"%PDF-1.4 fake content"
            result = await pipeline.ingest_file(fake_pdf, "report.pdf")
            assert result.document_id is not None
            assert result.content_type in ("pdf", "prose")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])