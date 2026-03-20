"""
Pytest configuration and fixtures for multiagent_rag_system tests.
"""
from __future__ import annotations

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch


@pytest.fixture
def test_settings():
    """Provide test settings with chunking configuration."""
    settings = Mock()
    settings.chunking = Mock()
    settings.chunking.chunk_size = 512
    settings.chunking.chunk_overlap = 50
    settings.chunking.min_chunk_size = 100
    return settings


@pytest.fixture
def mock_embed_model():
    """Provide a mock embedding model."""
    model = Mock()
    model.encode = Mock()
    return model


@pytest.fixture
def mock_vector_store():
    """Provide a mock vector store."""
    store = AsyncMock()
    store.add_chunks = AsyncMock()
    store.search = AsyncMock()
    return store


@pytest.fixture
def mock_dependencies(test_settings, mock_embed_model, mock_vector_store, monkeypatch):
    """Patch global dependency injection functions to use mocked versions."""
    # Mock get_settings to return test_settings
    async def mock_get_settings():
        return test_settings
    
    def sync_get_settings():
        return test_settings
    
    # Mock get_embedder to return a mock with the embed method
    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1] * 384] * 3)
    
    async def mock_get_embedder():
        return mock_embedder
    
    # Mock get_vector_store to return mock_vector_store
    async def mock_get_vector_store():
        return mock_vector_store
    
    # Patch the globals
    monkeypatch.setattr(
        "multiagent_rag_system.agent.doc_ingestion.get_settings",
        sync_get_settings
    )
    monkeypatch.setattr(
        "multiagent_rag_system.agent.doc_ingestion.get_embedder",
        mock_get_embedder
    )
    monkeypatch.setattr(
        "multiagent_rag_system.agent.doc_ingestion.get_vector_store",
        mock_get_vector_store
    )
    
    return {
        "settings": test_settings,
        "embedder": mock_embedder,
        "vector_store": mock_vector_store,
    }