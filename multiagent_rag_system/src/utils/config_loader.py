"""
src/core/config.py
Three-layer configuration:
  .env (secrets)  ->  config.yaml (tunable defaults)  ->  Settings (typed, validated)

Priority order (highest -> lowest):
  1. Real environment variables (shell / Docker)
  2. .env file (local dev secrets)
  3. config.yaml values (structured defaults)
"""
from __future__ import annotations

import yaml
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_BASE_DIR  = Path(__file__).resolve().parent.parent.parent
_YAML_PATH = _BASE_DIR / "config" / "configuration.yaml"


#YAML loader
def _load_yaml(path: Path = _YAML_PATH) -> dict[str, Any]:
    """
    Load and flatten configuration.yaml into a flat dict that pydantic-settings
    can consume directly as keyword arguments.

    Flattening rules:
      llm.active_provider  -> flat["active_provider"]
      llm.providers        -> flat["llm_providers"]
      <other section>.*    -> flat.*   (values merged at top level)
      scalar section       -> flat[section] = scalar
    """
    if not path.exists():
        return {}

    with path.open(encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f) or {}

    flat: dict[str, Any] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            if section == "llm":
                flat["active_provider"] = values.get("active_provider", "groq")
                flat["llm_providers"]   = values.get("providers", {})
            else:
                flat.update(values)
        else:
            flat[section] = values

    return flat


#Enums

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING     = "staging"
    PRODUCTION  = "production"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    GROQ      = "groq"
    OPENAI    = "openai"


class ExpansionStrategy(str, Enum):
    HYDE        = "hyde"
    MULTI_QUERY = "multi_query"
    BOTH        = "both"


# ── Sub-configs (one class per logical concern)

class LLMProviderConfig(BaseModel):
    """Config for a single LLM provider entry inside llm_providers."""
    base_url:          str   = "https://api.groq.com/openai/v1/chat/completions"
    model_name:        str   = "llama-3.3-70b-versatile"
    temperature:       float = Field(default=0.0,  ge=0.0, le=2.0)
    max_output_tokens: int   = Field(default=2048, ge=1)
    timeout_seconds:   int   = Field(default=30,   ge=1)
    stream:            bool  = True
    system_prompt:     str   = (
        "You are a precise, helpful assistant. "
        "Answer using ONLY the provided context. "
        "If the context does not contain the answer, say so clearly."
    )

    @field_validator("temperature")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        return v


class RerankerConfig(BaseModel):
    """Cross-encoder reranking stage."""
    enabled: bool = True
    top_n:int = 5
    model:str = "BAAI/bge-reranker-v2-m3"


class QueryExpansionConfig(BaseModel):
    """HyDE / multi-query rewriting before retrieval."""
    enabled: bool = True
    strategy: ExpansionStrategy = ExpansionStrategy.BOTH
    num_queries: int = 3
    hyde_temperature: float = 0.7


class EmbeddingsConfig(BaseModel):
    """Sentence-transformer embedding model settings."""
    model:str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 64


class ChunkingConfig(BaseModel):
    """Hybrid chunking strategy settings."""
    strategy: str  = "hybrid"   # hybrid | recursive | sentence
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_tokens:bool = False      # True = measure size in tokens (tiktoken)


class CacheConfig(BaseModel):
    """Redis semantic cache settings."""
    enabled: bool  = True
    ttl_seconds: int   = 3600
    session_ttl_seconds: int = 86400
    similarity_threshold: float = 0.92
    requests_per_minute: int = 60
    window_seconds:int = 60
    burst: int = 10


class VectorStoreConfig(BaseModel):
    """Qdrant vector store connection and HNSW index settings."""
    # Connection — server mode when url is set, local mode when url is ""
    url: str = "http://localhost:6333"
    api_key: str = ""           # override via QDRANT_API_KEY in .env
    collection_name: str = "rag_chunks"
    hnsw_m: int = 16   
    hnsw_ef_construct: int = 100  
    hnsw_ef: int = 128   
    local_path: str = "./data/qdrant"


class RetrieverConfig(BaseModel):
    """ANN retrieval settings."""
    top_k: int = 10
    similarity_threshold: float = 0.12   # minimum score to include a result
    hnsw_ef_search: int = 128
    ivf_nprobe:int = 32


class AuthConfig(BaseModel):
    """API key authentication."""
    enabled: bool  = True
    api_key_header: str = "X-API-Key"
    api_keys:list[str] = []


class ObservabilityConfig(BaseModel):
    """OpenTelemetry + LangSmith tracing."""
    enable_metrics: bool = True
    enable_tracing: bool = True
    otel_enabled:bool = False
    otlp_endpoint: str  = "http://localhost:4317"
    langsmith_enabled: bool = False
    sentry_dsn:str  = ""


class EvaluationConfig(BaseModel):
    """RAGAS quality evaluation — async, sampled."""
    enabled: bool = True
    sample_rate: float = 0.1    # fraction of live queries to evaluate


class AgentConfig(BaseModel):
    """Multi-agent pipeline thresholds."""
    consensus_n_agents:int = 5
    retrieval_relevance_threshold: float = 0.12
    claim_support_threshold: float = 0.20
    confidence_low_threshold: float = 0.65
    confidence_medium_threshold: float = 0.40


class ServerConfig(BaseModel):
    """Uvicorn server settings."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]


class JWTConfig(BaseModel):
    """JWT authentication settings."""
    algorithm: str = "HS256"
    expire_minutes: int = 60


#Root Settings
class Settings(BaseSettings):
    """
    Single source of truth for all application configuration.

    Construction:
        settings = get_settings()          # cached singleton, reads config.yaml + .env
        settings = Settings.from_yaml()    # explicit, useful in tests

    Secrets come from .env (or real env vars).
    Structured defaults come from config.yaml (loaded by _load_yaml()).
    All fields have safe defaults so the app starts even without a config file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",                # ignore YAML keys that don't match any field
    )

    #App identity
    app_name:str = "MultiAgentRAG"
    app_version:str= "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    log_level: str = "INFO"

    # Secrets (populated from .env or real environment variables)
    openai_api_key: SecretStr = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr = Field(default="", alias="ANTHROPIC_API_KEY")
    groq_api_key: SecretStr = Field(default="", alias="GROQ_API_KEY")
    qdrant_api_key: SecretStr = Field(default="", alias="QDRANT_API_KEY")
    langsmith_api_key: SecretStr = Field(default="", alias="LANGSMITH_API_KEY")
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    jwt_secret:SecretStr = Field(default="", alias="JWT_SECRET")

    # ── LLM provider registry ──────────────────────────────────────────────
    # active_provider selects which entry in llm_providers is used at runtime.
    # Both are populated from config.yaml's `llm:` section by _load_yaml().
    active_provider: LLMProvider = LLMProvider.GROQ
    llm_providers: dict[str, LLMProviderConfig] = {
        "groq": LLMProviderConfig(
            model_name="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1/chat/completions",
        ),
        "anthropic": LLMProviderConfig(
            model_name="claude-sonnet-4-20250514",
            base_url="https://api.anthropic.com/v1/messages",
        ),
        "openai": LLMProviderConfig(
            model_name="gpt-4o-mini",
            base_url="https://api.openai.com/v1/chat/completions",
        ),
    }

    # ── Structured sub-configs (populated from config.yaml sections) ───────
    server: ServerConfig = ServerConfig()
    jwt: JWTConfig = JWTConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    reranker: RerankerConfig = RerankerConfig()
    query_expansion: QueryExpansionConfig  = QueryExpansionConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    cache:CacheConfig = CacheConfig()
    vector_store: VectorStoreConfig = VectorStoreConfig()
    auth: AuthConfig = AuthConfig()
    observability: ObservabilityConfig = ObservabilityConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    agents: AgentConfig = AgentConfig()

    #Computed properties
    @property
    def active_llm(self) -> LLMProviderConfig:
        """Return the LLMProviderConfig for the currently active provider."""
        return self.llm_providers[self.active_provider.value]

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    @property
    def active_api_key(self) -> str:
        """Return the raw API key string for the active LLM provider."""
        mapping = {
            LLMProvider.GROQ:      self.groq_api_key,
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI:    self.openai_api_key,
        }
        secret = mapping[self.active_provider]
        return secret.get_secret_value() if secret else ""

    #Constructors
    @classmethod
    def from_yaml(cls, yaml_path: str | Path = _YAML_PATH) -> "Settings":
        """
        Load from a specific YAML file path.
        Useful in tests: Settings.from_yaml("tests/fixtures/config.yaml")
        """
        defaults = _load_yaml(Path(yaml_path))
        return cls(**defaults)

#Cached singleton
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton — returns the same Settings instance on every call.
    Call this everywhere instead of constructing Settings() directly.
    The cache means _load_yaml() and YAML parsing run exactly once per process.
    """
    defaults = _load_yaml()
    return Settings(**defaults)