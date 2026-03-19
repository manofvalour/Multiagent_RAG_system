from __future__ import annotations
import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..logger.logger import GLOBAL_LOGGER as logger
from ..exception.custom_exception import MulitagentragException

_BASE_DIR = Path(__file__).resolve().parent.parent
_YAML_PATH = _BASE_DIR/"config/configuration.yaml"


def _load_yaml()-> dict[str, Any]:
    try:    
        if not _YAML_PATH.exists():
            logger.error("Configuration file not found", path=str(_YAML_PATH))
            raise FileNotFoundError(f"Config file not found: {_YAML_PATH}")
        
        with _YAML_PATH.open(encoding= 'utf-8') as f:
            config: dict[str, Any]= yaml.safe_load(f) or {}

                ## log config summary (not actual content)
        top_keys = list(config.keys()) if isinstance(config,dict) else []
        logger.info("Configuration loaded successfully", path = str(_YAML_PATH), keys=top_keys)
        
        flat: dict[str, Any]= {}
        for section, values in config.items():
            if isinstance(values,dict):
                if section == 'llm':
                    flat['active_provider']= values.get('active_provider', 'groq')
                    flat['llm_providers']= values.get('providers', {})
                
                #elif section =="reranker":
                 #   flat['reranker']= values.get("reranker", {})

                else:
                    flat.update(values)
            else:
                flat[section] =values
        logger.info(f"yaml file loaded and flattened")

        return flat
    
    except Exception as e:
        logger.error(f"Failed to load the config file", error=str(e))
        raise MulitagentragException('Failed to load the config file')

class Environment(str, Enum):
    DEVELOPEMENT = 'developement'
    STAGING = 'staging'
    PRODUCTION = 'production'

class LLMProvider(str, Enum):
    ANTHROPIC = 'anthropic' 
    GROQ = 'groq'

class SystemPrompt(str, Enum):
    SYSTEMPROMPT= ""

class ExpansionStrategy(Optional[str], Enum):
    HYDE = 'hyde'
    MULTI_QUERY = "multi_query"
    BOTH = 'both'

#class VectorStoreType(str, Enum):
   # FAISS = 'faiss'
  #  PGVECTOR = 'pgvector'
 #   QDRANT = 'qdrant'

class RerankerConfig(BaseModel):
    """config for the reranker model"""
    enabled: bool = True
    top_n: int = 5
    model: str = "BAAI/bge-reranker-v2-m3"

class LLMProviderConfig(BaseModel):
    """ Config for a single LLM Provider."""
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    model_name:str = 'openai/gpt-oss-120b'
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2048, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)

class Settings(BaseSettings):
    """
    Priority order (highest -> lowest):
    1. Real environment variables (set in the shell or Docker)
    2. .env file (local dev secrets)
    3. config.yaml values (flattend into defaults below)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra = 'ignore',
    )

    ## app
    app_name: str = "MultiAgentRAG"
    app_version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPEMENT
    debug:bool = False
    log_level: str = "INFO"

    ##server
    host:str = "0.0.0.0"
    port: int = 8000
    workers: int =4
    cors_origins: list[str] = ["https://localhost:3000", "http://localhost:5173"]

    ## Redis
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    session_ttl_seconds: int = 86400

    ## Authentiation
    jwt_secret: SecretStr = Field(default="")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int =60
    api_key_header: str = "X-API-Key"

    ## LLM
    active_provider: LLMProvider = LLMProvider.GROQ
    system_prompt:SystemPrompt = SystemPrompt.SYSTEMPROMPT

    llm_providers: dict[str, LLMProviderConfig]= {
        'anthropic': LLMProviderConfig(model_name ="Claude-sonnet-4-20250514"),
        'groq': LLMProviderConfig(model_name='openai/gpt-oss-120b'),
    }

    anthropic_api_key: SecretStr = Field(default="", alias= "ANTHROPIC_API_KEY")
    groq_api_key: SecretStr = Field(default = "", alias = "GROQ_API_KEY")

    ## reranker
    reranker: dict[str, RerankerConfig]= {
        "reranker": RerankerConfig(model='')
    }

    ## Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int =64

    ##Chunking strategy
    chunking_strategy: str = "hybrid"        # hybrid | recursive | sentence
    chunk_size: int = 512
    chunk_overlap: int = 64
    use_tokens: bool = False

    ## Query Expansion
    enable_expansion:bool = True
    expansion_strategy: ExpansionStrategy
    hyde_temperature: str = 0.7
    num_queries:int = 3


    ## vector store (qdrant)
    top_k_retrieval: int=10
    qdrant_api_key: SecretStr = Field(default="", alias = "QDRANT_API_KEY")
    qdrant_url:str = "http://localhost:6333"
    collection_name: str = "rag_chunk"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 100        
    hnsw_ef: int  = 128
    qdrant_index_path: str  = "./data/qdrant"    

    ## Agents
    consensus_n_agents: int =5
    retrieval_relevance_threshold: float = 0.12
    claim_support_threshold: float =0.20
    confidence_low_threshold: float =0.65
    confidence_medium_threshold: float=0.40

    ## RAte Limiting
    rate_limit_requests: int =60
    rate_limit_window_seconds:int = 60

    ## Observability
    enable_metrics: bool =True
    enable_tracing: bool =True
    otlp_endpoint: str = "http://localhost:4317"
    sentry_dsn: str =""

    @property
    def active_llm(self) -> LLMProviderConfig:
        """Return the config of the provider that is currently active"""
        return self.llm_providers[self.active_provider.value]

    @property
    def is_production(self) -> LLMProviderConfig:
        return self.environment == Environment.PRODUCTION
    
@lru_cache
def get_settings()-> Settings:
    yaml_defaults = _load_yaml()
    return Settings(**yaml_defaults)

if __name__ == "__main__":
    try:
        config = get_settings()

        print(config.model_json_schema)
        logger.info("ConfigLoader test run completed succesfully")
    except Exception as e:
        raise MulitagentragException("failed to load the config file")
    
