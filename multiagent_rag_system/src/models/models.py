from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class HallucinationRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class AgentStatus(str,Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = 'done'
    FAILED = 'failed'


class ContentType(str, Enum):
    PROSE    = "prose"
    MARKDOWN = "markdown"
    CODE     = "code"

class IngestRequest(BaseModel):
    content:str = Field(..., min_length =10, description ="Raw document text")
    source: str = Field(..., description= "Human-readable source label")
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = Field(default=512, ge=64, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

class IngestResponse(BaseModel):
    document_id: str
    chunks_created: int
    processing_ms: float

##Document/chunk domain
class DocumentChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda:str(uuid.uuid4()))
    doc_id:str
    content: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int=0
    embedding: Optional[list[float]]=None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    vector_score: float=Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)

## setting up again trace
class AgentEvent(BaseModel):
    agent: str
    status: AgentStatus
    message: str
    duration_ms: float=0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

## claims
class Claim(BaseModel):
    id: str = Field(default_factory= lambda: str(uuid.uuid4()))
    text: str
    supported: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    supporting_chunks: list[RetrievedChunk] = Field(default_factory=list)

## query pipeline
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    include_trace: bool = True
    session_id: Optional[str]=None

class ConfidenceBreakdown(BaseModel):
    claim_support: float
    avg_relevance: float
    source_overlap:float
    final: float

class QueryResponse(BaseModel):
    request_id: str = Field(default_factory= lambda:str(uuid.uuid4()))
    query: str
    answer:str
    claims: list[Claim]
    retrieved_chunks: list[RetrievedChunk]
    confidence: ConfidenceBreakdown
    hallucination_risk: HallucinationRisk
    latency_ms: float
    agent_trace: list[AgentEvent] = Field(default_factory=list)
    cached: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

## checking the health
class HealthComponent(BaseModel):
    name: str
    healthy:bool
    latency_ms: Optional[float]=None
    detail: Optional[str]=None

class HealthResponse(BaseModel):
    status: str #"healthy, degraded, unhealthy"
    version:str
    components: list[HealthComponent]
    uptime_s:float

## analytics
class QueryMetrics(BaseModel):
    window_minutes:int
    total_queries: int
    avg_confidence: float
    avg_latency_ms: float
    risk_distribution: dict[str, int]
    cache_hit_rate: float
    top_sources: list[dict[str, Any]]

## Authentication
class TokenRequest(BaseModel):
    username:str
    password:str

class TokenResponse(BaseModel):
    access_token:str
    token_type: str = 'bearer'
    expires_in: int

class APIKeyCreate(BaseModel):
    name:str
    expires_days: Optional[int]=90

class APIKeyResponse(BaseModel):
    id: str
    name: str
    key: str
    created_at:datetime
    expires_at: Optional[datetime]=None