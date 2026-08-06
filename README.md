# Multi-Agent RAG System

A production-grade Retrieval-Augmented Generation (RAG) system featuring a 7-agent hallucination reduction pipeline. Built with FastAPI, Redis, Qdrant, and support for multiple LLM providers.

## Features

- **7-Stage Hallucination Reduction Pipeline**: Query expansion, retrieval, reranking, consensus generation, claim verification, confidence scoring, and RAGAS evaluation
- **Multi-LLM Support**: Groq (default), Anthropic Claude, and OpenAI GPT models
- **Semantic Caching**: Redis-based caching with similarity threshold matching
- **Vector Search**: Qdrant for high-performance ANN retrieval with HNSW indexes
- **Observability**: OpenTelemetry tracing, LangSmith integration, Prometheus metrics, and Grafana dashboards
- **Streaming Support**: Server-Sent Events (SSE) for real-time response streaming
- **Document Ingestion**: Support for PDF, DOCX, PPTX, and plain text with hybrid chunking

## Architecture

```
Query → Cache Check → Query Expansion → Retrieval → Reranking →
Consensus Generation → Claim Verification → Confidence Scoring → Response + Cache
```

### 7-Agent Pipeline

1. **QueryExpansionAgent**: HyDE / multi-query rewriting for improved retrieval recall
2. **ChunkRetrieval**: ANN vector search with Qdrant
3. **RerankerAgent**: Cross-encoder reranking using sentence-transformers
4. **ConsensusAgent**: Multi-agent answer generation with majority vote
5. **ClaimVerificationAgent**: Fact-checking against source documents
6. **ConfidenceScoringAgent**: Quality scoring with hallucination risk assessment
7. **RAGASEvaluator**: RAGAS quality metrics (faithfulness, answer relevancy, context precision)

## Quick Start

### Prerequisites

- Python 3.13+
- Docker & Docker Compose
- Redis (if running locally)
- Qdrant (if running locally)
- API keys for at least one LLM provider

### Local Development

```bash
# Clone the repository
git clone https://github.com/manofvalour/multiagent-rag-system.git
cd multiagent-rag-system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run locally
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Compose (Recommended)

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f api

# Stop services
docker compose down
```

## Configuration

The system uses a three-layer configuration system:

1. **Environment Variables** (highest priority)
2. **`.env` file** (local secrets)
3. **`config/configuration.yaml`** (tunable defaults)

### Environment Variables

```bash
# Required
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key

# Optional - defaults shown
QDRANT_ENDPOINT=http://localhost:6333
QDRANT_API_KEY=
REDIS_URL=redis://localhost:6379/0
ENVIRONMENT=development

# Observability
OTEL_ENABLED=false
LANGCHAIN_TRACING_V2=false
LANGSMITH_API_KEY=
LANGCHAIN_PROJECT=multiagent-rag

# Server
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
```

### Configuration File (`config/configuration.yaml`)

```yaml
# LLM Provider Settings
llm:
  active_provider: groq  # groq, anthropic, openai
  providers:
    groq:
      model_name: openai/gpt-oss-120b
      base_url: https://api.groq.com/openai/v1/chat/completions
      temperature: 0.0
    anthropic:
      model_name: claude-sonnet-4-20250514
      base_url: https://api.anthropic.com/v1/messages
    openai:
      model_name: gpt-4o-mini

# Reranker Settings
reranker:
  enabled: true
  top_n: 3
  model: cross-encoder/ms-marco-MiniLM-L-6-v2

# Query Expansion
query_expansion:
  enabled: true
  strategy: both  # hyde, multi_query, both
  num_queries: 3

# Redis Cache
cache:
  enabled: true
  ttl_seconds: 3600
  similarity_threshold: 0.92

# Vector Store
vector_store:
  collection_name: rag_chunks
  hnsw_m: 16
  hnsw_ef_construct: 100
```

## API Endpoints

### Query Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Submit a query for RAG processing |

### Document Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ingest/text` | Ingest raw text documents |
| POST | `/ingest/file` | Upload and ingest file (PDF, DOCX, PPTX) |
| GET | `/documents` | List all indexed documents |
| DELETE | `/documents/{id}` | Delete a document |

### System Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with component status |
| GET | `/metrics` | Prometheus metrics |
| GET | `/analytics` | Query analytics dashboard |

### Example Usage

```bash
# Query the system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key features of transformer models?"
    }'
```

### Response Format

```json
{
  "request_id": "uuid",
  "query": "What are the key features of transformer models?",
  "answer": "Transformer models feature...",
  "claims": [
    {
      "text": "Transformers use self-attention",
      "supported": true,
      "confidence": 0.95
    }
  ],
  "confidence": {
    "claim_support": 0.92,
    "avg_relevance": 0.88,
    "source_overlap": 0.85,
    "final": 0.88
  },
  "hallucination_risk": "LOW",
  "latency_ms": 1250.5,
  "cached": false
}
```

## Project Structure

```
multiagent_rag_system/
├── app.py                    # FastAPI application entry point
├── multiagent_rag_system/
│   ├── agent/
│   │   ├── agents/           # Individual agent implementations
│   │   │   ├── claim_verification_agent.py
│   │   │   ├── consensus_agent.py
│   │   │   ├── confidence_score_agent.py
│   │   │   ├── evaluator.py
│   │   │   ├── query_expansion.py
│   │   │   ├── reranker_agent.py
│   │   │   └── retrieval_agent.py
│   │   └── pipeline/
│   │       └── pipeline.py   # RAGOrchestrator wiring
│   └── src/
│       ├── cache/            # Redis semantic cache
│       ├── database/         # Qdrant vector store
│       ├── embedding/        # Sentence-transformer embeddings
│       ├── llm/              # LLM clients (Groq, Anthropic, OpenAI)
│       ├── models/           # Pydantic data models
│       ├── observability/    # OpenTelemetry + LangSmith tracing
│       ├── test/             # Unit and integration tests
│       └── utils/            # Config loader, utilities
├── backend/
│   └── Dockerfile            # Backend container definition
├── frontend/
│   ├── rag-frontend.html     # Static HTML frontend
│   ├── nginx.conf            # Nginx configuration
│   └── Dockerfile            # Frontend container
├── infra/
│   ├── nginx.conf            # Reverse proxy config
│   ├── prometheus.yml        # Prometheus configuration
│   └── grafana/              # Grafana dashboards and provisioning
├── config/
│   └── configuration.yaml    # Application configuration
├── notebook/
│   ├── experimentation.ipynb # Benchmark sweep (16 configs × 27 queries)
│   ├── analysis.ipynb        # Standalone post-hoc analysis (Groups 1–7)
│   ├── result/               # Raw JSON + summary CSV exports
├── docs/                     # report.md, plan.md, analysis_viz.md
├── docker-compose.yml        # Production stack
└── requirements.txt          # Python dependencies
```

## Benchmarking & Analysis

The pipeline is evaluated with a **2×2×4 factorial sweep** over 16 pipeline
configurations (**A–P**): reranker ON/OFF × consensus generators 1/3 × query
expansion `both`/`hyde`/`multi_query`/`off`, run against 27 fixed eval queries.

The benchmark workflow is split into two notebooks under `notebook/`:

| Notebook | Purpose |
|----------|---------|
| `notebook/experimentation.ipynb` | Initializes components, ingests docs, runs all 16 configs against the 27 queries, computes RAGAS + LLM-judge grounding, and exports raw results to `notebook/result/benchmark_raw.json` (expensive: calls external LLM APIs). |
| `notebook/analysis.ipynb` | **Standalone** — reloads `notebook/result/benchmark_raw.json`, builds summary tables, and runs the post-hoc Groups 1–7 analysis (grounding audit, latency characterization, quality/safety profile, main effects, balanced composite + Pareto, final recommendation). Also exports `notebook/result/benchmark_summary.csv`. |

> Run **Run All** on `experimentation.ipynb` first, then `analysis.ipynb` (which
> needs no re-run of the experiments). Analysis cells are robust to missing/NaN
> grounding scores.

### Report & docs (`docs/`)

- `report.md` — experiment design / results / analysis / conclusion.
- `plan.md` — analysis plan.
- `analysis_viz.md` — rationale for each visualization.

**Current best config: F (Reranker=ON, 1 gen, hyde). Fastest: D (ON, 3 gen, off).**

> ⚠ The `grounding_score` LLM-judge is **not discriminative** (73% of grades landed on
> 1.0, ρ≈0.09 vs faithfulness) — use it only as a flag, never for ranking. Cross-check
> hallucination-risk labels against RAGAS scores before trusting them.
> Latency includes external LLM API time + judge sleeps; compare with median/p95, not the mean.

## Testing

```bash
# Run all tests
pytest multiagent_rag_system/src/test/ -v

# Run specific test file
pytest multiagent_rag_system/src/test/test_pipeline.py -v

# Run with coverage
pytest multiagent_rag_system/src/test/ -cov=multiagent_rag_system -cov-report=html
```

## Monitoring

### Prometheus Metrics

Access at `http://localhost:9090`:
- `rag_query_total` - Total queries processed
- `rag_query_latency_seconds` - Query latency histogram
- `rag_cache_hit_total` - Cache hits/misses
- `rag_confidence_score` - Response confidence scores

### Grafana Dashboards

Access at `http://localhost:3001`:
- Query performance metrics
- Cache hit rates
- Error rates
- LLM token usage

## Deployment

### CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) handles:
1. Running tests on PR/push to main
2. Building and pushing Docker images to GitHub Container Registry
3. Deploying to production server via SSH

### Production Setup

```bash
# Build and deploy
docker compose -f docker-compose.yml build
docker compose -f docker-compose.yml up -d

# Scale services
docker compose scale api=4
```

### Environment Variables for Production

```bash
ENVIRONMENT=production
OTEL_ENABLED=true
LANGCHAIN_TRACING_V2=true
LOG_LEVEL=warning
```

## License

MIT License - see LICENSE file for details.