# Benchmark Analysis Plan — finding the best RAG config (latency + quality)

## Goal
Pick the best pipeline configuration among the 16 swept (A–P) for a **balanced** objective:
latency (speed) and generation accuracy/quality.

## Data
- `notebook/result/benchmark_raw.json` — per-query raw results (16 configs x 27 queries),
  with `reranked_chunks` stored as stringified reprs (re-parsed by the notebook's `transform()`).
- `notebook/result/benchmark_summary.csv` — per-config aggregate (already produced).

Factors swept:
- Reranker: ON / OFF
- Consensus generators: 3 / 1
- Query expansion: `both` / `hyde` / `multi_query` / `off`

Metrics per query: `latency_ms`, `n_retrieved`, `n_reranked`, `n_expanded`,
`n_claims`, `n_supported`, `claim_support`, `confidence_final`,
`hallucination_risk` (LOW/MED/HIGH), + RAGAS `faithfulness`,
`answer_relevancy`, `context_precision`, `context_recall`, and
LLM-judge `grounding_score` (1–5).

## Implemented plan (`notebook/analysis.ipynb` — new appended cells, grouped)

1. **Load tidy long DataFrame** — one row per (config, query) for grouped analysis.
2. **Grounding vs RAGAS discrepancy audit**
   - grounding value distribution; cross-tab faithfulness x grounding;
     spearman correlation of grounding with the other quality metrics.
   - Rationale: grounding is near-flat (~1.5) and non-discriminative — confirm and treat as a flag only.
3. **Latency characterization** — mean/median/p95/max per config; treat p95 (not mean) as the real signal
   since latency includes external LLM API time (retries/sleeps).
4. **Quality & safety profile** — config means of all RAGAS metrics + claim_support;
   hallucination-risk LOW/MED/HIGH counts.
5. **Main effects** — group each metric by each lever (reranker, generators, expansion) to isolate
   which lever drives latency vs quality.
6. **Balanced composite + Pareto frontier**
   - normalize 0–1; `composite = 0.5 * latency_norm + 0.5 * quality_norm`
     (quality = 0.8*RAGAS/claims + 0.2*low-risk fraction).
   - rank all 16 configs; Pareto-optimal set (min latency vs max quality);
     weight sweep (0.3/0.5/0.7/0.9) to test winner stability.
7. **Final recommendation** — balanced pick + fastest + highest-quality single-objective picks.

## Key caveats handled
- `grounding_score` had bugs (parse failure → coerced to worst 1.0; loose regex) → fixed the
  `grounding_eval` **function** for future runs and treated existing scores as a flag.
- Latency is noisy (external API); rely on median/p95 and relative comparisons.
- Faithfulness is saturated near 1.0 across configs → low discriminating power; rely on
  precision/recall/claims for quality ranking.
- Reranker=ON configs show more HIGH-risk flags than OFF in this dataset — cross-check against
  factual RAGAS scores rather than the pipeline's own risk label.

## Deliverable
A short verdict (recommended config) with the per-objective bests, plus the supporting
grouped cells retained in the notebook for review/re-do.

## Todo status
- [x] Fix `grounding_eval` for future runs
- [x] Append grouped analysis cells (groups 1–7)
- [x] Verify notebook loads and cells intact
- [ ] Run the notebook cells and confirm results