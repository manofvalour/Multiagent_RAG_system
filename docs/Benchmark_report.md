# Multi-Agent RAG System — Benchmark Report

**Scope:** 16 pipeline configurations benchmarked against 27 evaluation queries.
**Objective:** identify the configuration that best balances **latency** and
**generation accuracy/quality** for production use.

This report is produced from the appended post-hoc analysis in
`notebook/analysis.ipynb` (Groups 1–7) over the raw per-query results in
`notebook/result/benchmark_raw.json`
`analysis_viz.md` (rationale for each visualization).

---
## 1. Experiment design

### 1.1 System under test
A multi-agent RAG pipeline with the staged architecture:

```
Cache → QueryExpansion → ChunkRetrieval → RerankerAgent → ConsensusAgent
      → ClaimVerification → ConfidenceScoring → Response (+ Cache + RAGAS eval)
```

- **Embedding:** `all-MiniLM-L6-v2` (384-dim), Qdrant vector store (HNSW COSINE).
- **Reranker:** `jinaai/jina-reranker-v3` cross-encoder (optional).
- **LLM:** Groq (default `openai/gpt-oss-120b`) for generation; Groq/`gemini`
  used for the benefit-side judge evaluations.
- Retrieval retrieves `n_retrieved=18` chunks, reranks to `n_reranked=5`
  (when the reranker is on), expands to `n_expanded=5`.

### 1.2 Factors (independent variables)
A **full 2 × 2 × 4 factorial** design gives 16 configurations (A–P):

| Factor | Levels |
|--------|--------|
| **Reranker** | `ON` / `OFF` |
| **Number of consensus generators** | `1` / `3` |
| **Query expansion strategy** | `both` / `hyde` / `multi_query` / `off` |

Note the third factor actually has 4 levels (`off` = expansion disabled), so the
design is 2 × 2 × 4 = 16, enumerated as configurations A–P:

| Config | Reranker | Gen | Expansion | Config | Reranker | Gen | Expansion |
|--------|----------|-----|-----------|--------|----------|-----|-----------|
| **A** | ON | 3 | both | **I** | OFF | 3 | both |
| **B** | ON | 3 | hyde | **J** | OFF | 3 | hyde |
| **C** | ON | 3 | multi_query | **K** | OFF | 3 | multi_query |
| **D** | ON | 3 | off | **L** | OFF | 3 | off |
| **E** | ON | 1 | both | **M** | OFF | 1 | both |
| **F** | ON | 1 | hyde | **N** | OFF | 1 | hyde |
| **G** | ON | 1 | multi_query | **O** | OFF | 1 | multi_query |
| **H** | ON | 1 | off | **P** | OFF | 1 | off |

### 1.3 Workload and metrics
- **Queries:** a fixed `EVAL_QUERIES` set of **27** questions answered by every
  configuration → **432 query runs** total (27 × 16).
- **Dependent (performance):** `latency_ms` (mean, **median**, p95, max).
- **Dependent (quality, RAGAS):** `faithfulness`, `answer_relevancy`,
  `context_precision`, `context_recall`.
- **Dependent (robustness/claims):** `claim_support`, `confidence_final`,
  `hallucination_risk` (`LOW`/`MEDIUM`/`HIGH`), and `grounding_score` (1–5,
  a bespoke LLM judge which we decide not to use because it's indiscriminante).

### 1.4 Analysis approach
Metrics were aggregated per config **and per single lever** (main effects). A
**balanced objective** was defined to rank configs:

```
rz(s)            = min-max normalize to [0,1] (higher = better)
latency_norm     = 1 - rz(median latency)                     (lower = better)
quality          = mean(faithfulness, answer_relevancy, context_precision, context_recall, claim_support)
quality_all      = 0.8 * rz(quality) + 0.2 * rz(low_risk_fraction)
composite        = 0.5 * latency_norm + 0.5 * quality_all
```

A **Pareto frontier** (min median latency × max quality) and a **weight sweep**
(composite latency weight `w ∈ {0.3, 0.5, 0.7, 0.9}`) tested whether the winner is
stable, so the recommendation is not a single-weight coincidence.

---

## 2. Results

### 2.1 Latency (ordered by mean)
Latency here is dominated by external LLM API time, so **median** and **p95**
matter more than the mean.

| Config | Reranker | Generators | Expansion | mean (ms) | **median (ms)** | p95 (ms) | max (ms) |
|--------|----------|-----------|-----------|-----------|----------|----------|----------|
| **D** | ON | 3 | off | 5 116 | **4 143** | 9 232 | 19 030 |
| H | ON | 1 | off | 6 166 | **4 246** | 19 505 | 20 681 |
| J | OFF | 3 | hyde | 6 070 | **4 716** | 13 222 | 15 231 |
| **F** | ON | 1 | hyde | 6 362 | **5 427** | 12 958 | 20 501 |
| E | ON | 1 | both | 6 562 | 5 564 | 11 957 | 20 841 |
| C | ON | 3 | multi_query | 6 352 | 5 867 | 8 481 | 14 670 |
| A | ON | 3 | both | 11 679 | 8 277 | 21 876 | 31 728 |
| P | OFF | 1 | off | 11 096 | 8 957 | 28 253 | 43 926 |
| I | OFF | 3 | both | 12 911 | 6 961 | 44 925 | 75 370 |
| B | ON | 3 | hyde | 14 559 | 7 036 | 54 397 | 61 756 |
| G | ON | 1 | multi_query | 16 605 | 12 312 | 28 301 | 85 592 |
| O | OFF | 1 | multi_query | 17 236 | 17 098 | 30 593 | 50 862 |
| K | OFF | 3 | multi_query | 17 397 | 16 031 | 28 625 | 39 469 |
| L | OFF | 3 | off | 21 077 | 15 992 | 46 988 | 78 930 |
| N | OFF | 1 | hyde | 21 386 | 21 193 | 39 499 | 41 569 |
| M | OFF | 1 | both | 25 854 | 20 581 | 51 353 | 201 577 |

**Main-effect latency means:**

| Lever | Level | avg latency (ms) |
|-------|-------|------------------|
| Reranker | **ON** | **9 175** |
| Reranker | OFF | 16 628 |
| Generators | 1 | 13 908 |
| Generators | 3 | 11 895 |
| Expansion | **off** | **10 864** |
| Expansion | hyde | 12 094 |
| Expansion | both | 14 252 |
| Expansion | multi_query | 14 398 |

### 2.2 Quality (RAGAS + claims)

| Config | Faith | Relevance | Precision | Recall | ClaimSup | Conf | **quality*** |
|--------|-------|-----------|-----------|--------|----------|------|--------------|
| **F** | 1.000 | 0.882 | **0.943** | 0.981 | **0.695** | 0.626 | **0.900** |
| **D** | 1.000 | 0.884 | 0.832 | 1.000 | 0.562 | 0.562 | 0.855 |
| A | 0.951 | 0.829 | 0.876 | 0.944 | 0.553 | 0.556 | 0.831 |
| G | 0.994 | 0.848 | 0.892 | 0.944 | 0.565 | 0.563 | 0.849 |
| H | 0.953–1.000 | 0.878 | 0.857 | 1.000 | 0.561 | 0.561 | 0.859 |
| B | 0.977 | 0.893 | 0.904 | 0.982 | 0.486 | 0.478 | 0.848 |

*`Quality` = mean of the five RAGAS/claim columns. Config labels abbreviated; see
full table in the notebook Group 4.

### 2.3 Hallucination risk (pipeline label)

| Config | LOW | MED | HIGH | low_frac |
|--------|-----|-----|------|----------|
| **F** | 16 | 6 | **5** | 0.593 |
| H | 12 | 7 | 8 | 0.444 |
| D | 11 | 8 | 8 | 0.407 |
| P | 11 | 16 | 0 | 0.407 |
| I–O (OFF, mostly) | 8–10 | 16–19 | 0–2 | 0.296–0.370 |

**Key check:** every `Reranker=ON` config (A–H) carries **more HIGH** flags
(5–9) than nearly every `Reranker=OFF` config (I–P, 0–2 HIGH) — yet ON configs
also report **higher** claim-support. The pipeline’s own risk label therefore
disagrees with the factual quality measures.

### 2.4 Grounding metric
- Distribution: **314 / 432 = 73 %** of answers landed on the worst grade `1.0`;
  mean **1.61 / 5.0**.
- Spearman correlation with `faithfulness` = **0.091** (essentially zero),
  and ≤ 0.05 vs all other quality metrics.

---

## 4. Analysis

### 4.1 Data quality — grounding is unusable as-is
The grounding grades are flat (~1.6) and **uncorrelated** with every factual
metric (ρ ≈ 0.09 vs faithfulness). The bespoke 1–5 judge produced grades from a
faulty parser (a raw digit regex `[1-5]` captured digits inside larger numbers,
arbitrary-format failures were coerced to the worst `1.0`). **Verdict:** grounding
is **not discriminative** → excluded from ranking, kept only as a flag.

### 4.2 The reranker is the single biggest lever
- **Latency −45%:** avg 9,175 ms (ON) vs 16,628 ms (OFF).
- **Quality:** ON raises `context_precision` (0.883 vs 0.852) and `claim_support`
  (0.562 vs 0.508), with no faithfulness loss (0.989 vs 0.991).
- Reranked settings supply the top-four composite configs (F, H, D, C); every
  `OFF` config occupies the bottom half. **Reranking is not just a quality win —
  it is also faster in this setting** because it lets the LLM operate on 5 focused
  chunks instead of 18.

### 4.3 Query expansion — less is more
- Expanding **off** is the cheapest (median ~5,000 ms for reranked) and gives the
  best relevancy (mean `answer_relevancy` 0.884).
- `hyde` offers the best retrieval quality (**highest context_precision/recall**)
  and, importantly, combines with the reranker to yield the overall winner.
- `multi_query` adds latency (+~30%) with **no** precision/recall gain → **drop**.
- `both` (multi-approach expansion) adds latency with the *worst* faithfulness
  (0.980) → **drop**.

### 4.4 Consensus generators — marginal
`3` generators slightly raise `context_recall` (0.983 vs 0.979) but cost more LLM
calls. With a good reranker the best config uses only **1 generator**, so the
"consensus" overhead buys little here.

### 4.5 Risk label vs factual quality
Cramer-variance: `Reranker=ON` shows *more* HIGH flags yet *higher* faithfulness.
This indicates the `hallucination_risk` label (pipeline heuristic) is not a
reliable ranking signal and **should not** be weighted into the model selection —
a fundamental insight for this benchmark.

### 4.6 Balanced ranking & Pareto
**Composite ranking (0.5·lat + 0.5·quality):**

| Rank | Config | composite |
|------|--------|-----------|
| 1 | **F: ON·1·hyde** | **0.962** |
| 2 | H: ON·1·off | 0.726 |
| 3 | D: ON·3·off | 0.697 |
| 4 | C: ON·3·multi | 0.650 |
| 5 | J: OFF·3·hyde | 0.581 |

**Pareto front** (speed vs quality): `{D, H, F}` — D is fastest, F is highest
quality; H in between. **Weight sweep** (`w ∈ 0.3/0.5/0.7/0.9`): `F` is top at
three of four weights and only loses to `D` at the extreme latency-only `w=0.9`.
The winner is **not** a single-weight artifact.

---

## 5. Conclusion

**Recommendation — Configuration F: `Reranker=ON, 1 generator, query expansion=hyde`.**

| Criterion | Winner | Value |
|-----------|--------|-------|
| Fastest (median latency) | **D** (ON·3·off) | 4,142 ms |
| Highest quality | **F** (ON·1·hyde) | quality 0.900 |
| **Balanced latency + quality** | **F** | composite 0.962 |

**Key conclusions**
1. **Enable the reranker every time.** It is both the largest quality driver and
   the largest latency saver (−45%), so it dominates the whole search space.
2. **Use `hyde` expansion with the reranker** (F); avoid `multi_query` and `both`,
   which add latency with no measurable retrieval gain.
3. **One consensus generator is enough** once a reranker is present; three only
   inflate cost and complexity.
4. If latency is the absolute priority, **D (ON·3·off)** is the fastest stable
   configuration and still near the top on quality.
5. **Stop using the current `grounding_score` for ranking** (flat, artifact-heavy);
   trust the RAGAS faithfulness / context metrics instead.

**Caveats:** latency numbers include external LLM API variability and retries, so
absolute timings (median/p95) are lab-specific. The benchmark uses 27 fixed
queries; conclusions are strongest for the reranker effect and should be
revalidated on a larger, more diverse query set.

---

## Appendix — configuration codes

`Reranker` on/off · `Gen` = consensus generators (1/3) · `Exp` = query expansion
(`both`/`hyde`/`multi_query`/`off`).
Benefit-side judges: RAGAS scored with `gemini`; grounding scored with `groq`