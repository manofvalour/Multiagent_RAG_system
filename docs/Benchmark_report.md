# Multi-Agent RAG System — Benchmark Report

**Scope:** 16 pipeline configurations benchmarked against 27 evaluation queries,
**pooled across 3 runs** (3 × 27 × 16 = **1 296 query runs**).
**Objective:** identify the configuration that best balances **latency** and
**generation accuracy/quality** for production use.

This report is produced from the pooled post-hoc analysis in
`notebook/pooled_analysis.ipynb` (Groups 1–7) over the raw per-query results in
`notebook/benchmark_raw0.json`, `benchmark_raw1.json`, `benchmark_raw2.json`.
Aggregates are also exported to `notebook/result/benchmark_summary_pooled.csv`
(see `analysis_viz.md` for the rationale of each visualization).

> **Update note:** Pooling **reduces but does not remove** run-to-run noise:
> 7 of 16 configs (F, H, L, M, N, O, P) have an across-run spread (SD of per-run
> medians) greater than 50% of their pooled median — their medians must be read
> with care.

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
- **LLM:** Groq (default `openai/gpt-oss-120b`) for generation; `gemini`
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
  configuration, repeated across **3 runs** → **1 296 query runs** total
  (27 × 16 × 3).
- **Dependent (performance):** `latency_ms` (mean, **median**, p95, max, and
  across-run spread).
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

Because latency is dominated by external LLM API time, **median** (primary) and
**p95** (tail) are the reporting statistics, not the mean. A **noise gate**
flags configs whose across-run spread (SD of per-run medians) exceeds **50% of
the pooled median** — those pooled medians are treated as unreliable.

---

## 2. Results

### 2.1 Latency (pooled, ordered by median)

| Config | Reranker | Gen | Expansion | mean (ms) | **median (ms)** | p95 (ms) | max (ms) | run_med_sd |
|--------|----------|-----|-----------|-----------|----------|----------|----------|-----------|
| **P** | OFF | 1 | off | 6 170 | **3 989** | 13 831 | 43 926 | 3 101 |
| H | ON | 1 | off | 6 862 | **4 301** | 19 359 | 26 849 | 3 151 |
| **D** | ON | 3 | off | 7 246 | **5 057** | 17 949 | 26 214 | 1 584 |
| J | OFF | 3 | hyde | 7 214 | **5 809** | 15 231 | 21 698 | 888 |
| O | OFF | 1 | multi_query | 9 527 | **6 045** | 23 342 | 50 862 | 6 836 |
| **F** | ON | 1 | hyde | 8 870 | **6 143** | 20 996 | 27 237 | 3 451 |
| C | ON | 3 | multi_query | 7 496 | **6 226** | 14 670 | 29 121 | 734 |
| N | OFF | 1 | hyde | 11 065 | **6 314** | 32 465 | 41 569 | 9 156 |
| B | ON | 3 | hyde | 10 783 | **6 467** | 27 033 | 61 756 | 972 |
| M | OFF | 1 | both | 13 288 | **6 872** | 40 972 | 125 575 | 8 604 |
| I | OFF | 3 | both | 10 644 | **7 183** | 20 643 | 75 370 | 708 |
| L | OFF | 3 | off | 11 253 | **7 442** | 33 696 | 78 930 | 6 339 |
| A | ON | 3 | both | 11 417 | **9 333** | 22 023 | 31 728 | 1 438 |
| E | ON | 1 | both | 12 110 | **10 927** | 25 197 | 45 204 | 4 320 |
| G | ON | 1 | multi_query | 13 338 | **11 567** | 23 350 | 85 592 | 2 932 |
| K | OFF | 3 | multi_query | 16 550 | **13 122** | 29 649 | 127 402 | 4 486 |

**Main-effect latency means (pooled):**

| Lever | Level | avg latency (ms) |
|-------|-------|------------------|
| Reranker | ON | **9 765** |
| Reranker | OFF | 10 714 |
| Generators | 1 | 10 154 |
| Generators | 3 | 10 325 |
| Expansion | **off** | **7 883** |
| Expansion | hyde | 9 483 |
| Expansion | multi_query | 11 728 |
| Expansion | both | 11 865 |

**Effect size per lever** (max − min level mean): **expansion** dominates latency
(3 982 ms), then reranker (949 ms), then generators (171 ms — effectively neutral).

### 2.2 Quality (RAGAS + claims, pooled means)

| Config | Faith | Relevance | Precision | Recall | ClaimSup | Conf | **quality*** |
|--------|-------|-----------|-----------|--------|----------|------|--------------|
| **F** | 0.985 | **0.887** | 0.908 | 0.988 | **0.654** | **0.607** | **0.884** |
| A | 0.978 | 0.857 | **0.921** | 0.969 | 0.581 | 0.571 | 0.861 |
| B | 0.986 | 0.867 | 0.904 | **0.994** | 0.546 | 0.552 | 0.859 |
| C | 0.979 | 0.864 | 0.888 | 0.985 | 0.570 | 0.565 | 0.857 |
| D | 0.981 | 0.879 | 0.854 | 1.000 | 0.561 | 0.561 | 0.855 |
| M | 0.987 | 0.879 | 0.869 | 0.988 | 0.540 | 0.595 | 0.853 |
| P | 0.994 | 0.881 | 0.856 | 0.982 | 0.539 | 0.590 | 0.850 |
| H | 0.994 | 0.879 | 0.858 | 0.988 | 0.530 | 0.544 | 0.850 |

*`Quality` = mean of the five RAGAS/claim columns. Full 16-config table in
`benchmark_summary_pooled.csv` and the notebook Group 4.

### 2.3 Hallucination risk (pipeline label, pooled counts / 81)

| Config | LOW | MED | HIGH | low_frac |
|--------|-----|-----|------|----------|
| **F** | 43 | 18 | **20** | 0.531 |
| A | 35 | 21 | 25 | 0.432 |
| B | 33 | 23 | 25 | 0.407 |
| D | 33 | 24 | 24 | 0.407 |
| C | 31 | 26 | 24 | 0.383 |
| H | 32 | 22 | 27 | 0.395 |
| G | 32 | 23 | 26 | 0.395 |
| E | 29 | 26 | 26 | 0.358 |
| J | 24 | 55 | 2 | 0.296 |
| I / K / N / P / L / M / O (OFF) | 24–33 | 45–55 | 1–3 | 0.296–0.407 |

**Key check:** every `Reranker=ON` config (A–H) carries **20–27 HIGH** flags,
while every `Reranker=OFF` config (I–P) carries **1–3 HIGH** — yet ON configs also
report **higher** claim-support (0.562 vs 0.518 pooled). The pipeline's own risk
label therefore disagrees with the factual quality measures.

### 2.4 Grounding metric (pooled)
- Distribution: **1 019 / 1 296 = 79 %** of answers landed on the worst grade `1.0`;
  mean **1.42 / 5.0**.
- Spearman correlation with `faithfulness` = **0.061** (essentially zero), with
  `context_precision` = **−0.164** (slightly *negative*), and ≤ 0.10 vs all other
  quality metrics.

---

## 3. Analysis

### 3.1 Data quality — grounding is unusable as-is
The grounding grades are flat (~1.4) and **uncorrelated** with every factual
metric (ρ ≈ 0.06 vs faithfulness, slightly negative vs precision). The bespoke
1–5 judge produced grades from a faulty parser (a raw digit regex `[1-5]`
captured digits inside larger numbers, arbitrary-format failures were coerced to
the worst `1.0`). **Verdict:** grounding is **not discriminative** → excluded from
ranking, kept only as a flag.

### 3.2 The reranker is the biggest *quality* lever
Pooled effect sizes — `claim_support` **+0.044**, `context_precision` **+0.027**
(0.887 vs 0.860), a small `faithfulness` dip (−0.012) — make the reranker the
largest mover of generation quality. The latency effect is **modest** at the
pooled level (−9%: 9 765 ms ON vs 10 714 ms OFF; the −45% seen in the single
run did not reproduce across runs). ON configs still hold **5 of the top 6**
balanced-composite slots (F, D, H, B, C), so reranking remains on for quality.

### 3.3 Query expansion — less is more
- Expanding **off** is the cheapest (**7 883 ms** mean — the single largest
  latency lever, 3 982 ms) and gives the best relevancy (`answer_relevancy` 0.876)
  and faithfulness (0.991).
- `hyde` offers the best retrieval quality (context_precision 0.888,
  context_recall **0.991** — both best-in-class) and, importantly, combines with
  the reranker to yield the overall winner **F**.
- `multi_query` adds ~49% latency (11 728 ms) with **no** precision/recall gain
  over hyde → **drop**.
- `both` is the slowest expansion (11 865 ms) with the *lowest* relevancy
  (0.861) → **drop**.

### 3.4 Consensus generators — marginal
`1` generator (10 154 ms) vs `3` (10 325 ms) is nearly identical in latency, and
all quality effect sizes are ≤ 0.015. `3` generators slightly raise faithfulness
and recall; `1` raises claim-support. With a good reranker the best config uses
only **1 generator**, so the "consensus" overhead buys little here.

### 3.5 Risk label vs factual quality
The pooled data confirms the single-run finding: `Reranker=ON` shows *more* HIGH
flags (20–27) yet *higher* claim-support and RAGAS faithfulness. The
`hallucination_risk` label (pipeline heuristic) is **not** a reliable ranking
signal and **should not** be weighted into model selection — a fundamental insight
for this benchmark.

### 3.6 Balanced ranking & Pareto
**Composite ranking (0.5·lat + 0.5·quality):**

| Rank | Config | composite |
|------|--------|-----------|
| 1 | **F: ON·1·hyde** | **0.882** |
| 2 | P: OFF·1·off | 0.689 |
| 3 | D: ON·3·off | 0.674 |
| 4 | H: ON·1·off | 0.672 |
| 5 | B: ON·3·hyde | 0.630 |
| 6 | C: ON·3·multi | 0.616 |

**Pareto front** (speed vs quality): `{P, D, F}` — P is fastest, F is highest
quality, D in between. **Weight sweep** (`w ∈ 0.3/0.5/0.7/0.9`): `F` is top at
three of four weights and only loses to `P` at the extreme latency-only `w=0.9`.
The winner is **not** a single-weight artifact.

### 3.7 Noise gate — run-to-run variance
7 of 16 configs — **F, H, L, M, N, O, P** — have an across-run spread greater
than **50% of their pooled median**. Pooling averages the 3 runs but does not
eliminate this variance. This matters for the recommendation: both the balanced
winner **F** (0.562) and the nominal fastest **P** (0.777) **fail** the gate,
while **D** (0.313) passes and is the fastest stable configuration.

---

## 4. Conclusion

**Recommendation — Configuration F: `Reranker=ON, 1 generator, query expansion=hyde`.**
F wins the balanced objective and quality; treat its pooled latency as noisy
(below), or take **B (ON·3·hyde)** as the stable quality alternate.

| Criterion | Winner | Value |
|-----------|--------|-------|
| Fastest (median, nominal) | **P** (OFF·1·off) | 3 989 ms |
| Fastest **stable** (passes noise gate) | **D** (ON·3·off) | 5 057 ms |
| Highest quality | **F** (ON·1·hyde) | quality 0.884 |
| **Balanced latency + quality** | **F** | composite 0.882 |

**Key conclusions**
1. **Enable the reranker every time.** It is the largest *quality* lever
   (claim-support +0.044, precision +0.027) and ON configs dominate the top of
   the balanced ranking. (The large latency *saving* seen in the single run did
   not reproduce pooled; reranking is now justified on quality grounds.)
2. **Use `hyde` expansion with the reranker** (F); `off` is the cheapest with the
   best relevancy, but `hyde` has the best retrieval quality and wins the
   balanced objective. Avoid `multi_query` and `both`, which add latency with no
   measurable retrieval gain.
3. **One consensus generator is enough** once a reranker is present; three only
   inflate cost and complexity (effect sizes ≤ 0.015).
4. If latency is the absolute priority, **D (ON·3·off)** is the fastest *stable*
   configuration and still near the top on quality. The nominal fastest **P
   (OFF·1·off)** is faster but fails the noise gate.
5. **Stop using the current `grounding_score` for ranking** (flat, artifact-heavy);
   trust the RAGAS faithfulness / context metrics instead.

**Caveats:** pooled numbers (3 runs, 1 296 rows) reduce but do not eliminate
run-to-run variance — 7 configs (F, H, L, M, N, O, P) fail the 50% spread gate.
Latency includes external LLM API variability and retries, so absolute timings
(median/p95) are lab-specific. The benchmark uses 27 fixed queries; conclusions
are strongest for the expansion and reranker effects and should be revalidated on
a larger, more diverse query set.

---

## Appendix — configuration codes

`Reranker` on/off · `Gen` = consensus generators (1/3) · `Exp` = query expansion
(`both`/`hyde`/`multi_query`/`off`).
Benefit-side judges: RAGAS scored with `gemini`; grounding scored with `groq`