# Analysis & Visualization Guide — `notebook/analysis.ipynb`

This doc explains each appended analysis group, its inline-doc rationale, its
visualization, and **why** that visualization exists. It pairs 1:1 with the appended
cell block after the original benchmark (cells **33–55**).

Data: `notebook/result/benchmark_raw.json` (per-query) → the in-memory `output`
dict (ConfigResult objects). Factors: reranker ON/OFF, consensus generators 3/1,
query expansion both/hyde/multi_query/off. Goal: best config for **latency** and
**generation accuracy & quality** (balanced objective).

---

## Group 1 — Load tidy long DataFrame (cell 35)
**Analysis.** Flattens `output` (config → list[QueryResult]) into one row per
`(config, query)`. Every downstream metric (latency, RAGAS scores, risk, grounding)
then aggregates by config OR by a single lever with one code path.

**Why this gets a doc:** the choice of *tidy/long* layout is the design decision
everything else depends on — without it, grouped/comparative analysis would need ad‑hoc
loops everywhere.

**Visualization:** none (pure setup stage; nothing to plot yet).

---

## Group 2 — Grounding vs RAGAS discrepancy audit (cell 38) + plot (cell 39)

**Analysis.** Diagnoses `grounding_score` (1–5) before trusting it:
- (a) value distribution — earlier readings were near-flat (~1.5), contradicting
  RAGAS faithfulness (~0.95–1.0);
- (b) faithfulness × grounding cross-tab;
- (c) spearman correlation of grounding with the other quality metrics.

**Why the doc:** the grades were produced by a **buggy** `grounding_eval` (parse
failures coerced to the worst 1.0). Without this audit the metric would silently skew
every ranking. The doc records that grounding must be **disqualified** from ranking if
flat + uncorrelated.

**Visualization (cell 39).**
- Left histogram: grounding grade distribution — makes the "flat" pathology visible at
  a glance.
- Right scatter: faithfulness (RAGAS) vs grounding — shows the two measure different
  things (no relationship), justifying dropping grounding from ranking.

**Why this viz:** a table of a few hundred grades is unreadable; a histogram + scatter
prove the metric is non-discriminative in seconds, reinforcing the doc's "flag only"
call.

---

## Group 3 — Latency characterization (cell 41) + plot (cell 42)

**Analysis.** Per-config latency mean/median/p95/max, plus n_query. Latency here is
dominated by external LLM API time (retries + grading/ragas sleeps), so tail (p95) and
median are meaningful, the mean is inflated.

**Why the doc:** the metric-choice decision (prefer median/p95 over mean) is the only
way to read the table correctly and avoid ordering configs by an inflated mean.

**Visualization (why):** median (blue bars) with p95 (red markers) overlaid. Shows that
some configs have near-identical medians but wildly different tails — a speed/composite
choice must consider p95, not just median, to survive real load.

---

## Group 4 — Quality & safety profile (cell 44) + plot (cell 45)

**Analysis.** Splits configs into two axes: factual quality (RAGAS + claim-support) and
safety (`low_frac` = share of LOW-risk queries). Keeps `low_frac` separate from the
RAGAS means because a config can look high-quality yet still flag many HIGH-risk
answers. Documents the counter-intuitive observation that reranker=ON carries more HIGH
flags than OFF in this run.

**Why the doc:** it is the reason quality and safety are measured in isolation, and the
reason we verify the risk-label anomaly against RAGAS rather than trusting the pipeline
label.

**Visualization (why):**
- Left: stacked LOW/MED/HIGH risk bars per config — instantly shows the reranker=ON↔
  more-HIGH trend and where it comes from.
- Right: normalized RAGAS means (faithfulness/relevancy/precision/recall) — pure 1.0
  RAGAS values are saturated, so *normalizing* rescues the visualization and lets a
  viewer see which configs actually differ.

---

## 5 — Main effects (cell 47) + plot (cell 48)

**Analysis.** For each lever (reranker, generators, expansion) groups the mean of
latency + quality metrics per level. This isolates which single lever moves each
metric, before reading individual configs.

**Why the doc:** main-effects reads are only correct if the reader knows we are
averaging across all *other* levers; the doc states so and flags "don't over-read any
single config."

**Visualization (why):** a 2×3 grid that pairs the top row (latency — lower better) with
the bottom row (quality — higher better) per lever. Contradictory effects (e.g., a
lever cheap on latency but bad on quality) are seen at once, turning the main-effaction
tables above into a legible trade-off surface.

---

## 6 — Balanced composite + Pareto frontier (cell 50) + plot (cell 51)

**Analysis.** Ranking under your chosen balanced objective:
`rz()` normalizes to [0,1]; `quality = mean(RAGAS/claims)`;
`quality_all = 0.8·quality + 0.2·low_frac`;
`composite = 0.5·latency_norm + 0.5·quality_norm`.
A monotone Pareto scan returns configs not dominated on both axes; a weight sweep
(0.3/0.5/0.7/0.9) tests winner stability.

**Why the doc:** this cell encodes *decision logic* (weights, safety blend, Pareto
algorithm). Without the docstring the numbers are cryptic and the weighting is easy to
mis-trust.

**Visualization (why):**
- **Scatter + Pareto frontier** (median latency × quality): the frontier line makes
  explicit which configs trade latency against quality and which are merely dominated —
  visually the core "best-set" answer.
- **Horizontal composite bar ranking:** the final 0–1 score per configsorted, with
  values labelled, giving a single ordered eyeball of the winner.

---

## 7 — Final recommendation (cell 53)

**Analysis.** Subsets `df` (built in Group 6) to print the fastest, highest-quality, and
balanced-composite picks.

**Why doc:** it forwards decision from all prior groups; the doc records that results
are simple row selects from Group 6's already-sorted frame.

**Visualization:** none — the plots above are the visualization, this cell prints the
verdict text.

---

## Summary — why visualizations were added
| Plot | Purpose |
|------|---------|
| grounding histogram + scatter (cell 39) | Expose the flat/non-discriminative grounding metric |
| latency median vs p‑95 (cell 42) | Compare speed including tail cost |
| risk stacked + RAGAS normalized (cell 45) | Compare safety mix and rescale saturated RAGAS |
| main-effects grid (cell 48) | Trade-off per lever at a glance |
| Pareto scatter + composite ranking (cell 51) | Materialise the "best set" and the final ranking |

Each visualization exists because the underlying tables are dense and the key
takeaway is a **shape/trade-off** that a number alone hides.