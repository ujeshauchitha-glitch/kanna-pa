# Full Run Notebook — `full_run.ipynb`

This is the **canonical end-to-end run** of the model and evaluation. It runs the same
tested `src/` pipeline as the publication notebook with an explicit FP32 AdamW numerical
contract, plus cells for per-signal results, ablations, per-fold cross-validation, and
held-out Dovish/Neutral/Hawkish stance reporting.

## How to run (Colab, NVIDIA L4 20 GB)

1. Upload the repo to Drive at `/content/drive/MyDrive/market_supervised` (the committed
   layout: `configs/`, `src/`, `datasets/`, `notebooks/`). Run outputs land under
   `/content/drive/MyDrive/market_supervised_ecb`.
2. Open `notebooks/run/full_run.ipynb`.
3. Select `Runtime > Change runtime type > L4 GPU`.
4. In cell **[4]**, set `MODE = "MAIN"` (real DeBERTa-v3, no FinBERT) or `"FULL"`
   (adds FinBERT and every baseline; it needs network access or a warm Hugging Face cache).
5. Select `Runtime > Run all`. The data foundation, fine-tune, per-signal runs, and
   ablations are the long stages.

The committed configuration and notebook both keep:

```yaml
runtime:
  run_per_signal: true
  run_ablations: true
```

The notebook is orchestration-only: heavy work calls tested `src/` functions. Nothing is
fabricated; an unavailable stage records `NOT_RUN`, `SKIPPED`, or `INSUFFICIENT_SAMPLE`.

## Live console tracing (not a black box)

Execution is fully traced to the console. Cell **[3]** installs one flushing stdout
handler on the **root** logger, so every `src/` module streams its progress live:

- `pipeline.training` — per-epoch `train_loss` / `val_metric`, plus a per-`N`-step
  heartbeat (`progress_every_steps`, set in cell [4]) so a long fine-tune shows movement
  instead of looking hung;
- `src.run_council` — stage-by-stage orchestration transitions;
- `pipeline.market_data`, `pipeline.data_io`, `pipeline.checkpoint` — data assembly,
  I/O, and checkpoint saves.

Every heavy cell (data foundation, `run_council_pipeline`, per-signal, ablations,
cross-validation, final report) is wrapped in a `banner(...)` context manager that prints
a labelled `>>> START` / `<<< DONE` boundary with elapsed time, and emits `step(...)`
notes for each sub-operation. The final cell prints total wall time. So at any moment you
can see which stage is running, what it just did, and how long it took.

## Heavy run — CLI launch command

The heavy training and evaluation run is separate from the repository test suite and
requires the GPU described above. To launch it directly, run:

```bash
python -m src.run_council --config configs/default.yaml --output-dir "$OUTPUT_DIR"
```

Set `OUTPUT_DIR` to the run output root (the same location printed by cell [6]), for
example:

```bash
export OUTPUT_DIR=/content/drive/MyDrive/market_supervised_ecb
```

The pipeline preserves each event identifier and true date end to end. Prediction
parquets contain `event_id` and `event_date`, and `_load_primary_predictions` returns the
true dates rather than substituting row ordinals, so regime assignments remain aligned.

## Why the earlier run diverged, and what changed

The failed run reached its first accumulated optimizer boundary with a finite loss and
finite gradients. The first `AdamW.step()` then made parameters non-finite. Learning-rate
backoff continued down to approximately `7.8e-8` but did not repair the failure.

A minimal reproducer established the cause: the model parameters and AdamW moments were
FP16, while the default `eps=1e-8` underflowed at that precision. For a zero-gradient
coordinate, the effective denominator could become zero and the update could write NaN.
AMP/autocast configuration affects operation precision; it does not automatically promote
stored parameters or optimizer moments.

The primary fix is implemented in supervised training and DAPT:

| Optimizer contract | Previous behavior | Current behavior |
|---|---|---|
| Parameter storage before AdamW construction | Could remain FP16 from model loading | Explicit FP32 |
| AdamW moments | Followed parameter dtype | Audited FP32 |
| `optimizer.eps` | AdamW default `1e-8` | `1e-6` |
| `optimizer.foreach` / `optimizer.fused` | Backend-selected | Both `false` (scalar path) |
| First landed step | No moment-dtype invariant | Finite parameter/moment audit |
| Corruption after `step()` | LR retry could reuse poisoned state | Fail closed with diagnostics |

The notebook repeats that contract in cell **[4]**. Epoch 0 computes in FP32; later BF16
autocast may reduce operation precision while parameters and Adam moments remain FP32.
The run no longer deep-copies the full DeBERTa model and optimizer every step. An opt-in
resume attempt uses one temporary **CPU** copy of the fresh model only while validating a
matching checkpoint; if model, AdamW, or scheduler validation fails, the caller-visible
model is restored and a brand-new optimizer/scheduler pair is created before training.

Cell [4] also retains lower head LR, lower uncertainty weight, tighter gradient clipping,
a longer minimum chunk, and a tighter log-variance range as **secondary mitigations**.
Those settings are not claimed as the demonstrated root cause. If a failure recurs, inspect:

- `finetune/audit/training_nan_debug.jsonl`
- `finetune/audit/dapt_health.json`

Do not interpret LR reduction alone as a repair for non-finite optimizer state.

## Categorical and continuous stance outputs

The fine-tuned Council uses the shared market-supervised DeBERTa representation to predict
three probabilities in the fixed order **Dovish, Neutral, Hawkish**. The canonical
continuous stance is derived from that distribution:

```text
probability_score = P(Hawkish) - P(Dovish)    # dimensionless, range [-1, 1]
```

In `StanceInferencer`, the backward-compatible `stance_score`/`stance_scaled` fields
remain the natural-unit/standardized **market prediction** pair. The categorical-derived
continuous value is exposed separately as `probability_score`, and `stance_dominant`
contains the Dovish/Neutral/Hawkish argmax. This prevents a probability difference from
being misread as basis points or returns. In the dedicated held-out stance artifacts,
where the schema is explicitly categorical, the same probability difference remains the
`stance_score` column for the published stance time series.

The legacy independent scalar head remains only as a diagnostic parameter block; it is
not the canonical stance output. A full historical one-logit Council checkpoint cannot
represent a calibrated Neutral class and is explicitly rejected by ordinary inference
rather than silently reshaped. Use a newly trained three-way checkpoint (or an explicitly
versioned legacy architecture for historical analysis).

The published stance artifacts are scoped to held-out test events from a successful
fine-tune. The pipeline never substitutes an untrained frozen direction head. Each run
first invalidates prior canonical stance files, publishes the three outputs as one
validated set, and exposes them only when the current atomic manifest is `COMPLETED` and
its `stance_series` stage is `OK`; skipped/error reruns therefore cannot display stale
percentages.

Cell **[9a]** reads both the JSON summary and event parquet, verifies that every probability
row sums to one and that every score satisfies the formula, then prints:

- mean Hawkish/Dovish/Neutral probability mass;
- Hawkish/Dovish/Neutral dominant-class event shares (a distinct statistic);
- mean, median, standard deviation, and range of continuous stance;
- an event-level probability and score preview.

If the required fine-tune artifacts are absent, the cell prints `NOT_RUN` honestly.
Artifacts are:

| Scope | Artifact |
|---|---|
| Fine-tune evaluation | `finetune/evaluation/stance_event_predictions.parquet` |
| Fine-tune evaluation | `finetune/evaluation/stance_summary.json` |
| Published event predictions | `results/stance_event_predictions.parquet` |
| Published time series | `results/stance_series.csv` |
| Published summary | `results/stance_summary.json` |

## Output → paper table map

Paths are relative to the run's `OUTPUT_DIR` (printed in cell [6]) or `ARTIFACT_ROOT`.

| Output / paper table (`paper/sections/results.tex`) | Cell | Artifact |
|---|---:|---|
| Held-out categorical + continuous stance diagnostics | [9a] | `results/stance_summary.json`, `results/stance_event_predictions.parquet`, `results/stance_series.csv` |
| `tab:coverage` — mean tokens / segments per event | [12d] | `tables/corpus_stats.csv` |
| `tab:main` — **Council (fine-tuned)** row | [9] | `results/model_comparison.csv` (row `council_finetune`) |
| `tab:main` — Flat DeBERTa row (full metrics) | [12c] | `results/model_comparison_dm.csv` (row `flat_deberta`) |
| `tab:main` — DM column (vs Council) | [12c] | `results/model_comparison_dm.csv` (`dm_vs_council`) |
| `tab:ablation` — removed-objective + flat rows | [11] | `results/ablation_results.csv` |
| `tab:ablation` — fixed fusion, − stance, − DAPT | [12b] | `results/ablation_results.csv` (appended) |
| `tab:persignal` — per-signal R²/dir/weight | [10] | `per_signal/per_signal_metrics.csv` |
| `tab:cv` — rolling-origin + LOMO mean±std | [12] | `cross_validation/cv_summary.json` |
| `tab:regime`, `tab:leak` | [9], [13] | Produced by the base pipeline |
| **All pending rows, pre-formatted as LaTeX** | [15] | Printed to cell output |

Cell **[15]** prints every pending table row already formatted (for example,
`\textbf{Council (fine-tuned)} & $R^2$ & ... \\`). A missing value prints `\pending`, so
nothing is fabricated. Values come from the existing `evaluate_model`,
`diebold_mariano_test`, and `CrossValidator.aggregate` paths.

**DM limitation:** `compare_models` does not persist every baseline's per-event prediction,
so the DM p-value is computed only for models with saved test predictions
(`council_frozen`, `council_finetune`, `flat_deberta`). Sklearn, lexicon, or FinBERT DM
values remain `\pending` unless their per-event predictions are persisted.

## Notes and honest caveats

- The GPU auto-batch probe uses the production Council loss dispatcher, but routes its
  synthetic presence-audit writes to an automatically deleted temporary directory; probe
  rows can never become experiment evidence. It also preserves the resolved
  `gradient_checkpointing: false` stability setting on success and fallback.
- **Per-signal (cell [10])** and **per-fold CV (cell [12])** call the same tested training
  and aggregation functions per signal/fold. The CV cell trains the frozen Council per
  fold; `aggregate` requires at least two usable folds with at least `min_fold_events=5`
  evaluation events, otherwise it records a reason instead of a number.
- `fx` has zero coverage in the linked data, so its per-signal row is
  `INSUFFICIENT_SAMPLE` by design.
- Results are single-seed (42); multi-seed variance is not characterized.
- `notebooks/run/build_full_run.py` is the source of truth. Re-run it after any notebook
  cell change; the generated notebook intentionally contains no execution counts or
  outputs.
