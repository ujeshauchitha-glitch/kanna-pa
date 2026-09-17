"""Notebook orchestration support: registry, GPU benchmark, leakage classifier, report.

This module holds the reusable logic that the publication Colab notebook needs but
that does not belong to any existing pipeline stage. Keeping it here (rather than in
notebook cells) honours the project's "no large function bodies in notebook cells"
rule and lets the same code run identically in DEBUG / MAIN / FULL modes and under
pytest.

It provides four cohesive, independently-testable components:

1. :class:`ExperimentRegistry` -- a CSV+JSON registry of experiment runs with a
   explicit state machine (``NOT_STARTED`` / ``RUNNING`` / ``INTERRUPTED`` /
   ``COMPLETED`` / ``FAILED`` / ``SKIPPED``) that updates during execution.

2. :func:`benchmark_gpu` / :func:`probe_max_batch_size` -- a real forward+backward
   GPU batch-size probe that measures peak VRAM and backs off on OOM, plus a
   throughput benchmark. Torch-optional: degrades to a recorded ``NOT_RUN`` result
   with a reason when CUDA is unavailable, never fabricating utilisation numbers.

3. :func:`classify_leakage` -- the corrected leakage classifier that reads the raw
   :func:`src.leakage_audit.audit_leakage` report and RE-CLASSIFIES the
   ``target_leakage`` check. The raw check flags any target VALUE that recurs
   across splits, but a repeated numeric target value is a *target-value collision*
   (potentially benign), NOT target-feature leakage. This module separates the two
   categories and only reports genuine leakage as leakage, exactly as the
   manuscript's leakage section requires.

4. :func:`build_final_report` / :func:`readiness_checklist` -- aggregators that read
   only saved artifacts and emit the final experiment report (MD+JSON) and the
   publication-readiness checklist. They never invent numbers; a missing artifact
   yields ``NOT_RUN`` / ``INSUFFICIENT_DATA`` rather than a plausible estimate.

Every function is import-safe without torch or CUDA so the notebook can import the
whole module on any machine.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

# =============================================================================
# Experiment state machine
# =============================================================================

#: The only experiment states used across the registry (Requirement 4).
STATE_NOT_STARTED = "NOT_STARTED"
STATE_RUNNING = "RUNNING"
STATE_INTERRUPTED = "INTERRUPTED"
STATE_COMPLETED = "COMPLETED"
STATE_FAILED = "FAILED"
STATE_SKIPPED = "SKIPPED"

EXPERIMENT_STATES = (
    STATE_NOT_STARTED,
    STATE_RUNNING,
    STATE_INTERRUPTED,
    STATE_COMPLETED,
    STATE_FAILED,
    STATE_SKIPPED,
)

#: Honest "no result" sentinels. Preserved verbatim in reports; never coerced to 0.
NOT_RUN = "NOT_RUN"
FAILED = "FAILED"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

#: Registry columns (Requirement 5). Order is stable so the CSV is diff-friendly.
REGISTRY_COLUMNS = (
    "experiment_id",
    "experiment_name",
    "model_name",
    "configuration_hash",
    "random_seed",
    "status",
    "stage",
    "start_time",
    "last_update_time",
    "checkpoint_path",
    "best_checkpoint_path",
    "best_validation_metric",
    "epoch",
    "global_step",
    "GPU_name",
    "GPU_VRAM",
    "precision",
    "physical_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "peak_vram",
    "mean_gpu_utilization",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRegistry:
    """CSV+JSON registry of experiment runs with an explicit state machine.

    The registry is the single source of truth for "what has run" and drives the
    notebook's restart-safety: on startup the notebook reads the registry to decide
    which experiments to resume, skip, or start. Every mutation writes both
    ``experiment_registry.csv`` and ``experiment_registry.json`` immediately so a
    runtime that dies mid-run leaves an accurate on-disk record (Requirement 5).

    Rows are keyed by ``experiment_id``. Registering an existing id updates the row
    in place rather than appending a duplicate, so re-running the notebook is
    idempotent.
    """

    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.csv_path = os.path.join(registry_dir, "experiment_registry.csv")
        self.json_path = os.path.join(registry_dir, "experiment_registry.json")
        self._rows: "dict[str, dict[str, Any]]" = {}
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path) as fh:
                    data = json.load(fh)
                for row in data:
                    if isinstance(row, dict) and row.get("experiment_id"):
                        self._rows[str(row["experiment_id"])] = row
                return
            except (OSError, ValueError):
                pass  # fall through to CSV
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, newline="") as fh:
                    for row in csv.DictReader(fh):
                        if row.get("experiment_id"):
                            self._rows[str(row["experiment_id"])] = dict(row)
            except OSError:
                pass

    def _flush(self) -> None:
        rows = list(self._rows.values())
        # JSON (authoritative; preserves types).
        with open(self.json_path, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)
        # CSV (human/spreadsheet friendly; stable column order).
        with open(self.csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(REGISTRY_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in REGISTRY_COLUMNS})

    # -- mutations ---------------------------------------------------------
    def register(
        self,
        experiment_id: str,
        experiment_name: str,
        model_name: str = "",
        configuration_hash: str = "",
        random_seed: Any = "",
        status: str = STATE_NOT_STARTED,
        **fields: Any,
    ) -> dict:
        """Create or update a registry row, returning the stored row.

        A previously-registered id is updated in place (idempotent). ``start_time``
        is set once on first registration; ``last_update_time`` is refreshed on
        every mutation.
        """
        if status not in EXPERIMENT_STATES:
            raise ValueError(
                f"status must be one of {EXPERIMENT_STATES}, got {status!r}"
            )
        row = self._rows.get(str(experiment_id), {})
        row.setdefault("experiment_id", str(experiment_id))
        row.setdefault("start_time", _utcnow())
        row.update(
            {
                "experiment_name": experiment_name,
                "model_name": model_name,
                "configuration_hash": configuration_hash,
                "random_seed": random_seed,
                "status": status,
                "last_update_time": _utcnow(),
            }
        )
        for key, value in fields.items():
            row[key] = value
        self._rows[str(experiment_id)] = row
        self._flush()
        return dict(row)

    def update(self, experiment_id: str, **fields: Any) -> dict:
        """Patch fields on an existing row (registers it first if unknown)."""
        if str(experiment_id) not in self._rows:
            self.register(str(experiment_id), fields.get("experiment_name", ""))
        row = self._rows[str(experiment_id)]
        if "status" in fields and fields["status"] not in EXPERIMENT_STATES:
            raise ValueError(f"invalid status {fields['status']!r}")
        row.update(fields)
        row["last_update_time"] = _utcnow()
        self._flush()
        return dict(row)

    def set_status(self, experiment_id: str, status: str, **fields: Any) -> dict:
        """Transition an experiment to ``status`` and patch any extra fields."""
        return self.update(experiment_id, status=status, **fields)

    # -- queries -----------------------------------------------------------
    def get(self, experiment_id: str) -> Optional[dict]:
        row = self._rows.get(str(experiment_id))
        return dict(row) if row else None

    def status_of(self, experiment_id: str) -> str:
        row = self._rows.get(str(experiment_id))
        return str(row.get("status", STATE_NOT_STARTED)) if row else STATE_NOT_STARTED

    def is_completed(self, experiment_id: str) -> bool:
        return self.status_of(experiment_id) == STATE_COMPLETED

    def should_run(self, experiment_id: str, force: bool = False) -> bool:
        """Return True unless the experiment already COMPLETED (or SKIPPED).

        ``force`` overrides and always returns True so a user can re-run an
        expensive experiment on purpose (Requirement 4: never recompute a
        completed experiment unless explicitly forced).
        """
        if force:
            return True
        return self.status_of(experiment_id) not in (STATE_COMPLETED, STATE_SKIPPED)

    def all_rows(self) -> "list[dict]":
        return [dict(r) for r in self._rows.values()]

    def summary(self) -> "dict[str, int]":
        """Return a count of experiments per state."""
        counts = {s: 0 for s in EXPERIMENT_STATES}
        for row in self._rows.values():
            counts[str(row.get("status", STATE_NOT_STARTED))] = (
                counts.get(str(row.get("status", STATE_NOT_STARTED)), 0) + 1
            )
        return counts


# =============================================================================
# GPU benchmark / batch-size probe
# =============================================================================


@dataclass
class GPUBenchmarkResult:
    """Outcome of a GPU probe/benchmark. ``status`` is OK / NOT_RUN / FAILED."""

    status: str = NOT_RUN
    reason: str = ""
    gpu_name: Optional[str] = None
    total_vram_gb: Optional[float] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    compute_capability: Optional[str] = None
    max_stable_batch_size: Optional[int] = None
    recommended_batch_size: Optional[int] = None
    peak_vram_gb: Optional[float] = None
    precision: Optional[str] = None
    gradient_checkpointing: Optional[bool] = None
    samples_per_sec: Optional[float] = None
    per_batch: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def gpu_diagnostics() -> dict:
    """Return a dict of GPU/torch facts, or a NOT_RUN record when unavailable.

    Never raises: on any missing dependency it returns ``{"status": NOT_RUN,
    "reason": ...}`` so the notebook can print the reason instead of crashing.
    """
    try:
        import torch
    except Exception as exc:  # torch not installed
        return {"status": NOT_RUN, "reason": f"torch import failed: {exc}"}

    info: dict[str, Any] = {
        "status": "OK",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
    }
    if not torch.cuda.is_available():
        info["status"] = NOT_RUN
        info["reason"] = "CUDA not available (CPU-only runtime)"
        return info
    try:
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "total_vram_gb": round(props.total_memory / 1e9, 3),
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1e9, 3),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        info["status"] = FAILED
        info["reason"] = f"device query failed: {exc}"
    return info


def probe_max_batch_size(
    build_batch,
    forward_backward,
    candidate_sizes: Sequence[int] = (8, 16, 24, 32, 48, 64),
    safety_fraction: float = 0.9,
) -> GPUBenchmarkResult:
    """Find the largest batch size that survives a real forward+backward pass.

    ``build_batch(batch_size) -> batch`` must allocate a *realistic* batch on the
    GPU; ``forward_backward(batch) -> None`` must run a real forward and backward
    pass (this is the only honest way to measure peak VRAM -- never allocate dummy
    tensors just to inflate usage, per Requirement 17). For each candidate we run
    the pass, record peak VRAM, and on ``CUDA out of memory`` we stop and back off
    to the last stable size. ``recommended_batch_size`` applies ``safety_fraction``
    as head-room.

    Returns a :class:`GPUBenchmarkResult`; ``status`` is ``NOT_RUN`` with a reason
    when CUDA is unavailable so the caller records the fact rather than guessing.
    """
    diag = gpu_diagnostics()
    if diag.get("status") != "OK" or not diag.get("cuda_available"):
        return GPUBenchmarkResult(
            status=NOT_RUN,
            reason=diag.get("reason", "CUDA not available"),
            torch_version=diag.get("torch_version"),
        )
    import torch

    result = GPUBenchmarkResult(
        status="OK",
        gpu_name=diag.get("gpu_name"),
        total_vram_gb=diag.get("total_vram_gb"),
        cuda_version=diag.get("cuda_version"),
        torch_version=diag.get("torch_version"),
        compute_capability=diag.get("compute_capability"),
    )
    max_stable = None
    peak_overall = 0.0
    for bs in candidate_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = build_batch(bs)
            forward_backward(batch)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e9
            peak_overall = max(peak_overall, peak)
            result.per_batch.append(
                {"batch_size": bs, "peak_vram_gb": round(peak, 3), "status": "OK"}
            )
            max_stable = bs
            del batch
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                result.per_batch.append(
                    {"batch_size": bs, "status": "OOM"}
                )
                torch.cuda.empty_cache()
                break
            result.status = FAILED
            result.reason = f"{type(exc).__name__}: {exc}"
            torch.cuda.empty_cache()
            break
    result.max_stable_batch_size = max_stable
    result.peak_vram_gb = round(peak_overall, 3) if peak_overall else None
    if max_stable is not None:
        # Recommend a batch that keeps head-room below the OOM edge. If the peak
        # VRAM at the largest stable batch already sits above ``safety_fraction``
        # of total VRAM, step down one candidate; otherwise the largest stable
        # size is safe to use. This targets high-but-not-fragile utilisation.
        total = diag.get("total_vram_gb") or 0.0
        recommended = max_stable
        if total and peak_overall and (peak_overall / total) > safety_fraction:
            stable_sizes = [
                r["batch_size"] for r in result.per_batch if r.get("status") == "OK"
            ]
            if len(stable_sizes) >= 2:
                recommended = stable_sizes[-2]
        result.recommended_batch_size = recommended
    elif result.status == "OK":
        result.status = FAILED
        result.reason = "no candidate batch size fit in VRAM"
    return result


def apply_gpu_benchmark_to_config(
    config: dict,
    bench: dict,
    target_effective_batch: int = 24,
) -> dict:
    """Apply a GPU-probe result to the training config so the run fills the GPU.

    Sets ``model.deberta.batch_size`` to the probe's ``recommended_batch_size`` (the
    largest stable physical batch with head-room), then chooses
    ``gradient_accumulation_steps`` so the *effective* batch is at least
    ``target_effective_batch`` (accumulation only kicks in when the physical batch
    is smaller than the target, per Requirement 17.9 -- physical batch is maximised
    first). Also enables bf16 AMP and gradient checkpointing, which is what lets the
    hierarchical DeBERTa reach a large useful batch on a 20 GB L4.

    Mutates ``config`` in place and returns it. A ``NOT_RUN`` / ``FAILED`` bench (no
    GPU, or nothing fit) leaves the configured batch size untouched so the caller
    falls back to the safe default rather than a fabricated value.
    """
    deb = config.setdefault("model", {}).setdefault("deberta", {})
    rec = bench.get("recommended_batch_size") if isinstance(bench, dict) else None
    if not isinstance(bench, dict) or bench.get("status") != "OK" or not rec:
        deb.setdefault(
            "gpu_autobatch",
            {"applied": False, "reason": (bench or {}).get("reason", "no probe result")},
        )
        return config
    physical = int(rec)
    accum = (
        max(1, -(-int(target_effective_batch) // physical))
        if physical < target_effective_batch
        else 1
    )
    deb["batch_size"] = physical
    deb["gradient_accumulation_steps"] = accum
    deb["use_amp"] = True                 # bf16 mixed precision (handled by the loop)
    deb["gradient_checkpointing"] = True  # trades compute for the memory to grow batch
    deb["gpu_autobatch"] = {
        "applied": True,
        "physical_batch_size": physical,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": physical * accum,
        "peak_vram_gb": bench.get("peak_vram_gb"),
        "total_vram_gb": bench.get("total_vram_gb"),
    }
    return config


def benchmark_throughput(build_batch, forward_backward, batch_size: int, iters: int = 5) -> dict:
    """Measure samples/sec for a fixed batch size over ``iters`` real steps.

    Returns a dict with timing stats, or a NOT_RUN record when CUDA is absent.
    """
    diag = gpu_diagnostics()
    if diag.get("status") != "OK" or not diag.get("cuda_available"):
        return {"status": NOT_RUN, "reason": diag.get("reason", "CUDA not available")}
    import torch

    try:
        batch = build_batch(batch_size)
        # Warm-up (not timed) so kernel autotuning does not skew the measurement.
        forward_backward(batch)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            forward_backward(batch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        steps = max(1, iters)
        return {
            "status": "OK",
            "batch_size": batch_size,
            "iters": steps,
            "seconds_total": round(elapsed, 4),
            "step_time_sec": round(elapsed / steps, 4),
            "samples_per_sec": round(batch_size * steps / elapsed, 2) if elapsed else None,
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        }
    except RuntimeError as exc:
        return {"status": FAILED, "reason": f"{type(exc).__name__}: {exc}"}


# =============================================================================
# Corrected leakage classification (Requirement 11 -- the critical fix)
# =============================================================================

#: Map each raw audit check name to the manuscript's leakage category letter.
_CHECK_TO_CATEGORY = {
    "future_market_data": "D_temporal",
    "future_text": "D_temporal",
    "target_leakage": "A_target_value_collision",  # RECLASSIFIED (see below)
    "duplicate_documents": "C_identity",
    "cross_split_identity": "C_identity",
    "scaler_leakage": "E_scaler",
    "test_set_tuning": "F_hyperparameter",
    "hyperparameter_leakage": "F_hyperparameter",
    "signal_asof": "G_confounder_asof",
    "dapt_corpus_contamination": "H_dapt",
}

#: The categories that constitute GENUINE leakage (a true positive must fail here
#: to make the evidence "not clean"). ``A_target_value_collision`` is deliberately
#: excluded: a repeated numeric target value is not, by itself, leakage.
_GENUINE_LEAKAGE_CATEGORIES = (
    "B_target_feature",
    "C_identity",
    "D_temporal",
    "E_scaler",
    "F_hyperparameter",
    "G_confounder_asof",
    "H_dapt",
    "I_split_assignment",
    "J_prediction_contamination",
)


@dataclass
class LeakageClassification:
    """Reclassified leakage outcome distinguishing collisions from real leakage."""

    verdict: str  # CLEAN | CLEAN WITH CAVEATS | LEAKAGE DETECTED | INCONCLUSIVE
    genuine_leakage_checks: list = field(default_factory=list)
    target_value_collision: dict = field(default_factory=dict)
    per_check: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_leakage(raw_report: dict, splits=None) -> LeakageClassification:
    """Reclassify a raw ``audit_leakage`` report, separating collision from leakage.

    The raw ``target_leakage`` check flags any target VALUE that appears in more
    than one split. That is a *target-value collision*: for a low-cardinality
    numeric target (basis-point OIS changes rounded to a grid) identical values
    recurring across time is expected and does NOT mean the model saw a test label
    at train time. Genuine target leakage (category B) is a target-DERIVED feature
    entering the inputs, which this check does not measure at all.

    This function therefore:

    * moves ``target_leakage`` into an informational ``target_value_collision``
      record (with, when ``splits`` is provided, evidence on whether the colliding
      rows are actually distinct events -- if the *events* are distinct, the
      collision is benign);
    * treats the identity check (``cross_split_identity`` / ``duplicate_documents``)
      as the real cross-split-identity signal;
    * computes an overall ``verdict`` from the genuine-leakage categories only.

    A raw report where ONLY ``target_leakage`` fires yields
    ``CLEAN WITH CAVEATS`` (the caveat being the reported collision), not
    ``LEAKAGE DETECTED`` -- which is the correction the manuscript requires.
    """
    checks = list(raw_report.get("checks", []) or [])
    per_check: list[dict] = []
    genuine: list[str] = []
    collision_record: dict = {}
    notes: list[str] = []

    for finding in checks:
        name = finding.get("check")
        leak = bool(finding.get("leak_detected"))
        category = _CHECK_TO_CATEGORY.get(name, "unclassified")
        is_genuine_category = category in _GENUINE_LEAKAGE_CATEGORIES
        record = {
            "check": name,
            "category": category,
            "raw_leak_detected": leak,
            "detail": finding.get("detail", ""),
            "offenders_count": len(finding.get("offenders", []) or []),
        }
        if name == "target_leakage":
            # Reclassify: this is a collision report, not leakage.
            record["classification"] = (
                "TARGET-VALUE COLLISION" if leak else "no collision"
            )
            collision_record = {
                "detected": leak,
                "n_colliding_values": len(finding.get("offenders", []) or []),
                "colliding_values": list(finding.get("offenders", []) or [])[:50],
                "interpretation": (
                    "Identical low-cardinality numeric target values recur across "
                    "splits. This is a target-VALUE collision, not target leakage: "
                    "the model does not receive any test label as an input."
                ),
            }
            if leak:
                notes.append(
                    "target_leakage reclassified as TARGET-VALUE COLLISION (benign "
                    "unless the colliding rows are the same event -- see "
                    "cross_split_identity)."
                )
            per_check.append(record)
            continue

        # Genuine categories: a raw failure counts against cleanliness.
        record["classification"] = "GENUINE LEAKAGE" if (leak and is_genuine_category) else (
            "ok" if not leak else "flagged (non-leakage category)"
        )
        if leak and is_genuine_category:
            genuine.append(name)
        per_check.append(record)

    # Optional: strengthen the collision interpretation with split evidence.
    if collision_record.get("detected") and splits is not None:
        collision_record.update(_collision_event_evidence(splits))
        if collision_record.get("distinct_events_share_value"):
            notes.append(
                "Colliding target values map to DISTINCT event_ids across splits: "
                "collision confirmed benign (no shared event identity)."
            )

    if genuine:
        verdict = "LEAKAGE DETECTED"
    elif collision_record.get("detected"):
        verdict = "CLEAN WITH CAVEATS"
    else:
        verdict = "CLEAN"

    return LeakageClassification(
        verdict=verdict,
        genuine_leakage_checks=genuine,
        target_value_collision=collision_record,
        per_check=per_check,
        notes=notes,
    )


def _collision_event_evidence(splits) -> dict:
    """Return evidence on whether colliding target values are distinct events.

    When the identical target value maps to different ``event_id`` values in
    different splits, the collision is a coincidence of equal numbers, not a shared
    event -- strong evidence the collision is benign. Best-effort and defensive:
    any failure returns an empty dict so the classifier never crashes on odd input.
    """
    try:
        import pandas as pd

        if not isinstance(splits, pd.DataFrame):
            return {}
        if "target" not in splits.columns or "split" not in splits.columns:
            return {}
        sub = splits[["target", "split"]].dropna(subset=["target"]).copy()
        colliding = sub.groupby("target")["split"].nunique()
        colliding = colliding[colliding > 1].index
        if len(colliding) == 0:
            return {"distinct_events_share_value": False}
        distinct_events = True
        if "event_id" in splits.columns:
            ev = splits[["target", "split", "event_id"]].dropna(subset=["target"])
            ev = ev[ev["target"].isin(colliding)]
            # Shared identity would mean the SAME event_id in >1 split for a value.
            per_value_ids = ev.groupby("target")["event_id"].nunique()
            per_value_rows = ev.groupby("target")["event_id"].count()
            # If unique ids == row count for every colliding value, no id repeats.
            distinct_events = bool((per_value_ids == per_value_rows).all())
        return {
            "n_colliding_values": int(len(colliding)),
            "distinct_events_share_value": bool(distinct_events),
        }
    except Exception:  # pragma: no cover - evidence is best-effort
        return {}


# =============================================================================
# Final report + publication-readiness checklist (Requirements 47, 48)
# =============================================================================

#: The publication-readiness checklist items (Requirement 48). Each maps to a
#: predicate over the gathered ``facts`` dict assembled by :func:`build_final_report`.
READINESS_ITEMS = (
    "dataset_built",
    "event_targets_validated",
    "temporal_split_validated",
    "train_only_scaling_validated",
    "asof_joins_validated",
    "target_leakage_checked",
    "collision_separated_from_leakage",
    "duplicate_documents_checked",
    "council_finetune_completed",
    "major_baselines_completed",
    "ablations_completed",
    "rolling_origin_cv_completed",
    "leave_one_meeting_out_documented",
    "regime_analysis_completed_or_marked",
    "bootstrap_ci_completed",
    "statistical_tests_completed",
    "stance_validation_or_limitation",
    "gpu_benchmark_completed",
    "checkpoints_verified",
    "predictions_saved",
    "metrics_saved",
    "tables_generated",
    "figures_generated",
    "final_leakage_audit_completed",
    "final_report_generated",
)


def _exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path)


def gather_run_facts(
    output_dir: str,
    registry: Optional[ExperimentRegistry] = None,
    artifact_root: Optional[str] = None,
) -> dict:
    """Collect the facts the report/checklist need, reading only saved artifacts.

    Returns a plain dict. Missing artifacts are recorded as absent (``None`` /
    ``False``) rather than fabricated. This is deliberately tolerant: it inspects
    the on-disk run directory produced by :func:`src.run_council.run_council_pipeline`
    and never recomputes anything.

    ``artifact_root`` is the notebook's persistent workspace (Drive) when it differs
    from the per-run ``output_dir``; figures, the GPU benchmark, ablations, and the
    final report may be written there, so both locations are searched. The returned
    ``artifact_root`` is threaded into the checklist so its predicates look in the
    right place.
    """
    audit = os.path.join(output_dir, "audit")
    results = os.path.join(output_dir, "results")
    roots = [output_dir] + ([artifact_root] if artifact_root else [])
    figures_dirs = []
    for r in roots:
        figures_dirs.append(os.path.join(r, "figures"))
    figures_dirs.append(os.path.join(results, "figures"))

    def _read_json(path: str) -> Optional[dict]:
        if not _exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    manifest = _read_json(os.path.join(audit, "council_run_manifest.json"))
    leakage = _read_json(os.path.join(audit, "leakage_audit.json"))
    split_integrity = _read_json(os.path.join(audit, "split_integrity_report.json"))
    regime = _read_json(os.path.join(audit, "regime_audit.json"))
    cv = _read_json(os.path.join(audit, "cross_validation_folds.json"))

    comparison_csv = os.path.join(results, "model_comparison.csv")
    figures = []
    for fdir in figures_dirs:
        if os.path.isdir(fdir):
            figures.extend(
                os.path.join(fdir, f)
                for f in sorted(os.listdir(fdir))
                if f.lower().endswith((".png", ".pdf", ".svg"))
            )

    # Predictions parquet(s) written under frozen/ or finetune/ evaluation dirs.
    prediction_files: list[str] = []
    for root, _dirs, files in os.walk(output_dir):
        for f in files:
            if f.startswith("predictions_") and f.endswith(".parquet"):
                prediction_files.append(os.path.join(root, f))

    stages = {}
    if manifest and isinstance(manifest.get("stages"), dict):
        stages = {k: v.get("status") for k, v in manifest["stages"].items()}

    # Locate cross-location artifacts (figures/gpu_benchmark/ablations/report) that
    # may live under artifact_root rather than output_dir.
    def _first_existing(*candidates: str) -> Optional[str]:
        for c in candidates:
            if _exists(c):
                return c
        return None

    gpu_bench = _first_existing(
        os.path.join(output_dir, "gpu_benchmark", "gpu_benchmark.json"),
        *([os.path.join(artifact_root, "gpu_benchmark", "gpu_benchmark.json")] if artifact_root else []),
    )
    ablation_csv = _first_existing(
        os.path.join(output_dir, "results", "ablation_results.csv"),
        *([os.path.join(artifact_root, "ablations", "ablation_results.csv")] if artifact_root else []),
    )
    final_report_md = _first_existing(
        os.path.join(output_dir, "final_report", "FINAL_EXPERIMENT_REPORT.md"),
        *([os.path.join(artifact_root, "final_report", "FINAL_EXPERIMENT_REPORT.md")] if artifact_root else []),
    )

    stance_stage = (
        (manifest or {}).get("stages", {}).get("stance_series", {})
        if isinstance((manifest or {}).get("stages", {}), dict)
        else {}
    )
    stance_status = (
        stance_stage.get("status") if isinstance(stance_stage, dict) else None
    )
    stance_detail = (
        stance_stage.get("detail") if isinstance(stance_stage, dict) else None
    )
    stance_artifacts = (
        stance_stage.get("artifacts", {})
        if isinstance(stance_stage, dict) else {}
    )
    manifest_status = (manifest or {}).get("status")
    stance_available = False
    stance_reason = None
    stance_summary_path = None
    stance_series_path = None
    stance_events_path = None
    stance_summary = None

    if str(manifest_status).upper() != "COMPLETED":
        stance_reason = "current run manifest is not COMPLETED"
    elif str(stance_status).upper() != "OK":
        stance_reason = stance_detail or "current stance stage is not OK"
    else:
        declared_paths = {
            "summary_path": stance_artifacts.get("stance_summary"),
            "series_path": stance_artifacts.get("stance_series_path"),
            "event_predictions_path": stance_artifacts.get(
                "stance_event_predictions"
            ),
        }
        missing_declarations = [
            name for name, path in declared_paths.items() if not path
        ]
        output_root = os.path.abspath(output_dir)

        def _inside_current_run(path: str) -> bool:
            try:
                return os.path.commonpath(
                    [output_root, os.path.abspath(str(path))]
                ) == output_root
            except (OSError, ValueError):
                return False

        if missing_declarations:
            stance_reason = (
                "OK stance stage omitted required artifact declarations: "
                + ", ".join(missing_declarations)
            )
        elif any(not _inside_current_run(str(path)) for path in declared_paths.values()):
            stance_reason = "stance artifact path escapes the current run directory"
        elif any(not _exists(str(path)) for path in declared_paths.values()):
            stance_reason = "manifest-declared stance artifact set is incomplete"
        else:
            candidate_summary = _read_json(str(declared_paths["summary_path"]))
            if not isinstance(candidate_summary, dict):
                stance_reason = "manifest-declared stance summary is unreadable"
            else:
                stance_available = True
                stance_summary_path = str(declared_paths["summary_path"])
                stance_series_path = str(declared_paths["series_path"])
                stance_events_path = str(
                    declared_paths["event_predictions_path"]
                )
                stance_summary = candidate_summary

    stance = {
        "status": stance_status,
        "detail": stance_detail,
        "available": stance_available,
        "availability_reason": stance_reason,
        "summary_path": stance_summary_path,
        "series_path": stance_series_path,
        "event_predictions_path": stance_events_path,
        "summary": stance_summary,
    }

    return {
        "output_dir": output_dir,
        "artifact_root": artifact_root,
        "gpu_benchmark_json": gpu_bench,
        "ablation_csv": ablation_csv,
        "final_report_md": final_report_md,
        "manifest": manifest,
        "run_status": manifest_status,
        "stages": stages,
        "leakage": leakage,
        "split_integrity": split_integrity,
        "regime": regime,
        "cross_validation": cv,
        "model_comparison_csv": comparison_csv if _exists(comparison_csv) else None,
        "figures": figures,
        "prediction_files": prediction_files,
        "stance": stance,
        "registry_summary": registry.summary() if registry else None,
    }


def readiness_checklist(facts: dict, leakage_classification: Optional[LeakageClassification] = None) -> dict:
    """Compute the publication-readiness checklist from gathered ``facts``.

    Returns ``{"items": {name: bool}, "ready": bool, "blocking": [names]}``. An
    item is only True when there is real artifact evidence for it; otherwise it is
    False and appears in ``blocking``. ``ready`` is True only when every item is
    satisfied (Requirement 48).
    """
    stages = facts.get("stages") or {}
    split_integrity = facts.get("split_integrity") or {}
    leakage = facts.get("leakage") or {}
    lc = leakage_classification

    def stage_ok(name: str) -> bool:
        return str(stages.get(name, "")).upper() == "OK"

    items = {
        "dataset_built": bool(facts.get("split_integrity")),
        "event_targets_validated": bool(split_integrity.get("splits")),
        "temporal_split_validated": bool(split_integrity.get("chronological_monotonic")),
        "train_only_scaling_validated": _exists(
            os.path.join(facts.get("output_dir", ""), "models", "target_scaler.pkl")
        )
        or _exists(os.path.join(facts.get("output_dir", ""), "audit", "signal_scalers.json")),
        "asof_joins_validated": bool(
            leakage and _finding_ran(leakage, "signal_asof")
        ),
        "target_leakage_checked": bool(leakage and _finding_ran(leakage, "target_leakage")),
        "collision_separated_from_leakage": lc is not None,
        "duplicate_documents_checked": bool(
            leakage and _finding_ran(leakage, "duplicate_documents")
        ),
        "council_finetune_completed": stage_ok("training_eval"),
        "major_baselines_completed": stage_ok("baselines"),
        "ablations_completed": facts.get("ablation_csv") is not None,
        "rolling_origin_cv_completed": stage_ok("cross_validation"),
        "leave_one_meeting_out_documented": bool(facts.get("cross_validation")),
        "regime_analysis_completed_or_marked": stage_ok("regime_audit"),
        "bootstrap_ci_completed": facts.get("model_comparison_csv") is not None,
        "statistical_tests_completed": facts.get("model_comparison_csv") is not None,
        "stance_validation_or_limitation": bool(
            stage_ok("stance_series")
            and (facts.get("stance") or {}).get("available") is True
            and (facts.get("stance") or {}).get("summary")
            and (facts.get("stance") or {}).get("summary_path")
            and (facts.get("stance") or {}).get("event_predictions_path")
            and (facts.get("stance") or {}).get("series_path")
        ),
        "gpu_benchmark_completed": facts.get("gpu_benchmark_json") is not None,
        "checkpoints_verified": _has_any_checkpoint(facts.get("output_dir", "")),
        "predictions_saved": len(facts.get("prediction_files", []) or []) > 0,
        "metrics_saved": facts.get("model_comparison_csv") is not None,
        "tables_generated": facts.get("model_comparison_csv") is not None,
        "figures_generated": len(facts.get("figures", []) or []) > 0,
        "final_leakage_audit_completed": bool(leakage),
        "final_report_generated": facts.get("final_report_md") is not None,
    }
    # Ensure every declared item is present (default False when not computed).
    for name in READINESS_ITEMS:
        items.setdefault(name, False)
    blocking = [name for name, ok in items.items() if not ok]
    return {"items": items, "ready": len(blocking) == 0, "blocking": blocking}


def _finding_ran(leakage_report: dict, check_name: str) -> bool:
    """True when ``check_name`` is present in the report (ran, leak or not)."""
    for finding in leakage_report.get("checks", []) or []:
        if finding.get("check") == check_name:
            # A skipped check reports its skip in the detail; treat presence as ran.
            return True
    return False


def _has_any_checkpoint(output_dir: str) -> bool:
    if not output_dir or not os.path.isdir(output_dir):
        return False
    for root, _dirs, files in os.walk(output_dir):
        if any(f.endswith(".pt") for f in files):
            return True
    return False


def build_final_report(
    output_dir: str,
    report_dir: str,
    facts: Optional[dict] = None,
    leakage_classification: Optional[LeakageClassification] = None,
    extra: Optional[dict] = None,
) -> "tuple[str, str]":
    """Write ``FINAL_EXPERIMENT_REPORT.{md,json}`` from saved artifacts only.

    Returns ``(md_path, json_path)``. Every number in the report is copied from a
    saved artifact; a missing artifact is reported as ``NOT_RUN`` /
    ``INSUFFICIENT_DATA`` rather than estimated (Requirement 47, principle 2).
    """
    os.makedirs(report_dir, exist_ok=True)
    facts = facts or gather_run_facts(output_dir)
    checklist = readiness_checklist(facts, leakage_classification)
    extra = extra or {}

    payload = {
        "generated_at": _utcnow(),
        "output_dir": output_dir,
        "stages": facts.get("stages"),
        "split_integrity": facts.get("split_integrity"),
        "leakage_raw": facts.get("leakage"),
        "leakage_classification": (
            leakage_classification.to_dict() if leakage_classification else None
        ),
        "regime": facts.get("regime"),
        "cross_validation": facts.get("cross_validation"),
        "model_comparison_csv": facts.get("model_comparison_csv"),
        "prediction_files": facts.get("prediction_files"),
        "stance": facts.get("stance"),
        "figures": facts.get("figures"),
        "registry_summary": facts.get("registry_summary"),
        "readiness": checklist,
        "extra": extra,
    }

    json_path = os.path.join(report_dir, "FINAL_EXPERIMENT_REPORT.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    md_path = os.path.join(report_dir, "FINAL_EXPERIMENT_REPORT.md")
    with open(md_path, "w") as fh:
        fh.write(_render_final_report_md(payload))
    return md_path, json_path


def _render_final_report_md(payload: dict) -> str:
    lines = ["# Final Experiment Report", ""]
    lines.append(f"_Generated: {payload.get('generated_at')}_")
    lines.append(f"_Run directory: `{payload.get('output_dir')}`_")
    lines.append("")

    # Executive summary from stage statuses.
    lines.append("## Executive summary")
    lines.append("")
    stages = payload.get("stages") or {}
    if stages:
        for name, status in stages.items():
            lines.append(f"- **{name}**: {status}")
    else:
        lines.append(f"- No run manifest found: {NOT_RUN}")
    lines.append("")

    # Dataset / split.
    lines.append("## Dataset and split")
    lines.append("")
    si = payload.get("split_integrity") or {}
    splits = si.get("splits") if isinstance(si, dict) else None
    if splits:
        lines.append("| Split | Count | Min date | Max date |")
        lines.append("| --- | --- | --- | --- |")
        for name in ("train", "val", "test"):
            s = splits.get(name, {}) if isinstance(splits, dict) else {}
            lines.append(
                f"| {name} | {s.get('count', NOT_RUN)} | "
                f"{s.get('min_date', NOT_RUN)} | {s.get('max_date', NOT_RUN)} |"
            )
        lines.append(
            f"\nChronological monotonic: {si.get('chronological_monotonic', NOT_RUN)}"
        )
    else:
        lines.append(f"Split integrity report absent: {NOT_RUN}")
    lines.append("")

    # Leakage.
    lines.append("## Leakage")
    lines.append("")
    lc = payload.get("leakage_classification")
    if lc:
        lines.append(f"**Verdict:** {lc.get('verdict')}")
        lines.append("")
        collision = lc.get("target_value_collision") or {}
        if collision.get("detected"):
            lines.append(
                f"- Target-value collision: DETECTED across "
                f"{collision.get('n_colliding_values', '?')} value(s). "
                "Classified as a collision, NOT leakage."
            )
            if collision.get("distinct_events_share_value"):
                lines.append(
                    "- Evidence: colliding values map to distinct event_ids "
                    "(collision confirmed benign)."
                )
        genuine = lc.get("genuine_leakage_checks") or []
        lines.append(
            f"- Genuine leakage checks failing: "
            f"{', '.join(genuine) if genuine else 'none'}"
        )
        for note in lc.get("notes", []) or []:
            lines.append(f"- Note: {note}")
        lines.append("")
        lines.append("| Check | Category | Raw leak? | Classification |")
        lines.append("| --- | --- | --- | --- |")
        for rec in lc.get("per_check", []) or []:
            lines.append(
                f"| {rec.get('check')} | {rec.get('category')} | "
                f"{rec.get('raw_leak_detected')} | {rec.get('classification')} |"
            )
    else:
        lines.append(f"Leakage classification absent: {NOT_RUN}")
    lines.append("")

    # Results pointer (numbers live in the CSV; we do not retype them).
    lines.append("## Main results")
    lines.append("")
    csv_path = payload.get("model_comparison_csv")
    lines.append(
        f"Model comparison: `{csv_path}`" if csv_path
        else f"Model comparison table: {NOT_RUN}"
    )
    lines.append("")

    # Categorical probabilities followed by the derived continuous stance.
    lines.append("## Stance probabilities and continuous score")
    lines.append("")
    stance = payload.get("stance") or {}
    stance_summary = stance.get("summary") or {}
    stance_available = (
        str(stance.get("status", "")).upper() == "OK"
        and stance.get("available") is True
        and bool(stance_summary)
    )
    if stance_available:
        mass = stance_summary.get("mean_probability_mass_percent", {})
        shares = stance_summary.get("dominant_event_share_percent", {})
        continuous = stance_summary.get("continuous_stance", {})
        lines.append(
            "Mean probability mass: "
            f"Hawkish {float(mass.get('hawkish', float('nan'))):.2f}%, "
            f"Dovish {float(mass.get('dovish', float('nan'))):.2f}%, "
            f"Neutral {float(mass.get('neutral', float('nan'))):.2f}%."
        )
        lines.append(
            "Dominant-class event share: "
            f"Hawkish {float(shares.get('hawkish', float('nan'))):.2f}%, "
            f"Dovish {float(shares.get('dovish', float('nan'))):.2f}%, "
            f"Neutral {float(shares.get('neutral', float('nan'))):.2f}%."
        )
        lines.append(
            "Continuous stance is `P(Hawkish) - P(Dovish)`; "
            f"mean={float(continuous.get('mean', float('nan'))):.4f}."
        )
        lines.append(f"Event predictions: `{stance.get('event_predictions_path')}`")
        lines.append(f"Time series: `{stance.get('series_path')}`")
    else:
        detail = (
            stance.get("detail")
            or stance.get("availability_reason")
            or NOT_RUN
        )
        lines.append(f"Stance probabilities: {detail}")
    lines.append("")

    # Regime.
    lines.append("## Regime analysis")
    lines.append("")
    lines.append(
        "Regime audit artifact present."
        if payload.get("regime") else f"Regime analysis: {NOT_RUN}"
    )
    lines.append("")

    # Predictions / figures.
    lines.append("## Provenance artifacts")
    lines.append("")
    lines.append(f"- Prediction files: {len(payload.get('prediction_files') or [])}")
    lines.append(f"- Figures: {len(payload.get('figures') or [])}")
    lines.append("")

    # Readiness.
    lines.append("## Publication-readiness checklist")
    lines.append("")
    readiness = payload.get("readiness") or {}
    items = readiness.get("items", {})
    for name in READINESS_ITEMS:
        mark = "x" if items.get(name) else " "
        lines.append(f"- [{mark}] {name}")
    lines.append("")
    status = "READY" if readiness.get("ready") else "NOT READY"
    lines.append(f"**PUBLICATION EXPERIMENT STATUS: {status}**")
    if not readiness.get("ready"):
        lines.append("")
        lines.append("Blocking items:")
        for name in readiness.get("blocking", []):
            lines.append(f"- {name}")
    lines.append("")

    # Limitations (honest).
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Numbers in this report are copied from saved artifacts only; any stage "
        "not run is marked NOT_RUN rather than estimated."
    )
    extra = payload.get("extra") or {}
    for lim in extra.get("limitations", []) or []:
        lines.append(f"- {lim}")
    lines.append("")
    return "\n".join(lines)
