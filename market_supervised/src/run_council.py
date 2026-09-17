"""Config-driven end-to-end orchestrator for the Council of Supervision (Task 21.1).

This module wires the individual Council stages -- each of which already exists in
its own module -- into a single pipeline that runs **entirely from the resolved
configuration file with no manual code edits** (Requirement 13.5). Every stage
parameter is read from the config; nothing is hard-coded per experiment.

Stage chain (design "Data flow" + stage sections):

    DAPT
      -> SignalNormalizer            (fit inside build_dataloaders, train-only)
      -> training                    (frozen headline + fine-tune ablation)
      -> Evaluation                  (compare_models rows per mode)
      -> Cross-Validation            (rolling-origin / LOMO aggregate)
      -> Regime audit                (ZLB vs post-hiking)
      -> Leakage audit               (signal as-of + DAPT contamination + ...)
      -> Baselines                   (lexicon / ridge / RF / xgb / single / DCS)
      -> comparison report           (model_comparison.{csv,tex})

Artifacts emitted (design + Task 21.1):

    * ``model_comparison.{csv,tex}``            (Evaluation_Suite.compare_models)
    * ``leakage_audit.{json,md}``               (Leakage_Auditor.write_report)
    * ``stance_series.csv``                     (Stance_Space.series)
    * audit JSONL/parquet artifacts             (signal weights/exclusions/presence,
                                                 DAPT corpus audit, regime/CV JSON)

The resolved config + its ``config_hash`` are persisted via
``ConfigManager.save_resolved_config`` (Requirement 14.4).

Consistent with the codebase's Requirement-33 owner-run convention, the heavy /
full-scale execution is **owner-run** (Task 21.2 CPU-smokes it). This module
**authors and wires** the orchestrator: it is importable without torch, every
stage is invoked from config, and stages that require torch / real data degrade
gracefully into recorded ``SKIPPED`` stage results rather than aborting the run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import ConfigManager
from .market_supervised import _cfg_get

_LOGGER = logging.getLogger(__name__)

try:  # torch is optional at import time; stages that need it degrade gracefully.
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch-free inspection path
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# Stage / pipeline result containers
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Outcome of a single orchestrated stage.

    ``status`` is one of ``"OK"``, ``"SKIPPED"`` (a prerequisite such as torch or
    an upstream artifact was unavailable), or ``"ERROR"`` (the stage raised). No
    stage aborts the pipeline: a failure is recorded here and the chain continues,
    mirroring the honest-status discipline of ``DaptResult`` / ``TrainResult``.
    """

    name: str
    status: str = "OK"
    detail: str = ""
    artifacts: dict[str, str] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "artifacts": self.artifacts,
        }


@dataclass
class CouncilRunResult:
    """Structured outcome of a full ``run_council`` invocation."""

    output_dir: str
    config_hash: Optional[str] = None
    status: str = "RUNNING"
    stages: dict[str, StageResult] = field(default_factory=dict)

    def add(self, stage: StageResult) -> StageResult:
        self.stages[stage.name] = stage
        return stage

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "config_hash": self.config_hash,
            "status": self.status,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
        }


# ---------------------------------------------------------------------------
# Small config/path helpers (resolved from config only -- no hard-coding)
# ---------------------------------------------------------------------------

def _audit_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "audit")


def _results_dir(output_dir: str) -> str:
    return os.path.join(output_dir, "results")


def _stance_result_paths(output_dir: str) -> dict[str, str]:
    """Return the complete canonical stance artifact set for one run."""
    results_dir = _results_dir(output_dir)
    return {
        "stance_event_predictions": os.path.join(
            results_dir, "stance_event_predictions.parquet"
        ),
        "stance_series_path": os.path.join(results_dir, "stance_series.csv"),
        "stance_summary": os.path.join(results_dir, "stance_summary.json"),
    }


def _clear_stance_result_artifacts(output_dir: str) -> None:
    """Remove canonical/temporary stance outputs before a new publication."""
    for path in _stance_result_paths(output_dir).values():
        for candidate in (path, f"{path}.tmp"):
            try:
                os.remove(candidate)
            except FileNotFoundError:
                pass


def _write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    """Replace a JSON artifact atomically within its destination directory."""
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    os.replace(temp_path, path)


def _select_task_path(cfg: Any, output_dir: str) -> Optional[str]:
    """Resolve the primary task parquet path used to drive the model stages.

    Prefers an explicit ``data.task_parquet`` config key; otherwise looks for the
    single primary target's ``model_<primary_target>.parquet`` under the run's
    ``models`` directory. Returns ``None`` when no task parquet can be resolved so
    the caller can record a graceful ``SKIPPED`` rather than fabricating data.
    """
    explicit = _cfg_get(cfg, "data.task_parquet", None)
    if explicit and os.path.exists(str(explicit)):
        return str(explicit)

    primary = _cfg_get(cfg, "model.deberta.stance.primary_target", None)
    models_dir = os.path.join(output_dir, "models")
    # One canonical masked union is the default Council task. Scalar target
    # parquets remain diagnostics/optional per-signal experiments only.
    candidates = [os.path.join(models_dir, "model_council.parquet")]
    if primary:
        candidates.append(os.path.join(models_dir, f"model_{primary}.parquet"))
    # Fall back to any model_*.parquet already present in the run's models dir.
    if os.path.isdir(models_dir):
        for fn in sorted(os.listdir(models_dir)):
            if fn.startswith("model_") and fn.endswith(".parquet"):
                candidates.append(os.path.join(models_dir, fn))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _primary_target_name(cfg: Any) -> str:
    return str(_cfg_get(cfg, "model.deberta.stance.primary_target", "target"))


def _cfg_set(cfg: Any, dotted: str, value: Any) -> None:
    """Set ``dotted`` (e.g. ``"model.deberta.encoder_override"``) on a dict or object.

    Mirrors :func:`_cfg_get`: walks nested dict/attribute segments, creating dict
    segments as needed, and assigns ``value`` at the final segment. Objects along
    the path are traversed by attribute; the leaf is assigned by key when its
    container is a dict, else by attribute.
    """
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if isinstance(node, dict):
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            node = child
        else:
            child = getattr(node, part, None)
            if child is None:
                child = {}
                setattr(node, part, child)
            node = child
    leaf = parts[-1]
    if isinstance(node, dict):
        node[leaf] = value
    else:
        setattr(node, leaf, value)


def _apply_dapt_encoder_override(cfg: Any, dapt_stage: StageResult) -> Optional[str]:
    """Forward the DAPT-adapted encoder into the supervised stages (Req 8).

    When the DAPT stage completed successfully and saved an encoder, set
    ``model.deberta.encoder_override`` to that ``dapt_encoder`` path so
    ``_build_encoder`` initializes the supervised model from the adapted weights
    (Req 8.1), and log the encoder provenance (Req 8.3). When DAPT is unavailable
    (skipped/errored or no saved encoder), leave ``encoder_override`` untouched so
    ``_build_encoder`` falls back to ``base_model``/stub (Req 8.2).

    Returns the forwarded ``dapt_encoder`` path when set, else ``None``.
    """
    if dapt_stage.status != "OK":
        return None
    dapt_encoder = dapt_stage.artifacts.get("dapt_encoder")
    if not dapt_encoder or not os.path.exists(str(dapt_encoder)):
        return None
    _cfg_set(cfg, "model.deberta.encoder_override", str(dapt_encoder))
    _LOGGER.info(
        "Supervised encoder initialized from DAPT encoder at %s", dapt_encoder
    )
    dapt_stage.data["encoder_override"] = str(dapt_encoder)
    return str(dapt_encoder)


# ---------------------------------------------------------------------------
# Individual stage runners. Each returns a StageResult and never raises.
# ---------------------------------------------------------------------------

def _run_dapt_stage(
    cfg: Any, output_dir: str, master: Optional[Any], splits: Optional[Any]
) -> StageResult:
    """DAPT stage: build the Council corpus and run domain-adaptive pretraining.

    Delegates to :func:`src.dapt.run_dapt_with_external_corpus`, which assembles
    the corpus (external corpora concatenated, test-split docs excluded, corpus
    audit written) and applies the ``min_corpus_docs`` gate. Skips gracefully when
    torch or the master events table is unavailable (Req 5.1-5.5).
    """
    stage = StageResult(name="dapt")
    if not bool(_cfg_get(cfg, "runtime.run_dapt", True)):
        stage.status = "SKIPPED"
        stage.detail = "runtime.run_dapt=false; DAPT not run"
        return stage
    if not _HAS_TORCH:
        stage.status = "SKIPPED"
        stage.detail = "torch unavailable; DAPT authored but owner-run at scale"
        return stage
    if master is None:
        stage.status = "SKIPPED"
        stage.detail = "no master events table supplied; DAPT skipped"
        return stage
    try:
        from .dapt import run_dapt_with_external_corpus

        audit_path = os.path.join(_audit_dir(output_dir), "dapt_corpus_audit.json")
        result = run_dapt_with_external_corpus(
            master, cfg, output_dir=output_dir, splits=splits, audit_path=audit_path
        )
        stage.status = result.status
        stage.detail = (
            f"DAPT status={result.status} n_docs={result.n_documents} "
            f"successful_steps={getattr(result, 'steps', 0)} "
            f"attempted_steps={getattr(result, 'attempted_steps', 0)} "
            f"nonfinite_loss={getattr(result, 'nonfinite_loss_steps', 0)} "
            f"nonfinite_grad={getattr(result, 'nonfinite_gradient_steps', 0)} "
            f"nonfinite_params={getattr(result, 'nonfinite_parameter_steps', 0)}"
        )
        if getattr(result, "reason", None):
            stage.detail += f" reason={result.reason}"
        if getattr(result, "encoder_path", None):
            stage.artifacts["dapt_encoder"] = result.encoder_path
        if os.path.exists(audit_path):
            stage.artifacts["dapt_corpus_audit"] = audit_path
        health_path = os.path.join(_audit_dir(output_dir), "dapt_health.json")
        if os.path.exists(health_path):
            stage.artifacts["dapt_health"] = health_path
        stage.data["result"] = result
    except Exception as exc:  # never abort the pipeline
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


def _run_training_eval_stage(
    cfg: Any, output_dir: str, task_path: Optional[str]
) -> StageResult:
    """Training + Evaluation stage: frozen headline AND fine-tune ablation.

    Delegates to :func:`src.training.evaluate_frozen_and_finetune`, which runs
    BOTH Requirement-6 modes on the SAME eval split, scores each via
    :func:`src.evaluation.evaluate_model`, and feeds both records to
    :func:`src.evaluation.compare_models` so ``model_comparison.{csv,tex}`` gains a
    ``council_frozen`` and a ``council_finetune`` row (Req 6.3, 6.4). The
    ``SignalNormalizer`` (train-split-only z-scoring) is fit inside
    ``build_dataloaders`` on this path (Req 4.1).

    Returns the per-model ``MetricRecord`` mapping in ``stage.data['records']`` so
    the baselines stage can score every model on the *same* comparison table.
    """
    stage = StageResult(name="training_eval")
    if not _HAS_TORCH:
        stage.status = "SKIPPED"
        stage.detail = "torch unavailable; training authored but owner-run at scale"
        return stage
    if not task_path:
        stage.status = "SKIPPED"
        stage.detail = "no task parquet resolved from config; training skipped"
        return stage
    if not bool(_cfg_get(cfg, "runtime.run_frozen", True)) and not bool(
        _cfg_get(cfg, "runtime.run_finetune", True)
    ):
        stage.status = "SKIPPED"
        stage.detail = "runtime.run_frozen=false and runtime.run_finetune=false"
        stage.data["task_path"] = task_path
        return stage
    try:
        from .training import evaluate_frozen_and_finetune

        results_dir = _results_dir(output_dir)
        os.makedirs(results_dir, exist_ok=True)
        out = evaluate_frozen_and_finetune(
            task_path,
            cfg,
            output_dir,
            _primary_target_name(cfg),
            results_dir=results_dir,
        )
        stage.data["records"] = out.get("records", {})
        stage.data["comparison"] = out.get("comparison")
        stage.data["task_path"] = task_path
        # Surface each mode's training outcome (status + optim_steps) WITHOUT
        # retaining the heavy prediction arrays, so the manifest and the notebook
        # summary can explain a run that produced no checkpoints/progress
        # (e.g. INSUFFICIENT_SAMPLE / UNTRAINED_NO_STEP) rather than looking silent.
        for _mode_key in ("frozen", "finetune"):
            _mode_out = out.get(_mode_key) or {}
            _tr = _mode_out.get("train_result") or {}
            stage.data[f"{_mode_key}_summary"] = {
                "status": _mode_out.get("status"),
                "optim_steps": _tr.get("optim_steps"),
                "n_samples": _tr.get("n_samples"),
                "best_epoch": _tr.get("best_epoch"),
                "best_metric": _tr.get("best_metric"),
                "nonfinite_events": _tr.get("nonfinite_events"),
                "checkpoint_dir": _mode_out.get("checkpoint_dir"),
            }
            # Persist lightweight paths required by downstream stages. stage.data
            # itself is runtime-only, so copy each real path into artifacts for
            # the manifest as well; never advertise a planned/nonexistent file.
            _mode_artifacts: dict[str, str] = {}
            for _artifact_key in (
                "predictions_path",
                "stance_predictions_path",
                "stance_summary_path",
                "best_checkpoint",
                "target_scaler_path",
                "diagnostics_path",
            ):
                _artifact_path = _mode_out.get(_artifact_key)
                if _artifact_path and os.path.exists(str(_artifact_path)):
                    _mode_artifacts[_artifact_key] = str(_artifact_path)
                    stage.artifacts[
                        f"{_mode_key}_{_artifact_key}"
                    ] = str(_artifact_path)
            stage.data[f"{_mode_key}_artifacts"] = _mode_artifacts
        for ext in ("csv", "tex"):
            p = os.path.join(results_dir, f"model_comparison.{ext}")
            if os.path.exists(p):
                stage.artifacts[f"model_comparison_{ext}"] = p
        mode_statuses = {
            key: (out.get(key) or {}).get("status")
            for key in ("frozen", "finetune")
        }
        summary_path = os.path.join(_audit_dir(output_dir), "training_stage_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump({
                "task_path": task_path,
                "primary_target": _primary_target_name(cfg),
                "mode_statuses": mode_statuses,
                "record_names": sorted(stage.data["records"].keys()),
                "frozen": stage.data.get("frozen_summary"),
                "finetune": stage.data.get("finetune_summary"),
                "frozen_artifacts": stage.data.get("frozen_artifacts", {}),
                "finetune_artifacts": stage.data.get("finetune_artifacts", {}),
            }, fh, indent=2, default=str)
        stage.artifacts["training_stage_summary"] = summary_path
        if stage.data["records"]:
            stage.detail = (
                f"scored {len(stage.data['records'])} council mode(s): "
                f"{sorted(stage.data['records'].keys())}; statuses={mode_statuses}"
            )
        else:
            stage.status = "ERROR"
            stage.detail = f"no Council mode produced valid predictions; statuses={mode_statuses}"
    except Exception as exc:
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


def _split_frames_from_task(task_path: Optional[str]) -> Optional[Any]:
    """Load a task parquet as a splits frame for the CV / regime / leakage stages.

    Returns the raw task-parquet DataFrame (which carries ``split`` / ``target`` /
    date columns) or ``None`` when unavailable so the callers can record a
    graceful ``SKIPPED``.
    """
    if not task_path or not os.path.exists(task_path):
        return None
    try:
        import pandas as pd

        return pd.read_parquet(task_path)
    except Exception:
        return None


def _primary_observed_rows(frame):
    """Filter canonical union rows to finite, genuinely observed primary labels."""
    if frame is None:
        return None
    try:
        import numpy as np
        import pandas as pd

        out = frame
        if "primary_present" in out.columns:
            out = out[out["primary_present"].fillna(False).astype(bool)]
        if "target" in out.columns:
            finite = np.isfinite(pd.to_numeric(out["target"], errors="coerce"))
            out = out[finite]
        return out.reset_index(drop=True)
    except Exception:
        return frame


def _run_cross_validation_stage(
    cfg: Any, output_dir: str, training_stage: StageResult
) -> StageResult:
    """Build the chronological CV plan and aggregate genuine OOF artifacts.

    The fold plan is always persisted. With ``runtime.run_expensive_cv=false``
    (the bounded default), the stage returns ``OK`` as an explicitly plan-only
    result and publishes no predictive metrics. When predictive CV is enabled,
    aggregation accepts only predictions already stamped with both their true
    ``fold_id`` and ``event_id``; an ordinary final-test prediction artifact is
    never relabeled as out-of-fold evidence. If those artifacts are unavailable,
    the plan remains auditable and the stage reports that no OOF metrics exist.
    """
    stage = StageResult(name="cross_validation")
    run_expensive_cv = bool(_cfg_get(cfg, "runtime.run_expensive_cv", False))
    task_path = training_stage.data.get("task_path")
    events = _primary_observed_rows(_split_frames_from_task(task_path))
    if events is None:
        stage.status = "SKIPPED"
        stage.detail = "no events frame available for cross-validation"
        return stage
    try:
        from .cross_validation import CrossValidator, build_folds

        cfg_cv = _cfg_get(cfg, "cross_validation", {}) or {}
        scheme = str(cfg_cv.get("scheme", "rolling_origin"))
        folds = build_folds(events, cfg_cv)
        validator = CrossValidator(
            scheme=scheme,
            min_fold_events=int(cfg_cv.get("min_fold_events", 5)),
        )
        stage.data["folds"] = folds
        stage.data["validator"] = validator

        audit_dir = _audit_dir(output_dir)
        os.makedirs(audit_dir, exist_ok=True)
        plan_path = os.path.join(audit_dir, "cross_validation_folds.json")
        with open(plan_path, "w") as f:
            json.dump(
                {
                    "scheme": scheme,
                    "n_folds": len(folds),
                    "fold_ids": [
                        getattr(fold, "fold_id", i)
                        for i, fold in enumerate(folds)
                    ],
                    "predictive_fold_training_enabled": run_expensive_cv,
                },
                f,
                indent=2,
                default=str,
            )
        stage.artifacts["cross_validation_folds"] = plan_path
        if not run_expensive_cv:
            stage.status = "OK"
            stage.detail = (
                f"{scheme}: built {len(folds)} fold plan(s) only; "
                "runtime.run_expensive_cv=false, so no predictive CV metrics exist"
            )
            return stage

        # Consume only explicitly fold-stamped OOF predictions. A normal final-
        # test artifact is never relabeled as cross-validation evidence.
        oof = _oof_predictions_for_folds(output_dir, folds)
        if oof is None:
            # No usable predictions -> record the fold plan for traceability
            # rather than fabricating metrics, and skip gracefully.
            cv_path = os.path.join(audit_dir, "cross_validation_folds.json")
            with open(cv_path, "w") as f:
                json.dump(
                    {
                        "scheme": scheme,
                        "n_folds": len(folds),
                        "fold_ids": [
                            getattr(fold, "fold_id", i)
                            for i, fold in enumerate(folds)
                        ],
                    },
                    f,
                    indent=2,
                    default=str,
                )
            stage.artifacts["cross_validation_folds"] = cv_path
            stage.detail = (
                f"{scheme}: built {len(folds)} fold(s); no OOF predictions available"
            )
            return stage

        aggregate = validator.aggregate(
            folds, oof, cfg=cfg, model="council"
        )
        stage.data["aggregate"] = aggregate

        cv_path = os.path.join(audit_dir, "cross_validation.json")
        with open(cv_path, "w") as f:
            json.dump(
                {
                    "scheme": aggregate.scheme,
                    "metrics": list(aggregate.metrics),
                    "mean": aggregate.mean,
                    "std": aggregate.std,
                    "n_usable_folds": aggregate.n_usable_folds,
                    "usable_fold_ids": aggregate.usable_fold_ids,
                    "excluded_fold_ids": aggregate.excluded_fold_ids,
                    "fold_records": {
                        fid: record.to_row()
                        for fid, record in aggregate.fold_records.items()
                    },
                },
                f,
                indent=2,
                default=str,
            )
        stage.artifacts["cross_validation"] = cv_path
        stage.detail = (
            f"{scheme}: {aggregate.n_usable_folds} usable fold(s) of {len(folds)}"
        )
    except Exception as exc:
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


def _oof_predictions_for_folds(output_dir: str, folds: "list") -> "Optional[dict]":
    """Return genuine fold-stamped OOF predictions keyed by ``fold_id``.

    Every consumed row must carry both ``fold_id`` and ``event_id``. For each
    planned fold, all declared evaluation IDs must be present in that fold's own
    artifact. Returns ``None`` for missing or incomplete evidence; it never uses
    row-order alignment and never repurposes ordinary final-test predictions.
    """
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        return None

    preds_df = _load_prediction_frame(output_dir)
    if preds_df is None or preds_df.empty:
        return None
    # A normal frozen/finetune test artifact is NOT out-of-fold evidence. Only
    # consume predictions explicitly stamped with the fold that trained them.
    # This prevents relabeling one final-test run as multiple CV evaluations.
    if "fold_id" not in preds_df.columns or "event_id" not in preds_df.columns:
        return None

    oof: dict = {}
    fold_labels = preds_df["fold_id"].astype(str)
    for fold in folds:
        fold_id = getattr(fold, "fold_id", None)
        eval_ids = list(getattr(fold, "eval_ids", []) or [])
        if fold_id is None or not eval_ids:
            return None
        fold_df = preds_df[fold_labels == str(fold_id)]
        if fold_df.empty:
            return None
        by_id = fold_df.drop_duplicates("event_id").set_index("event_id")
        if any(event_id not in by_id.index for event_id in eval_ids):
            return None
        aligned = by_id.loc[eval_ids]
        oof[fold_id] = (
            aligned["y_true"].to_numpy(dtype="float64"),
            aligned["y_pred"].to_numpy(dtype="float64"),
        )

    return oof if oof else None


def _load_prediction_frame(output_dir: str):
    """Load the frozen/finetune predictions parquet as a DataFrame or ``None``.

    Mirrors :func:`_load_primary_predictions` but returns the full frame (with
    ``event_id`` / ``event_date`` identity columns when present) so the CV stage
    can align predictions to each fold's held-out events.
    """
    try:
        import pandas as pd
    except Exception:
        return None
    for sub in ("frozen", "finetune"):
        eval_dir = os.path.join(output_dir, sub, "evaluation")
        if not os.path.isdir(eval_dir):
            continue
        for fn in sorted(os.listdir(eval_dir)):
            if fn.startswith("predictions_") and fn.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(eval_dir, fn))
                if "y_true" in df.columns and "y_pred" in df.columns:
                    return df
    return None


def _run_regime_audit_stage(
    cfg: Any, output_dir: str, training_stage: StageResult
) -> StageResult:
    """Regime audit stage: ZLB vs post-hiking partition metrics.

    Scores the primary evaluated model's ``(y_true, y_pred)`` across the two
    regimes via :func:`src.regime_audit.audit_regimes` using the configured
    boundary dates. Skips gracefully when no usable predictions / event dates are
    available (Req 10.1-10.4).
    """
    stage = StageResult(name="regime_audit")
    task_path = training_stage.data.get("task_path")
    records = training_stage.data.get("records", {})
    events = _primary_observed_rows(_split_frames_from_task(task_path))
    if events is None or not records:
        stage.status = "SKIPPED"
        stage.detail = "no predictions/event dates available for regime audit"
        return stage
    try:
        import numpy as np

        from .regime_audit import audit_regimes

        # Recover the evaluation predictions written by the training stage.
        preds = _load_primary_predictions(output_dir)
        if preds is None:
            stage.status = "SKIPPED"
            stage.detail = "no persisted predictions parquet found for regime audit"
            return stage
        y_true, y_pred, dates = preds
        result = audit_regimes(
            y_true, y_pred, dates, cfg=cfg, model="council"
        )
        regime_path = os.path.join(_audit_dir(output_dir), "regime_audit.json")
        os.makedirs(os.path.dirname(regime_path), exist_ok=True)
        payload = result.to_dict() if hasattr(result, "to_dict") else str(result)
        with open(regime_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        stage.artifacts["regime_audit"] = regime_path
        stage.detail = "regime audit computed (ZLB vs post-hiking)"
    except Exception as exc:
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


def _load_primary_predictions(output_dir: str):
    """Load ``(y_true, y_pred, event_dates)`` from the frozen predictions parquet.

    Returns ``None`` when no predictions parquet is present so regime/other stages
    skip gracefully rather than fabricating values.
    """
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        return None
    # evaluate_frozen_and_finetune writes frozen predictions under output/frozen.
    for sub in ("frozen", "finetune"):
        eval_dir = os.path.join(output_dir, sub, "evaluation")
        if not os.path.isdir(eval_dir):
            continue
        for fn in sorted(os.listdir(eval_dir)):
            if fn.startswith("predictions_") and fn.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(eval_dir, fn))
                y_true = df.get("y_true")
                y_pred = df.get("y_pred")
                if y_true is None or y_pred is None:
                    continue
                dates = df.get("event_date")
                if dates is not None:
                    # True event dates are present -> return them unchanged and
                    # never substitute an ordinal (Req 5.2).
                    dates = pd.Series(dates)
                else:
                    # Last-resort fallback for a genuinely date-less legacy
                    # artifact: synthesize an ordinal timeline so the partition
                    # still runs (regime boundaries then split on order). This
                    # is a degradation from true-date regime assignment, so log
                    # it rather than substituting silently (Req 5.2).
                    _LOGGER.warning(
                        "Degradation: predictions artifact %s has no 'event_date' "
                        "column; falling back to a row-ordinal timeline for regime "
                        "partitioning. Regime boundaries will split on row order "
                        "rather than true dates.",
                        os.path.join(eval_dir, fn),
                    )
                    dates = pd.Series(range(len(df)))
                return (
                    np.asarray(y_true, dtype="float64"),
                    np.asarray(y_pred, dtype="float64"),
                    list(dates),
                )
    return None


def _run_leakage_audit_stage(
    cfg: Any, output_dir: str, training_stage: StageResult
) -> StageResult:
    """Leakage audit stage: signal as-of + DAPT contamination + the base checks.

    Runs :func:`src.leakage_audit.audit_leakage` over the split assignments, the
    fitted signal-scaler metadata, and the DAPT corpus audit, then writes
    ``leakage_audit.{json,md}`` via :func:`src.leakage_audit.write_report`
    (Req 11.1-11.4). Skips gracefully when no splits frame is available.
    """
    stage = StageResult(name="leakage_audit")
    task_path = training_stage.data.get("task_path")
    splits = _split_frames_from_task(task_path)
    if splits is None:
        stage.status = "SKIPPED"
        stage.detail = "no splits frame available for leakage audit"
        return stage
    try:
        from .leakage_audit import audit_leakage, write_report

        audit_dir = _audit_dir(output_dir)
        os.makedirs(audit_dir, exist_ok=True)

        scaler_meta = _load_signal_scaler_meta(audit_dir)
        dapt_audit = os.path.join(audit_dir, "dapt_corpus_audit.json")
        run_meta = {
            "selection_split": "val",
            "hyperparameter_search_split": "val",
        }
        report = audit_leakage(
            splits,
            scaler_meta,
            run_meta,
            signal_windows=None,
            dapt_corpus_audit=dapt_audit if os.path.exists(dapt_audit) else None,
        )
        write_report(report, audit_dir)
        for ext in ("json", "md"):
            p = os.path.join(audit_dir, f"leakage_audit.{ext}")
            if os.path.exists(p):
                stage.artifacts[f"leakage_audit_{ext}"] = p
        stage.data["report"] = report
        stage.detail = (
            "LEAK DETECTED" if report.get("leak_detected") else "no leakage detected"
        ) + f" ({report.get('n_checks', 0)} checks)"
    except Exception as exc:
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


def _load_signal_scaler_meta(audit_dir: str) -> Optional[dict]:
    """Load ``signal_scalers.json`` if present (fit_on=='train' scaler metadata)."""
    path = os.path.join(audit_dir, "signal_scalers.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    # audit_leakage's scaler check expects a single {mean,std,n,fit_on} record;
    # surface the first configured signal's stats (all share fit_on=='train').
    if isinstance(data, dict) and data:
        first = next(iter(data.values()))
        if isinstance(first, dict) and "fit_on" in first:
            return first
    return data


def _run_baselines_stage(
    cfg: Any, output_dir: str, training_stage: StageResult
) -> StageResult:
    """Baselines + comparison report stage.

    Runs every baseline via :func:`src.baselines.all_baselines` on the SAME
    train/test splits, scores each with :func:`src.evaluation.evaluate_model`, and
    merges them with the Council records from the training stage into one
    :func:`src.evaluation.compare_models` table so ``model_comparison.{csv,tex}``
    reports the Council model AND every baseline on identical splits (Req 12.5).
    Skips gracefully when no splits frame is available.
    """
    stage = StageResult(name="baselines")
    task_path = training_stage.data.get("task_path")
    frame = _primary_observed_rows(_split_frames_from_task(task_path))
    if frame is None:
        stage.status = "SKIPPED"
        stage.detail = "no task frame available for baselines"
        return stage
    try:
        from .baselines import all_baselines
        from .evaluation import compare_models, evaluate_model

        train = frame[frame["split"] == "train"] if "split" in frame else frame
        test = frame[frame["split"] == "test"] if "split" in frame else frame

        baseline_preds = all_baselines(train, test, cfg)

        # Score every baseline against the same test targets and merge with the
        # Council records so all models share one comparison table.
        records = dict(training_stage.data.get("records", {}))
        y_true = _targets_of(test)
        for name, pred in baseline_preds.items():
            y_pred = getattr(pred, "regression", None)
            if y_pred is None or y_true is None or len(y_true) == 0:
                continue
            try:
                records[name] = evaluate_model(
                    y_true,
                    y_pred,
                    cfg=cfg,
                    model=name,
                    target_name=_primary_target_name(cfg),
                )
            except Exception:
                continue

        results_dir = _results_dir(output_dir)
        os.makedirs(results_dir, exist_ok=True)
        comparison = compare_models(records, results_dir=results_dir)
        stage.data["comparison"] = comparison
        for ext in ("csv", "tex"):
            p = os.path.join(results_dir, f"model_comparison.{ext}")
            if os.path.exists(p):
                stage.artifacts[f"model_comparison_{ext}"] = p
        stage.detail = f"scored {len(baseline_preds)} baseline(s); comparison rows={len(records)}"
    except Exception as exc:
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


def _targets_of(frame):
    """Best-effort 1-D target array from a task frame."""
    try:
        import numpy as np

        if frame is None:
            return None
        if hasattr(frame, "get") and "target" in getattr(frame, "columns", []):
            return np.asarray(frame["target"], dtype="float64")
    except Exception:
        return None
    return None


def _run_stance_series_stage(
    cfg: Any, output_dir: str, training_stage: StageResult
) -> StageResult:
    """Publish validated held-out stance probabilities and continuous series.

    The fine-tune prediction pass already used the trained checkpoint state held
    by ``run_training_loop``. This stage consumes only that aligned artifact; it
    never falls back to the frozen model's untrained direction head. Canonical
    outputs are cleared before every attempt and committed as one validated set,
    so a skipped/error rerun cannot expose stale probabilities.
    """
    stage = StageResult(name="stance_series", status="SKIPPED")
    _clear_stance_result_artifacts(output_dir)

    finetune_summary = training_stage.data.get("finetune_summary") or {}
    if finetune_summary.get("status") != "OK":
        stage.detail = (
            "fine-tune did not produce OK stance predictions; "
            f"status={finetune_summary.get('status', 'NOT_AVAILABLE')}"
        )
        return stage

    finetune_artifacts = training_stage.data.get("finetune_artifacts") or {}
    source_path = finetune_artifacts.get("stance_predictions_path") or (
        training_stage.artifacts.get("finetune_stance_predictions_path")
    )
    if not source_path or not os.path.exists(source_path):
        stage.detail = "fine-tune stance_event_predictions.parquet is unavailable"
        return stage

    try:
        import pandas as pd

        from .training import print_stance_summary, summarize_stance_predictions

        frame = pd.read_parquet(source_path)
        required = {
            "event_id",
            "event_date",
            "stance_prob_dovish",
            "stance_prob_neutral",
            "stance_prob_hawkish",
            "stance_score",
            "stance_dominant",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(
                f"stance prediction artifact missing required columns: {missing}"
            )
        if frame.empty:
            raise ValueError("stance prediction artifact is empty")

        # Recompute and validate all identities from the persisted probabilities.
        summary = summarize_stance_predictions(frame)
        summary.update({
            "artifact_contract": "held_out_council_stance_v1",
            "scope": "held_out_test_events",
            "source_artifact": str(source_path),
            "primary_target": _primary_target_name(cfg),
        })
        frame = frame.copy()
        frame["_event_sort_date"] = pd.to_datetime(
            frame["event_date"], errors="coerce"
        )
        if frame["_event_sort_date"].isna().any():
            raise ValueError("stance event_date contains unparseable values")
        frame = frame.sort_values(
            ["_event_sort_date", "event_id"], kind="stable"
        ).drop(columns=["_event_sort_date"])

        paths = _stance_result_paths(output_dir)
        os.makedirs(_results_dir(output_dir), exist_ok=True)
        temp_paths = {key: f"{path}.tmp" for key, path in paths.items()}

        event_columns = [
            "event_id",
            "event_date",
            "stance_prob_dovish",
            "stance_prob_neutral",
            "stance_prob_hawkish",
            "stance_score",
            "stance_dominant",
        ]
        optional_columns = [
            column for column in ("y_true", "y_pred") if column in frame.columns
        ]
        frame[event_columns + optional_columns].to_parquet(
            temp_paths["stance_event_predictions"], index=False
        )
        # Keep the historical event_date/stance_score pair first for compatible
        # readers, then append the categorical evidence additively.
        series_columns = [
            "event_date",
            "stance_score",
            "event_id",
            "stance_prob_dovish",
            "stance_prob_neutral",
            "stance_prob_hawkish",
            "stance_dominant",
        ]
        frame[series_columns].to_csv(
            temp_paths["stance_series_path"], index=False
        )
        with open(
            temp_paths["stance_summary"], "w", encoding="utf-8"
        ) as fh:
            json.dump(summary, fh, indent=2, default=str)

        # Publish only after every temporary artifact was written successfully.
        for key in (
            "stance_event_predictions", "stance_series_path", "stance_summary"
        ):
            os.replace(temp_paths[key], paths[key])

        print_stance_summary(summary)
        stage.status = "OK"
        stage.data["summary"] = summary
        stage.artifacts.update(paths)
        stage.detail = (
            f"published {summary['n_events']} held-out event stance predictions "
            "with Dovish/Neutral/Hawkish probabilities and continuous scores"
        )
    except Exception as exc:
        _clear_stance_result_artifacts(output_dir)
        stage.status = "ERROR"
        stage.detail = f"{type(exc).__name__}: {exc}"
    return stage


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_council_pipeline(
    cfg: dict,
    output_dir: str,
    master: Optional[Any] = None,
    splits: Optional[Any] = None,
    config_hash: Optional[str] = None,
) -> CouncilRunResult:
    """Run the full Council stage chain against an already-resolved config.

    This is the low-level entry point used by :func:`run_council` once the config
    has been loaded/validated/resolved and the experiment directory created. Every
    stage parameter is read from ``cfg``; nothing is hard-coded per experiment
    (Req 13.5).

    Args:
        cfg: The fully-resolved configuration mapping.
        output_dir: The experiment output directory (already created).
        master: Optional master events table for the DAPT corpus + task assembly.
        splits: Optional ``temporal_split`` frame defining test-split ids.
        config_hash: Optional stable hash of the resolved config for the result.

    Returns:
        A :class:`CouncilRunResult` recording every stage's status + artifacts.
        No stage aborts the pipeline; failures are recorded as ``ERROR`` /
        ``SKIPPED`` stage results.
    """
    os.makedirs(_audit_dir(output_dir), exist_ok=True)
    os.makedirs(_results_dir(output_dir), exist_ok=True)

    result = CouncilRunResult(output_dir=output_dir, config_hash=config_hash)
    manifest_path = os.path.join(
        _audit_dir(output_dir), "council_run_manifest.json"
    )
    # Invalidate any prior completed manifest before touching stage outputs. If
    # this invocation is interrupted, readers see RUNNING and cannot publish
    # artifacts from a previous attempt as current evidence.
    _write_json_atomic(manifest_path, result.to_dict())
    _clear_stance_result_artifacts(output_dir)

    # 1. DAPT -> adapted encoder + corpus audit.
    dapt_stage = result.add(_run_dapt_stage(cfg, output_dir, master, splits))
    # Forward the DAPT-adapted encoder into the supervised stages via
    # model.deberta.encoder_override (Req 8). When DAPT is unavailable this is a
    # no-op and _build_encoder falls back to base_model/stub.
    _apply_dapt_encoder_override(cfg, dapt_stage)

    # Resolve the task parquet driving the model stages (from config / run dir).
    task_path = _select_task_path(cfg, output_dir)

    # 2. training (frozen + finetune) -> 3. Evaluation (compare_models rows).
    training_stage = result.add(_run_training_eval_stage(cfg, output_dir, task_path))
    # Ensure the resolved task path is threaded to downstream stages.
    training_stage.data.setdefault("task_path", task_path)

    # 4. Cross-Validation.
    result.add(_run_cross_validation_stage(cfg, output_dir, training_stage))

    # 5. Regime audit.
    result.add(_run_regime_audit_stage(cfg, output_dir, training_stage))

    # 6. Leakage audit -> leakage_audit.{json,md}.
    result.add(_run_leakage_audit_stage(cfg, output_dir, training_stage))

    # 7. Baselines -> comparison report (model_comparison.{csv,tex}).
    result.add(_run_baselines_stage(cfg, output_dir, training_stage))

    # 8. Stance_Space series -> stance_series.csv.
    result.add(_run_stance_series_stage(cfg, output_dir, training_stage))

    # Atomically publish the final manifest only after every stage has recorded
    # its status. Readers require COMPLETED + stance_series=OK + the full
    # manifest-declared artifact set before exposing stance values.
    result.status = "COMPLETED"
    _write_json_atomic(manifest_path, result.to_dict())

    return result


def run_council(
    config_path: str,
    project_root: Optional[str] = None,
    master: Optional[Any] = None,
    splits: Optional[Any] = None,
) -> CouncilRunResult:
    """Config-driven end-to-end Council orchestrator (Req 6.3, 13.5, 14.4).

    Loads + validates + resolves the configuration at ``config_path`` (halting on
    a missing required input file, Req 14.5), creates the experiment output
    directory tree, persists the resolved config + ``config_hash`` via
    :meth:`ConfigManager.save_resolved_config` (Req 14.4), and then runs the full
    stage chain via :func:`run_council_pipeline` -- every stage resolved from the
    config with no manual code edits (Req 13.5):

        DAPT -> SignalNormalizer -> training (frozen + finetune) -> Evaluation
        -> Cross-Validation -> Regime audit -> Leakage audit -> Baselines
        -> comparison report.

    Emits ``model_comparison.{csv,tex}``, ``leakage_audit.{json,md}``,
    ``stance_series.csv``, and the audit JSONL/parquet artifacts under the
    experiment output directory.

    Args:
        config_path: Path to the experiment YAML config.
        project_root: Optional root used to rebase Colab-style paths onto the
            current machine (passed to :meth:`ConfigManager.load_config`).
        master: Optional master events table for the DAPT / task-assembly stages.
        splits: Optional ``temporal_split`` frame defining test-split ids.

    Returns:
        The :class:`CouncilRunResult` for the run.
    """
    manager = ConfigManager()
    # load_config validates, resolves defaults, rebases paths, and halts on a
    # missing required input file naming it (Req 14.5).
    cfg = manager.load_config(config_path, project_root=project_root)

    # Create the experiment output tree and persist the resolved config + hash
    # alongside the artifacts (Req 14.4).
    run = manager.create_experiment(cfg)
    manager.save_resolved_config(run)

    config_hash = None
    hash_path = os.path.join(run.output_dir, "config_hash.txt")
    if os.path.exists(hash_path):
        with open(hash_path) as f:
            config_hash = f.read().strip()

    return run_council_pipeline(
        cfg,
        run.output_dir,
        master=master,
        splits=splits,
        config_hash=config_hash,
    )
