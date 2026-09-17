"""Configuration loading, path rebasing, validation, and experiment/environment metadata.

Adapts ``ConfigManager``, ``EnvironmentInfo``, ``ExperimentRun``, ``_get_nested`` and
``_set_nested`` from ``notebooks/utils.py``. Adds the master-retention override of
``splits.data_start`` / ``splits.require_primary_target`` (Requirement 2.7), the
config-driven target registry (Requirement 12), and a stable ``config_hash`` used for
checkpoint invalidation (Requirement 21).

Satisfies: 18.3, 2.7, 12.1, 27.2.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import yaml

# =============================================================================
# Custom Exceptions (PipelineError hierarchy) - re-exported from errors
# =============================================================================

from .errors import ConfigError, PipelineError  # noqa: F401


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class EnvironmentInfo:
    """Captured runtime environment information."""

    python_version: str
    packages: dict[str, str]
    gpu_name: Optional[str]
    gpu_driver: Optional[str]
    platform_info: str
    timestamp: str


@dataclass
class ExperimentRun:
    """An experiment run with unique ID and configuration."""

    experiment_id: str
    config: dict[str, Any]
    environment: EnvironmentInfo
    output_dir: str
    created_at: str


@dataclass
class TargetSpec:
    """Declarative specification of a modelling target (Data Models §a).

    Targets are declared in config; the default task parquets are generated from
    the registry. Adding a target requires only a new registry entry - no code
    change (Requirement 12).
    """

    name: str
    source_dataset: str
    source_series: str
    path: str  # "intraday" | "daily"
    event_window: dict[str, int]
    formula: str
    unit: str
    timezone: str
    min_observations: int
    has_flag_column: str
    target_column: str
    #: For EA-MPD daily targets: which event window sheet the series comes from
    #: ("press_release" | "press_conference" | "monetary_event"). None for the
    #: intraday EA-CED targets.
    mpd_window: "Optional[str]" = None


# =============================================================================
# Config field metadata
# =============================================================================

# Committed Colab base prefix. Paths under this base are rebased onto the current
# machine's ``project_root`` by ``rebase_paths``; the required-file existence
# check (Req 14.5) is skipped while ``data.datasets_dir`` still resolves under it,
# because those paths do not exist on disk until rebased.
_COLAB_BASE = "/content/drive/MyDrive/market_supervised"

_REQUIRED_CONFIG_KEYS = [
    "schema_version",
    "experiment",
    "environment",
    "data",
    "splits",
    "market_windows",
    "model",
]

_REQUIRED_NESTED_FIELDS = [
    "experiment.name",
    "experiment.seed",
    "experiment.primary_metric",
    "environment.platform",
    "environment.base_path",
    "data.datasets_dir",
    "data.outputs_dir",
    "data.required_files",
    "splits.train_end",
    "splits.val_start",
    "splits.val_end",
    "splits.test_start",
    "market_windows.start_offset_minutes",
    "market_windows.end_offset_minutes",
    # Council-of-supervision required blocks (Req 1, 2, 9, 10). These carry no
    # safe default: fusion_mode/scheme select behaviour and the regime boundaries
    # partition the sample, so an absent value must halt fail-fast rather than be
    # silently substituted. council.signals is additionally range-checked (>= 2)
    # in validate() (Req 1.5).
    "council.fusion_mode",
    "council.signals",
    "cross_validation.scheme",
    "regimes.zlb_end",
    "regimes.hiking_start",
]

_DEFAULTS: dict[str, Any] = {
    "market_windows.contrastive_threshold_bp": 0.5,
    "market_windows.material_move_bp": 1.0,
    "market_windows.material_move_by_target": {
        "ois2y": 1.0,
        "ois5y": 1.0,
        "ois10y": 1.0,
        "dgs2": 1.0,
        "dgs10": 1.0,
        "dgs30": 1.0,
        "ea_mpd_target": 1.0,
        "ea_mpd_path": 1.0,
        "ea_mpd_timing": 1.0,
        "equity": 0.001,
        "fx": 0.0005,
        "vix": 0.5,
    },
    "model.deberta.base_model": "microsoft/deberta-v3-base",
    "model.deberta.max_chunks": 8,
    "model.deberta.chunk_size": 512,
    "model.deberta.chunk_overlap": 64,
    "model.deberta.encoder_mode": "hierarchical",
    "model.deberta.max_epochs": 8,
    "model.deberta.early_stopping_patience": 3,
    # Explicit optimizer-numerics contract. Trainable parameters and AdamW
    # moments stay in FP32; mixed precision is applied only via autocast.
    # eps=1e-6 is representable in FP16 as an additional safeguard, while the
    # conservative scalar backend avoids version/device-dependent foreach/fused
    # update behavior.
    "model.deberta.optimizer.parameter_dtype": "float32",
    "model.deberta.optimizer.eps": 1.0e-6,
    "model.deberta.optimizer.foreach": False,
    "model.deberta.optimizer.fused": False,
    "model.deberta.learning_rate": 2.0e-5,
    "model.deberta.batch_size": 16,
    "model.deberta.loss_weights.regression": 1.0,
    "model.deberta.loss_weights.direction": 0.5,
    "model.deberta.loss_weights.uncertainty": 0.3,
    "model.deberta.loss_weights.temporal": 0.2,
    "model.deberta.loss_weights.contrastive": 0.3,
    "model.deberta.loss_weights.council": 0.5,
    # Stance objective weight in the six-objective total (Req 8.7).
    "model.deberta.loss_weights.stance": 0.4,
    # Continuous stance-scoring head defaults (Req 8.1, 8.7). The Stance_Scorer
    # reads these exclusively from config; resolve_defaults populates any absent
    # optional key so the head never falls back to a hard-coded value.
    "model.deberta.stance.enabled": True,        # attach the StanceHead + stance objective
    "model.deberta.stance.temperature": 1.0,     # softmax temperature (>0)
    "model.deberta.stance.fusion_hidden": 128,   # hidden width of the fusion MLP
    "model.deberta.stance.neutral_band_bp": 1.0, # neutral-band width (bp); defaults to material_move_bp
    "model.deberta.stance.sharpness": 1.0,       # steepness of the signed-magnitude squash
    "model.deberta.stance.default_emotion": 0.0,      # substituted when the emotion signal is absent
    "model.deberta.stance.default_expectation": 0.0,  # substituted when the expectation signal is absent
    "model.deberta.stance.primary_target": "ois2y",   # short-rate series driving the soft label
    "model.deberta.stance.promote_signals": ["vix", "dgs2", "dgs10", "dgs30", "equity"],  # datasets promoted to supervised signal
    "model.deberta.temporal.near_days": 5,
    "model.deberta.temporal.far_days": 30,
    "model.deberta.temporal.margin": 0.5,
    "model.deberta.nan_recovery.lr_reduction_factor": 0.5,
    "model.deberta.nan_recovery.max_retries": 3,
    "model.deberta.nan_recovery.lr_backoff_after_steps": 5,
    "model.deberta.nan_recovery.abort_after_events": 100,
    "model.deberta.nan_recovery.min_lr": 1.0e-8,
    "model.deberta.min_chunk_tokens": 2,
    "model.deberta.cache_tokenization": True,
    # Disabled as a numerical-stability mitigation. It is not claimed as the
    # proven root cause without a controlled same-batch A/B diagnostic.
    "model.deberta.gradient_checkpointing": False,
    # Resume is opt-in: stale checkpoints from a different signal set/model
    # contract must never be loaded silently. Compatible checkpoints carry a
    # fingerprint and can be resumed by explicitly setting resume=true.
    "model.deberta.resume": False,
    # Checkpoint footprint controls. Safe defaults write only best weights;
    # owner runs can opt into rolling optimizer checkpoints when needed.
    "model.deberta.checkpoint.save_latest": False,
    "model.deberta.checkpoint.save_optimizer_state": False,
    # Market-context signal sources for the stance head (Req 8.7). Emotion =
    # risk/volatility (VIX), expectation = rate-expectation (MPS; US = context-only).
    "market_context.emotion_source": "vix_indices",
    "market_context.expectation_source": "us_mps",
    # Council fusion bounds (Req 2.4). Reuse the uncertainty log-variance clamp
    # convention so exp(-log_var) weights stay finite and bounded.
    "council.log_var_min": -8.0,
    "council.log_var_max": 2.0,
    # Frozen-encoder headline vs fine-tuning ablation switch (Req 6.1/6.2/6.3).
    "model.deberta.frozen_encoder": True,
    # Continuous Stance_Score head toggle for the Council pipeline (Req 7).
    "model.deberta.stance_score.enabled": True,
    "model.dapt.masking_probability": 0.15,
    "model.dapt.min_corpus_docs": 100,
    "model.dapt.epochs": 1,
    "model.dapt.max_steps": 300,
    "model.dapt.min_successful_steps": 25,
    # Linear LR warmup steps for the DAPT MLM head (0 => auto: a small fraction
    # of the step budget). Stabilises the randomly re-initialised MLM decoder.
    "model.dapt.warmup_steps": 0,
    # Runtime stage gates bound the default experiment to one canonical Council
    # fit. Expensive diagnostic matrices remain explicit opt-ins.
    "runtime.run_dapt": True,
    "runtime.run_frozen": True,
    "runtime.run_finetune": True,
    "runtime.run_per_signal": False,
    "runtime.run_ablations": False,
    "runtime.run_expensive_cv": False,
    "runtime.run_checkpointing_diagnostic": False,
    # Optional additional large corpora for domain-adaptive pretraining (Req 5.2).
    "model.dapt.external_corpus_paths": [],
    # Cross-validation optional settings (Req 9). scheme is required (see
    # _REQUIRED_NESTED_FIELDS); these govern fold layout and viability.
    "cross_validation.rolling_origin_boundaries": [
        "2016-01-01",
        "2018-01-01",
        "2020-01-01",
        "2022-01-01",
    ],
    "cross_validation.min_fold_events": 5,
    # Regime-shift low-confidence event threshold (Req 10.4). The zlb_end /
    # hiking_start boundaries are required (see _REQUIRED_NESTED_FIELDS).
    "regimes.min_events": 5,
    # Rigorous evaluation metrics (Req 8.2, 8.5).
    "evaluation.bootstrap_alpha": 0.05,
    "evaluation.bootstrap_iterations": 1000,
    "evaluation.min_n_stable_r2": 30,
    "model.baselines.historical_mean.rolling_window": 8,
    "visualization.dpi": 300,
    "visualization.format": ["pdf", "png"],
    "visualization.figsize_default": [8, 6],
    "visualization.font_family": "serif",
    "visualization.color_palette": "tab10",
}

# Stance parameters the Stance_Scorer reads exclusively from config (Req 8.1):
# the promoted signals, fusion settings, stance loss weight, and thresholds.
# validate() fails-fast naming any of these that is absent from both the config
# and _DEFAULTS (Req 8.2). Every entry below currently has a documented default
# in _DEFAULTS, so in practice they fall back rather than halting; the check
# guards against a parameter being introduced without a safe default.
_REQUIRED_STANCE_PARAMETERS = [
    "model.deberta.loss_weights.stance",
    "model.deberta.stance.enabled",
    "model.deberta.stance.temperature",
    "model.deberta.stance.fusion_hidden",
    "model.deberta.stance.neutral_band_bp",
    "model.deberta.stance.sharpness",
    "model.deberta.stance.default_emotion",
    "model.deberta.stance.default_expectation",
    "model.deberta.stance.primary_target",
    "model.deberta.stance.promote_signals",
    "market_context.emotion_source",
    "market_context.expectation_source",
]

# Council parameters the Council_Supervisor / Signal_Fusion read exclusively from
# config (Req 1, 2). validate() fails-fast naming any of these that is absent from
# both the config and _DEFAULTS, mirroring _REQUIRED_STANCE_PARAMETERS. fusion_mode
# and signals carry no safe default (also listed in _REQUIRED_NESTED_FIELDS);
# log_var_min / log_var_max have documented defaults, so in practice they fall back
# rather than halting -- the check guards against a Council parameter being
# introduced without a safe default.
_REQUIRED_COUNCIL_PARAMETERS = [
    "council.fusion_mode",
    "council.signals",
    "council.log_var_min",
    "council.log_var_max",
]

# Valid Council fusion modes (Req 2.1, 2.2). uncertainty = learned log-variance
# weighting; fixed = configured per-signal weights.
_VALID_COUNCIL_FUSION_MODES = ("uncertainty", "fixed")

# The provenance fields every council.signals[i] entry must carry so the
# Council_Supervisor can read each signal's source and identifier from config
# (Req 1.3). name identifies the Signal_Head; source/identifier map the signal to
# the stance-signal registry.
_REQUIRED_COUNCIL_SIGNAL_FIELDS = ("name", "source", "identifier")

# Master-retention override (Requirement 2.7 / Assumptions §4). Applied to the
# master parquet so pre-2004 and coverage-gap events are retained.
_MASTER_RETENTION_OVERRIDE: dict[str, Any] = {
    "splits.data_start": "2004-01-01",
    "splits.require_primary_target": True,
}

# Default target registry (Data Models §a). Additional targets (OIS-1Y, policy
# surprise, realised volatility, yield-curve slope/curvature) can be added under
# ``targets`` in config without code changes.
_DEFAULT_TARGET_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "ois2y",
        "source_dataset": "EA-CED",
        "source_series": "OIS_2Y",
        "path": "intraday",
        "event_window": {"start_offset_minutes": -15, "end_offset_minutes": 60},
        "formula": "within_window_change",
        "unit": "bp",
        "timezone": "unknown",
        "min_observations": 1,
        "has_flag_column": "has_ois_2y",
        "target_column": "ois_2y_change_bp",
    },
    {
        "name": "ois5y",
        "source_dataset": "EA-CED",
        "source_series": "OIS_5Y",
        "path": "intraday",
        "event_window": {"start_offset_minutes": -15, "end_offset_minutes": 60},
        "formula": "within_window_change",
        "unit": "bp",
        "timezone": "unknown",
        "min_observations": 1,
        "has_flag_column": "has_ois_5y",
        "target_column": "ois_5y_change_bp",
    },
    {
        "name": "ois10y",
        "source_dataset": "EA-CED",
        "source_series": "OIS_10Y",
        "path": "intraday",
        "event_window": {"start_offset_minutes": -15, "end_offset_minutes": 60},
        "formula": "within_window_change",
        "unit": "bp",
        "timezone": "unknown",
        "min_observations": 1,
        "has_flag_column": "has_ois_10y",
        "target_column": "ois_10y_change_bp",
    },
    {
        "name": "equity",
        "source_dataset": "EA-CED",
        "source_series": "EUROSTOXX",
        "path": "intraday",
        "event_window": {"start_offset_minutes": -15, "end_offset_minutes": 60},
        "formula": "within_window_return",
        "unit": "return",
        "timezone": "unknown",
        "min_observations": 1,
        "has_flag_column": "has_equity",
        "target_column": "eurostoxx_return",
    },
    {
        # EA-CED Sheet 2 contains a genuine per-event EURUSD return stamped with
        # the same ID/date/hour/minute as the IMC event. Keep it on the intraday
        # identity-aware path; the old NOT_AVAILABLE mapping discarded ~4.4k
        # observed returns and incorrectly produced an empty FX task.
        "name": "fx",
        "source_dataset": "EA-CED",
        "source_series": "EURUSD",
        "path": "intraday",
        "event_window": {"start_offset_minutes": -15, "end_offset_minutes": 60},
        "formula": "within_window_return",
        "unit": "return",
        "timezone": "unknown",
        "min_observations": 1,
        "has_flag_column": "has_fx",
        "target_column": "eurusd_return",
    },
]

# =============================================================================
# Promotable stance-signal registry (Design §5, Req 5.1, 8.7).
#
# Documented defaults for the market-context datasets that can be promoted from
# confounder-only controls to supervised stance signals. A dataset is promoted
# only when its ``name`` appears in ``model.deberta.stance.promote_signals``;
# each promotion becomes exactly one ``TargetSpec`` registry entry via the
# existing registry machinery (so build_targets / write_definition_report /
# standardize_units apply unchanged). Each entry documents the four provenance
# fields plus the derived unit/formula/flag/target column:
#   - source_dataset : originating dataset key
#   - source_series  : series identifier within that dataset
#   - unit           : native unit standardized to bp via standardize_units
#   - formula        : within-window change/return applied by construct_one
#   - has_flag_column: presence flag column written to the target parquet
#   - target_column  : column holding the standardized value
#
# The US monetary-policy-surprise ("mps") entry is an ECB-window EXPECTATION
# context signal only; it is NEVER emitted as an ECB stance label (Req 5.2) and
# so is intentionally omitted from the default ``promote_signals`` list.
# =============================================================================

_STANCE_SIGNAL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "vix",
        "source_dataset": "INDICES",
        "source_series": "^VIX",
        "path": "daily",
        # Daily VIX close changes are index points, not basis points.
        "unit": "index_point",
        "formula": "event_window_change",
        "has_flag_column": "has_vix",
        "target_column": "vix_change",
    },
    {
        "name": "dgs2",
        "source_dataset": "FRED",
        "source_series": "DGS2",
        "path": "daily",
        "unit": "bp",
        "formula": "within_window_change",
        "has_flag_column": "has_dgs2",
        "target_column": "dgs2_change_bp",
    },
    {
        "name": "dgs10",
        "source_dataset": "FRED",
        "source_series": "DGS10",
        "path": "daily",
        "unit": "bp",
        "formula": "within_window_change",
        "has_flag_column": "has_dgs10",
        "target_column": "dgs10_change_bp",
    },
    {
        "name": "dgs30",
        "source_dataset": "FRED",
        "source_series": "DGS30",
        "path": "daily",
        "unit": "bp",
        "formula": "within_window_change",
        "has_flag_column": "has_dgs30",
        "target_column": "dgs30_change_bp",
    },
    {
        "name": "equity",
        "source_dataset": "EA-CED",
        "source_series": "EUROSTOXX",
        "path": "intraday",
        "unit": "return",
        "formula": "within_window_return",
        "has_flag_column": "has_equity",
        "target_column": "eurostoxx_return",
    },
    {
        # ECB-window market-EXPECTATION context signal (US FOMC surprise). Used
        # as context only via build_expectation_signal; never promoted as an ECB
        # stance label (Req 5.2), hence excluded from the default promote list.
        "name": "mps",
        "source_dataset": "US_MPS",
        "source_series": "mps",
        "path": "daily",
        "unit": "bp",
        "formula": "within_window_change",
        "has_flag_column": "has_mps",
        "target_column": "mps_change_bp",
    },
]

# The four provenance fields every promoted stance spec must carry before it can
# become a Target_Registry entry (Req 5.3, 5.4). A spec missing any of the four
# is excluded from promotion and the corresponding enumerated reason below is
# recorded, mirroring the enumerated missing-reason convention in
# ``src/target_construction.py`` (no ad-hoc strings).
_PROMOTION_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_dataset",
    "source_series",
    "unit",
    "event_window",
)

# Enumerated exclusion reasons for a promoted spec missing a provenance field
# (Req 5.4). One code per required field, keyed by field name in
# ``_MISSING_PROVENANCE_REASON`` so the builder can look up the exact code.
REASON_MISSING_PROVENANCE_SOURCE_DATASET = "REASON_MISSING_PROVENANCE_SOURCE_DATASET"
REASON_MISSING_PROVENANCE_SOURCE_SERIES = "REASON_MISSING_PROVENANCE_SOURCE_SERIES"
REASON_MISSING_PROVENANCE_UNIT = "REASON_MISSING_PROVENANCE_UNIT"
REASON_MISSING_PROVENANCE_EVENT_WINDOW = "REASON_MISSING_PROVENANCE_EVENT_WINDOW"

_MISSING_PROVENANCE_REASON: dict[str, str] = {
    "source_dataset": REASON_MISSING_PROVENANCE_SOURCE_DATASET,
    "source_series": REASON_MISSING_PROVENANCE_SOURCE_SERIES,
    "unit": REASON_MISSING_PROVENANCE_UNIT,
    "event_window": REASON_MISSING_PROVENANCE_EVENT_WINDOW,
}

# Non-ECB source datasets that are market CONTEXT only and must never be emitted
# as an ECB stance label (Req 5.2). The US monetary-policy-surprise series
# ("mps" / source_dataset "US_MPS") is a US FOMC signal used as expectation
# context via ``build_expectation_signal`` -- promoting it as an ECB label would
# mislabel a non-ECB event, so it is filtered out regardless of ``promote_signals``.
_NON_ECB_CONTEXT_DATASETS: frozenset[str] = frozenset({"US_MPS"})

# Default event window applied to a promoted spec that does not declare its own,
# matching the default used by ``ConfigManager.target_registry`` so promoted
# specs share the standard [-15, +60] minute window unless overridden.
_PROMOTION_DEFAULT_EVENT_WINDOW: dict[str, int] = {
    "start_offset_minutes": -15,
    "end_offset_minutes": 60,
}


def build_promoted_stance_specs(
    config: dict[str, Any],
) -> tuple[list[TargetSpec], list[dict[str, Any]]]:
    """Build promoted ``TargetSpec`` entries from ``_STANCE_SIGNAL_REGISTRY``.

    For each dataset name listed in ``model.deberta.stance.promote_signals`` this
    looks up the matching ``_STANCE_SIGNAL_REGISTRY`` entry and, using the same
    registry machinery as ``target_registry`` (so ``build_targets``,
    ``write_definition_report`` and ``standardize_units`` apply unchanged),
    produces **exactly one** ``TargetSpec`` per promotable dataset (Req 5.1).

    A dataset whose ``source_dataset`` is a non-ECB context dataset (e.g. the US
    ``US_MPS`` monetary-policy-surprise series) is never promoted as an ECB stance
    label (Req 5.2) -- it is skipped even if named in ``promote_signals``.

    A promotable entry missing any of the four provenance fields (source dataset,
    source series, unit, event window) is excluded from promotion and an
    enumerated ``REASON_MISSING_PROVENANCE_<field>`` reason is recorded (Req 5.4).

    Returns:
        A ``(specs, exclusions)`` tuple where ``specs`` is the list of promoted
        ``TargetSpec`` entries (one per successfully promoted dataset) and
        ``exclusions`` is a list of ``{"name", "reason", "field"}`` records for
        every requested dataset that could not be promoted.
    """
    requested = _get_nested(config, "model.deberta.stance.promote_signals", []) or []
    by_name = {entry["name"]: entry for entry in _STANCE_SIGNAL_REGISTRY}

    specs: list[TargetSpec] = []
    exclusions: list[dict[str, Any]] = []

    for name in requested:
        entry = by_name.get(name)
        if entry is None:
            exclusions.append(
                {"name": name, "reason": "REASON_UNKNOWN_PROMOTE_SIGNAL", "field": None}
            )
            continue

        # Req 5.2: a non-ECB dataset is retained as context only and never
        # promoted as an ECB stance label.
        if entry.get("source_dataset") in _NON_ECB_CONTEXT_DATASETS:
            exclusions.append(
                {"name": name, "reason": "REASON_NON_ECB_CONTEXT_ONLY", "field": None}
            )
            continue

        # Req 5.4: exclude a spec missing any of the four provenance fields,
        # recording the enumerated reason for the first missing field.
        missing_field = _first_missing_provenance_field(entry)
        if missing_field is not None:
            exclusions.append(
                {
                    "name": name,
                    "reason": _MISSING_PROVENANCE_REASON[missing_field],
                    "field": missing_field,
                }
            )
            continue

        specs.append(_promoted_spec_from_entry(entry))

    return specs, exclusions


def _provenance_value_present(value: Any) -> bool:
    """True when a provenance field carries a usable (non-empty) value."""
    if value is None:
        return False
    if isinstance(value, str) and value.strip() == "":
        return False
    if isinstance(value, dict) and not value:
        return False
    return True


def _first_missing_provenance_field(entry: dict[str, Any]) -> Optional[str]:
    """Return the first missing/empty provenance field name, or ``None``.

    The event-window field defaults to the standard promotion window, so it is
    considered present unless the entry declares it explicitly with an empty
    value; the other three fields must be supplied by the registry entry.
    """
    for field_name in _PROMOTION_PROVENANCE_FIELDS:
        if field_name == "event_window":
            # event_window is optional in the registry entry and defaults to the
            # standard window; only an explicitly-present-but-empty value counts
            # as missing.
            if "event_window" in entry and not _provenance_value_present(
                entry["event_window"]
            ):
                return "event_window"
            continue
        if not _provenance_value_present(entry.get(field_name)):
            return field_name
    return None


def _promoted_spec_from_entry(entry: dict[str, Any]) -> TargetSpec:
    """Build one ``TargetSpec`` from a promotable registry entry.

    Uses the same field-resolution defaults as ``ConfigManager.target_registry``
    so promoted specs are indistinguishable from config-declared targets to
    ``build_targets`` / ``write_definition_report`` / ``standardize_units``.
    """
    return TargetSpec(
        name=entry["name"],
        source_dataset=entry["source_dataset"],
        source_series=entry["source_series"],
        path=entry.get("path", "intraday"),
        event_window=entry.get("event_window", dict(_PROMOTION_DEFAULT_EVENT_WINDOW)),
        formula=entry.get("formula", "within_window_change"),
        unit=entry["unit"],
        timezone=entry.get("timezone", "unknown"),
        min_observations=int(entry.get("min_observations", 1)),
        has_flag_column=entry.get("has_flag_column", f"has_{entry['name']}"),
        target_column=entry.get("target_column", entry["name"]),
        mpd_window=entry.get("mpd_window"),
    )


# =============================================================================
# Council signal registry (Design §5, Req 1.2, 1.3).
#
# The Council Market_Signal_Vector is defined by ``council.signals`` in config,
# where each entry carries {name, source, identifier, weight}. To turn each entry
# into exactly one ``TargetSpec`` we resolve its (source, identifier) against the
# known signal registries, reusing the same registry machinery as
# ``build_promoted_stance_specs`` / ``_promoted_spec_from_entry`` so
# ``build_targets`` / ``construct_one`` / ``standardize_units`` apply unchanged.
#
# Resolution draws from three sources keyed by (source_dataset, source_series):
#   - ``_STANCE_SIGNAL_REGISTRY``   (FRED DGS2/10/30, INDICES ^VIX, EA-CED equity)
#   - the EA-MPD multi-window matrix (EA-MPD OIS_1Y/2Y/5Y__ME target/path/timing)
#   - the default target registry    (EA-CED OIS_2Y and friends)
#   - ``_COUNCIL_SIGNAL_REGISTRY`` below supplies any (source, identifier) pair a
#     configured council signal needs that none of the above already provides.
#
# This guarantees all nine configured signals (DGS2/10/30, EA-MPD
# target/path/timing, VIX, equity, OIS) resolve to a spec. ``source`` maps to
# ``source_dataset``, ``identifier`` to ``source_series``, and ``weight`` to the
# fixed-mode weight carried on the produced spec's metadata (Req 1.3).
# =============================================================================

# Supplemental council registry entries: (source_dataset, source_series) pairs a
# configured council signal may reference that are not already covered by the
# stance/EA-MPD/default registries. EA-CED OIS_2Y is the canonical short-rate
# ("ois") signal; the EA-MPD OIS windows are covered by the EA-MPD matrix but are
# duplicated here defensively so council resolution never depends on matrix
# construction order.
_COUNCIL_SIGNAL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "ois",
        "source_dataset": "EA-CED",
        "source_series": "OIS_2Y",
        "path": "intraday",
        "unit": "bp",
        "formula": "within_window_change",
        "has_flag_column": "has_ois",
        "target_column": "ois_change_bp",
    },
    {
        "name": "ea_mpd_target",
        "source_dataset": "EA-MPD",
        "source_series": "OIS_2Y__ME",
        "path": "daily",
        "mpd_window": "monetary_event",
        "unit": "bp",
        "formula": "event_window_change",
        "has_flag_column": "has_ea_mpd_target",
        "target_column": "ea_mpd_target_change_bp",
    },
    {
        "name": "ea_mpd_path",
        "source_dataset": "EA-MPD",
        "source_series": "OIS_5Y__ME",
        "path": "daily",
        "mpd_window": "monetary_event",
        "unit": "bp",
        "formula": "event_window_change",
        "has_flag_column": "has_ea_mpd_path",
        "target_column": "ea_mpd_path_change_bp",
    },
    {
        "name": "ea_mpd_timing",
        "source_dataset": "EA-MPD",
        "source_series": "OIS_1Y__ME",
        "path": "daily",
        "mpd_window": "monetary_event",
        "unit": "bp",
        "formula": "event_window_change",
        "has_flag_column": "has_ea_mpd_timing",
        "target_column": "ea_mpd_timing_change_bp",
    },
]

# The three provenance fields a configured council.signals[i] entry must carry so
# it can be turned into a TargetSpec (Req 1.3). ``name`` identifies the
# Signal_Head; ``source``/``identifier`` map to source_dataset/source_series.
_REQUIRED_COUNCIL_SIGNAL_ENTRY_FIELDS: tuple[str, ...] = ("name", "source", "identifier")

# Enumerated exclusion reasons for a council signal that cannot be promoted to a
# TargetSpec (Req 1.3). No ad-hoc strings -- mirror the enumerated-reason
# convention used by the stance-promotion path and target_construction.
REASON_COUNCIL_MISSING_NAME = "REASON_COUNCIL_MISSING_NAME"
REASON_COUNCIL_MISSING_SOURCE = "REASON_COUNCIL_MISSING_SOURCE"
REASON_COUNCIL_MISSING_IDENTIFIER = "REASON_COUNCIL_MISSING_IDENTIFIER"
REASON_COUNCIL_UNKNOWN_SIGNAL = "REASON_COUNCIL_UNKNOWN_SIGNAL"

_COUNCIL_MISSING_ENTRY_FIELD_REASON: dict[str, str] = {
    "name": REASON_COUNCIL_MISSING_NAME,
    "source": REASON_COUNCIL_MISSING_SOURCE,
    "identifier": REASON_COUNCIL_MISSING_IDENTIFIER,
}


def _council_signal_lookup() -> dict[tuple[str, str], dict[str, Any]]:
    """Build a (source_dataset, source_series) -> registry-entry lookup.

    Combines the stance-signal registry, the EA-MPD multi-window matrix, the
    default target registry, and the supplemental council registry so every
    configured council signal's (source, identifier) resolves to a base entry.
    Earlier sources win on collision; the supplemental council registry is added
    last only for pairs not already present.
    """
    lookup: dict[tuple[str, str], dict[str, Any]] = {}

    def _add(entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            key = (entry.get("source_dataset"), entry.get("source_series"))
            if key[0] is None or key[1] is None:
                continue
            lookup.setdefault(key, entry)

    _add(_STANCE_SIGNAL_REGISTRY)
    _add(_ea_mpd_target_registry())
    _add(_DEFAULT_TARGET_REGISTRY)
    _add(_COUNCIL_SIGNAL_REGISTRY)
    return lookup


def build_council_signal_specs(
    config: dict[str, Any],
) -> tuple[list[TargetSpec], list[dict[str, Any]]]:
    """Build one ``TargetSpec`` per configured Council signal (Req 1.2, 1.3).

    Reads ``council.signals`` and, for each entry, resolves its ``source`` /
    ``identifier`` against the combined signal registry (stance + EA-MPD matrix +
    default targets + supplemental council entries) to obtain the provenance
    (unit, formula, path, window, flag/target columns). Each resolved signal
    becomes **exactly one** ``TargetSpec`` built via ``_promoted_spec_from_entry``
    -- the same machinery used for promoted stance specs -- so ``build_targets`` /
    ``construct_one`` / ``standardize_units`` apply unchanged.

    ``source`` maps to ``source_dataset``, ``identifier`` to ``source_series``,
    and ``weight`` (fixed-mode weight) is carried through on the produced spec's
    ``name`` so the fixed-fusion path can read ``w(name)``.

    Returns:
        A ``(specs, exclusions)`` tuple. ``specs`` holds one ``TargetSpec`` per
        successfully resolved council signal; ``exclusions`` holds
        ``{"name", "reason", "field"}`` records for any signal missing a required
        entry field or whose (source, identifier) is unknown, with an enumerated
        ``REASON_COUNCIL_*`` reason.
    """
    signals = _get_nested(config, "council.signals", []) or []
    lookup = _council_signal_lookup()

    specs: list[TargetSpec] = []
    exclusions: list[dict[str, Any]] = []

    for signal in signals:
        signal = signal if isinstance(signal, dict) else {}

        # Req 1.3: every council signal must carry name/source/identifier.
        missing_field = next(
            (
                f
                for f in _REQUIRED_COUNCIL_SIGNAL_ENTRY_FIELDS
                if not _provenance_value_present(signal.get(f))
            ),
            None,
        )
        if missing_field is not None:
            exclusions.append(
                {
                    "name": signal.get("name"),
                    "reason": _COUNCIL_MISSING_ENTRY_FIELD_REASON[missing_field],
                    "field": missing_field,
                }
            )
            continue

        name = signal["name"]
        source = signal["source"]
        identifier = signal["identifier"]

        base = lookup.get((source, identifier))
        if base is None:
            exclusions.append(
                {
                    "name": name,
                    "reason": REASON_COUNCIL_UNKNOWN_SIGNAL,
                    "field": None,
                }
            )
            continue

        # Build the entry that _promoted_spec_from_entry consumes: inherit the
        # resolved provenance but override name/source/identifier from config so
        # the produced spec uses the council signal's own name (drives the
        # Signal_Head and the fixed-mode weight key) and its configured source /
        # identifier (Req 1.3).
        entry = dict(base)
        entry["name"] = name
        entry["source_dataset"] = source
        entry["source_series"] = identifier
        # Default the presence/target columns to the signal name when the base
        # entry did not name them explicitly, so distinct council names never
        # collide on a shared base series.
        entry.setdefault("has_flag_column", f"has_{name}")
        entry.setdefault("target_column", name)

        specs.append(_promoted_spec_from_entry(entry))

    return specs, exclusions


# =============================================================================
# EA-MPD multi-window target matrix (Bund / OIS / equity / EURUSD across the
# Press Release, Press Conference, and Monetary Event windows). DATE-KEYED daily
# event-window changes (no intraday timestamp). Each entry uses a UNIQUE
# source_series "<COL>__<WIN>" so per-window series never collide. US monetary-
# policy-surprise files are NOT targets (US FOMC data; confounder controls only).
# =============================================================================

EA_MPD_WINDOWS: dict[str, str] = {
    "press_release": "PR",
    "press_conference": "PC",
    "monetary_event": "ME",
}

_EA_MPD_TARGET_COLUMNS: list[tuple[str, str, str, str]] = [
    ("DE2Y", "bund2y", "bp", "event_window_change"),
    ("DE5Y", "bund5y", "bp", "event_window_change"),
    ("DE10Y", "bund10y", "bp", "event_window_change"),
    ("OIS_1Y", "ois1y", "bp", "event_window_change"),
    ("OIS_2Y", "mpd_ois2y", "bp", "event_window_change"),
    ("OIS_5Y", "mpd_ois5y", "bp", "event_window_change"),
    ("OIS_10Y", "mpd_ois10y", "bp", "event_window_change"),
    ("STOXX50", "stoxx50", "return", "event_window_return"),
    ("EURUSD", "eurusd", "return", "event_window_return"),
]


def _ea_mpd_target_registry() -> list[dict[str, Any]]:
    """Build the full EA-MPD target matrix: 3 windows x target columns."""
    entries: list[dict[str, Any]] = []
    for window_key, tag in EA_MPD_WINDOWS.items():
        for col, stem, unit, formula in _EA_MPD_TARGET_COLUMNS:
            name = f"{stem}_{tag.lower()}"
            series_key = f"{col}__{tag}"
            entries.append({
                "name": name,
                "source_dataset": "EA-MPD",
                "source_series": series_key,
                "path": "daily",
                "mpd_window": window_key,
                "event_window": {"start_offset_minutes": -15, "end_offset_minutes": 60},
                "formula": formula,
                "unit": unit,
                "timezone": "unknown",
                "min_observations": 1,
                "has_flag_column": f"has_{name}",
                "target_column": f"{name}_change" if unit == "bp" else f"{name}_return",
            })
    return entries


_VALID_PRIMARY_METRICS = [
    "mse",
    "mae",
    "r_squared",
    "directional_accuracy",
    "spearman_correlation",
]


# =============================================================================
# Nested dict helpers
# =============================================================================

def _get_nested(config: dict, dotpath: str, default: Any = None) -> Any:
    """Retrieve a nested value from a dict using dot-notation path."""
    keys = dotpath.split(".")
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _set_nested(config: dict, dotpath: str, value: Any) -> None:
    """Set a nested value in a dict using dot-notation path."""
    keys = dotpath.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


# =============================================================================
# ConfigManager
# =============================================================================

class ConfigManager:
    """Central configuration, reproducibility, and experiment tracking.

    Loads a YAML experiment config, validates required fields, resolves defaults,
    rebases paths onto ``project_root``, applies the master-retention override,
    assigns UUID v4 experiment IDs, and records the runtime environment.
    """

    def load_config(self, path: str, project_root: Optional[str] = None) -> dict[str, Any]:
        """Load, validate, resolve defaults, rebase paths, and apply the override.

        Raises:
            ConfigError: If the file is missing, unreadable, empty, or invalid.
        """
        if not os.path.exists(path):
            raise ConfigError(f"Configuration file not found: {path}")

        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in configuration file: {e}")

        if config is None:
            raise ConfigError("Configuration file is empty")

        errors = self.validate(config)
        if errors:
            raise ConfigError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {err}" for err in errors)
            )

        config = self.resolve_defaults(config)
        if project_root is not None:
            config = self.rebase_paths(config, project_root)
        config = self.apply_master_retention_override(config)

        # Missing-file halt (Req 14.5): name any absent required input dataset
        # file before any stage runs. Skipped while datasets still resolve under
        # the committed Colab base (paths that only exist on disk once rebased
        # onto the current machine via project_root), so the check runs against
        # real, machine-local paths rather than the un-rebased Colab template.
        datasets_dir = _get_nested(config, "data.datasets_dir") or ""
        if not datasets_dir.startswith(_COLAB_BASE):
            file_errors = self._validate_required_files(config)
            if file_errors:
                raise ConfigError(
                    "Configuration validation failed:\n"
                    + "\n".join(f"  - {err}" for err in file_errors)
                )
        return config

    def rebase_paths(self, config: dict[str, Any], project_root: str) -> dict[str, Any]:
        """Rebase Colab-style absolute paths onto ``project_root``.

        Any path under the committed Colab base
        (``/content/drive/MyDrive/market_supervised``) is rewritten to sit under
        ``project_root`` so datasets and outputs resolve on the current machine.
        """
        colab_base = _COLAB_BASE
        project_root = project_root.rstrip("/")
        if project_root == colab_base:
            return config

        def _rebase(value: Any) -> Any:
            if isinstance(value, str) and value.startswith(colab_base):
                return project_root + value[len(colab_base):]
            return value

        for dotpath in ("environment.base_path", "data.datasets_dir", "data.outputs_dir"):
            current = _get_nested(config, dotpath)
            if current is not None:
                _set_nested(config, dotpath, _rebase(current))

        return config

    def resolve_defaults(self, config: dict[str, Any]) -> dict[str, Any]:
        """Fill missing optional fields with documented defaults."""
        for dotpath, default_value in _DEFAULTS.items():
            if _get_nested(config, dotpath) is None:
                _set_nested(config, dotpath, default_value)
        return config

    def apply_master_retention_override(self, config: dict[str, Any]) -> dict[str, Any]:
        """Apply the master-retention override (Requirement 2.7).

        Forces ``splits.data_start=2004-01-01`` and
        ``splits.require_primary_target=true`` so the master parquet retains all
        pre-2004 and coverage-gap events. These values still govern the training
        splits for supervised fine-tuning (Assumptions §4).
        """
        for dotpath, value in _MASTER_RETENTION_OVERRIDE.items():
            _set_nested(config, dotpath, value)
        return config

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Return a list of validation error messages (empty if valid).

        Includes target-registry checks (Requirement 12).
        """
        errors: list[str] = []

        for key in _REQUIRED_CONFIG_KEYS:
            if key not in config:
                errors.append(f"Missing required top-level key: '{key}'")

        for dotpath in _REQUIRED_NESTED_FIELDS:
            if _get_nested(config, dotpath) is None:
                errors.append(f"Missing required field: '{dotpath}'")

        start_offset = _get_nested(config, "market_windows.start_offset_minutes")
        end_offset = _get_nested(config, "market_windows.end_offset_minutes")
        if start_offset is not None and end_offset is not None:
            if start_offset >= end_offset:
                errors.append(
                    f"Invalid market window: start_offset_minutes ({start_offset}) "
                    f"must be less than end_offset_minutes ({end_offset})"
                )

        # experiment.seed must be a non-negative integer (Req 8.4). A missing
        # seed is already reported by the _REQUIRED_NESTED_FIELDS check above; an
        # invalid (non-int or negative) seed halts here without substituting a
        # default. bool is an int subclass, so reject it explicitly.
        seed = _get_nested(config, "experiment.seed")
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                errors.append(
                    f"experiment.seed must be a non-negative integer, got: "
                    f"{type(seed).__name__}"
                )
            elif seed < 0:
                errors.append(
                    f"experiment.seed must be a non-negative integer, got: {seed}"
                )

        primary_metric = _get_nested(config, "experiment.primary_metric")
        if primary_metric is not None and primary_metric not in _VALID_PRIMARY_METRICS:
            errors.append(
                f"experiment.primary_metric '{primary_metric}' not recognized. "
                f"Valid options: {_VALID_PRIMARY_METRICS}"
            )

        errors.extend(self._validate_target_registry(config))
        errors.extend(self._validate_stance_parameters(config))
        errors.extend(self._validate_council_parameters(config))
        return errors

    def _validate_council_parameters(self, config: dict[str, Any]) -> list[str]:
        """Validate the Council_Supervisor / Signal_Fusion parameters (Req 1, 2).

        Every Council parameter is read exclusively from config. A required
        Council parameter that is absent from both the config and the documented
        ``_DEFAULTS`` has no safe fallback, so validation fails-fast naming the
        missing parameter, mirroring ``_validate_stance_parameters``. Parameters
        absent from the config but present in ``_DEFAULTS`` (``log_var_min`` /
        ``log_var_max``) are optional -- ``resolve_defaults`` populates them, so
        they are not reported here (validate runs before resolve_defaults).

        Beyond the missing-key check this enforces:

        * ``council.signals`` is a list of at least two heterogeneous signals
          (Req 1.5); a present-but-too-small set halts initialization with a
          descriptive error.
        * every ``council.signals[i]`` carries ``name`` / ``source`` /
          ``identifier`` so the Council_Supervisor can read each signal's source
          and identifier from config (Req 1.3).
        * ``council.fusion_mode`` is one of ``uncertainty`` / ``fixed``
          (Req 2.1, 2.2).
        * ``council.log_var_min < council.log_var_max`` so the log-variance clamp
          bounds a non-empty interval and ``exp(-log_var)`` weights stay finite
          and bounded (Req 2.4).
        """
        errors: list[str] = []

        # Fail-fast naming any required Council key with no safe default (mirrors
        # _validate_stance_parameters). An absent key present in _DEFAULTS is
        # optional and populated by resolve_defaults, so it is not reported.
        for dotpath in _REQUIRED_COUNCIL_PARAMETERS:
            if _get_nested(config, dotpath) is None and dotpath not in _DEFAULTS:
                errors.append(
                    f"Missing required council parameter '{dotpath}' with no "
                    f"safe default; add it to the configuration file"
                )

        # council.signals: at least two signals, each carrying name/source/
        # identifier (Req 1.3, 1.5). An absent set is already reported by the
        # required-key check above.
        council_signals = _get_nested(config, "council.signals")
        if council_signals is not None:
            if not isinstance(council_signals, list):
                errors.append(
                    "council.signals must be a list of signal definitions, got: "
                    f"{type(council_signals).__name__}"
                )
            else:
                if len(council_signals) < 2:
                    errors.append(
                        f"council.signals must contain at least two signals, got: "
                        f"{len(council_signals)}"
                    )
                for i, signal in enumerate(council_signals):
                    if not isinstance(signal, dict):
                        errors.append(
                            f"council.signals[{i}] must be a mapping with "
                            f"{list(_REQUIRED_COUNCIL_SIGNAL_FIELDS)}, got: "
                            f"{type(signal).__name__}"
                        )
                        continue
                    missing = [
                        f for f in _REQUIRED_COUNCIL_SIGNAL_FIELDS if not signal.get(f)
                    ]
                    if missing:
                        errors.append(
                            f"council.signals[{i}] "
                            f"('{signal.get('name', '?')}') missing required "
                            f"fields: {missing}"
                        )

        # council.fusion_mode: uncertainty | fixed (Req 2.1, 2.2). An absent mode
        # is already reported by the required-key check above.
        fusion_mode = _get_nested(config, "council.fusion_mode")
        if fusion_mode is not None and fusion_mode not in _VALID_COUNCIL_FUSION_MODES:
            errors.append(
                f"council.fusion_mode '{fusion_mode}' not recognized. Valid "
                f"options: {list(_VALID_COUNCIL_FUSION_MODES)}"
            )

        # council.log_var_min < council.log_var_max (Req 2.4). validate runs
        # before resolve_defaults, so fall back to the documented defaults for an
        # absent bound to mirror the effective run behavior.
        log_var_min = _get_nested(config, "council.log_var_min")
        if log_var_min is None:
            log_var_min = _DEFAULTS.get("council.log_var_min")
        log_var_max = _get_nested(config, "council.log_var_max")
        if log_var_max is None:
            log_var_max = _DEFAULTS.get("council.log_var_max")
        if (
            isinstance(log_var_min, (int, float))
            and not isinstance(log_var_min, bool)
            and isinstance(log_var_max, (int, float))
            and not isinstance(log_var_max, bool)
            and log_var_min >= log_var_max
        ):
            errors.append(
                f"council.log_var_min ({log_var_min}) must be less than "
                f"council.log_var_max ({log_var_max})"
            )

        return errors

    def _validate_required_files(self, config: dict[str, Any]) -> list[str]:
        """Return errors naming any absent required input dataset file (Req 14.5).

        Resolves each ``data.required_files`` entry against ``data.datasets_dir``
        and reports every file that is not present on disk, so a missing input is
        named before any stage runs. Called from ``load_config`` only once paths
        have been rebased onto the current machine (i.e. when a ``project_root``
        is supplied), because the committed config carries Colab-style absolute
        paths that do not resolve until rebasing.
        """
        errors: list[str] = []
        datasets_dir = _get_nested(config, "data.datasets_dir")
        required = _get_nested(config, "data.required_files") or []
        if not isinstance(required, list):
            errors.append(
                "data.required_files must be a list of dataset file paths, got: "
                f"{type(required).__name__}"
            )
            return errors
        for rel in required:
            path = os.path.join(datasets_dir, rel) if datasets_dir else rel
            if not os.path.exists(path):
                errors.append(f"Required input dataset file is absent: '{path}'")
        return errors

    def _validate_stance_parameters(self, config: dict[str, Any]) -> list[str]:
        """Validate the stance-scoring parameters (Req 8.1, 8.2, 8.3).

        Every stance parameter is read exclusively from config (Req 8.1). A
        required stance parameter that is absent from both the config and the
        documented ``_DEFAULTS`` has no safe fallback, so validation fails-fast
        naming the missing parameter (Req 8.2). Parameters that are absent from
        the config but present in ``_DEFAULTS`` are optional: ``resolve_defaults``
        populates them, so they are not reported here (validate runs before
        resolve_defaults). This means a stance parameter never falls back to a
        hard-coded value silently, yet the documented defaults still apply.
        """
        errors: list[str] = []
        for dotpath in _REQUIRED_STANCE_PARAMETERS:
            if _get_nested(config, dotpath) is None and dotpath not in _DEFAULTS:
                errors.append(
                    f"Missing required stance parameter '{dotpath}' with no "
                    f"safe default; add it to the configuration file"
                )

        # Resolve the market-context source keys against data.confounder_files
        # (Req 9.3). Only enforced when stance is enabled; validate runs before
        # resolve_defaults, so an unset enabled flag falls back to its documented
        # default to mirror the effective run behavior.
        enabled = _get_nested(config, "model.deberta.stance.enabled")
        if enabled is None:
            enabled = _DEFAULTS.get("model.deberta.stance.enabled", False)
        if enabled:
            confounder_files = _get_nested(config, "data.confounder_files")
            confounder_keys = (
                confounder_files if isinstance(confounder_files, dict) else {}
            )
            for source_dotpath in (
                "market_context.emotion_source",
                "market_context.expectation_source",
            ):
                source_key = _get_nested(config, source_dotpath)
                if source_key is None:
                    source_key = _DEFAULTS.get(source_dotpath)
                if source_key is not None and source_key not in confounder_keys:
                    errors.append(
                        f"Unresolved stance source key: '{source_dotpath}' names "
                        f"'{source_key}', which is not present in "
                        f"data.confounder_files"
                    )
        return errors

    def _validate_target_registry(self, config: dict[str, Any]) -> list[str]:
        """Validate any user-declared target registry entries (Requirement 12)."""
        errors: list[str] = []
        raw = _get_nested(config, "targets")
        if raw is None:
            return errors  # Falls back to the default registry.
        if not isinstance(raw, list):
            errors.append("config 'targets' must be a list of target specs")
            return errors
        required_fields = {
            "name",
            "source_dataset",
            "source_series",
            "path",
            "unit",
            "has_flag_column",
            "target_column",
        }
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                errors.append(f"targets[{i}] must be a mapping")
                continue
            missing = required_fields - set(entry.keys())
            if missing:
                errors.append(
                    f"targets[{i}] ('{entry.get('name', '?')}') missing fields: "
                    f"{sorted(missing)}"
                )
            if entry.get("path") not in (None, "intraday", "daily"):
                errors.append(
                    f"targets[{i}] path must be 'intraday' or 'daily', "
                    f"got '{entry.get('path')}'"
                )
        return errors

    def target_registry(self, config: dict[str, Any]) -> list[TargetSpec]:
        """Return the list of ``TargetSpec`` for this config.

        Uses the config's ``targets`` list when present, otherwise the default
        registry (``ois2y``, ``ois5y``, ``ois10y``, ``equity``, ``fx``). Additional
        registry entries produce additional specs with no code change.
        """
        raw = _get_nested(config, "targets")
        entries = raw if isinstance(raw, list) and raw else _DEFAULT_TARGET_REGISTRY
        specs: list[TargetSpec] = []
        for entry in entries:
            specs.append(
                TargetSpec(
                    name=entry["name"],
                    source_dataset=entry["source_dataset"],
                    source_series=entry["source_series"],
                    path=entry.get("path", "intraday"),
                    event_window=entry.get(
                        "event_window",
                        {"start_offset_minutes": -15, "end_offset_minutes": 60},
                    ),
                    formula=entry.get("formula", "within_window_change"),
                    unit=entry["unit"],
                    timezone=entry.get("timezone", "unknown"),
                    min_observations=int(entry.get("min_observations", 1)),
                    has_flag_column=entry["has_flag_column"],
                    target_column=entry["target_column"],
                    mpd_window=entry.get("mpd_window"),
                )
            )
        return specs

    def promoted_stance_specs(
        self, config: dict[str, Any]
    ) -> tuple[list[TargetSpec], list[dict[str, Any]]]:
        """Return promoted stance ``TargetSpec`` entries and exclusion records.

        Thin wrapper over :func:`build_promoted_stance_specs`. For each dataset
        enabled in ``model.deberta.stance.promote_signals`` this yields exactly
        one ``TargetSpec`` built via the existing registry machinery (Req 5.1),
        never promotes a non-ECB dataset as an ECB label (Req 5.2), and excludes
        any spec missing one of the four provenance fields with an enumerated
        ``REASON_MISSING_PROVENANCE_<field>`` reason (Req 5.4). The returned specs
        can be concatenated with ``target_registry`` output so ``build_targets``,
        ``write_definition_report`` and ``standardize_units`` apply unchanged.
        """
        return build_promoted_stance_specs(config)

    def council_signal_specs(
        self, config: dict[str, Any]
    ) -> tuple[list[TargetSpec], list[dict[str, Any]]]:
        """Return one ``TargetSpec`` per configured Council signal + exclusions.

        Thin wrapper over :func:`build_council_signal_specs`. Reads
        ``council.signals`` and resolves each entry's ``source`` / ``identifier``
        against the combined signal registry, producing exactly one ``TargetSpec``
        per resolved signal via the same registry machinery as the promoted stance
        specs (Req 1.2, 1.3). Unpromotable signals are returned as enumerated
        ``REASON_COUNCIL_*`` exclusion records. The returned specs can be
        concatenated with ``target_registry`` output so ``build_targets`` /
        ``construct_one`` / ``standardize_units`` apply unchanged.
        """
        return build_council_signal_specs(config)

    def create_experiment(self, config: dict[str, Any]) -> ExperimentRun:
        """Assign a UUID, create the output directory tree, and record environment."""
        experiment_id = str(uuid.uuid4())
        outputs_dir = _get_nested(config, "data.outputs_dir", "outputs")
        output_dir = os.path.join(outputs_dir, experiment_id)

        for sub in ("", "intermediate", "models", "evaluation", "figures", "final"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

        environment = self.record_environment()
        run = ExperimentRun(
            experiment_id=experiment_id,
            config=config,
            environment=environment,
            output_dir=output_dir,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        env_path = os.path.join(output_dir, "environment.json")
        with open(env_path, "w") as f:
            json.dump(
                {
                    "python_version": environment.python_version,
                    "packages": environment.packages,
                    "gpu_name": environment.gpu_name,
                    "gpu_driver": environment.gpu_driver,
                    "platform_info": environment.platform_info,
                    "timestamp": environment.timestamp,
                },
                f,
                indent=2,
            )
        return run

    def save_resolved_config(self, run: ExperimentRun) -> None:
        """Save the fully-resolved config to the experiment output directory.

        Persists the fully-resolved configuration (after ``resolve_defaults`` /
        ``rebase_paths`` / master-retention override) alongside the experiment
        artifacts as ``config.yaml`` (Req 14.4). A stable ``config_hash`` of the
        resolved config is written to a companion ``config_hash.txt`` so the
        exact configuration that produced a run is recoverable and checkpoints
        can be invalidated when the resolved config changes.
        """
        config_path = os.path.join(run.output_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(run.config, f, default_flow_style=False, sort_keys=False)

        # Persist the resolved config's stable hash alongside the config so the
        # config that produced this run is recoverable and verifiable (Req 14.4).
        hash_path = os.path.join(run.output_dir, "config_hash.txt")
        with open(hash_path, "w") as f:
            f.write(config_hash(run.config) + "\n")

    def record_environment(self) -> EnvironmentInfo:
        """Capture Python version, installed packages, GPU name, and driver."""
        python_version = platform.python_version()
        platform_info = f"{platform.system()} {platform.release()}"
        timestamp = datetime.now(timezone.utc).isoformat()

        packages: dict[str, str] = {}
        try:
            import subprocess

            result = subprocess.run(
                ["pip", "freeze"], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "==" in line:
                        name, version = line.split("==", 1)
                        packages[name] = version
        except Exception:
            pass  # Best-effort; environment recording is non-fatal.

        gpu_name = None
        gpu_driver = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_driver = torch.version.cuda
        except ImportError:
            pass

        return EnvironmentInfo(
            python_version=python_version,
            packages=packages,
            gpu_name=gpu_name,
            gpu_driver=gpu_driver,
            platform_info=platform_info,
            timestamp=timestamp,
        )


def config_hash(config: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash of a config for checkpoint invalidation.

    The config is serialized with sorted keys so semantically-equal configs hash
    identically regardless of key ordering (Requirement 21).
    """
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
