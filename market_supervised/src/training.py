"""Runnable training support for the market-supervised model.

This module supplies the pieces the orchestration notebook needs to actually
train (owner-run, Requirement 33): a tokenizing ``TaskDataset`` over the task
parquets, ``build_dataloaders`` to construct train/val/test loaders for one
target, ``train_target`` to run the full loop for a single target and persist
predictions, and ``predict`` to score a split.

The heavy numerical-stability policy (two-phase FP32->BF16 precision, gradient
clipping, ``assert_finite`` guards on inputs/targets/outputs/loss/gradients,
offending-batch logging, minimum sample-size gates, and contrastive-group
thresholds) lives in :func:`run_training_loop`, which is what
``src.market_supervised.train`` delegates to.

Everything here requires torch + transformers at RUN time; the module stays
importable without them so the pipeline can be inspected in a torch-free
environment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import pandas as pd

from .market_supervised import (
    TrainResult,
    assert_finite,
    build_model,
    compute_losses,
    compute_council_losses,
    CouncilModel,
    COUNCIL_STANCE_CLASS_ORDER,
    make_contrastive_groups,
    _cfg_get,
)
from .stance import (
    build_soft_labels,
    build_emotion_signal,
    build_expectation_signal,
    market_to_stance_soft,
    seed_stance_pipeline,
)

try:  # torch is required at run time; guard so the module stays importable.
    import torch
    from torch.utils.data import DataLoader, Dataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = object  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False


_logger = logging.getLogger("pipeline.training")


_MIN_SAFE_ADAMW_EPS = 1.0e-6
_ADAMW_MOMENT_KEYS = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq"})


def _stable_adamw_kwargs(cfg: Any) -> dict[str, Any]:
    """Return the enforced AdamW numerical contract.

    FP16 rounds AdamW's usual ``1e-8`` epsilon to zero, so a zero second moment
    can produce ``0 / 0`` on the first update.  The model is separately forced
    to FP32, and this function pins a representable epsilon plus the conservative
    scalar backend for reproducible behavior across CUDA/PyTorch versions.
    """
    parameter_dtype = str(
        _cfg_get(cfg, "model.deberta.optimizer.parameter_dtype", "float32")
    ).lower()
    if parameter_dtype not in {"float32", "fp32", "torch.float32"}:
        raise ValueError(
            "model.deberta.optimizer.parameter_dtype must be 'float32'; "
            f"got {parameter_dtype!r}"
        )
    eps = float(_cfg_get(cfg, "model.deberta.optimizer.eps", _MIN_SAFE_ADAMW_EPS))
    if not math.isfinite(eps) or eps < _MIN_SAFE_ADAMW_EPS:
        raise ValueError(
            "model.deberta.optimizer.eps must be finite and >= 1e-6; "
            f"got {eps!r}"
        )
    foreach = bool(_cfg_get(cfg, "model.deberta.optimizer.foreach", False))
    fused = bool(_cfg_get(cfg, "model.deberta.optimizer.fused", False))
    if foreach or fused:
        raise ValueError(
            "stable training requires model.deberta.optimizer.foreach=false "
            "and model.deberta.optimizer.fused=false"
        )
    return {"eps": eps, "foreach": False, "fused": False}


def _move_model_to_fp32(model: Any, device: Any, cfg: Any) -> Any:
    """Move ``model`` to ``device`` while enforcing FP32 master parameters."""
    # Validate the config before allocating/copying the full model.
    _stable_adamw_kwargs(cfg)
    before = sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()})
    model = model.to(device=device, dtype=torch.float32)
    invalid = [
        (name, str(param.dtype))
        for name, param in model.named_parameters()
        if param.is_floating_point() and param.dtype != torch.float32
    ]
    if invalid:
        name, dtype = invalid[0]
        raise RuntimeError(
            "FP32 optimizer invariant failed after model device move: "
            f"parameter {name!r} has dtype {dtype}"
        )
    _logger.info(
        "optimizer numerics: parameter_dtype=float32 source_dtypes=%s "
        "adamw_eps=%g foreach=false fused=false",
        before or ["none"],
        _stable_adamw_kwargs(cfg)["eps"],
    )
    return model


def _coerce_and_validate_adamw_state(optimizer: Any) -> None:
    """Validate resumed AdamW state and convert moment tensors to FP32."""
    for parameter, state in optimizer.state.items():
        for key, value in list(state.items()):
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"non-finite AdamW state tensor: {key}")
            if key in _ADAMW_MOMENT_KEYS and value.dtype != torch.float32:
                state[key] = value.to(device=parameter.device, dtype=torch.float32)


def _first_invalid_adamw_state(model: Any, optimizer: Any) -> Optional[dict]:
    """Return the first non-finite or non-FP32 AdamW moment diagnostic."""
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if not torch.is_tensor(value) or not value.is_floating_point():
                continue
            if not bool(torch.isfinite(value).all()):
                return {
                    "parameter_name": names.get(id(parameter), "<unknown>"),
                    "optimizer_state_key": str(key),
                    "optimizer_state_dtype": str(value.dtype),
                    "reason": "non_finite_optimizer_state_after_step",
                }
            if key in _ADAMW_MOMENT_KEYS and value.dtype != torch.float32:
                return {
                    "parameter_name": names.get(id(parameter), "<unknown>"),
                    "optimizer_state_key": str(key),
                    "optimizer_state_dtype": str(value.dtype),
                    "reason": "non_fp32_optimizer_moment_after_step",
                }
    return None


# =============================================================================
# Tokenizer construction
# =============================================================================

def build_tokenizer(cfg: Any):
    """Load the tokenizer for ``model.deberta.base_model`` (or the override).

    Falls back to a tiny whitespace-hash tokenizer when transformers/weights are
    unavailable and ``model.deberta.allow_stub_encoder`` is true, so a CPU smoke
    test needs no network access.
    """
    override = _cfg_get(cfg, "model.deberta.encoder_override", None)
    allow_stub = bool(_cfg_get(cfg, "model.deberta.allow_stub_encoder", False))
    if override == "__stub__" or _cfg_get(cfg, "model.deberta.force_stub", False):
        vocab = int(_cfg_get(cfg, "model.deberta.stub_vocab_size", 4096))
        return _StubTokenizer(vocab_size=vocab)
    base_model = override or _cfg_get(
        cfg, "model.deberta.base_model", "microsoft/deberta-v3-base"
    )
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(base_model)
    except Exception:
        if allow_stub:
            vocab = int(_cfg_get(cfg, "model.deberta.stub_vocab_size", 4096))
            return _StubTokenizer(vocab_size=vocab)
        raise


class _StubTokenizer:
    """Deterministic hashing tokenizer for offline smoke tests only."""

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size
        self.cls_id = 1
        self.pad_id = 0

    def __call__(self, text, max_length=64, truncation=True, padding="max_length"):
        toks = str(text).split()
        if truncation:
            toks = toks[: max_length - 1]
        ids = [self.cls_id] + [
            (hash(t) % (self.vocab_size - 2)) + 2 for t in toks
        ]
        mask = [1] * len(ids)
        if padding == "max_length" and len(ids) < max_length:
            pad = max_length - len(ids)
            ids += [self.pad_id] * pad
            mask += [0] * pad
        return {"input_ids": ids[:max_length], "attention_mask": mask[:max_length]}


# =============================================================================
# Dataset + segmentation
# =============================================================================

def _segment_text(text: str, chunk_words: int, max_chunks: int, overlap_words: int) -> list[str]:
    """Split ``text`` into up to ``max_chunks`` overlapping word chunks.

    Hierarchical encoding needs the document as ordered segments; here we chunk
    on whitespace so no external sentence splitter is required. Always returns at
    least one (possibly empty) chunk.
    """
    words = str(text).split()
    if not words:
        return [""]
    step = max(1, chunk_words - overlap_words)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if len(chunks) >= max_chunks:
            break
    return chunks or [""]


if _HAS_TORCH:

    class TaskDataset(Dataset):
        """Tokenized view over one target's task parquet rows.

        Each item yields per-segment ``input_ids``/``attention_mask``
        (``[S, L]`` in hierarchical mode, ``[L]`` in flat mode), a
        ``segment_mask`` marking present segments, the scaled regression
        ``target``, and the raw target for reporting.
        """

        def __init__(self, frame: "pd.DataFrame", tokenizer, cfg: Any):
            self.rows = frame.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.mode = _cfg_get(cfg, "model.deberta.encoder_mode", "hierarchical")
            self.max_len = int(_cfg_get(cfg, "model.deberta.chunk_size", 512))
            self.max_chunks = int(_cfg_get(cfg, "model.deberta.max_chunks", 4))
            self.overlap = int(_cfg_get(cfg, "model.deberta.chunk_overlap", 64))
            # Words-per-chunk approximated from token budget (~0.75 words/token).
            self.chunk_words = max(16, int(self.max_len * 0.75))
            self.overlap_words = max(0, int(self.overlap * 0.75))
            # Prefer the pre-scaled target; fall back to raw when absent.
            scaled = self.rows.get("target_scaled")
            if scaled is None or scaled.isna().all():
                self.rows["target_scaled"] = self.rows["target"].astype("float64")
            # Gate stance-field emission on the enabled flag (Req 1.1/1.8). When
            # false, __getitem__ returns exactly the pre-feature dict so a
            # disabled run is byte-identical to a baseline (Req 10.1).
            self.stance_enabled = bool(
                _cfg_get(cfg, "model.deberta.stance.enabled", False)
            )
            # Configured defaults substituted when a signal is absent (Req 1.6/1.7).
            self._stance_default_emotion = float(
                _cfg_get(cfg, "model.deberta.stance.default_emotion", 0.0)
            )
            self._stance_default_expectation = float(
                _cfg_get(cfg, "model.deberta.stance.default_expectation", 0.0)
            )
            # Gate council-field emission on whether any council signal is
            # configured (Req 1.2). When no signal is configured -- or the
            # attach step left the frame without the Market_Signal_Vector
            # columns -- __getitem__ omits the council fields so a non-Council
            # run is byte-identical to today's batch.
            self._council_signal_names = _council_signal_names(cfg)
            self.council_enabled = bool(self._council_signal_names) and (
                "signal_targets" in self.rows.columns
            )
            self._default_material_threshold = float(
                _cfg_get(cfg, "market_windows.material_move_bp", 0.0)
            )
            self._cache_tokenization = bool(
                _cfg_get(cfg, "model.deberta.cache_tokenization", True)
            )
            self._token_cache: dict[int, dict[str, "torch.Tensor"]] = {}

        def __len__(self) -> int:
            return len(self.rows)

        def _encode_segment(self, text: str):
            enc = self.tokenizer(
                text, max_length=self.max_len, truncation=True, padding="max_length"
            )
            return enc["input_ids"], enc["attention_mask"]

        def _stance_fields(self, row) -> dict:
            """Build the six Stance_Batch_Fields for one event row (Req 1).

            ``stance_mask`` gates the soft label: when the label is undefined
            (or missing) the emitted vector is the all-zero placeholder so it
            contributes exactly 0.0 to the masked stance loss (Req 1.3). Signals
            fall back to the configured defaults when absent (Req 1.6/1.7).
            """
            mask = bool(row.get("stance_mask", False))
            soft = row.get("stance_soft_label")
            # Placeholder when masked/undefined or the column is absent (Req 1.3).
            if mask and soft is not None:
                soft_vec = tuple(float(v) for v in soft)
            else:
                soft_vec = _STANCE_PLACEHOLDER_LABEL
            emotion = _stance_value_or_default(
                row.get("stance_emotion"), self._stance_default_emotion
            )
            expectation = _stance_value_or_default(
                row.get("stance_expectation"), self._stance_default_expectation
            )
            return {
                "stance_soft_label": torch.tensor(soft_vec, dtype=torch.float32),
                "stance_mask": torch.tensor(mask, dtype=torch.bool),
                "stance_emotion": torch.tensor(emotion, dtype=torch.float32),
                "stance_emotion_present": torch.tensor(
                    bool(row.get("stance_emotion_present", False)), dtype=torch.bool
                ),
                "stance_expectation": torch.tensor(expectation, dtype=torch.float32),
                "stance_expectation_present": torch.tensor(
                    bool(row.get("stance_expectation_present", False)), dtype=torch.bool
                ),
            }

        def _council_fields(self, row) -> dict:
            """Build the three Market_Signal_Vector batch fields for one row.

            Multi-signal analogue of :meth:`_stance_fields` (Req 1.2, 3.4). The
            per-event columns are materialized upstream by
            :func:`attach_council_fields`:

              * ``signal_targets`` -> ``[N]`` float tensor of normalized
                per-signal values (placeholder ``0.0`` where absent -- never a
                fabricated market value),
              * ``signal_present``  -> ``[N]`` bool tensor, the per-signal
                presence mask,
              * ``stance_target``   -> scalar float tensor in ``[-1, 1]``.

            The ``[N]`` vectors emit in the configured signal order, so the
            default torch collate stacks them into ``[B, N]`` / ``[B]``
            index-aligned with ``input_ids`` / ``target``. A row missing the
            attached columns (or carrying a mis-sized vector) falls back to the
            all-absent placeholder of the configured width -- never fabricating
            a value, never emitting a ragged vector that would break collation.
            """
            n = len(self._council_signal_names)

            raw_targets = row.get("signal_targets")
            raw_present = row.get("signal_present")

            targets_vec: list[float] = []
            present_vec: list[bool] = []
            if raw_targets is not None and raw_present is not None:
                try:
                    targets_vec = [float(v) for v in raw_targets]
                    present_vec = [bool(v) for v in raw_present]
                except (TypeError, ValueError):
                    targets_vec = []
                    present_vec = []

            # Enforce the configured width so every item collates into a
            # rectangular [B, N] tensor (an absent/short vector pads with the
            # neutral placeholder + False, never a fabricated market value).
            if len(targets_vec) != n or len(present_vec) != n:
                targets_vec = [_COUNCIL_ABSENT_PLACEHOLDER] * n
                present_vec = [False] * n

            stance_target = _stance_value_or_default(row.get("stance_target"), 0.0)

            return {
                "signal_targets": torch.tensor(targets_vec, dtype=torch.float32),
                "signal_present": torch.tensor(present_vec, dtype=torch.bool),
                "stance_target": torch.tensor(stance_target, dtype=torch.float32),
            }

        def __getitem__(self, idx: int):
            row = self.rows.iloc[idx]
            target_scaled = float(row.get("target_scaled"))
            target_raw = float(row.get("target"))
            cached = self._token_cache.get(idx) if self._cache_tokenization else None

            if cached is None:
                text = row.get("text") or ""
                if self.mode == "flat":
                    ids, mask = self._encode_segment(text)
                    cached = {
                        "input_ids": torch.tensor(ids, dtype=torch.long),
                        "attention_mask": torch.tensor(mask, dtype=torch.long),
                    }
                else:
                    segs = _segment_text(
                        text, self.chunk_words, self.max_chunks, self.overlap_words
                    )
                    seg_ids, seg_masks = [], []
                    for segment in segs:
                        ids, mask = self._encode_segment(segment)
                        seg_ids.append(ids)
                        seg_masks.append(mask)
                    present = len(seg_ids)
                    pad_id = (
                        self.tokenizer.pad_id
                        if hasattr(self.tokenizer, "pad_id")
                        else getattr(self.tokenizer, "pad_token_id", 0) or 0
                    )
                    while len(seg_ids) < self.max_chunks:
                        seg_ids.append([pad_id] * self.max_len)
                        seg_masks.append([0] * self.max_len)
                    segment_mask = [1] * present + [0] * (self.max_chunks - present)
                    cached = {
                        "input_ids": torch.tensor(
                            seg_ids[: self.max_chunks], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            seg_masks[: self.max_chunks], dtype=torch.long
                        ),
                        "segment_mask": torch.tensor(
                            segment_mask, dtype=torch.long
                        ),
                    }
                if self._cache_tokenization:
                    self._token_cache[idx] = cached

            item = dict(cached)
            item.update({
                "target": torch.tensor(target_scaled, dtype=torch.float32),
                "target_raw": torch.tensor(target_raw, dtype=torch.float32),
                "primary_present": torch.tensor(
                    bool(row.get("primary_present", True)), dtype=torch.bool
                ),
                "material_threshold": torch.tensor(
                    _stance_value_or_default(
                        row.get("material_threshold"), self._default_material_threshold
                    ),
                    dtype=torch.float32,
                ),
            })
            event_id = row.get("event_id")
            if event_id is not None:
                item["event_id"] = str(event_id)
            if self.stance_enabled:
                item.update(self._stance_fields(row))
            if self.council_enabled:
                item.update(self._council_fields(row))
            return item


def _load_task_frame(task_path: str) -> "pd.DataFrame":
    """Read a task parquet into a DataFrame (empty frame if the file is empty)."""
    frame = pd.read_parquet(task_path)
    return frame


# =============================================================================
# Stance-field materialization (Req 3, 7) -- Task 1.1
#
# ``attach_stance_fields`` merges market-derived soft labels and leakage-safe
# market-context signals onto a per-event task frame, keyed strictly by
# ``event_id``. It reuses the existing stance APIs (``build_soft_labels``,
# ``build_emotion_signal``, ``build_expectation_signal``) and NEVER reimplements
# stance logic or reads any new stance parameter. It is a no-op passthrough when
# ``model.deberta.stance.enabled`` is false, so a disabled run is byte-identical
# to a pre-feature baseline (Req 1.8, 10.1).
# =============================================================================

# The six stance columns materialized onto the task frame (Stance_Batch_Fields).
_STANCE_PLACEHOLDER_LABEL = (0.0, 0.0, 0.0)


def _stance_value_or_default(value: Any, default: float) -> float:
    """Return ``value`` as a float, or the configured ``default`` when the value
    is absent (None) or non-finite NaN (Req 1.6/1.7). This keeps the emitted
    signal defined even if an absent entry was stored as NaN rather than the
    configured default in the task frame."""
    if value is None:
        return float(default)
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if fv != fv else fv  # fv != fv is True only for NaN


def attach_stance_fields(frame: "pd.DataFrame", cfg: Any) -> "pd.DataFrame":
    """Materialize market-derived soft labels + market-context signals onto a
    per-event task ``frame``, keyed strictly by ``event_id`` (Req 3, 7).

    Returns ``frame`` unchanged when ``model.deberta.stance.enabled`` is false
    (no-op passthrough, Req 1.8/10.1). When enabled, invokes each existing
    stance builder exactly once, merges each output onto the row whose
    ``event_id`` equals the entry's ``event_id`` (strict event-id join,
    Req 3.4), preserves the original row count (Req 3.6), and adds the six
    stance columns:

      * ``stance_soft_label``        : 3-tuple ``(hawkish, dovish, neutral)`` when
        the label is defined; ``(0.0, 0.0, 0.0)`` placeholder otherwise (Req 3.3).
      * ``stance_mask``              : bool = ``MarketDerivedLabel.defined``.
      * ``stance_emotion``           : float = ``SignalValue.value`` when present
        else the configured ``default_emotion`` (Req 1.4/1.6/10.3).
      * ``stance_emotion_present``   : bool = ``SignalValue.present``.
      * ``stance_expectation``       : float = ``SignalValue.value`` when present
        else the configured ``default_expectation`` (Req 1.5/1.7/10.3).
      * ``stance_expectation_present`` : bool = ``SignalValue.present``.

    Never fabricates a numeric stance value for an absent entry (Req 3.3, 7.3):
    an absent label/signal carries the builder's absence indicator
    (``defined=False`` / ``present=False``), the placeholder / configured
    default, and ``stance_mask`` false.

    On a zero-event frame, invokes each builder and returns a zero-event frame
    with the stance columns present and no fabricated values (Req 3.7). If any
    builder raises or returns no mapping, halts assembly of the affected stance
    fields, leaves non-stance columns unchanged, and raises an error naming the
    failed builder (Req 3.8).
    """
    # No-op passthrough when stance is disabled (Req 1.8, 10.1). No optional
    # market-context source is accessed on this path (Req 10.2).
    if not bool(_cfg_get(cfg, "model.deberta.stance.enabled", False)):
        return frame

    # Read only already-defined stance parameters (Req 9.1, 9.4).
    primary_target = _cfg_get(cfg, "model.deberta.stance.primary_target", None)
    neutral_band_bp = _cfg_get(cfg, "model.deberta.stance.neutral_band_bp", None)
    default_emotion = float(_cfg_get(cfg, "model.deberta.stance.default_emotion", 0.0))
    default_expectation = float(
        _cfg_get(cfg, "model.deberta.stance.default_expectation", 0.0)
    )

    # Resolve leakage-safe market-context source paths relative to datasets_dir
    # (mirrors data_io.locate_datasets). The keys are validated to exist in
    # data.confounder_files by ConfigManager.validate (Req 9.3); resolve them
    # here without substituting any hard-coded fallback (Req 9.1).
    datasets_dir = _cfg_get(cfg, "data.datasets_dir", "")
    confounder_files = _cfg_get(cfg, "data.confounder_files", {}) or {}
    emotion_source = _cfg_get(cfg, "market_context.emotion_source", None)
    expectation_source = _cfg_get(cfg, "market_context.expectation_source", None)
    indices_path = os.path.join(
        datasets_dir, confounder_files.get(emotion_source, "")
    )
    mps_path = os.path.join(
        datasets_dir, confounder_files.get(expectation_source, "")
    )

    # Invoke each existing stance builder EXACTLY ONCE (Req 3.1, 3.2, 3.5). A
    # builder raising or returning no mapping halts assembly of the affected
    # stance fields, leaves non-stance columns unchanged, and surfaces an error
    # naming the failed builder (Req 3.8).
    labels = _invoke_builder(
        "build_soft_labels",
        lambda: build_soft_labels(
            frame, primary_target, material_move_bp=neutral_band_bp, cfg=cfg
        ),
        allow_empty_frame=(len(frame) == 0),
    )
    emotion = _invoke_builder(
        "build_emotion_signal",
        lambda: build_emotion_signal(frame, indices_path, cfg),
        allow_empty_frame=(len(frame) == 0),
    )
    expectation = _invoke_builder(
        "build_expectation_signal",
        lambda: build_expectation_signal(frame, mps_path, cfg),
        allow_empty_frame=(len(frame) == 0),
    )

    # Merge strictly by event-id equality via row-wise dictionary lookup. This
    # guarantees no cross-event contamination (Req 3.4), at most one entry per
    # builder per event (Req 3.6), and exact row-count preservation -- no row is
    # added or dropped (Req 3.6, 3.7). A copy keeps the caller's frame intact.
    out = frame.copy()

    event_ids = out["event_id"].tolist() if "event_id" in out.columns else []

    soft_labels: list = []
    masks: list = []
    emotions: list = []
    emotion_present: list = []
    expectations: list = []
    expectation_present: list = []

    for eid in event_ids:
        label = labels.get(eid)
        if label is not None and getattr(label, "defined", False):
            soft_labels.append((float(label.hawkish), float(label.dovish), float(label.neutral)))
            masks.append(True)
        else:
            # Absent / undefined -> placeholder, mask false; no fabrication
            # (Req 3.3, 7.3).
            soft_labels.append(_STANCE_PLACEHOLDER_LABEL)
            masks.append(False)

        emo = emotion.get(eid)
        if emo is not None and getattr(emo, "present", False):
            emotions.append(float(emo.value))
            emotion_present.append(True)
        else:
            emotions.append(default_emotion)
            emotion_present.append(False)

        exp = expectation.get(eid)
        if exp is not None and getattr(exp, "present", False):
            expectations.append(float(exp.value))
            expectation_present.append(True)
        else:
            expectations.append(default_expectation)
            expectation_present.append(False)

    # Assign the six columns. On a zero-event frame the lists are empty, so the
    # columns are present with zero rows and no fabricated values (Req 3.7).
    out["stance_soft_label"] = pd.Series(soft_labels, index=out.index, dtype=object)
    out["stance_mask"] = pd.Series(masks, index=out.index, dtype=bool)
    out["stance_emotion"] = pd.Series(emotions, index=out.index, dtype="float64")
    out["stance_emotion_present"] = pd.Series(emotion_present, index=out.index, dtype=bool)
    out["stance_expectation"] = pd.Series(expectations, index=out.index, dtype="float64")
    out["stance_expectation_present"] = pd.Series(
        expectation_present, index=out.index, dtype=bool
    )

    return out


def _invoke_builder(name: str, call, allow_empty_frame: bool) -> dict:
    """Invoke a stance builder, surfacing a named error on failure (Req 3.8).

    Wraps the builder call so that a raised exception or a missing mapping is
    reported with the failing builder's name. An empty mapping is only tolerated
    for a zero-event frame (Req 3.7); for a non-empty frame a builder returning
    no mapping is a failure that halts assembly (Req 3.8).
    """
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 -- re-raised with builder identity
        raise RuntimeError(
            f"stance builder {name!r} failed during attach_stance_fields: {exc}"
        ) from exc
    if result is None or (not isinstance(result, dict)):
        raise RuntimeError(
            f"stance builder {name!r} returned no mapping during attach_stance_fields"
        )
    if len(result) == 0 and not allow_empty_frame:
        raise RuntimeError(
            f"stance builder {name!r} returned an empty mapping for a non-empty "
            f"task frame during attach_stance_fields"
        )
    return result


# =============================================================================
# Signal normalization (train-split-only z-scoring)  (Req 4.1)
# =============================================================================

def _council_signal_names(cfg: Any) -> list[str]:
    """Return the configured Council signal names (``council.signals[i].name``).

    Reads the signal set straight from configuration (Req 1.3) and returns the
    per-signal ``name`` values in configured order. A malformed entry (not a
    mapping, or missing/blank ``name``) is skipped rather than fabricated -- the
    fusion / masking path resolves the authoritative spec set elsewhere; here we
    only need the column names to normalize.
    """
    signals = _cfg_get(cfg, "council.signals", []) or []
    names: list[str] = []
    for sig in signals:
        if not isinstance(sig, dict):
            continue
        name = sig.get("name")
        if name is None:
            continue
        name = str(name).strip()
        if name:
            names.append(name)
    return names


def _council_audit_dir(cfg: Any) -> str:
    """Resolve the ``outputs/audit`` directory for Council artifacts.

    Mirrors the ``data.outputs_dir`` convention used elsewhere; falls back to a
    relative ``outputs`` root when unset so the normalizer never depends on a
    resolved experiment directory to persist its scaler metadata.
    """
    outputs_dir = _cfg_get(cfg, "data.outputs_dir", "outputs") or "outputs"
    return os.path.join(outputs_dir, "audit")


class SignalNormalizer:
    """Per-signal z-scoring fit on the TRAINING split only (Req 4.1).

    The Council supervises the shared representation against a heterogeneous
    Market_Signal_Vector whose raw scales differ wildly (basis points vs return
    vs VIX points); un-normalized, that heterogeneity drove the encoder's
    disentangled-attention path into non-finite gradients. ``SignalNormalizer``
    standardizes each configured signal to zero-mean/unit-std using statistics
    computed **only** from training-split rows, then applies the frozen stats to
    val/test. This is the same discipline ``_check_scaler_leakage`` enforces
    (``fit_on == "train"``): validation/test values never influence the fitted
    mean/std.

    Fitted stats are stored per signal name as ``{"mean","std","n","fit_on"}``
    and persisted to ``outputs/audit/signal_scalers.json`` as
    ``{"<signal>": {...}}``, consumed unchanged by the leakage audit's scaler
    check.
    """

    #: ``fit_on`` marker every persisted scaler carries; the leakage audit fails
    #: any scaler whose value is not exactly ``"train"``.
    FIT_ON = "train"

    def __init__(self, signal_names: Optional[list[str]] = None):
        self.signal_names: list[str] = list(signal_names or [])
        # signal name -> {"mean","std","n","fit_on"}
        self.stats: dict[str, dict[str, Any]] = {}
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: Any) -> "SignalNormalizer":
        """Construct a normalizer for the configured ``council.signals`` set."""
        return cls(_council_signal_names(cfg))

    # ------------------------------------------------------------------
    @staticmethod
    def _fit_one(values) -> Optional[dict[str, Any]]:
        """Fit ``{"mean","std","n","fit_on"}`` on the finite ``values`` only.

        Mirrors ``dataset_builder.fit_target_scaler``: the std is floored to a
        small positive value so a degenerate (constant) train signal never
        yields a zero/non-finite std. Returns ``None`` when no finite training
        value exists for the signal (the signal is simply left un-normalized
        rather than fabricating statistics).
        """
        arr = np.asarray(list(values), dtype="float64").ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if not math.isfinite(std) or std <= 0.0:
            std = 1.0
        return {
            "mean": mean,
            "std": std,
            "n": int(arr.size),
            "fit_on": SignalNormalizer.FIT_ON,
        }

    def fit(self, train_frame: "pd.DataFrame") -> "SignalNormalizer":
        """Fit per-signal mean/std over TRAINING-split rows only (Req 4.1).

        ``train_frame`` must already be restricted to the training split (the
        caller passes it after ``temporal_split`` and before any val/test rows
        are materialized) so no validation/test value can influence the fitted
        statistics. Only configured signal columns present in the frame are
        fit; a missing column or an all-non-finite column is skipped.
        """
        self.stats = {}
        for name in self.signal_names:
            if train_frame is None or name not in getattr(train_frame, "columns", []):
                continue
            fitted = self._fit_one(train_frame[name].to_numpy())
            if fitted is not None:
                self.stats[name] = fitted
        self._fitted = True
        return self

    def apply(self, frame: "pd.DataFrame") -> "pd.DataFrame":
        """Z-score ``frame``'s signal columns with the fitted train stats.

        Applies ``(x - mean) / std`` per configured signal using the statistics
        frozen at ``fit`` time, so val/test rows are standardized with
        train-only stats (never their own). A signal without fitted stats (never
        seen a finite train value) is left unchanged. The caller's frame is not
        mutated; a copy carrying the standardized columns is returned.
        """
        if not self._fitted:
            raise RuntimeError("SignalNormalizer.apply called before fit")
        if frame is None or len(getattr(frame, "columns", [])) == 0:
            return frame
        out = frame.copy()
        for name, st in self.stats.items():
            if name not in out.columns:
                continue
            mean = st["mean"]
            std = st["std"] or 1.0
            out[name] = (out[name].astype("float64") - mean) / std
        return out

    # ------------------------------------------------------------------
    def to_metadata(self) -> dict[str, dict[str, Any]]:
        """Return the fitted stats as ``{"signal": {"mean","std","n","fit_on"}}``."""
        return {name: dict(st) for name, st in self.stats.items()}

    def save(self, audit_dir: str) -> Optional[str]:
        """Persist the fitted stats to ``<audit_dir>/signal_scalers.json``.

        Writes ``{"signal": {"mean","std","n","fit_on":"train"}}`` -- the exact
        shape ``_check_scaler_leakage`` consumes. Returns the written path, or
        ``None`` when no ``audit_dir`` is supplied.
        """
        if not audit_dir:
            return None
        os.makedirs(audit_dir, exist_ok=True)
        path = os.path.join(audit_dir, "signal_scalers.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_metadata(), fh, indent=2, default=str)
        return path


# =============================================================================
# Council-field materialization (Req 1.2, 3.4) -- Task 4.1
#
# ``attach_council_fields`` is the multi-signal analogue of
# ``attach_stance_fields``. Where the stance path materializes a single soft
# label + two market-context signals per event, the Council path materializes
# the per-event Market_Signal_Vector: one normalized target per configured
# ``council.signals`` entry, its per-signal presence flag, and one continuous
# scalar ``stance_target`` in ``[-1, 1]``.
#
# It reuses the same NEVER-FABRICATE discipline as the stance path (Req 3.3,
# design "Never fabricate data"): an absent signal carries the placeholder
# ``0.0`` in ``signal_targets`` and ``False`` in ``signal_present`` -- the model
# masks the term to a clean zero rather than imputing a value. Presence is read
# from the same provenance the pipeline already tracks: an explicit
# ``has_<name>`` coverage-flag column (as ``TargetResult.has_flag_column`` names
# it) when present, otherwise the finiteness of the signal's own value column
# (``TargetResult`` finiteness / ``SignalValue.present`` convention).
# =============================================================================

#: Placeholder emitted in ``signal_targets`` for an absent signal (Req 3.3): a
#: neutral ``0.0`` that contributes exactly zero once masked out. NEVER a
#: fabricated market value.
_COUNCIL_ABSENT_PLACEHOLDER = 0.0


def _council_present_and_value(row, name: str) -> tuple[bool, float]:
    """Resolve ``(present, value)`` for one council signal on one event row.

    Presence follows the pipeline's existing provenance convention, checked in
    priority order and NEVER fabricated (Req 3.3, 3.4):

    1. An explicit ``has_<name>`` coverage-flag column (the
       ``TargetResult.has_flag_column`` naming) -- when present and truthy the
       signal contributed; when present and falsy it is absent.
    2. Otherwise, the finiteness of the signal's own value column ``<name>``
       (the ``TargetResult`` finiteness / ``SignalValue.present`` convention):
       a finite value means present, a missing/NaN/Inf value means absent.

    Returns ``(present, value)`` where ``value`` is the (already normalized)
    signal value when present, or the ``0.0`` placeholder when absent -- so an
    absent signal never emits a fabricated number.
    """
    has_flag_col = f"has_{name}"
    flag = row.get(has_flag_col)

    raw = row.get(name)
    try:
        fv = float(raw)
    except (TypeError, ValueError):
        fv = float("nan")
    value_finite = math.isfinite(fv)

    if flag is not None:
        # Explicit coverage flag governs presence; a present flag with a
        # non-finite value is still treated as absent to keep the emitted
        # target finite (never fabricate, never emit NaN).
        present = bool(flag) and value_finite
    else:
        present = value_finite

    if present:
        return True, fv
    return False, _COUNCIL_ABSENT_PLACEHOLDER


def _council_stance_target(row, cfg: Any) -> float:
    """Derive the monotone scalar ``stance_target`` in ``[-1, 1]`` for one event row.

    Monotone Stance_Score supervision target (Req 7.1/7.2/7.5, design
    "Interpretable Continuous Stance Space"). The event's primary within-window
    market reaction is mapped through ``market_to_stance_soft``'s signed-
    magnitude ``tanh`` map and the soft triple is collapsed to the signed
    directional mass ``hawkish - dovish``. That value is:

    * **monotone non-decreasing in the raw move** (Req 7.2): a more-hawkish /
      more-positive short-rate reaction maps to a higher score, a more-dovish /
      more-negative reaction to a lower score. The ``hawkish - dovish`` collapse
      of the signed-magnitude ``tanh`` map is odd and non-decreasing in
      ``move_bp``, so ``r_a < r_b => s(r_a) <= s(r_b)`` (design Property 8),
    * **bounded to ``[-1, 1]``** (Req 7.1) by the ``tanh`` squash,
    * **neutral ``0.0`` for an immaterial move** (``|move| < material_move_bp``,
      Req 7.5): the smooth ``tanh`` map alone is exactly neutral only at
      ``move == 0``, so the ``material_move_bp`` threshold is applied here as an
      explicit neutral band -- any move whose magnitude does not reach the
      configured ``material_move_bp`` is assigned the neutral ``0.0`` Stance_Score
      (the design's "neutral on immaterial moves" contract). Zeroing a
      sub-threshold move preserves monotonicity: the band is symmetric about
      zero, so a lower move never receives a higher score.

    The direction convention (positive short-rate reaction -> hawkish -> higher
    score) matches ``market_to_stance_soft`` in ``src/stance.py``.

    The primary reaction is the event's raw ``target`` (the primary within-window
    move in bp for this task's target). An absent / non-finite primary reaction
    yields a neutral ``0.0`` -- never a fabricated stance.
    """
    raw = row.get("target")
    try:
        move_bp = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(move_bp):
        return 0.0

    material_move_bp = float(_cfg_get(cfg, "market_windows.material_move_bp", 1.0))
    if not math.isfinite(material_move_bp) or material_move_bp <= 0:
        material_move_bp = 1.0

    # Neutral band (Req 7.5): an immaterial move -- one whose magnitude does not
    # reach the configured material_move_bp threshold -- is a neutral Stance_Score
    # of exactly 0.0. Applied BEFORE the tanh collapse because the smooth map is
    # neutral only at move == 0, not across the whole [-material, +material) band.
    if abs(move_bp) < material_move_bp:
        return 0.0

    neutral_band_bp = _cfg_get(cfg, "model.deberta.stance.neutral_band_bp", None)
    sharpness = float(_cfg_get(cfg, "model.deberta.stance.sharpness", 1.0))

    hawkish, dovish, _neutral = market_to_stance_soft(
        move_bp,
        material_move_bp,
        neutral_band_bp=neutral_band_bp,
        sharpness=sharpness,
    )
    # Signed directional mass: monotone in move, bounded [-1, 1] (Req 7.1/7.2).
    return float(hawkish - dovish)


def attach_council_fields(
    frame: "pd.DataFrame", cfg: Any, normalizer: Optional["SignalNormalizer"] = None
) -> "pd.DataFrame":
    """Materialize the per-event Market_Signal_Vector onto a task ``frame``.

    Multi-signal analogue of :func:`attach_stance_fields` (Req 1.2, 3.4). For a
    configured signal set ``council.signals = {s_1..s_N}`` it emits, keyed
    strictly by row (index-aligned with ``target`` / ``input_ids``), three
    columns:

      * ``signal_targets`` : ``[N]`` float vector of normalized per-signal
        values, with the placeholder ``0.0`` where a signal is absent -- NEVER a
        fabricated market value (Req 3.3).
      * ``signal_present``  : ``[N]`` bool vector, the per-signal presence mask
        (``TargetResult`` finiteness / ``SignalValue.present`` convention,
        Req 3.4).
      * ``stance_target``   : scalar float in ``[-1, 1]``, the monotone
        Stance_Score supervision target derived from the primary reaction via
        :func:`_council_stance_target` -- monotone non-decreasing in the move,
        bounded ``[-1, 1]``, neutral ``0.0`` for an immaterial move
        (``|move| < material_move_bp``) (Req 7.1/7.2/7.5).

    Normalization: when a fitted :class:`SignalNormalizer` is supplied (the
    train-only z-scoring fit in :func:`build_dataloaders`), the frame is
    z-scored through ``normalizer.apply`` BEFORE the signal values are read, so
    ``signal_targets`` carries normalized values while val/test never influence
    the statistics (Req 4.1). Presence is resolved on the ORIGINAL frame so a
    genuine zero-valued observation is never mistaken for an absent signal after
    centering.

    No-op passthrough when no council signal is configured (byte-identical to a
    non-Council run). Preserves the caller's row count exactly (a copy keeps the
    caller's frame intact); a zero-event frame returns the three columns with
    zero rows and no fabricated values.
    """
    signal_names = _council_signal_names(cfg)
    if not signal_names:
        # No council signal configured -> no-op passthrough (Req: config-driven).
        return frame

    # Resolve presence on the ORIGINAL (un-normalized) frame so a true zero move
    # is not confused with an absent signal after centering, then read the
    # normalized values from the standardized frame (Req 4.1).
    presence_frame = frame
    value_frame = frame
    if normalizer is not None and getattr(normalizer, "_fitted", False):
        value_frame = normalizer.apply(frame)

    out = frame.copy()

    signal_targets: list = []
    signal_present: list = []
    stance_targets: list = []

    n_rows = len(out)
    for pos in range(n_rows):
        prow = presence_frame.iloc[pos]
        vrow = value_frame.iloc[pos]

        targets_vec: list[float] = []
        present_vec: list[bool] = []
        for name in signal_names:
            present, _val = _council_present_and_value(prow, name)
            if present:
                # Read the (possibly normalized) value from the value frame; fall
                # back to the presence-frame value if the column is absent there.
                _, norm_val = _council_present_and_value(vrow, name)
                targets_vec.append(float(norm_val))
                present_vec.append(True)
            else:
                targets_vec.append(_COUNCIL_ABSENT_PLACEHOLDER)
                present_vec.append(False)

        signal_targets.append(targets_vec)
        signal_present.append(present_vec)
        stance_targets.append(_council_stance_target(prow, cfg))

    out["signal_targets"] = pd.Series(signal_targets, index=out.index, dtype=object)
    out["signal_present"] = pd.Series(signal_present, index=out.index, dtype=object)
    out["stance_target"] = pd.Series(stance_targets, index=out.index, dtype="float64")

    return out


def build_dataloaders(task_path: str, tokenizer, cfg: Any) -> dict:
    """Build train/val/test DataLoaders for one target's task parquet.

    Returns ``{"train","val","test": DataLoader or None, "n_train","n_val",
    "n_test","frames"}``. A split with zero rows yields ``None`` for that loader.
    """
    if not _HAS_TORCH:
        raise RuntimeError("build_dataloaders requires PyTorch.")
    frame = _load_task_frame(task_path)
    batch_size = int(_cfg_get(cfg, "model.deberta.batch_size", 4))
    loaders: dict[str, Any] = {"frames": {}}

    # Reconstruct the authoritative primary-target scaler from this task's own
    # training rows. This removes the notebook's former shared-scaler bug (the
    # OIS2Y scaler was passed to equity/FX/other-tenor tasks) and gives validation
    # an automatic same-scale inverse transform.
    train_rows = frame[frame["split"] == "train"] if len(frame) else frame
    if "primary_present" in train_rows.columns:
        train_rows = train_rows[
            train_rows["primary_present"].fillna(False).astype(bool)
        ]
    train_values = pd.to_numeric(
        train_rows.get("target", pd.Series(dtype="float64")), errors="coerce"
    ).to_numpy(dtype="float64")
    train_values = train_values[np.isfinite(train_values)]
    if train_values.size:
        target_mean = float(np.mean(train_values))
        target_std = float(np.std(train_values))
        if not math.isfinite(target_std) or target_std <= 0.0:
            target_std = 1.0
        loaders["target_scaler"] = {
            "mean": target_mean,
            "std": target_std,
            "n": int(train_values.size),
            "fit_on": "train",
            "source": "task_local_train_rows",
        }
    else:
        loaders["target_scaler"] = None

    # The scaler above is authoritative for this task. Recompute target_scaled
    # for every split instead of trusting a parquet column that may have been
    # produced with another target's scaler by an older notebook run. Primary-
    # absent Council rows keep a finite zero placeholder behind their explicit
    # mask and therefore cannot influence any primary objective.
    frame = frame.copy()
    raw_target = pd.to_numeric(
        frame.get("target", pd.Series(index=frame.index, dtype="float64")),
        errors="coerce",
    )
    if loaders["target_scaler"] is not None:
        target_mean = float(loaders["target_scaler"]["mean"])
        target_std = float(loaders["target_scaler"]["std"])
        frame["target_scaled"] = (raw_target - target_mean) / target_std
    else:
        frame["target_scaled"] = raw_target
    if "primary_present" in frame.columns:
        primary_present = frame["primary_present"].fillna(False).astype(bool)
        frame.loc[~primary_present, "target_scaled"] = 0.0
    frame["target_scaled"] = pd.to_numeric(
        frame["target_scaled"], errors="coerce"
    ).fillna(0.0)

    # Fit per-signal z-scoring on the TRAINING split only, AFTER split
    # assignment (frame["split"]) and BEFORE any signal materialization, so
    # val/test values never influence the fitted stats -- the same discipline
    # _check_scaler_leakage enforces (fit_on == "train") (Req 4.1). The fitted
    # normalizer is exposed on the returned loaders dict so the council-field
    # attachment (attach_council_fields, built concurrently) consumes it to
    # standardize each split's signal_targets, and the stats are persisted to
    # outputs/audit/signal_scalers.json for the leakage audit's scaler check.
    signal_normalizer = SignalNormalizer.from_config(cfg)
    if signal_normalizer.signal_names:
        train_frame = frame[frame["split"] == "train"] if len(frame) else frame
        signal_normalizer.fit(train_frame)
        signal_normalizer.save(_council_audit_dir(cfg))
    loaders["signal_normalizer"] = signal_normalizer

    for split in ("train", "val", "test"):
        sub = frame[frame["split"] == split] if len(frame) else frame
        # Materialize market-derived soft labels + market-context signals onto
        # this split's per-event frame before constructing the dataset (Req 2.5).
        # Gated inside attach_stance_fields: a no-op passthrough when
        # model.deberta.stance.enabled is false, so the disabled path is
        # byte-identical to today. The default torch collate then stacks each
        # per-example stance tensor into stance_soft_label [N,3], stance_mask
        # [N] bool, stance_emotion/stance_expectation [N] float, and the present
        # flags [N] bool, index-aligned with input_ids / target (Req 2.1-2.3).
        sub = attach_stance_fields(sub, cfg)
        # Materialize the per-event Market_Signal_Vector onto this split's frame
        # (Req 1.2, 3.4). Uses the SignalNormalizer fitted on the TRAINING split
        # ONLY (above) so signal_targets carries train-only z-scored values --
        # val/test never influence the statistics (Req 4.1). A no-op passthrough
        # when no council signal is configured, so the disabled path is
        # byte-identical to today. The default torch collate then stacks each
        # per-example council tensor into signal_targets [B, N] float,
        # signal_present [B, N] bool, and stance_target [B] float, index-aligned
        # with input_ids / target.
        sub = attach_council_fields(sub, cfg, signal_normalizer)
        loaders["frames"][split] = sub
        loaders[f"n_{split}"] = int(len(sub))
        if len(sub) == 0:
            loaders[split] = None
            continue
        ds = TaskDataset(sub, tokenizer, cfg)
        split_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=False,
        )
        # Make the task-local scaler discoverable by predict() even when callers
        # pass only an individual DataLoader rather than the enclosing mapping.
        split_loader.target_scaler = loaders.get("target_scaler")
        loaders[split] = split_loader
    return loaders


# =============================================================================
# The training loop (delegated to by src.market_supervised.train)
# =============================================================================

def _move_batch(batch: dict, device) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def _log_offending_batch(debug_log_path: str, payload: dict) -> None:
    """Append an offending (non-finite) batch record to the NaN debug log."""
    if not debug_log_path:
        return
    os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)
    with open(debug_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


#: Filename under ``outputs/audit`` that collects the Council's per-step
#: effective per-signal weights (Req 2.3).
_COUNCIL_SIGNAL_WEIGHTS_FILE = "council_signal_weights.jsonl"


def _extract_council_signal_weights(outputs: Any) -> Optional[dict]:
    """Return the ``{signal_name: float}`` effective-weight map for a step.

    ``SignalFusion.fuse`` returns ``(fused_loss, effective_weights)`` where
    ``effective_weights`` is a ``{signal_name: float}`` dict of the effective
    per-signal weight applied this step (``exp(-clamp(log_var_i))`` in
    uncertainty mode, the configured weight in fixed mode -- Req 2.3). Once the
    Council forward (task 8.1) wires that fusion into the step, it surfaces the
    map on the model outputs under ``"council_signal_weights"``.

    This is intentionally defensive: it returns the map only when ``outputs`` is
    a mapping that carries a non-empty ``council_signal_weights`` dict, and
    ``None`` otherwise. A non-Council run (or a Council run before 8.1 lands)
    therefore never triggers any logging, so its behavior is byte-identical to
    today's loop.
    """
    if not isinstance(outputs, dict):
        return None
    weights = outputs.get("council_signal_weights")
    if not isinstance(weights, dict) or not weights:
        return None
    coerced: dict[str, float] = {}
    for name, value in weights.items():
        try:
            coerced[str(name)] = float(value)
        except (TypeError, ValueError):
            # Skip an unconvertible entry rather than fabricate a weight; the
            # remaining signals still record their effective weight.
            continue
    return coerced or None


def _log_council_signal_weights(weights_log_path: str, payload: dict) -> None:
    """Append one step's effective per-signal Council weights (Req 2.3).

    Mirrors :func:`_log_offending_batch`'s append-JSONL pattern: a falsy path is
    a no-op, the parent directory is created on demand, and one JSON object is
    appended per line so the file grows into a per-step audit trail consumed
    downstream. ``payload`` carries the ``epoch``, ``step`` and the
    ``weights`` map for the step.
    """
    if not weights_log_path:
        return
    os.makedirs(os.path.dirname(weights_log_path), exist_ok=True)
    with open(weights_log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _temporal_contrastive_terms(outputs: dict, batch: dict, cfg: Any):
    """Compute optional temporal and contrastive terms from raw primary labels.

    Both objectives are restricted to rows whose primary target is genuinely
    observed. Directional groups use raw market units and the task-specific
    material threshold, never the sign of a train-standardized target.
    """
    terms: dict[str, Any] = {}
    doc = outputs.get("document")
    target = batch.get("target_raw", batch.get("target"))
    if doc is None or target is None:
        return terms

    target = target.reshape(-1).to(doc.device, doc.dtype)
    primary_present = batch.get("primary_present")
    if primary_present is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    else:
        valid = primary_present.reshape(-1).to(target.device).bool()
    valid = valid & torch.isfinite(target)
    valid_idx = valid.nonzero(as_tuple=False).reshape(-1)
    if valid_idx.numel() == 0:
        return terms

    margin = float(_cfg_get(cfg, "model.deberta.temporal.margin", 0.5))
    reg = outputs.get("primary_prediction", outputs.get("regression"))
    if reg is not None and valid_idx.numel() >= 2:
        reg = reg.reshape(-1)
        valid_target = target.index_select(0, valid_idx)
        order = torch.argsort(valid_target.abs())
        lo = valid_idx[order[0]]
        hi = valid_idx[order[-1]]
        terms["temporal"] = torch.relu(
            reg.new_tensor(margin) - (reg[hi].abs() - reg[lo].abs())
        )

    threshold = batch.get("material_threshold")
    if threshold is None:
        threshold = torch.full_like(
            target,
            float(_cfg_get(cfg, "market_windows.contrastive_threshold_bp", 0.5)),
        )
    elif not torch.is_tensor(threshold):
        threshold = torch.as_tensor(
            threshold, dtype=target.dtype, device=target.device
        )
    threshold = threshold.reshape(-1).to(target.device, target.dtype)
    if threshold.numel() == 1:
        threshold = threshold.expand_as(target)
    threshold = torch.clamp(
        torch.nan_to_num(threshold, nan=0.0, posinf=0.0, neginf=0.0), min=0.0
    )

    positive = (valid & (target > threshold)).nonzero(as_tuple=False).reshape(-1)
    negative = (valid & (target < -threshold)).nonzero(as_tuple=False).reshape(-1)
    min_pairs = int(_cfg_get(cfg, "market_windows.contrastive_min_pairs", 1))
    if positive.numel() >= min_pairs and negative.numel() >= min_pairs:
        pos_c = doc.index_select(0, positive).mean(dim=0)
        neg_c = doc.index_select(0, negative).mean(dim=0)
        # A zero centroid makes cosine gradients ill-defined. Skip that optional
        # term rather than letting a degenerate text batch poison backward.
        eps = 1e-8
        if float(pos_c.detach().norm()) > eps and float(neg_c.detach().norm()) > eps:
            sim = torch.nn.functional.cosine_similarity(
                pos_c.unsqueeze(0), neg_c.unsqueeze(0), eps=eps
            )
            terms["contrastive"] = torch.relu(sim).squeeze()
    return terms


def _select_and_compute_loss(model, outputs: dict, batch: dict, cfg: Any) -> dict:
    """Select and compute the training loss for the current model (Req 2.1-2.4).

    Dispatches on the model type so the Council path optimizes the multi-signal
    uncertainty-fused loss while a standard :class:`MarketSupervisedModel` keeps
    its existing single-target loss with the temporal/contrastive augmentation:

    * ``CouncilModel`` -> :func:`compute_council_losses` with the model's
      ``signal_fusion`` and per-signal ``council_heads.log_var`` (Req 2.1). In
      ``uncertainty`` mode ``SignalFusion.fuse`` clamps each predicted
      log-variance to ``[log_var_min, log_var_max]`` before the Gaussian NLL
      (Req 2.4), combines only observed-signal contributions, and augments the
      composite objective with raw-primary temporal/contrastive terms when the
      batch contains enough masked observations.
    * otherwise -> :func:`compute_losses` (Req 2.2), using the same raw-primary
      temporal/contrastive helper.

    The returned dict carries the ``effective_weights`` (Council path) that
    continue to flow to :func:`_extract_council_signal_weights` for the per-step
    weights log.
    """
    if isinstance(model, CouncilModel):
        # The Council retains the same optional representation-shaping terms as
        # the standard model, but they are now derived only from observed raw
        # primary labels. compute_council_losses applies their configured weights.
        outputs.update(_temporal_contrastive_terms(outputs, batch, cfg))
        return compute_council_losses(
            outputs, batch, cfg,
            fusion=model.signal_fusion,
            log_var=model.council_heads.log_var,
        )
    outputs.update(_temporal_contrastive_terms(outputs, batch, cfg))
    return compute_losses(outputs, batch, cfg)


def _primary_metric_value(y_true: "np.ndarray", y_pred: "np.ndarray", cfg: Any) -> float:
    """Lower-is-better validation metric (scaled-space MSE) for selection."""
    if len(y_true) == 0:
        return float("inf")
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def _finite_max_abs(tensor: Any) -> Optional[float]:
    """Return the max absolute value over the FINITE entries of ``tensor``.

    Used as the ``grad_stat`` summary for a (partially) non-finite gradient: the
    non-finite entries themselves carry no magnitude, so we summarize the finite
    tail that survived. Returns ``None`` when the tensor has no finite entry
    (rather than fabricating a magnitude).
    """
    try:
        finite_mask = torch.isfinite(tensor)
        if not bool(finite_mask.any()):
            return None
        return float(tensor[finite_mask].abs().max().detach().cpu())
    except Exception:
        return None


def _resolve_loss_component(parameter_name: str, losses: Any) -> str:
    """Resolve the contributing ``loss_component`` for ``parameter_name`` from the
    per-signal / ``total`` loss dict (Req 4.1).

    Best-effort attribution: a task head's parameter path usually embeds the
    component (or per-signal) name that produced its gradient (e.g. a Council
    per-signal head or the ``regression``/``direction``/``uncertainty`` heads).
    We match the longest loss-dict key that appears in the parameter path so a
    more specific per-signal component wins over a generic one, and fall back to
    ``"total"`` when nothing matches (the loss that every parameter contributes
    to via the backward on ``losses["total"]``).
    """
    if not isinstance(losses, dict) or not losses:
        return "total"
    name_lc = str(parameter_name).lower()
    best: Optional[str] = None
    for key in losses:
        if key == "total":
            continue
        token = str(key).lower()
        if token and token in name_lc:
            if best is None or len(token) > len(str(best)):
                best = str(key)
    if best is not None:
        return best
    return "total" if "total" in losses else next(iter(losses))


def _event_batch_id(batch: Any) -> Any:
    """Read the batch's event identifier(s) from ``batch["event_id"]`` (Req 4.1).

    Returns the raw ids (list-ified for tensors) when present, else ``None`` --
    the batch collate does not always carry ``event_id`` (e.g. the flat/segment
    task loaders), so this is defensive rather than mandatory.
    """
    if not isinstance(batch, dict):
        return None
    eid = batch.get("event_id")
    if eid is None:
        return None
    try:
        if torch.is_tensor(eid):
            return eid.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(eid, (list, tuple)):
        return list(eid)
    return eid


def _first_nonfinite_gradient(model, losses: Any, batch: Any) -> Optional[dict]:
    """Return the diagnostic record for the FIRST non-finite gradient this step,
    or ``None`` when every gradient is finite (Req 4.1).

    Scans ``model.named_parameters()`` in definition order and, at the first
    parameter whose ``.grad`` contains a non-finite entry, records:

    * ``parameter_name`` -- the failing parameter's name,
    * ``loss_component`` -- the contributing loss component resolved from the
      per-signal / ``total`` loss dict (:func:`_resolve_loss_component`),
    * ``grad_stat`` -- a finite max-abs summary of the gradient
      (:func:`_finite_max_abs`), and
    * ``event_batch_id`` -- the batch's event id(s) from ``batch["event_id"]``.
    """
    for name, param in model.named_parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        if bool(torch.isfinite(grad).all()):
            continue
        return {
            "parameter_name": name,
            "loss_component": _resolve_loss_component(name, losses),
            "grad_stat": _finite_max_abs(grad),
            "event_batch_id": _event_batch_id(batch),
        }
    return None


def _maybe_nan_recover(optimizer, step_retries: int, max_retries: int,
                       lr_reduction_factor: float, debug_log_path: str,
                       epoch: int, step: int) -> "tuple[bool, bool]":
    """Single reconciled per-step NaN-recovery policy (Req 4.3-4.5).

    Superseding the previous *two* independent safeguards (a consecutive-step LR
    back-off and a separate cumulative ``abort_after_events`` abort), this is the
    ONE recovery mechanism: on a non-finite loss/grad for a step, reduce every
    param-group learning rate by ``lr_reduction_factor`` and signal the caller to
    retry the *same* step, up to ``max_retries`` retries (Req 4.3). Once the
    retries for a step reach ``max_retries``, signal an abort and report the
    exhausted-retry condition (Req 4.4). There is no cumulative-count abort
    (Req 4.5).

    ``step_retries`` is the number of retries already attempted for the current
    step (0 on the first non-finite event for that step).

    Returns ``(should_retry, should_abort)`` -- exactly one is ``True``.
    """
    if step_retries >= max_retries:
        _log_offending_batch(debug_log_path, {
            "epoch": epoch, "step": step, "stage": "nan_recovery",
            "action": "abort", "reason": "retries_exhausted",
            "step_retries": step_retries, "max_retries": max_retries,
        })
        return False, True

    for g in optimizer.param_groups:
        g["lr"] = g["lr"] * lr_reduction_factor
    _log_offending_batch(debug_log_path, {
        "epoch": epoch, "step": step, "stage": "nan_recovery",
        "action": "reduce_lr_and_retry",
        "new_lr": optimizer.param_groups[0]["lr"],
        "step_retries": step_retries + 1, "max_retries": max_retries,
    })
    return True, False


def _checkpoint_contract(model: Any, cfg: Any) -> dict[str, Any]:
    """Return the semantic/model-shape contract required for safe resume."""
    state_schema = {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in model.state_dict().items()
    }
    raw_signals = _cfg_get(cfg, "council.signals", []) or []
    signals = []
    for signal in raw_signals:
        if isinstance(signal, dict):
            signals.append({
                key: signal.get(key)
                for key in ("name", "source", "identifier")
            })
        else:
            signals.append(str(signal))
    model_family = "council_3way" if isinstance(model, CouncilModel) else "market_binary"
    return {
        "version": 2,
        "model_family": model_family,
        "direction_semantics": (
            "dovish_neutral_hawkish_v1"
            if model_family == "council_3way" else "binary_up_v1"
        ),
        "stance_class_order": (
            list(COUNCIL_STANCE_CLASS_ORDER)
            if model_family == "council_3way" else None
        ),
        "continuous_stance_semantics": (
            "probability_hawkish_minus_dovish_v1"
            if model_family == "council_3way" else "market_regression_v1"
        ),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "state_schema": state_schema,
        "base_model": _cfg_get(cfg, "model.deberta.base_model", None),
        "encoder_override": _cfg_get(cfg, "model.deberta.encoder_override", None),
        "encoder_mode": _cfg_get(cfg, "model.deberta.encoder_mode", "hierarchical"),
        "max_chunks": _cfg_get(cfg, "model.deberta.max_chunks", None),
        "chunk_size": _cfg_get(cfg, "model.deberta.chunk_size", None),
        "primary_target": _cfg_get(
            cfg, "model.deberta.stance.primary_target", None
        ),
        "signals": signals,
        "fusion_mode": _cfg_get(cfg, "council.fusion_mode", None),
        "loss_weights": _cfg_get(cfg, "model.deberta.loss_weights", {}) or {},
        "optimizer_numerics": {
            "parameter_dtype": "float32",
            **_stable_adamw_kwargs(cfg),
        },
    }


def _checkpoint_fingerprint(model: Any, cfg: Any) -> tuple[str, dict[str, Any]]:
    contract = _checkpoint_contract(model, cfg)
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), contract


def run_training_loop(model, loaders: dict, cfg: Any, debug_log_path: str,
                      ckpt_dir: Optional[str] = None, resume: bool = False,
                      save_every_steps: int = 0) -> "TrainResult":
    """Execute the full training loop honoring the numerical-stability policy.

    - **Two-phase precision** (17.2/17.3): epoch 0 forces FP32 (AMP disabled)
      regardless of ``use_amp``; BF16 AMP is enabled from epoch 1 onward only if
      the FP32 warmup produced all-finite loss.
    - **Gradient clipping** every optimiser step (17.3).
    - **assert_finite** guards on inputs, targets, outputs, loss, and gradients
      (17.1); a non-finite detection appends the offending batch to
      ``debug_log_path`` and skips the step (17.4).
    - **Minimum sample-size gate** (17.5): fewer than ``min_observations`` train
      rows records ``status='INSUFFICIENT_SAMPLE'``.
    - **Contrastive-group thresholds** via :func:`make_contrastive_groups`.
    - Early stopping on the validation metric with ``early_stopping_patience``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("run_training_loop requires PyTorch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Autocast controls operation precision only; it does not convert parameters
    # that a checkpoint loader materialized in FP16. Enforce FP32 master weights
    # before AdamW creates moments so its epsilon cannot underflow to zero.
    model = _move_model_to_fp32(model, device, cfg)

    # Effective per-signal Council weights are appended here each step when the
    # forward exposes them (Req 2.3). Resolved from the same ``outputs/audit``
    # convention as the other Council artifacts; logging stays a no-op for a
    # non-Council run because ``_extract_council_signal_weights`` returns None.
    council_weights_log_path = os.path.join(
        _council_audit_dir(cfg), _COUNCIL_SIGNAL_WEIGHTS_FILE
    )

    train_loader = loaders.get("train")
    val_loader = loaders.get("val")
    n_train = int(loaders.get("n_train", 0) or 0)

    min_obs = int(_cfg_get(cfg, "model.deberta.min_observations", 1))
    if train_loader is None or n_train < max(1, min_obs):
        _logger.warning(
            "training loop skipped: INSUFFICIENT_SAMPLE (n_train=%d < min_observations=%d)",
            n_train, max(1, min_obs),
        )
        return TrainResult(status="INSUFFICIENT_SAMPLE", n_samples=n_train)

    max_epochs = int(_cfg_get(cfg, "model.deberta.max_epochs", 50))
    patience = int(_cfg_get(cfg, "model.deberta.early_stopping_patience", 5))
    lr = float(_cfg_get(cfg, "model.deberta.learning_rate", 2e-5))
    accum = max(1, int(_cfg_get(cfg, "model.deberta.gradient_accumulation_steps", 1)))
    clip = float(_cfg_get(cfg, "model.deberta.grad_clip_norm", 1.0))
    want_amp = bool(_cfg_get(cfg, "model.deberta.use_amp", False)) and device.type == "cuda"
    # Opt-in live progress printing (default off so existing runs/tests stay quiet).
    # ``verbose`` prints an epoch header + per-epoch train/val summary; when
    # ``progress_every_steps`` > 0 it also prints a per-N-step heartbeat so a long
    # fine-tune run shows movement in the notebook/stdout log instead of looking hung.
    verbose = bool(_cfg_get(cfg, "model.deberta.verbose", False))
    progress_every = int(_cfg_get(cfg, "model.deberta.progress_every_steps", 0))
    # Single reconciled per-step NaN-recovery policy (Req 4.3-4.5): on a
    # non-finite loss/grad for a step, reduce every param-group LR by
    # ``lr_reduction_factor`` and retry the SAME step, up to ``max_retries``
    # retries; abort the run once a step's retries are exhausted. This supersedes
    # the old ad-hoc "100 events" cumulative abort and the separate
    # consecutive-step LR back-off -- retry exhaustion is the only abort trigger.
    lr_reduction = float(_cfg_get(cfg, "model.deberta.nan_recovery.lr_reduction_factor", 0.5))
    max_retries = int(_cfg_get(cfg, "model.deberta.nan_recovery.max_retries", 3))
    optim_steps_taken = 0       # real optimiser steps that actually landed
    # Count of steps left UNRECOVERED after exhausting their retry budget
    # (the single reconciled policy's abort trigger, Req 4.4). A step that
    # recovers within ``max_retries`` retries does not count. A clean run keeps
    # this at zero; reaching a nonzero value means the run aborted.
    nonfinite_events = 0

    # Two-group optimiser: the pretrained encoder keeps the base ``learning_rate``
    # while the randomly-initialised task heads get a higher LR so they actually
    # move off their initialisation within the few effective optimiser steps a
    # small per-target split affords. Heads stuck at init emit near-constant
    # predictions (the observed mean-collapse); a head LR multiplier > 1 gives
    # them the gradient budget to fit. The multiplier is config-driven
    # (``model.deberta.head_lr_multiplier``, default 10.0); set it to 1.0 to
    # recover the previous single-LR behaviour.
    head_lr_mult = float(_cfg_get(cfg, "model.deberta.head_lr_multiplier", 10.0))
    _head_prefixes = (
        "regression_head", "direction_head", "uncertainty_head",
        "stance_head", "pool",
        "council_heads", "stance_score_head",
    )
    head_params, base_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if pname.startswith(_head_prefixes) else base_params).append(p)
    adamw_kwargs = _stable_adamw_kwargs(cfg)

    def _build_optimizer_and_scheduler():
        """Create a fresh optimizer/scheduler pair bound to the live model."""
        if head_params and base_params and head_lr_mult != 1.0:
            built_optimizer = torch.optim.AdamW([
                {"params": base_params, "lr": lr},
                {"params": head_params, "lr": lr * head_lr_mult},
            ], **adamw_kwargs)
        else:
            built_optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, **adamw_kwargs
            )
        built_scheduler = torch.optim.lr_scheduler.StepLR(
            built_optimizer,
            step_size=max(1, max_epochs // 3 or 1),
            gamma=0.5,
        )
        return built_optimizer, built_scheduler

    optimizer, scheduler = _build_optimizer_and_scheduler()

    # Checkpoint footprint controls (Drive-limit mitigation). ``save_latest``
    # off => write only best.pt (no resumable rolling checkpoint), halving the
    # files per target. ``save_optimizer_state`` off => store just the model
    # weights, which is the bulk-independent part but still shrinks each file
    # and, more importantly, lets best.pt stay small for a weights-only reload.
    save_latest = bool(_cfg_get(cfg, "model.deberta.checkpoint.save_latest", True))
    save_optim = bool(_cfg_get(cfg, "model.deberta.checkpoint.save_optimizer_state", True))

    # Resumable checkpointing: latest.pt (most recent) + best.pt (best val MSE).
    latest_path = (os.path.join(ckpt_dir, "latest.pt")
                   if (ckpt_dir and save_latest) else None)
    best_path = os.path.join(ckpt_dir, "best.pt") if ckpt_dir else None
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    global_step = 0
    start_epoch = 0

    history: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = -1
    best_state = None
    epochs_no_improve = 0
    used_amp = False
    checkpoint_fingerprint, checkpoint_contract = _checkpoint_fingerprint(model, cfg)

    def _save_ckpt(path, epoch, step):
        if not path:
            return
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "global_step": step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "history": history,
            "checkpoint_fingerprint": checkpoint_fingerprint,
            "checkpoint_contract": checkpoint_contract,
        }
        # Optimizer + scheduler state roughly double a checkpoint's size (Adam
        # keeps two moment tensors per parameter). Only persist them when
        # resumable training is actually wanted.
        if save_optim:
            payload["optimizer"] = optimizer.state_dict()
            payload["scheduler"] = scheduler.state_dict()
        torch.save(payload, path)

    if resume and latest_path and os.path.exists(latest_path):
        try:
            ckpt = torch.load(latest_path, map_location=device)
            if not isinstance(ckpt, dict):
                raise TypeError(
                    f"checkpoint payload must be a mapping, got {type(ckpt).__name__}"
                )
        except Exception as exc:
            # A truncated/corrupt opt-in resume file must not abort the run before
            # its rejection can be audited. The untouched in-memory model and
            # freshly-created optimizer remain at their initial state.
            _log_offending_batch(debug_log_path, {
                "stage": "resume",
                "action": "reject_checkpoint_start_fresh",
                "reason": "checkpoint_deserialization_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "checkpoint_path": latest_path,
            })
        else:
            stored_fingerprint = ckpt.get("checkpoint_fingerprint")
            if stored_fingerprint != checkpoint_fingerprint:
                _log_offending_batch(debug_log_path, {
                    "stage": "resume",
                    "action": "reject_checkpoint_start_fresh",
                    "reason": (
                        "missing_checkpoint_fingerprint"
                        if stored_fingerprint is None
                        else "checkpoint_fingerprint_mismatch"
                    ),
                    "expected_fingerprint": checkpoint_fingerprint,
                    "stored_fingerprint": stored_fingerprint,
                    "checkpoint_path": latest_path,
                })
            else:
                # A matching fingerprint permits an opt-in resume attempt, but
                # loading is transactional. Keep one CPU snapshot of the fresh
                # model only around this load boundary; this is not the removed
                # per-optimizer-step snapshot and consumes no additional GPU
                # memory. A failed model/optimizer/scheduler validation restores
                # the caller-visible model and constructs brand-new optimizer
                # state before training starts.
                fresh_model_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                try:
                    candidate_start_epoch = int(ckpt.get("epoch", -1)) + 1
                    candidate_global_step = int(ckpt.get("global_step", 0))
                    candidate_best_metric = float(
                        ckpt.get("best_metric", best_metric)
                    )
                    candidate_best_epoch = int(
                        ckpt.get("best_epoch", best_epoch)
                    )
                    candidate_history = list(ckpt.get("history", history))

                    checkpoint_model_state = ckpt.get("model")
                    if not hasattr(checkpoint_model_state, "items"):
                        raise TypeError("checkpoint model state must be a mapping")
                    for state_name, state_value in checkpoint_model_state.items():
                        if (
                            torch.is_tensor(state_value)
                            and state_value.is_floating_point()
                            and not bool(torch.isfinite(state_value).all())
                        ):
                            raise ValueError(
                                f"non-finite checkpoint model tensor: {state_name}"
                            )

                    model.load_state_dict(checkpoint_model_state)
                    # Optimizer/scheduler may be absent from a weights-only
                    # checkpoint; in that case resume model weights with fresh
                    # optimizer state, but only after fingerprint compatibility.
                    if "optimizer" in ckpt:
                        optimizer.load_state_dict(ckpt["optimizer"])
                        # Old checkpoints may carry half-precision Adam moments.
                        # Convert finite moments to the current FP32 contract and
                        # reject poisoned state before another update can use it.
                        _coerce_and_validate_adamw_state(optimizer)
                    if "scheduler" in ckpt:
                        scheduler.load_state_dict(ckpt["scheduler"])
                except Exception as exc:
                    model.load_state_dict(fresh_model_state)
                    optimizer, scheduler = _build_optimizer_and_scheduler()
                    start_epoch = 0
                    global_step = 0
                    best_metric = float("inf")
                    best_epoch = -1
                    history = []
                    _log_offending_batch(debug_log_path, {
                        "stage": "resume",
                        "action": "reject_checkpoint_start_fresh",
                        "reason": "compatible_checkpoint_load_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "checkpoint_path": latest_path,
                    })
                else:
                    start_epoch = candidate_start_epoch
                    global_step = candidate_global_step
                    best_metric = candidate_best_metric
                    best_epoch = candidate_best_epoch
                    history = candidate_history
                    _log_offending_batch(debug_log_path, {
                        "stage": "resume",
                        "action": "checkpoint_loaded",
                        "from_epoch": start_epoch,
                        "global_step": global_step,
                        "checkpoint_fingerprint": checkpoint_fingerprint,
                    })
                finally:
                    del fresh_model_state

    _logger.info(
        "training loop start: device=%s n_train=%d epochs=%d->%d ckpt_dir=%s "
        "save_latest=%s resume=%s",
        device.type, n_train, start_epoch, max_epochs, ckpt_dir or "(none)",
        bool(latest_path), resume,
    )

    for epoch in range(start_epoch, max_epochs):
        # Two-phase precision: epoch 0 is always FP32; AMP allowed afterwards.
        amp_enabled = want_amp and epoch >= 1
        used_amp = used_amp or amp_enabled
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        n_steps = 0
        saw_nonfinite = False
        if verbose:
            print(f"[train] epoch {epoch + 1}/{max_epochs} "
                  f"(amp={'bf16' if amp_enabled else 'fp32'})", flush=True)

        for step, batch in enumerate(train_loader):
            if verbose and progress_every and step > 0 and step % progress_every == 0:
                print(f"[train]   epoch {epoch + 1} step {step} "
                      f"running_loss={epoch_loss / max(1, n_steps):.4f}", flush=True)
            batch = _move_batch(batch, device)
            try:
                assert_finite("input_ids", batch["input_ids"].float())
                assert_finite("targets", batch["target"])
            except ValueError:
                _log_offending_batch(debug_log_path, {
                    "epoch": epoch, "step": step, "stage": "input", "reason": "non_finite_input",
                })
                continue

            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if amp_enabled else _nullcontext()
            )
            # Single reconciled per-step retry policy (Req 4.3-4.5): re-run the
            # SAME step on a non-finite loss/grad after reducing every
            # param-group LR by ``lr_reduction_factor``, up to ``max_retries``
            # retries; abort once a step's retries are exhausted. ``step_retries``
            # counts retries already spent on the current step.
            step_retries = 0
            aborted = False
            step_loss: Optional[float] = None
            while True:
                try:
                    with ctx:
                        outputs = model(batch)
                        # Select the loss by model type (Req 2.1/2.2): a
                        # CouncilModel optimizes the multi-signal uncertainty-fused
                        # Council loss; a standard model keeps compute_losses with
                        # its temporal/contrastive augmentation. _select_and_compute_loss
                        # applies the temporal/contrastive terms only on the standard
                        # path so the Council loss stays per-signal-MSE composed.
                        losses = _select_and_compute_loss(model, outputs, batch, cfg)
                        loss = losses.get("total")
                        if loss is None:
                            break
                        assert_finite("loss", loss)
                    loss_to_back = loss / accum
                    loss_to_back.backward()
                    # Record this step's effective per-signal Council weights (Req
                    # 2.3) once the forward produced finite outputs. Defensive: a
                    # non-Council run (or a Council run before the fusion is wired
                    # into the forward) exposes no weights, so this is a no-op.
                    _step_council_weights = _extract_council_signal_weights(outputs)
                    if _step_council_weights is not None:
                        _log_council_signal_weights(council_weights_log_path, {
                            "epoch": epoch, "step": step,
                            "weights": _step_council_weights,
                        })
                except ValueError:
                    # Non-finite forward/loss: log the offending batch and recover.
                    saw_nonfinite = True
                    _log_offending_batch(debug_log_path, {
                        "epoch": epoch, "step": step, "stage": "forward_or_loss",
                        "reason": "non_finite_loss", "step_retries": step_retries,
                    })
                    optimizer.zero_grad(set_to_none=True)
                    _should_retry, aborted = _maybe_nan_recover(
                        optimizer, step_retries, max_retries, lr_reduction,
                        debug_log_path, epoch, step)
                    if _should_retry:
                        step_retries += 1
                        continue
                    break

                if (step + 1) % accum == 0:
                    # Guard gradients before the optimiser step. Find and record the
                    # FIRST non-finite gradient with its diagnostic fields (Req
                    # 4.1/4.2) rather than only asking "is any grad non-finite?".
                    nonfinite_grad = _first_nonfinite_gradient(model, losses, batch)
                    if nonfinite_grad is not None:
                        saw_nonfinite = True
                        _log_offending_batch(debug_log_path, {
                            "epoch": epoch, "step": step, "stage": "gradients",
                            "reason": "non_finite_gradient",
                            "step_retries": step_retries, **nonfinite_grad,
                        })
                        optimizer.zero_grad(set_to_none=True)
                        _should_retry, aborted = _maybe_nan_recover(
                            optimizer, step_retries, max_retries, lr_reduction,
                            debug_log_path, epoch, step)
                        if _should_retry:
                            step_retries += 1
                            continue
                        break
                    # Clip the accumulated gradient. Capture the returned total
                    # norm: when any gradient element is non-finite the norm is
                    # inf/nan and, depending on the torch version, the in-place
                    # scaling can leave non-finite gradients in place. Guard on the
                    # norm and, if it is non-finite, treat the step as a
                    # recoverable non-finite event instead of letting AdamW consume
                    # a poisoned gradient (which would push exp_avg_sq -> inf and
                    # corrupt every later update).
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), clip
                    )
                    if not bool(torch.isfinite(total_norm).all()):
                        saw_nonfinite = True
                        _log_offending_batch(debug_log_path, {
                            "epoch": epoch, "step": step, "stage": "gradients",
                            "reason": "non_finite_clipped_grad_norm",
                            "step_retries": step_retries,
                        })
                        optimizer.zero_grad(set_to_none=True)
                        _should_retry, aborted = _maybe_nan_recover(
                            optimizer, step_retries, max_retries, lr_reduction,
                            debug_log_path, epoch, step)
                        if _should_retry:
                            step_retries += 1
                            continue
                        break
                    # With FP32 master parameters/moments and eps>=1e-6, finite
                    # parameters + finite clipped gradients have a finite AdamW
                    # update. Keep the post-step checks as a last-resort guard,
                    # but do not clone the full model and two moment tensors on
                    # every healthy step (that tripled DeBERTa's optimizer-step
                    # memory footprint). A post-step failure is irreversible
                    # without such a snapshot, so fail closed instead of retrying
                    # a corrupted optimizer state.
                    optimizer.step()
                    _first_bad_param = None
                    for _pname, _p in model.named_parameters():
                        if not bool(torch.isfinite(_p).all()):
                            _first_bad_param = {
                                "parameter_name": _pname,
                                "loss_component": _resolve_loss_component(
                                    _pname, losses
                                ),
                                "pre_step_grad_stat": _finite_max_abs(_p.grad)
                                if _p.grad is not None else None,
                            }
                            break
                    _bad_state = (
                        _first_invalid_adamw_state(model, optimizer)
                        if optim_steps_taken == 0 else None
                    )
                    if _first_bad_param is not None or _bad_state is not None:
                        saw_nonfinite = True
                        if _first_bad_param is not None:
                            _log_offending_batch(debug_log_path, {
                                "epoch": epoch, "step": step,
                                "stage": "parameters",
                                "reason": "non_finite_parameters_after_step",
                                "step_retries": step_retries,
                                **_first_bad_param,
                            })
                        if _bad_state is not None:
                            _log_offending_batch(debug_log_path, {
                                "epoch": epoch, "step": step,
                                "stage": "optimizer_state",
                                "step_retries": step_retries,
                                **_bad_state,
                            })
                        optimizer.zero_grad(set_to_none=True)
                        _log_offending_batch(debug_log_path, {
                            "epoch": epoch, "step": step,
                            "stage": "nan_recovery", "action": "abort",
                            "reason": "post_step_state_corrupted",
                            "step_retries": step_retries,
                        })
                        aborted = True
                        break
                    optimizer.zero_grad(set_to_none=True)
                    optim_steps_taken += 1
                    global_step += 1
                    if save_every_steps and latest_path and global_step % save_every_steps == 0:
                        _save_ckpt(latest_path, epoch, global_step)

                step_loss = float(loss.detach().cpu())
                break

            if aborted:
                # Retries exhausted for this step -> the single reconciled
                # policy's only abort trigger (Req 4.4/4.5). Record the one
                # unrecovered step and stop the run.
                nonfinite_events += 1
                return TrainResult(
                    status="NAN_ABORTED", best_epoch=best_epoch,
                    best_metric=best_metric, history=history,
                    used_amp=used_amp, n_samples=n_train,
                    optim_steps=optim_steps_taken, nonfinite_events=nonfinite_events)
            if step_loss is None:
                continue

            epoch_loss += step_loss
            n_steps += 1

        # Flush the final partial accumulation window (Req 4.7). The in-loop
        # optimiser step only fires on ``(step + 1) % accum == 0``, so when the
        # epoch's batch count is not a multiple of ``accum`` the last window's
        # gradients are still pending on the parameters after the loop. Perform
        # one more optimiser step here so the effective optimiser-step count is
        # ``ceil(n_steps / accum)`` rather than ``floor``. Reuse the exact
        # gradient-guard / instrumentation / recovery logic so a non-finite
        # gradient in the flush is handled the same way as an in-loop step.
        if n_steps > 0 and n_steps % accum != 0:
            flush_step = n_steps - 1
            nonfinite_grad = _first_nonfinite_gradient(model, losses, batch)
            if nonfinite_grad is not None:
                _log_offending_batch(debug_log_path, {
                    "epoch": epoch, "step": flush_step, "stage": "gradients",
                    "reason": "non_finite_gradient", "window": "final_partial",
                    "step_retries": 0, **nonfinite_grad,
                })
                optimizer.zero_grad(set_to_none=True)
                _log_offending_batch(debug_log_path, {
                    "epoch": epoch, "step": flush_step,
                    "stage": "nan_recovery", "action": "abort",
                    "reason": "final_partial_window_cannot_be_replayed",
                })
                nonfinite_events += 1
                return TrainResult(
                    status="NAN_ABORTED", best_epoch=best_epoch,
                    best_metric=best_metric, history=history,
                    used_amp=used_amp, n_samples=n_train,
                    optim_steps=optim_steps_taken,
                    nonfinite_events=nonfinite_events)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not bool(torch.isfinite(total_norm).all()):
                _log_offending_batch(debug_log_path, {
                    "epoch": epoch, "step": flush_step, "stage": "gradients",
                    "reason": "non_finite_clipped_grad_norm",
                    "window": "final_partial", "step_retries": 0,
                })
                optimizer.zero_grad(set_to_none=True)
                nonfinite_events += 1
                return TrainResult(
                    status="NAN_ABORTED", best_epoch=best_epoch,
                    best_metric=best_metric, history=history,
                    used_amp=used_amp, n_samples=n_train,
                    optim_steps=optim_steps_taken,
                    nonfinite_events=nonfinite_events)

            optimizer.step()
            first_bad_param = None
            for parameter_name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    first_bad_param = {
                        "parameter_name": parameter_name,
                        "loss_component": _resolve_loss_component(
                            parameter_name, losses
                        ),
                        "pre_step_grad_stat": _finite_max_abs(parameter.grad)
                        if parameter.grad is not None else None,
                    }
                    break
            bad_state = (
                _first_invalid_adamw_state(model, optimizer)
                if optim_steps_taken == 0 else None
            )
            if first_bad_param is not None or bad_state is not None:
                if first_bad_param is not None:
                    _log_offending_batch(debug_log_path, {
                        "epoch": epoch, "step": flush_step,
                        "stage": "parameters",
                        "reason": "non_finite_parameters_after_step",
                        "window": "final_partial", **first_bad_param,
                    })
                if bad_state is not None:
                    _log_offending_batch(debug_log_path, {
                        "epoch": epoch, "step": flush_step,
                        "stage": "optimizer_state",
                        "window": "final_partial", **bad_state,
                    })
                optimizer.zero_grad(set_to_none=True)
                nonfinite_events += 1
                return TrainResult(
                    status="NAN_ABORTED", best_epoch=best_epoch,
                    best_metric=best_metric, history=history,
                    used_amp=used_amp, n_samples=n_train,
                    optim_steps=optim_steps_taken,
                    nonfinite_events=nonfinite_events)

            optimizer.zero_grad(set_to_none=True)
            optim_steps_taken += 1
            global_step += 1
            if save_every_steps and latest_path and global_step % save_every_steps == 0:
                _save_ckpt(latest_path, epoch, global_step)

        # Per-epoch NaN summary. In-epoch recovery (LR backoff + abort) already
        # ran inside the step loop via _maybe_nan_recover, so we no longer skip
        # validation on a warmup NaN -- if any real optimiser steps landed this
        # epoch, the model has genuinely updated and deserves a validation pass.
        if saw_nonfinite:
            _log_offending_batch(debug_log_path, {
                "epoch": epoch, "stage": "epoch_summary",
                "nonfinite_events_total": nonfinite_events,
                "optim_steps_this_run": optim_steps_taken,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Validation-based selection + early stopping.
        val_metric = float("inf")
        if val_loader is not None:
            y_true, y_pred = _predict_loader(
                model,
                val_loader,
                device,
                scaler=loaders.get("target_scaler"),
            )
            val_metric = _primary_metric_value(y_true, y_pred, cfg)
        else:
            val_metric = epoch_loss / max(1, n_steps)

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss / max(1, n_steps),
            "val_metric": val_metric,
            "amp": amp_enabled,
        })
        if verbose:
            print(f"[train] epoch {epoch + 1}/{max_epochs} done "
                  f"train_loss={epoch_loss / max(1, n_steps):.4f} "
                  f"val_metric={val_metric:.4f} "
                  f"optim_steps={optim_steps_taken}", flush=True)
        # Mirror the per-epoch summary to the pipeline log so progress is visible
        # even when stdout prints are captured/buffered (e.g. notebook logs).
        _logger.info(
            "epoch %d/%d done train_loss=%.4f val_metric=%.4f optim_steps=%d",
            epoch + 1, max_epochs, epoch_loss / max(1, n_steps), val_metric,
            optim_steps_taken,
        )

        scheduler.step()
        _save_ckpt(latest_path, epoch, global_step)
        if latest_path:
            _logger.info("saved checkpoint: %s (epoch %d, step %d)",
                         latest_path, epoch, global_step)

        if val_metric < best_metric - 1e-9:
            best_metric = val_metric
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            _save_ckpt(best_path, epoch, global_step)
            if best_path:
                _logger.info("saved best checkpoint: %s (epoch %d, val_metric=%.4f)",
                             best_path, epoch, val_metric)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # If not a single real optimiser step ever landed, the model still holds its
    # initial weights and will emit (near-)constant predictions. Report that
    # honestly instead of letting a degenerate run masquerade as a valid result
    # (this is precisely what produced the y_pred std==0 collapse before).
    if optim_steps_taken == 0:
        _log_offending_batch(debug_log_path, {
            "stage": "run_summary", "reason": "no_optimizer_step_landed",
            "nonfinite_events_total": nonfinite_events,
        })
        return TrainResult(
            status="UNTRAINED_NO_STEP", best_epoch=best_epoch,
            best_metric=best_metric, history=history, used_amp=used_amp,
            n_samples=n_train, optim_steps=0, nonfinite_events=nonfinite_events)

    return TrainResult(
        status="OK",
        best_epoch=best_epoch,
        best_metric=best_metric,
        history=history,
        used_amp=used_amp,
        n_samples=n_train,
        optim_steps=optim_steps_taken,
        nonfinite_events=nonfinite_events,
    )


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _predict_loader(
    model,
    loader,
    device,
    scaler: Optional[dict] = None,
    return_details: bool = False,
):
    """Return aligned raw predictions, optionally with Council stance details.

    Council union rows without a genuine primary target are excluded through
    ``primary_present``.  When ``return_details`` is true, the third return value
    carries source-row indices plus Dovish/Neutral/Hawkish probabilities and the
    derived score. Source indices use the unfiltered loader-frame order, so
    identity/date alignment remains correct even if a non-finite batch is
    omitted.
    """
    model.eval()
    trues, preds = [], []
    detail_rows, detail_probs, detail_scores = [], [], []
    row_cursor = 0
    mean = float((scaler or {}).get("mean", 0.0))
    std = float((scaler or {}).get("std", 1.0)) or 1.0
    with torch.no_grad():
        for batch in loader:
            batch_size = int(batch["target"].reshape(-1).shape[0])
            source_rows = np.arange(
                row_cursor, row_cursor + batch_size, dtype="int64"
            )
            row_cursor += batch_size
            batch = _move_batch(batch, device)
            try:
                out = model(batch)
            except ValueError:
                # A non-finite forward remains an explicit omitted batch rather
                # than a validation crash; source_rows still advances so later
                # event identities cannot shift onto the wrong predictions.
                continue
            pred_tensor = out.get("primary_prediction", out.get("regression"))
            if pred_tensor is None:
                continue
            pred = pred_tensor.detach().float().cpu().numpy().reshape(-1)
            if scaler is not None:
                pred = pred * std + mean
            raw = batch.get("target_raw", batch["target"])
            true = raw.detach().float().cpu().numpy().reshape(-1)
            present = batch.get("primary_present")
            if present is None:
                observed = np.ones_like(true, dtype=bool)
            else:
                observed = present.detach().bool().cpu().numpy().reshape(-1)
            finite = observed & np.isfinite(true) & np.isfinite(pred)
            if not finite.any():
                continue
            preds.append(pred[finite])
            trues.append(true[finite])

            if return_details:
                probs = np.full((batch_size, 3), np.nan, dtype="float64")
                class_order = tuple(out.get("stance_class_order", ()))
                probs_tensor = out.get(
                    "primary_direction_probs", out.get("stance_probs")
                )
                if (
                    class_order == COUNCIL_STANCE_CLASS_ORDER
                    and torch.is_tensor(probs_tensor)
                    and probs_tensor.ndim == 2
                    and tuple(probs_tensor.shape) == (batch_size, 3)
                ):
                    probs = (
                        probs_tensor.detach().float().cpu().numpy().astype(
                            "float64", copy=False
                        )
                    )
                # Derive the persisted score from probabilities rather than
                # trusting a second model output, preserving the exact identity.
                scores = probs[:, 2] - probs[:, 0]
                detail_rows.append(source_rows[finite])
                detail_probs.append(probs[finite])
                detail_scores.append(scores[finite])

    if not preds:
        y_true = np.array([], dtype="float64")
        y_pred = np.array([], dtype="float64")
    else:
        y_true = np.concatenate(trues)
        y_pred = np.concatenate(preds)
    if not return_details:
        return y_true, y_pred

    if detail_rows:
        rows = np.concatenate(detail_rows).astype("int64", copy=False)
        probs = np.concatenate(detail_probs).astype("float64", copy=False)
        scores = np.concatenate(detail_scores).astype("float64", copy=False)
    else:
        rows = np.array([], dtype="int64")
        probs = np.empty((0, 3), dtype="float64")
        scores = np.array([], dtype="float64")
    details = {
        "row_indices": rows,
        "stance_probs": probs,
        "stance_score": scores,
        "stance_class_order": COUNCIL_STANCE_CLASS_ORDER,
        "stance_available": bool(
            len(rows) == len(y_true)
            and probs.shape == (len(y_true), 3)
            and np.isfinite(probs).all()
            and np.isfinite(scores).all()
        ),
    }
    return y_true, y_pred, details


def predict(
    model,
    loader,
    cfg: Any,
    scaler: Optional[dict] = None,
    return_details: bool = False,
):
    """Predict raw outcomes and optionally aligned Council stance details.

    A task-local scaler attached by :func:`build_dataloaders` takes precedence
    over a legacy caller-supplied scaler. The default two-array return contract
    remains unchanged; ``return_details=True`` adds a third details mapping.
    """
    if not _HAS_TORCH:
        raise RuntimeError("predict requires PyTorch.")
    device = next(model.parameters()).device
    task_scaler = getattr(loader, "target_scaler", None)
    effective_scaler = task_scaler if task_scaler is not None else scaler
    return _predict_loader(
        model,
        loader,
        device,
        scaler=effective_scaler,
        return_details=return_details,
    )


# =============================================================================
# Task-data assembly with stance-signal promotion (Req 4) -- Task 5.1
#
# ``assemble_task_parquets`` is the single task-assembly entry point that hooks
# dataset promotion into the existing ``build_targets`` machinery. It calls
# ``ConfigManager.promoted_stance_specs(cfg)`` (which itself reuses
# ``build_promoted_stance_specs``), concatenates the returned promoted specs with
# ``ConfigManager.target_registry(cfg)``, and drives the unchanged
# ``build_task_parquets`` registry loop so each promoted dataset yields exactly
# one additional ``model_<name>`` task parquet. The promotion, provenance
# validation, and exclusion logic all live in ``src/config.py`` and are consumed
# unchanged here -- this function reimplements none of it (Req 4.2).
# =============================================================================

#: Column order of the promotion exclusion audit record (mirrors the enumerated
#: ``{"name","reason","field"}`` records returned by ``build_promoted_stance_specs``).
_PROMOTION_EXCLUSION_COLUMNS = ["name", "reason", "field"]


def _write_promotion_exclusions(exclusions: list, audit_dir: str) -> Optional[str]:
    """Write the promotion ``exclusions`` to ``promotion_exclusions.csv`` (Req 4.4/4.5).

    Each record carries the requested signal ``name``, the enumerated ``reason``
    (e.g. ``REASON_NON_ECB_CONTEXT_ONLY`` for a non-ECB context dataset such as
    ``US_MPS``, or ``REASON_MISSING_PROVENANCE_<field>`` for a spec missing one of
    the four provenance fields), and the offending ``field`` (or empty). When
    there are no exclusions a HEADER-ONLY file is written so the audit output is
    always present and consistent. Returns the written path, or ``None`` when no
    ``audit_dir`` was supplied.
    """
    if not audit_dir:
        return None
    os.makedirs(audit_dir, exist_ok=True)
    path = os.path.join(audit_dir, "promotion_exclusions.csv")
    frame = pd.DataFrame(list(exclusions or []), columns=_PROMOTION_EXCLUSION_COLUMNS)
    frame = frame.reindex(columns=_PROMOTION_EXCLUSION_COLUMNS)
    frame.to_csv(path, index=False)
    return path


def assemble_task_parquets(
    master: "pd.DataFrame",
    splits: Any,
    cfg: Any,
    models_dir: Optional[str] = None,
    audit_dir: Optional[str] = None,
) -> dict:
    """Assemble per-target task parquets, hooking in promoted stance signals (Req 4).

    Calls ``ConfigManager.promoted_stance_specs(cfg)`` to obtain ``(specs,
    exclusions)``, concatenates the promoted ``specs`` with
    ``ConfigManager.target_registry(cfg)``, and feeds the combined registry to the
    existing ``build_task_parquets`` (``build_targets``) machinery so each
    successfully promoted enabled dataset produces exactly one additional task
    parquet (Req 4.1). The returned ``exclusions`` -- a non-ECB context dataset
    recorded as ``REASON_NON_ECB_CONTEXT_ONLY`` (never an ECB registry entry), or
    a spec missing a provenance field recorded as
    ``REASON_MISSING_PROVENANCE_<field>`` -- are written to the audit output
    (Req 4.4/4.5). When ``model.deberta.stance.promote_signals`` is empty,
    ``promoted_stance_specs`` returns no specs, so the combined registry equals the
    base registry and the task data is unchanged with respect to promotion
    (Req 4.3).

    Reuses ``build_promoted_stance_specs`` / ``ConfigManager.promoted_stance_specs``
    and reimplements no promotion, provenance-validation, or exclusion logic
    (Req 4.2).

    Returns ``{"task_paths", "exclusions", "promoted_specs", "exclusion_log",
    "promotion_exclusions_path", "registry"}`` where ``task_paths`` maps
    ``model_<target>`` (base + promoted) to the written parquet path.
    """
    from .config import ConfigManager
    from .dataset_builder import build_task_parquets

    manager = ConfigManager()
    base_registry = manager.target_registry(cfg)
    promoted_specs, exclusions = manager.promoted_stance_specs(cfg)

    # Concatenate promoted specs onto the base registry so the existing
    # build_task_parquets loop produces one additional model_<name> parquet per
    # promoted dataset (Req 4.1). Empty promote_signals -> no promoted specs ->
    # combined registry == base registry -> task data unchanged (Req 4.3).
    registry = list(base_registry) + list(promoted_specs)

    task_paths, exclusion_log = build_task_parquets(
        master, registry, splits, models_dir=models_dir,
    )

    promotion_exclusions_path = _write_promotion_exclusions(exclusions, audit_dir)

    return {
        "task_paths": task_paths,
        "exclusions": exclusions,
        "promoted_specs": promoted_specs,
        "exclusion_log": exclusion_log,
        "promotion_exclusions_path": promotion_exclusions_path,
        "registry": registry,
    }


# =============================================================================
# Per-target orchestration convenience
# =============================================================================

# =============================================================================
# Frozen-encoder headline vs fine-tuning ablation (Req 6) -- Task 11.1
#
# A single config switch, ``model.deberta.frozen_encoder`` (default true),
# selects the training mode with NO per-experiment code edit (Req 6.3):
#
#   * true  (headline)  -> freeze every Shared_Encoder + ``_AttentionPool``
#                          parameter (requires_grad=False), extract the pooled
#                          ``doc`` representation for each event, and fit simple
#                          ridge / linear heads on those FROZEN features
#                          (Req 6.1). Stable on the small sample because only a
#                          handful of head parameters are learned; the encoder is
#                          never updated (Property 7).
#   * false (ablation)  -> joint fine-tune the full ``CouncilModel`` (encoder +
#                          Signal_Heads) through :func:`run_training_loop`
#                          (Req 6.2).
#
# ``train_target`` branches on the switch; ``train_target_frozen`` implements the
# frozen path, reusing the pooled ``doc`` extractor (:func:`encode_latents`) and
# the ridge/linear fitters from :mod:`src.baselines`.
# =============================================================================

#: Head estimator selected by ``model.deberta.frozen_head`` for the frozen path.
_FROZEN_HEAD_DEFAULT = "ridge"


def _freeze_encoder(model) -> int:
    """Freeze the encoder and initialize frozen pooling as a deterministic mean.

    The learned attention scorer is untrained on the frozen path; leaving its
    random initialization in place makes headline features seed-dependent and
    can collapse them. Zero scorer weights produce uniform softmax weights over
    present chunks, i.e. an explicit deterministic mean pool.
    """
    pool = getattr(model, "pool", None)
    score = getattr(pool, "score", None)
    if score is not None:
        with torch.no_grad():
            score.weight.zero_()
            if score.bias is not None:
                score.bias.zero_()
        model._frozen_pooling_mode = "uniform_mean"
    else:
        model._frozen_pooling_mode = "flat_single_segment"

    frozen = 0
    for module in (getattr(model, "encoder", None), pool):
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = False
            frozen += 1
    model.eval()
    return frozen


def _fit_frozen_head(name: str, x_tr, y_tr, x_te, cfg: Any):
    """Fit a simple ridge / linear head on FROZEN pooled features (Req 6.1).

    Reuses the same estimators :func:`src.baselines.fit_predict_baseline` wires
    for its ``"ridge"`` / ``"linear"`` baselines (``RidgeCV`` / ``LinearRegression``
    from scikit-learn), but fits them directly on the pooled ``doc`` feature
    matrix rather than on TF-IDF text features -- these ARE the frozen encoder
    features (Req 6.1, 12.2). Returns ``(preds [N_test], estimator_name)``.

    Falls back to the training-mean predictor (the same graceful degradation
    ``fit_predict_baseline`` uses) when scikit-learn is unavailable or the fit
    raises, so the frozen path never crashes a run.
    """
    y_tr = np.asarray(y_tr, dtype="float64").reshape(-1)
    grids = cfg.get("model", {}).get("baselines", {}) or {} if isinstance(cfg, dict) else {}
    try:
        if name == "linear":
            from sklearn.linear_model import LinearRegression

            est = LinearRegression()
        else:  # ridge (default)
            from sklearn.linear_model import RidgeCV

            alphas = grids.get("ridge", {}).get("alpha", [0.01, 0.1, 1.0, 10.0, 100.0])
            est = RidgeCV(alphas=np.asarray(alphas, dtype=float))
        est.fit(x_tr, y_tr)
        preds = np.asarray(est.predict(x_te), dtype="float64").reshape(-1)
        if not np.all(np.isfinite(preds)):
            raise ValueError("non-finite predictions from frozen head estimator")
        return preds, type(est).__name__
    except Exception as exc:  # sklearn missing / fit failure -> graceful fallback
        mean_value = float(np.mean(y_tr)) if y_tr.size else 0.0
        preds = np.full(int(x_te.shape[0]) if hasattr(x_te, "shape") else 0,
                        mean_value, dtype="float64")
        return preds, f"historical_mean (degraded: {type(exc).__name__}: {exc})"


def _latent_diagnostics(values: Any) -> dict[str, Any]:
    """Summarize frozen feature spread/rank without hiding degeneration."""
    arr = np.asarray(values, dtype="float64")
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return {
            "n_rows": int(arr.shape[0]) if arr.ndim >= 1 else 0,
            "n_features": int(arr.shape[1]) if arr.ndim == 2 else 0,
            "finite": bool(np.isfinite(arr).all()) if arr.size else True,
            "centered_rank": 0,
            "variance_mean": 0.0,
            "variance_min": 0.0,
            "variance_max": 0.0,
            "zero_variance_fraction": 1.0,
        }
    finite = bool(np.isfinite(arr).all())
    clean = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    variance = np.var(clean, axis=0)
    centered = clean - clean.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    return {
        "n_rows": int(clean.shape[0]),
        "n_features": int(clean.shape[1]),
        "finite": finite,
        "centered_rank": rank,
        "max_possible_centered_rank": int(min(max(clean.shape[0] - 1, 0), clean.shape[1])),
        "variance_mean": float(np.mean(variance)),
        "variance_min": float(np.min(variance)),
        "variance_max": float(np.max(variance)),
        "zero_variance_fraction": float(np.mean(variance <= 1e-12)),
    }


def _constant_prediction_diagnostics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype="float64").reshape(-1)
    pred = np.asarray(y_pred, dtype="float64").reshape(-1)
    pred_std = float(np.std(pred)) if pred.size else float("nan")
    target_std = float(np.std(truth)) if truth.size else float("nan")
    relative_floor = 1e-6 * (target_std if math.isfinite(target_std) and target_std > 0 else 1.0)
    constant = bool(
        pred.size > 0
        and math.isfinite(pred_std)
        and pred_std <= max(1e-12, relative_floor)
    )
    return {
        "prediction_std": pred_std,
        "target_std": target_std,
        "constant_prediction": constant,
        "constant_floor": float(max(1e-12, relative_floor)),
    }


def _persist_target_scaler(
    output_dir: str, target_name: str, scaler: Optional[dict]
) -> Optional[str]:
    """Persist one task-local train-only scaler beside that target's artifacts."""
    if scaler is None:
        return None
    safe_target = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(target_name)
    )
    path = os.path.join(output_dir, "models", "scalers", f"{safe_target}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(scaler), fh, indent=2, default=str)
    return path


def _observed_primary_frame(frame: Optional["pd.DataFrame"]) -> Optional["pd.DataFrame"]:
    """Return frame rows aligned to primary-filtered prediction/latent arrays."""
    if frame is None:
        return None
    out = frame
    if "primary_present" in out.columns:
        out = out[out["primary_present"].fillna(False).astype(bool)]
    if "target" in out.columns:
        finite = np.isfinite(pd.to_numeric(out["target"], errors="coerce"))
        out = out[finite]
    return out.reset_index(drop=True)


def _build_predictions_frame(
    y_true: Any,
    y_pred: Any,
    test_frame: Optional["pd.DataFrame"] = None,
    *,
    details: Optional[dict] = None,
) -> "pd.DataFrame":
    """Assemble aligned market and Council stance prediction rows.

    The legacy ``y_true``/``y_pred`` schema is preserved. When detailed Council
    outputs are supplied, source-row indices align identities against the
    unfiltered test frame and append the canonical stance columns in explicit
    Dovish/Neutral/Hawkish order.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    data: dict[str, Any] = {"y_true": y_true, "y_pred": y_pred}
    n = len(y_true)

    aligned_frame = None
    if test_frame is not None and n > 0:
        row_indices = np.asarray(
            (details or {}).get("row_indices", []), dtype="int64"
        ).reshape(-1)
        if (
            len(row_indices) == n
            and bool((row_indices >= 0).all())
            and bool((row_indices < len(test_frame)).all())
        ):
            aligned_frame = test_frame.iloc[row_indices].reset_index(drop=True)
        elif len(test_frame) == n:
            aligned_frame = test_frame.reset_index(drop=True)

    if aligned_frame is not None:
        if "event_id" in aligned_frame.columns:
            data["event_id"] = np.asarray(
                aligned_frame["event_id"].tolist(), dtype=object
            )
        if "event_date" in aligned_frame.columns:
            dates = pd.to_datetime(aligned_frame["event_date"], errors="coerce")
        elif "event_timestamp" in aligned_frame.columns:
            dates = pd.to_datetime(
                aligned_frame["event_timestamp"], errors="coerce"
            ).dt.normalize()
        else:
            dates = None
        if dates is not None:
            data["event_date"] = np.asarray(
                [date.date() if pd.notna(date) else None for date in dates],
                dtype=object,
            )

    if details and bool(details.get("stance_available", False)):
        probs = np.asarray(details.get("stance_probs"), dtype="float64")
        if probs.shape != (n, 3):
            raise ValueError(
                f"stance_probs must have shape ({n}, 3), got {probs.shape}"
            )
        if not np.isfinite(probs).all():
            raise ValueError("stance probabilities must be finite")
        row_sums = probs.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=1e-5):
            raise ValueError("stance probabilities must sum to one per event")
        if bool((probs < -1e-7).any()) or bool((probs > 1.0 + 1e-7).any()):
            raise ValueError("stance probabilities must lie in [0, 1]")
        derived_score = probs[:, 2] - probs[:, 0]
        supplied_score = np.asarray(
            details.get("stance_score", derived_score), dtype="float64"
        ).reshape(-1)
        if supplied_score.shape != (n,) or not np.allclose(
            supplied_score, derived_score, atol=1e-6, rtol=1e-6
        ):
            raise ValueError(
                "stance_score must equal P(hawkish) - P(dovish) for every event"
            )
        dominant = np.asarray(COUNCIL_STANCE_CLASS_ORDER, dtype=object)[
            np.argmax(probs, axis=1)
        ]
        data.update({
            "stance_prob_dovish": probs[:, 0],
            "stance_prob_neutral": probs[:, 1],
            "stance_prob_hawkish": probs[:, 2],
            "stance_score": derived_score,
            "stance_dominant": dominant,
        })

    return pd.DataFrame(data)


_STANCE_EVENT_COLUMNS = (
    "event_id",
    "event_date",
    "stance_prob_dovish",
    "stance_prob_neutral",
    "stance_prob_hawkish",
    "stance_score",
    "stance_dominant",
)


def summarize_stance_predictions(frame: "pd.DataFrame") -> dict[str, Any]:
    """Summarize persisted Council stance probabilities and dominant classes."""
    required = set(_STANCE_EVENT_COLUMNS[2:])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"stance prediction frame missing columns: {missing}")
    if len(frame) == 0:
        raise ValueError("cannot summarize an empty stance prediction frame")

    probability_columns = [
        "stance_prob_dovish",
        "stance_prob_neutral",
        "stance_prob_hawkish",
    ]
    probs = frame[probability_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype="float64")
    scores = pd.to_numeric(
        frame["stance_score"], errors="coerce"
    ).to_numpy(dtype="float64")
    if not np.isfinite(probs).all() or not np.isfinite(scores).all():
        raise ValueError("stance prediction artifact contains non-finite values")
    row_sum_error = float(np.max(np.abs(probs.sum(axis=1) - 1.0)))
    score_error = float(np.max(np.abs(scores - (probs[:, 2] - probs[:, 0]))))
    if row_sum_error > 1e-5 or score_error > 1e-6:
        raise ValueError(
            "stance prediction identities failed: "
            f"simplex_error={row_sum_error}, score_error={score_error}"
        )

    labels = np.asarray(COUNCIL_STANCE_CLASS_ORDER, dtype=object)
    dominant = labels[np.argmax(probs, axis=1)]
    n_events = int(len(frame))
    means = probs.mean(axis=0)
    counts = {label: int(np.sum(dominant == label)) for label in labels}
    shares = {label: counts[label] / n_events for label in labels}
    mean_mass = {label: float(means[index]) for index, label in enumerate(labels)}
    summary = {
        "class_order": list(COUNCIL_STANCE_CLASS_ORDER),
        "n_events": n_events,
        "mean_probability_mass": mean_mass,
        "mean_probability_mass_percent": {
            label: 100.0 * value for label, value in mean_mass.items()
        },
        "dominant_event_count": counts,
        "dominant_event_share": shares,
        "dominant_event_share_percent": {
            label: 100.0 * value for label, value in shares.items()
        },
        "continuous_stance": {
            "formula": "P(hawkish) - P(dovish)",
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        },
        "identity_checks": {
            "probability_sum_max_abs_error": row_sum_error,
            "score_formula_max_abs_error": score_error,
            "passed": True,
        },
    }
    return summary


def print_stance_summary(summary: dict[str, Any]) -> None:
    """Print probability mass and dominant-class event shares to stdout."""
    mass = summary.get("mean_probability_mass_percent", {})
    shares = summary.get("dominant_event_share_percent", {})
    score = summary.get("continuous_stance", {})
    print("\nSTANCE DISTRIBUTION (held-out events)", flush=True)
    print(
        "  Mean probability mass: "
        f"Hawkish {float(mass.get('hawkish', float('nan'))):.2f}% | "
        f"Dovish {float(mass.get('dovish', float('nan'))):.2f}% | "
        f"Neutral {float(mass.get('neutral', float('nan'))):.2f}%",
        flush=True,
    )
    print(
        "  Dominant-class event share: "
        f"Hawkish {float(shares.get('hawkish', float('nan'))):.2f}% | "
        f"Dovish {float(shares.get('dovish', float('nan'))):.2f}% | "
        f"Neutral {float(shares.get('neutral', float('nan'))):.2f}%",
        flush=True,
    )
    print(
        "  Continuous stance: "
        f"mean={float(score.get('mean', float('nan'))):.4f} "
        f"range=[{float(score.get('min', float('nan'))):.4f}, "
        f"{float(score.get('max', float('nan'))):.4f}] "
        "(P(Hawkish) - P(Dovish))",
        flush=True,
    )


def train_target_frozen(task_path: str, cfg: Any, output_dir: str, target_name: str,
                        scaler: Optional[dict] = None) -> dict:
    """Frozen-encoder headline for ONE target (Req 6.1): freeze + fit simple heads.

    Loads the model (the DAPT encoder body when configured), FREEZES every
    encoder + ``_AttentionPool`` parameter (``requires_grad=False``, Req 6.1),
    extracts the pooled ``doc`` representation for the train and test splits via
    :func:`encode_latents`, and fits a ridge (default) or linear head on the
    frozen features via :func:`_fit_frozen_head` -- the encoder is never updated
    (Property 7). Test predictions are inverse-transformed with ``scaler`` when
    supplied and persisted exactly like :func:`train_target` so downstream
    evaluation is mode-agnostic.

    Skips gracefully (status ``INSUFFICIENT_SAMPLE``) when the target has no
    usable train/test rows.
    """
    if not _HAS_TORCH:
        raise RuntimeError("train_target_frozen requires PyTorch.")
    # Seed the stance path before any stochastic step (mirrors train_target).
    seed_stance_pipeline(cfg)
    tokenizer = build_tokenizer(cfg)
    loaders = build_dataloaders(task_path, tokenizer, cfg)
    # Always prefer the scaler reconstructed from this task's train rows. The
    # optional argument is retained only for legacy artifacts without train data.
    task_scaler = loaders.get("target_scaler")
    effective_scaler = task_scaler if task_scaler is not None else scaler
    scaler_path = _persist_target_scaler(output_dir, target_name, effective_scaler)

    model = build_model(cfg)
    # Run feature extraction on the GPU when available. build_model() leaves the
    # model on CPU, and encode_latents() infers its device from the model's
    # parameters -- so without this move the frozen headline extracts pooled
    # features for the ENTIRE corpus on CPU (hierarchical DeBERTa forward over
    # batch x max_chunks x chunk_size), which is 10-50x slower and looks hung.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    n_frozen = _freeze_encoder(model)

    n_train = int(loaders.get("n_train", 0) or 0)
    n_primary_train = int((task_scaler or {}).get("n", 0))
    min_obs = int(_cfg_get(cfg, "model.deberta.min_observations", 1))
    out: dict[str, Any] = {
        "target": target_name,
        "mode": "frozen",
        "frozen_params": n_frozen,
        "pooling_mode": getattr(model, "_frozen_pooling_mode", "unknown"),
        "target_scaler": effective_scaler,
        "target_scaler_path": scaler_path,
        "n_union_train": n_train,
        "n_primary_train": n_primary_train,
    }
    if loaders.get("train") is None or n_primary_train < max(1, min_obs) \
            or loaders.get("test") is None:
        out["status"] = "INSUFFICIENT_SAMPLE"
        out["n_test"] = int(loaders.get("n_test", 0) or 0)
        out["train_result"] = asdict(TrainResult(
            status="INSUFFICIENT_SAMPLE", n_samples=n_primary_train
        ))
        return out

    # Extract pooled ``doc`` features on the frozen encoder for train + test.
    # encode_latents removes canonical Council rows lacking a real primary label.
    x_tr, y_tr = encode_latents(model, loaders["train"], device=device)
    x_te, y_te = encode_latents(model, loaders["test"], device=device)
    if len(y_tr) < max(1, min_obs) or len(y_te) == 0:
        out["status"] = "INSUFFICIENT_SAMPLE"
        out["n_test"] = int(len(y_te))
        out["train_result"] = asdict(TrainResult(
            status="INSUFFICIENT_SAMPLE", n_samples=int(len(y_tr))
        ))
        return out

    head_name = str(_cfg_get(cfg, "model.deberta.frozen_head", _FROZEN_HEAD_DEFAULT))
    if head_name not in ("ridge", "linear"):
        head_name = _FROZEN_HEAD_DEFAULT
    y_pred, estimator = _fit_frozen_head(head_name, x_tr, y_tr, x_te, cfg)

    # Frozen heads are fit directly against raw labels; predictions are already
    # in natural target units and must not be inverse-transformed again.
    y_true = np.asarray(y_te, dtype="float64").reshape(-1)
    train_latent_diag = _latent_diagnostics(x_tr)
    test_latent_diag = _latent_diagnostics(x_te)
    prediction_diag = _constant_prediction_diagnostics(y_true, y_pred)
    degradation_reasons: list[str] = []
    if "degraded:" in estimator:
        degradation_reasons.append(estimator)
    if not train_latent_diag["finite"] or train_latent_diag["centered_rank"] == 0:
        degradation_reasons.append("train_latents_nonfinite_or_rank_zero")
    if prediction_diag["constant_prediction"]:
        degradation_reasons.append("constant_prediction")
    diagnostics = {
        "target": target_name,
        "pooling_mode": getattr(model, "_frozen_pooling_mode", "unknown"),
        "head": head_name,
        "estimator": estimator,
        "degraded": bool(degradation_reasons),
        "degradation_reasons": degradation_reasons,
        "train_latents": train_latent_diag,
        "test_latents": test_latent_diag,
        "predictions": prediction_diag,
    }

    preds_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(preds_dir, exist_ok=True)
    pred_path = os.path.join(preds_dir, f"predictions_{target_name}.parquet")
    test_frame = _observed_primary_frame(loaders.get("frames", {}).get("test"))
    _build_predictions_frame(y_true, y_pred, test_frame).to_parquet(
        pred_path, index=False
    )
    diagnostics_path = os.path.join(preds_dir, f"frozen_diagnostics_{target_name}.json")
    with open(diagnostics_path, "w", encoding="utf-8") as fh:
        json.dump(diagnostics, fh, indent=2, default=str)

    out.update({
        "status": "OK",
        "train_result": asdict(TrainResult(
            status="OK", n_samples=int(len(y_tr))
        )),
        "n_test": int(len(y_true)),
        "predictions_path": pred_path,
        "diagnostics_path": diagnostics_path,
        "diagnostics": diagnostics,
        "degraded": bool(degradation_reasons),
        "degradation_reasons": degradation_reasons,
        "head": head_name,
        "estimator": estimator,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    return out


def train_target(task_path: str, cfg: Any, output_dir: str, target_name: str,
                  scaler: Optional[dict] = None) -> dict:
    """Build loaders, train, evaluate the model for ONE target, and persist.

    Branches on the ``model.deberta.frozen_encoder`` switch (default true,
    Req 6.3) with NO per-experiment code edit:

    * ``frozen_encoder=true``  -> the frozen-encoder headline: delegate to
      :func:`train_target_frozen`, which freezes the encoder + ``_AttentionPool``
      and fits ridge/linear heads on the pooled features (Req 6.1).
    * ``frozen_encoder=false`` -> the fine-tuning ablation: joint fine-tune the
      full ``CouncilModel`` (encoder + Signal_Heads) via
      :func:`run_training_loop` (Req 6.2).

    Returns a dict with the :class:`TrainResult` (as a dict), the test
    predictions, and the paths written. Skips gracefully (status
    ``INSUFFICIENT_SAMPLE``) when the target has no usable rows (e.g. fx).
    """
    if not _HAS_TORCH:
        raise RuntimeError("train_target requires PyTorch.")

    # Config-switch only (Req 6.3): the frozen-encoder headline is the default.
    if bool(_cfg_get(cfg, "model.deberta.frozen_encoder", True)):
        return train_target_frozen(task_path, cfg, output_dir, target_name,
                                    scaler=scaler)
    # Seed the stance path before any stochastic stance step (dataset build /
    # model init) so stance supervision inputs are reproducible from the single
    # ``experiment.seed`` (Req 8.1, 8.2, 8.3, 11.4). ``seed_stance_pipeline``
    # derives every stance-path RNG seed deterministically from that seed and
    # fails fast (ValueError) on a missing/null/non-integer/negative seed or any
    # RNG that fails to seed (Req 8.4, 8.5) -- before the first stochastic step.
    seed_stance_pipeline(cfg)
    tokenizer = build_tokenizer(cfg)
    loaders = build_dataloaders(task_path, tokenizer, cfg)
    task_scaler = loaders.get("target_scaler")
    effective_scaler = task_scaler if task_scaler is not None else scaler
    scaler_path = _persist_target_scaler(output_dir, target_name, effective_scaler)

    model = build_model(cfg)
    ckpt_dir = os.path.join(output_dir, "models", "checkpoints", target_name)
    result = run_training_loop(
        model, loaders, cfg,
        os.path.join(output_dir, "audit", "training_nan_debug.jsonl"),
        ckpt_dir=ckpt_dir,
        resume=bool(_cfg_get(cfg, "model.deberta.resume", False)),
        save_every_steps=int(_cfg_get(cfg, "model.deberta.save_every_steps", 0)),
    )

    out: dict[str, Any] = {
        "target": target_name,
        "train_result": asdict(result),
        "checkpoint_dir": ckpt_dir,
        "target_scaler": effective_scaler,
        "target_scaler_path": scaler_path,
    }

    if result.status != "OK" or loaders.get("test") is None:
        out["status"] = result.status
        out["n_test"] = int(loaders.get("n_test", 0) or 0)
        return out

    y_true, y_pred, prediction_details = predict(
        model,
        loaders["test"],
        cfg,
        scaler=effective_scaler,
        return_details=True,
    )
    if len(y_true) == 0:
        out["status"] = "INSUFFICIENT_SAMPLE"
        out["n_test"] = 0
        out["reason"] = "no observed primary targets in test split"
        return out

    preds_dir = os.path.join(output_dir, "evaluation")
    os.makedirs(preds_dir, exist_ok=True)
    pred_path = os.path.join(preds_dir, f"predictions_{target_name}.parquet")
    # Detailed source-row indices align against the unfiltered canonical test
    # frame (including auxiliary-only rows); _build_predictions_frame applies the
    # exact observed/finite prediction mask before attaching event identities.
    test_frame = loaders.get("frames", {}).get("test")
    prediction_frame = _build_predictions_frame(
        y_true, y_pred, test_frame, details=prediction_details
    )
    prediction_frame.to_parquet(pred_path, index=False)

    stance_predictions_path = None
    stance_summary_path = None
    stance_summary = None
    if bool(prediction_details.get("stance_available", False)):
        missing_stance_columns = [
            column
            for column in _STANCE_EVENT_COLUMNS
            if column not in prediction_frame.columns
        ]
        if missing_stance_columns:
            raise ValueError(
                "cannot persist aligned stance artifact; missing columns: "
                f"{missing_stance_columns}"
            )
        stance_predictions_path = os.path.join(
            preds_dir, "stance_event_predictions.parquet"
        )
        rich_columns = list(_STANCE_EVENT_COLUMNS) + ["y_true", "y_pred"]
        prediction_frame[rich_columns].to_parquet(
            stance_predictions_path, index=False
        )
        stance_summary = summarize_stance_predictions(prediction_frame)
        stance_summary_path = os.path.join(preds_dir, "stance_summary.json")
        with open(stance_summary_path, "w", encoding="utf-8") as fh:
            json.dump(stance_summary, fh, indent=2, default=str)

    prediction_diagnostics = _constant_prediction_diagnostics(y_true, y_pred)
    diagnostics_path = os.path.join(
        preds_dir, f"prediction_diagnostics_{target_name}.json"
    )
    with open(diagnostics_path, "w", encoding="utf-8") as fh:
        json.dump(prediction_diagnostics, fh, indent=2, default=str)

    best_checkpoint = os.path.join(ckpt_dir, "best.pt")
    out.update({
        "status": "OK",
        "n_test": int(len(y_true)),
        "predictions_path": pred_path,
        "stance_predictions_path": stance_predictions_path,
        "stance_summary_path": stance_summary_path,
        "stance_summary": stance_summary,
        "prediction_diagnostics": prediction_diagnostics,
        "diagnostics_path": diagnostics_path,
        "degraded": bool(prediction_diagnostics["constant_prediction"]),
        "degradation_reasons": (
            ["constant_prediction"]
            if prediction_diagnostics["constant_prediction"] else []
        ),
        "best_checkpoint": best_checkpoint if os.path.exists(best_checkpoint) else None,
        "y_true": y_true,
        "y_pred": y_pred,
    })
    return out


# =============================================================================
# High-level helpers used by the orchestration notebook (keep cells thin).
# =============================================================================

def smoke_config(base: Any) -> dict:
    """Return a fast, self-contained smoke variant of ``base`` config.

    Forces the tiny stub encoder (no download), a tiny hidden size, and a couple
    of short epochs so the whole pipeline runs in seconds on CPU.
    """
    import copy as _copy
    cfg = _copy.deepcopy(base)
    d = cfg.setdefault("model", {}).setdefault("deberta", {})
    d.update({"force_stub": True, "allow_stub_encoder": True, "stub_vocab_size": 512,
              "hidden_size": 32, "max_chunks": 2, "chunk_size": 24, "chunk_overlap": 4,
              "batch_size": 4, "max_epochs": 2, "early_stopping_patience": 2,
              "gradient_accumulation_steps": 1, "use_amp": False, "min_observations": 1})
    return cfg


def _synthetic_task_parquet(path: str) -> str:
    """Write a tiny synthetic task parquet (used when no target has train rows)."""
    rng = np.random.default_rng(0)
    def rows(n, split):
        return pd.DataFrame({"event_id": [f"{split}{i}" for i in range(n)],
            "event_timestamp": pd.Timestamp("2020-01-01"),
            "text": [" ".join([f"w{int(rng.integers(0,40))}" for _ in range(int(rng.integers(6,30)))]) for _ in range(n)],
            "target": rng.normal(0, 2, n), "target_scaled": rng.normal(0, 1, n),
            "split": split, "is_primary_test": split == "test",
            "event_type": "IMC_speech", "speaker_normalized": "x"})
    pd.concat([rows(16, "train"), rows(6, "val"), rows(6, "test")], ignore_index=True).to_parquet(path, index=False)
    return path


def run_smoke_test(task_paths: dict, cfg: Any, output_dir: str) -> dict:
    """End-to-end tiny-slice smoke test used by the notebook's SMOKE TEST cell.

    Asserts: batch/target shapes align, loss is finite, gradients flow (nonzero
    norm), the short training loop returns OK with a finite metric, and a saved
    checkpoint reloads and reproduces predictions within 1e-4. Raises
    ``AssertionError`` on any failure; returns a small results dict on success.
    """
    if not _HAS_TORCH:
        raise RuntimeError("run_smoke_test requires PyTorch.")
    chosen = None
    for name, path in sorted((task_paths or {}).items()):
        fr = pd.read_parquet(path)
        if len(fr[fr["split"] == "train"]) >= 4:
            chosen = (name, path)
            break
    if chosen is None:
        tmp = os.path.join(output_dir, "intermediate", "_smoke_task.parquet")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        chosen = ("model_smoke", _synthetic_task_parquet(tmp))

    name, path = chosen
    cfg = smoke_config(cfg)
    tokenizer = build_tokenizer(cfg)
    loaders = build_dataloaders(path, tokenizer, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbg = os.path.join(output_dir, "audit", "training_nan_debug.jsonl")

    # (a) forward/backward: shapes align, loss finite, gradients flow.
    model = build_model(cfg).to(device)
    batch = _move_batch(next(iter(loaders["train"])), device)
    outputs = model(batch)
    assert outputs["regression"].shape == batch["target"].shape, (
        outputs["regression"].shape, batch["target"].shape)
    loss = compute_losses(outputs, batch, cfg)["total"]
    assert bool(torch.isfinite(loss).all()), "loss is not finite"
    loss.backward()
    gnorm = float(sum((p.grad.detach().norm() ** 2) for p in model.parameters()
                      if p.grad is not None) ** 0.5)
    assert gnorm > 0, "gradient norm is zero (no gradient flow)"

    # (b) short training loop -> OK with finite metric.
    res = run_training_loop(build_model(cfg).to(device), loaders, cfg, dbg)
    assert res.status == "OK", res.status
    assert np.isfinite(res.best_metric), "non-finite best_metric"

    # (c) checkpoint reload reproducibility (< 1e-4).
    m = build_model(cfg).to(device)
    run_training_loop(m, loaders, cfg, dbg)
    ckpt = os.path.join(output_dir, "models", "_smoke_ckpt.pt")
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(m.state_dict(), ckpt)
    _, y1 = predict(m, loaders["test"], cfg)
    m2 = build_model(cfg).to(device)
    m2.load_state_dict(torch.load(ckpt, map_location=device))
    _, y2 = predict(m2, loaders["test"], cfg)
    max_diff = float(np.max(np.abs(y1 - y2))) if len(y1) else 0.0
    assert max_diff < 1e-4, f"checkpoint reload mismatch: max diff {max_diff}"

    return {"target": name, "loss": float(loss.detach()), "grad_norm": gnorm,
            "status": res.status, "best_metric": float(res.best_metric),
            "reload_max_diff": max_diff}


def encode_latents(model, loader, device=None):
    """Return primary-observed latent rows and raw labels from a loader.

    The canonical Council parquet is a union over auxiliary signals, so rows with
    ``primary_present=False`` are intentionally retained for Council fine-tuning
    but must not enter a frozen primary-target estimator or its diagnostics.
    """
    if not _HAS_TORCH:
        raise RuntimeError("encode_latents requires PyTorch.")
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    docs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            b = _move_batch(batch, device)
            out = model(b)
            doc = out["document"].float()
            raw = b.get("target_raw", b["target"]).float().reshape(-1)
            present = b.get("primary_present")
            if present is None:
                keep = torch.ones_like(raw, dtype=torch.bool)
            else:
                keep = present.reshape(-1).bool()
            keep = keep & torch.isfinite(raw) & torch.isfinite(doc).all(dim=1)
            if not bool(keep.any()):
                continue
            docs.append(doc[keep].cpu().numpy())
            ys.append(raw[keep].cpu().numpy())
    if not docs:
        hidden = int(getattr(model, "hidden_size", 1) or 1)
        return np.zeros((0, hidden), dtype="float32"), np.zeros((0,), dtype="float32")
    return np.concatenate(docs), np.concatenate(ys)


def load_target_scaler(models_dir: str) -> Optional[dict]:
    """Load the train-fit target scaler dict saved under ``models_dir``.

    Reads ``<models_dir>/target_scaler.pkl`` (written by
    :func:`src.dataset_builder.save_target_scaler`) and returns the
    ``{"mean","std",...}`` mapping, or ``None`` when absent so callers can
    proceed in scaled space.
    """
    from .data_io import load_scaler
    path = os.path.join(models_dir, "target_scaler.pkl")
    if not os.path.exists(path):
        return None
    try:
        scaler = load_scaler(path)
    except Exception:
        return None
    if isinstance(scaler, dict):
        return scaler
    # Tolerate a fitted sklearn StandardScaler.
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is not None and scale is not None:
        return {"mean": float(np.ravel(mean)[0]), "std": float(np.ravel(scale)[0])}
    return None


# =============================================================================
# Frozen vs fine-tune comparison rows (Req 6.4) -- Task 11.2
#
# ``evaluate_frozen_and_finetune`` runs BOTH modes on the SAME eval split for one
# target and scores each as a separate ``compare_models`` row:
#
#   * frozen headline  -> ``model="council_frozen"``
#   * fine-tune ablation -> ``model="council_finetune"``
#
# It is config-driven and reuses the existing mode paths without duplicating any
# training logic: the frozen row comes from :func:`train_target_frozen`; the
# fine-tune row comes from :func:`train_target` with the ``frozen_encoder``
# switch forced false (Req 6.2, 6.3). Each mode's ``(y_true, y_pred)`` on the
# same test split is scored via ``evaluation.evaluate_model`` with the matching
# model name, and both records are fed into ``evaluation.compare_models`` so the
# comparison table (``model_comparison.{csv,tex}``) carries both rows (Req 6.4).
# =============================================================================

#: Canonical comparison-row names for the two Requirement-6 modes.
COUNCIL_FROZEN_MODEL = "council_frozen"
COUNCIL_FINETUNE_MODEL = "council_finetune"


def _cfg_with_frozen(cfg: Any, frozen: bool) -> Any:
    """Return a deep copy of ``cfg`` with ``model.deberta.frozen_encoder`` forced.

    Orchestrating both modes on the same eval split requires flipping only the
    single Requirement-6.3 switch without mutating the caller's config. A deep
    copy keeps the caller's ``cfg`` intact and leaves every other setting -- the
    eval split, scaler, seed, signal set -- identical across the two runs so the
    two rows are strictly comparable.
    """
    import copy as _copy

    out = _copy.deepcopy(cfg)
    if isinstance(out, dict):
        out.setdefault("model", {}).setdefault("deberta", {})["frozen_encoder"] = bool(frozen)
        return out
    # Object-style config: set through the nested attribute chain when present.
    model = getattr(out, "model", None)
    deberta = getattr(model, "deberta", None) if model is not None else None
    if deberta is not None:
        try:
            setattr(deberta, "frozen_encoder", bool(frozen))
        except Exception:  # pragma: no cover - immutable config object
            pass
    return out


def evaluate_frozen_and_finetune(
    task_path: str,
    cfg: Any,
    output_dir: str,
    target_name: str,
    scaler: Optional[dict] = None,
    results_dir: Optional[str] = None,
) -> dict:
    """Run BOTH Requirement-6 modes on the SAME eval split and score each row.

    Trains the frozen-encoder headline and the fine-tuning ablation for one
    ``target_name`` on the same task parquet (hence the same test split), scores
    each mode's ``(y_true, y_pred)`` via :func:`src.evaluation.evaluate_model`
    with the matching model name (``"council_frozen"`` / ``"council_finetune"``),
    and feeds both records into :func:`src.evaluation.compare_models` so the
    comparison table gains one row per mode (Req 6.4). Config-driven: the mode is
    selected by flipping only the ``model.deberta.frozen_encoder`` switch on a
    copy of ``cfg`` (Req 6.3), never mutating the caller's config.

    Frozen artifacts are written under ``output_dir/frozen`` and fine-tune
    artifacts under ``output_dir/finetune`` so their per-target prediction
    parquets and checkpoints do not collide. When ``results_dir`` is supplied,
    ``compare_models`` writes ``model_comparison.{csv,tex}`` there.

    Returns ``{"frozen", "finetune", "records", "comparison"}`` where ``frozen``
    / ``finetune`` are the raw :func:`train_target_frozen` / :func:`train_target`
    result dicts, ``records`` maps each present model name to its
    :class:`~src.evaluation.MetricRecord`, and ``comparison`` is the
    ``compare_models`` DataFrame. A mode that produces no usable predictions
    (e.g. status ``INSUFFICIENT_SAMPLE``) is skipped in the comparison rather
    than fabricating an empty metric row.
    """
    if not _HAS_TORCH:
        raise RuntimeError("evaluate_frozen_and_finetune requires PyTorch.")

    from .evaluation import compare_models, evaluate_model

    run_frozen = bool(_cfg_get(cfg, "runtime.run_frozen", True))
    run_finetune = bool(_cfg_get(cfg, "runtime.run_finetune", True))

    if run_frozen:
        _logger.info(
            "[frozen] training target=%s (deterministic mean pool + ridge head)",
            target_name,
        )
        frozen_cfg = _cfg_with_frozen(cfg, True)
        frozen_out = train_target_frozen(
            task_path,
            frozen_cfg,
            os.path.join(output_dir, "frozen"),
            target_name,
            scaler=scaler,
        )
        _logger.info(
            "[frozen] done target=%s status=%s",
            target_name,
            frozen_out.get("status"),
        )
    else:
        frozen_out = {
            "target": target_name,
            "mode": "frozen",
            "status": "NOT_RUN",
            "reason": "runtime.run_frozen=false",
        }

    if run_finetune:
        _logger.info(
            "[finetune] training target=%s (end-to-end DeBERTa fine-tune)",
            target_name,
        )
        finetune_cfg = _cfg_with_frozen(cfg, False)
        finetune_out = train_target(
            task_path,
            finetune_cfg,
            os.path.join(output_dir, "finetune"),
            target_name,
            scaler=scaler,
        )
        _tr = finetune_out.get("train_result") or {}
        _logger.info(
            "[finetune] done target=%s status=%s optim_steps=%s n_samples=%s "
            "best_epoch=%s",
            target_name,
            finetune_out.get("status"),
            _tr.get("optim_steps"),
            _tr.get("n_samples"),
            _tr.get("best_epoch"),
        )
    else:
        finetune_out = {
            "target": target_name,
            "mode": "finetune",
            "status": "NOT_RUN",
            "reason": "runtime.run_finetune=false",
        }

    records: dict[str, Any] = {}
    for model_name, out in (
        (COUNCIL_FROZEN_MODEL, frozen_out),
        (COUNCIL_FINETUNE_MODEL, finetune_out),
    ):
        y_true = out.get("y_true")
        y_pred = out.get("y_pred")
        # Only score a mode that produced usable predictions on the eval split;
        # never fabricate an empty metric row for a skipped/insufficient mode.
        if (
            out.get("status") == "OK"
            and y_true is not None
            and y_pred is not None
            and len(y_true) > 0
        ):
            records[model_name] = evaluate_model(
                y_true,
                y_pred,
                cfg=cfg,
                model=model_name,
                target_name=target_name,
            )

    comparison = compare_models(records, results_dir=results_dir)

    return {
        "frozen": frozen_out,
        "finetune": finetune_out,
        "records": records,
        "comparison": comparison,
    }
