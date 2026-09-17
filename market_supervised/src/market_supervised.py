"""Market-supervised DeBERTa model: hierarchical encoder + five objectives.

Hierarchical document architecture (per-segment encoding + attention pooling) is
the DEFAULT (``encoder_mode='hierarchical'``); ``'flat'`` is a distinct ablation-only
variant. NaN/Inf protection, two-phase precision, gradient clipping, and
contrastive-group thresholds are wired here. Interface + numerical wiring only;
training is executed by the owner (Requirement 33). Implemented in Task 20.

Only the pure, shape-level helpers are runnable/agent-testable without training:
``assert_finite`` (Requirement 17.1), ``encode_segments`` and ``attention_pool``
(Property 13 shape contract), and ``make_contrastive_groups`` (threshold logic).
Model construction (``build_model``/``build_deberta_baseline``) and ``train`` wire
the numerical-stability policy but are NOT executed by the agent.

Satisfies: 17.1-17.5, 22.1, 23.1-23.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

# ---------------------------------------------------------------------------
# Optional torch import. torch is PREFERRED. The pure helpers (assert_finite,
# encode_segments, attention_pool) also accept numpy arrays / plain sequences so
# they remain testable in a torch-free environment.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly by whichever backend is present
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _HAS_TORCH = False

try:  # numpy is a hard project dependency, but guard anyway.
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# =============================================================================
# Config access helper (config is a nested dict in this codebase; tolerate
# attribute-style access too so the design's ``cfg.model.encoder_mode`` reads
# and the implemented dict-based config both work).
# =============================================================================

def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``dotted`` (e.g. ``"model.deberta.encoder_mode"``) from a dict or object.

    Supports nested dict access, attribute access, and mixtures of the two.
    Returns ``default`` when any segment along the path is missing.
    """
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        else:
            if not hasattr(node, part):
                return default
            node = getattr(node, part)
    return node if node is not None else default


# =============================================================================
# Numerical-stability helper (Requirement 17.1) - PURE, runnable, torch-first.
# =============================================================================

def assert_finite(name: str, tensor: Any) -> None:
    """Raise if ``tensor`` contains any non-finite value (NaN or +/-Inf).

    This is the finite-value guard applied during training to inputs, attention
    masks, targets, model outputs, loss, and gradients (Requirement 17.1). It is
    a pure, side-effect-free helper: it inspects the value and either returns
    ``None`` (all finite) or raises ``ValueError``.

    torch is preferred: if ``tensor`` is a ``torch.Tensor`` it is checked with
    ``torch.isfinite``. numpy arrays and plain Python scalars / nested sequences
    are also accepted so the guard is testable without torch installed.

    Args:
        name: Human-readable label used in the raised message (e.g. ``"loss"``,
            ``"gradients"``) so an offending tensor is identifiable in logs.
        tensor: The value to check. May be a torch tensor, numpy array, Python
            scalar, or an array-like (nested list/tuple) of numbers.

    Raises:
        ValueError: If any element is NaN, +Inf, or -Inf.
        TypeError: If a non-numeric element is encountered.
    """
    # --- torch tensors (preferred path) ------------------------------------
    if _HAS_TORCH and torch.is_tensor(tensor):
        if not bool(torch.isfinite(tensor).all()):
            n_nan = int(torch.isnan(tensor).sum().item())
            n_posinf = int(torch.isposinf(tensor).sum().item())
            n_neginf = int(torch.isneginf(tensor).sum().item())
            raise ValueError(
                f"Non-finite values detected in '{name}': "
                f"{n_nan} NaN, {n_posinf} +Inf, {n_neginf} -Inf "
                f"(shape={tuple(tensor.shape)})"
            )
        return

    # --- numpy arrays / numpy scalars --------------------------------------
    if _HAS_NUMPY and isinstance(tensor, (np.ndarray, np.generic)):
        arr = np.asarray(tensor)
        if arr.size == 0:
            return
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"assert_finite('{name}', ...) received non-numeric dtype {arr.dtype}"
            )
        if not np.isfinite(arr).all():
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            raise ValueError(
                f"Non-finite values detected in '{name}': "
                f"{n_nan} NaN, {n_inf} Inf (shape={arr.shape})"
            )
        return

    # --- plain python scalars ----------------------------------------------
    if isinstance(tensor, bool):
        return  # booleans are always finite
    if isinstance(tensor, (int, float)):
        if not math.isfinite(float(tensor)):
            raise ValueError(f"Non-finite value detected in '{name}': {tensor}")
        return

    # --- array-likes (nested sequences) ------------------------------------
    if isinstance(tensor, (list, tuple)):
        if _HAS_NUMPY:
            try:
                arr = np.asarray(tensor, dtype=float)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"assert_finite('{name}', ...) could not interpret sequence "
                    f"as numeric: {exc}"
                ) from exc
            assert_finite(name, arr)
            return
        # numpy unavailable: recurse element-wise.
        for element in tensor:
            assert_finite(name, element)
        return

    raise TypeError(
        f"assert_finite('{name}', ...) received unsupported type "
        f"{type(tensor).__name__}; expected torch.Tensor, numpy array, number, "
        f"or nested sequence."
    )


# =============================================================================
# Hierarchical encoder - pure shape logic (Property 13, Requirements 23.3/23.4).
# encode_segments / attention_pool work with a stubbed/mock callable encoder and
# plain tensors/arrays, so no DeBERTa weights or training are involved.
# =============================================================================

def _segment_text(segment: Any) -> str:
    """Extract the encodable text from a Segment / mapping / raw string."""
    if isinstance(segment, str):
        return segment
    if isinstance(segment, dict):
        return segment.get("segment_text", "")
    return getattr(segment, "segment_text", "")


def _stack_embeddings(rows: list[Any]) -> Any:
    """Stack a list of per-segment embedding vectors into a matrix.

    Uses torch.stack for tensors, otherwise numpy.stack, otherwise returns the
    list of lists unchanged. All rows must share the same hidden size.
    """
    if _HAS_TORCH and rows and torch.is_tensor(rows[0]):
        return torch.stack(rows, dim=0)
    if _HAS_NUMPY:
        return np.stack([np.asarray(r, dtype=float) for r in rows], axis=0)
    return rows


def encode_segments(segments: Sequence[Any], encoder: Callable[[str], Any]) -> Any:
    """Encode every available segment with a single shared encoder.

    Runs the ONE shared segment encoder over each available segment of a document
    and returns per-segment embeddings shaped ``[num_segments, hidden]``. Weight
    sharing is intrinsic: the same ``encoder`` callable is applied to every
    segment (Requirement 23.3). The number of returned embeddings equals the
    number of available segments - no segment is dropped (Property 13).

    ``encoder`` is any callable mapping a segment's text to a fixed-width
    embedding vector (a real DeBERTa module in production, or a stubbed/mock
    callable in tests). The result is a torch tensor when the encoder returns
    tensors, a numpy array when it returns array-likes.

    Args:
        segments: Ordered available segments (``Segment`` objects, mappings, or
            plain strings). Absent/padded segments must be filtered out before
            this call so that ``len(result) == len(segments)``.
        encoder: Shared callable ``str -> embedding vector``.

    Returns:
        Per-segment embeddings of shape ``[len(segments), hidden]``.

    Raises:
        ValueError: If ``segments`` is empty (hierarchical mode requires at least
            one available segment).
    """
    if segments is None or len(segments) == 0:
        raise ValueError("encode_segments requires at least one available segment")

    rows = [encoder(_segment_text(seg)) for seg in segments]
    return _stack_embeddings(rows)


def _row_count(matrix: Any) -> int:
    if _HAS_TORCH and torch.is_tensor(matrix):
        return int(matrix.shape[0])
    if _HAS_NUMPY and isinstance(matrix, np.ndarray):
        return int(matrix.shape[0])
    return len(matrix)


def _softmax_scores(scores: Any) -> Any:
    """Numerically stable softmax over the first (segment) axis."""
    if _HAS_TORCH and torch.is_tensor(scores):
        return torch.softmax(scores, dim=0)
    arr = np.asarray(scores, dtype=float)
    shifted = arr - np.max(arr)
    exp = np.exp(shifted)
    return exp / exp.sum()


def attention_pool(segment_embeddings: Any, mask: Optional[Any] = None) -> Any:
    """Pool per-segment embeddings into ONE document vector via learned attention.

    Computes a scalar score per segment, masks padded/absent segments, applies a
    softmax so the attention weights over the present segments **sum to 1** and are
    finite, and returns the weighted sum ``[hidden]`` (Requirement 23.3 - learned
    weighted aggregation, not mean/first-token truncation). The scoring here is a
    deterministic pure reference (score = mean of the embedding) so the shape and
    weight-normalisation contract is testable without a trained attention head;
    the production model swaps in a learned scoring layer.

    Args:
        segment_embeddings: ``[num_segments, hidden]`` per-segment embeddings from
            :func:`encode_segments`.
        mask: Optional ``[num_segments]`` boolean/0-1 mask of PRESENT segments.
            When omitted, all segments are treated as present.

    Returns:
        A single ``[hidden]`` document representation.

    Raises:
        ValueError: If no segment is present, or if any pooled value / attention
            weight is non-finite.
    """
    n = _row_count(segment_embeddings)
    if n == 0:
        raise ValueError("attention_pool requires at least one segment embedding")

    if _HAS_TORCH and torch.is_tensor(segment_embeddings):
        emb = segment_embeddings.to(torch.float32)
        scores = emb.mean(dim=1)  # [num_segments] pure reference scorer
        if mask is not None:
            mask_t = mask if torch.is_tensor(mask) else torch.as_tensor(mask)
            mask_bool = mask_t.to(torch.bool)
            if not bool(mask_bool.any()):
                raise ValueError("attention_pool requires at least one present segment")
            scores = scores.masked_fill(~mask_bool, float("-inf"))
        weights = torch.softmax(scores, dim=0)
        assert_finite("attention_weights", weights)
        doc = (weights.unsqueeze(1) * emb).sum(dim=0)
        assert_finite("document_representation", doc)
        return doc

    # numpy / array-like path
    emb = np.asarray(segment_embeddings, dtype=float)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    scores = emb.mean(axis=1)
    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool)
        if not mask_arr.any():
            raise ValueError("attention_pool requires at least one present segment")
        scores = np.where(mask_arr, scores, -np.inf)
    weights = _softmax_scores(scores)
    assert_finite("attention_weights", weights)
    doc = (weights.reshape(-1, 1) * emb).sum(axis=0)
    assert_finite("document_representation", doc)
    return doc


# =============================================================================
# Contrastive-group construction (Requirement 23.5, thresholds from config).
# =============================================================================

@dataclass
class ContrastiveGroups:
    """Positive/negative/neutral index groupings for the contrastive objective.

    ``positive``/``negative`` hold indices of events whose target magnitude is a
    material move in each direction; ``neutral`` holds near-zero moves excluded
    from pairing. ``threshold_bp`` records the applied config threshold and
    ``valid`` reports whether the minimum pair count was met.
    """

    positive: list[int] = field(default_factory=list)
    negative: list[int] = field(default_factory=list)
    neutral: list[int] = field(default_factory=list)
    threshold_bp: float = 0.5
    min_pairs: int = 1
    valid: bool = False


def _batch_targets(batch: Any) -> list[float]:
    """Extract a flat list of target values from a batch dict / object / sequence."""
    values: Any = None
    if isinstance(batch, dict):
        for key in ("target", "targets", "y", "labels"):
            if key in batch:
                values = batch[key]
                break
    elif hasattr(batch, "target"):
        values = getattr(batch, "target")
    else:
        values = batch

    if values is None:
        return []
    if _HAS_TORCH and torch.is_tensor(values):
        return [float(v) for v in values.detach().reshape(-1).tolist()]
    if _HAS_NUMPY and isinstance(values, (np.ndarray, np.generic)):
        return [float(v) for v in np.asarray(values).reshape(-1).tolist()]
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


def make_contrastive_groups(batch: Any, cfg: Any) -> ContrastiveGroups:
    """Partition a batch into positive/negative/neutral groups by target magnitude.

    Uses ``market_windows.contrastive_threshold_bp`` (default 0.5) as the
    positive/negative threshold: events with target > +threshold are positive,
    < -threshold are negative, and near-zero moves in ``[-threshold, +threshold]``
    are neutral and excluded from pairing. ``min_pairs`` (config
    ``market_windows.contrastive_min_pairs``, default 1) sets the minimum number of
    positive AND negative members required for the groups to be ``valid``.

    Pure/runnable helper - no model or training involved.
    """
    threshold = float(_cfg_get(cfg, "market_windows.contrastive_threshold_bp", 0.5))
    min_pairs = int(_cfg_get(cfg, "market_windows.contrastive_min_pairs", 1))

    targets = _batch_targets(batch)
    groups = ContrastiveGroups(threshold_bp=threshold, min_pairs=min_pairs)
    for idx, value in enumerate(targets):
        if not math.isfinite(value):
            continue  # non-finite targets never form a contrastive pair
        if value > threshold:
            groups.positive.append(idx)
        elif value < -threshold:
            groups.negative.append(idx)
        else:
            groups.neutral.append(idx)

    groups.valid = (
        len(groups.positive) >= min_pairs and len(groups.negative) >= min_pairs
    )
    return groups


# =============================================================================
# Model construction + loss composition + training wiring.
# These wire the architecture / numerical-stability policy but are NOT executed
# by the agent (Requirement 33). They require torch.
# =============================================================================

_VALID_ENCODER_MODES = ("hierarchical", "flat")

# Canonical Council stance-class order. The three direction logits are trained
# against raw market moves as down/neutral/up, which for the configured rate
# target maps to dovish/neutral/hawkish. Legacy ``StanceHead`` outputs remain in
# their historical (hawkish, dovish, neutral) order; this constant applies only
# to CouncilModel's primary direction/stance outputs.
COUNCIL_STANCE_CLASS_ORDER = ("dovish", "neutral", "hawkish")


def _require_torch(what: str) -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            f"{what} requires PyTorch, which is not installed in this environment."
        )


if _HAS_TORCH:

    def _first_slot_like(mask_bool: "torch.Tensor") -> "torch.Tensor":
        """Boolean tensor same shape as ``mask_bool`` with only the last-dim
        index 0 set True. Used to force one attended slot on an all-absent row."""
        idx = torch.zeros_like(mask_bool)
        # Select index 0 along the segment dimension for every leading position.
        idx.index_fill_(-1, torch.tensor([0], device=mask_bool.device), True)
        return idx

    class _AttentionPool(nn.Module):
        """Learned attention over segments producing one document vector.

        Scores each segment with a linear head, masks absent segments to -inf,
        softmaxes so present-segment weights sum to 1 and are finite, and returns
        the weighted sum. Mirrors the pure :func:`attention_pool` contract.
        """

        def __init__(self, hidden_size: int):
            super().__init__()
            self.score = nn.Linear(hidden_size, 1)

        def forward(self, segment_embeddings, mask=None, return_weights=False):
            scores = self.score(segment_embeddings).squeeze(-1)  # [B, S] or [S]
            if mask is not None:
                mask_bool = mask.to(torch.bool)
                # A row with NO present segment would softmax over an all -inf
                # row and yield NaN. Guarantee at least one attended segment per
                # row by forcing the first slot present where the whole row is
                # absent; that segment's embedding is a zero vector (degenerate
                # chunks were zeroed upstream), so the pooled doc is just 0 for
                # such a row -- finite, and carrying no spurious signal.
                all_absent = ~mask_bool.any(dim=-1, keepdim=True)  # [..., 1]
                mask_bool = mask_bool | (all_absent & _first_slot_like(mask_bool))
                scores = scores.masked_fill(~mask_bool, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            assert_finite("attention_weights", weights)
            doc = (weights.unsqueeze(-1) * segment_embeddings).sum(dim=-2)
            assert_finite("document_representation", doc)
            # ``return_weights`` is an opt-in interpretability hook (Req 7.4): the
            # normalized per-segment attention weights sum to 1 over present
            # segments and are exactly the weights ``Stance_Space.attributions``
            # attributes each event's stance to. The default (``False``) preserves
            # the original single-tensor contract every existing caller relies on.
            if return_weights:
                return doc, weights
            return doc

    class MarketSupervisedModel(nn.Module):
        """Market-supervised DeBERTa model (hierarchical-by-default).

        Encoder mode is read from ``cfg.model.deberta.encoder_mode``:
        ``"hierarchical"`` (default) encodes every segment with a shared encoder
        and attention-pools to one document vector; ``"flat"`` is the distinct
        truncated-single-segment ablation variant (never the default). The
        encoder body itself is created by the owner at train time; this class
        wires heads and the pooling policy.
        """

        def __init__(self, cfg: Any, encoder: Optional["nn.Module"] = None):
            super().__init__()
            mode = _cfg_get(cfg, "model.deberta.encoder_mode", "hierarchical")
            if mode not in _VALID_ENCODER_MODES:
                raise ValueError(
                    f"Unknown encoder_mode '{mode}'; expected one of {_VALID_ENCODER_MODES}"
                )
            self.encoder_mode = mode
            self._cfg = cfg  # retained for forward-time config lookups (min_chunk_tokens)
            self.encoder = encoder  # DeBERTa body; built by build_model when None
            # Prefer the encoder's true hidden size when an encoder is supplied.
            hidden = None
            if encoder is not None and hasattr(encoder, "config"):
                hidden = getattr(encoder.config, "hidden_size", None)
            if hidden is None:
                hidden = int(_cfg_get(cfg, "model.deberta.hidden_size", 768))
            self.hidden_size = int(hidden)
            self.pool = _AttentionPool(self.hidden_size) if mode == "hierarchical" else None
            # Five-objective heads.
            self.regression_head = nn.Linear(self.hidden_size, 1)
            self.direction_head = nn.Linear(self.hidden_size, 1)
            self.uncertainty_head = nn.Linear(self.hidden_size, 1)  # log-variance
            # Diagnostic: number of non-finite entries the encoder emitted for
            # PRESENT segments on the most recent forward pass. Scrubbed before
            # pooling (so the forward stays finite); the training loop reads this
            # to log/skip a genuinely degenerate batch instead of crashing.
            self.last_nonfinite_segments = 0

            # Feature: continuous-stance-scoring -- optional stance head (Req 6.1).
            # When ``model.deberta.stance.enabled`` is true, attach a StanceHead
            # over the SAME pooled document vector the five existing heads read;
            # the encoder and _AttentionPool parameters are left in place. Import
            # StanceHead lazily (not at module top) to avoid any import-order
            # coupling with src/stance.py, which imports ``assert_finite`` from
            # this module inside its own torch guard.
            self.stance_head = None
            # Diagnostic (mirrors ``last_nonfinite_segments``): number of
            # non-finite entries the stance path emitted for PRESENT (supervised)
            # examples on the most recent forward pass. Scrubbed before the
            # outputs leave forward (so the forward stays finite); the training
            # loop reads this to append the offending batch to the training debug
            # log instead of crashing (Req 9.5).
            self.last_nonfinite_stance = 0
            if bool(_cfg_get(cfg, "model.deberta.stance.enabled", False)):
                from .stance import StanceHead

                self.stance_head = StanceHead(self.hidden_size, cfg)

        def is_hierarchical(self) -> bool:
            return self.encoder_mode == "hierarchical"

        def _encode_document(self, input_ids, attention_mask, segment_mask=None,
                             return_attention=False):
            """Encode a padded batch of segments into one document vector per row.

            Inputs are shaped ``[B, S, L]`` (batch, segments, tokens) in
            hierarchical mode or ``[B, L]`` in flat mode. The shared encoder is
            applied to every segment (weight sharing is intrinsic - one encoder),
            each segment is reduced to its ``[CLS]``/first-token vector, and the
            per-segment vectors are attention-pooled (hierarchical) or passed
            through directly (flat). Returns ``[B, hidden]``.

            When ``return_attention`` is true, returns ``(doc, weights)`` where
            ``weights`` is ``[B, S]`` (hierarchical) or ``[B, 1]`` (flat) of the
            normalized per-segment attention weights the pool applied (summing to
            1 over present segments). This opt-in hook feeds the interpretability
            attributions (Req 7.4); the default (``False``) keeps the original
            ``[B, hidden]``-only contract every training/forward caller relies on.
            """
            if self.encoder is None:
                raise RuntimeError(
                    "MarketSupervisedModel.encoder is None; build the model via "
                    "build_model(cfg) so the DeBERTa body is constructed."
                )

            # Encoder hidden states may come back in a different precision
            # (some pretrained bodies default to fp16). Cast to the head dtype so
            # the downstream Linear layers never hit a dtype mismatch.
            head_dtype = self.regression_head.weight.dtype

            if self.encoder_mode == "flat":
                out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                doc = out.last_hidden_state[:, 0, :].to(head_dtype)  # [B, hidden]
                assert_finite("document_representation", doc)
                if return_attention:
                    # Flat mode encodes exactly ONE (truncated) segment, so the
                    # trivial attention over that lone segment is a weight of 1.0
                    # for every row -- there is no multi-segment weighting to
                    # attribute (Req 7.4 attributions are only meaningful in the
                    # hierarchical default).
                    weights = torch.ones(
                        doc.shape[0], 1, dtype=doc.dtype, device=doc.device
                    )
                    return doc, weights
                return doc

            # hierarchical: input_ids [B, S, L]
            b, s, length = input_ids.shape
            flat_ids = input_ids.reshape(b * s, length)
            flat_mask = attention_mask.reshape(b * s, length)

            # A fully-padded segment (attention_mask all-zero) makes the encoder's
            # self-attention softmax over an all -inf row, which yields NaN - and a
            # NaN here poisons the backward pass even after we later zero it
            # (NaN * 0 == NaN in the gradient). So BEFORE encoding, force every
            # row's first token to be attended (mask[:, 0] = 1). These rows are
            # absent segments; their encoder output is discarded below via the
            # segment mask, so unmasking a dummy token changes nothing downstream
            # while keeping the whole tensor finite.
            # A degenerate chunk is the documented NaN trigger: DeBERTa-v3's
            # disentangled attention drives the position-bias path into a NaN
            # for a PRESENT chunk that has very few real tokens (a short chunk
            # padded to chunk_size). Scrubbing the *value* afterwards with
            # nan_to_num does NOT clean the *gradient*: the derivative w.r.t. the
            # encoder weights is computed through the original NaN-producing op,
            # so a non-finite gradient still poisons the backward pass (this is
            # exactly the 941 non_finite_gradient events observed at epoch 0 in
            # FP32). The only robust fix is to keep the NaN out of the graph:
            # detect chunks whose real-token count is below ``min_chunk_tokens``
            # (all-pad rows included) and route them AROUND the encoder. Their
            # encoder output is replaced with a constant zero vector (detached
            # from the encoder) and they are marked absent via the segment mask,
            # so they contribute nothing to the pool or the gradient.
            real_tokens = flat_mask.sum(dim=1)  # [B*S]
            min_chunk_tokens = int(_cfg_get(
                getattr(self, "_cfg", None) or {}, "model.deberta.min_chunk_tokens", 2))
            degenerate = real_tokens < max(1, min_chunk_tokens)  # [B*S] bool

            # TRUE bypass: call the encoder only for non-degenerate rows, then
            # scatter their vectors back into a zero tensor. Post-hoc
            # ``torch.where`` is insufficient because the invalid rows still
            # execute inside the encoder and can contribute ``NaN * 0`` during
            # backward. Here they are absent from the graph entirely.
            valid = ~degenerate
            valid_idx = valid.nonzero(as_tuple=False).reshape(-1)
            if valid_idx.numel() > 0:
                valid_ids = flat_ids.index_select(0, valid_idx)
                valid_mask = flat_mask.index_select(0, valid_idx)
                out = self.encoder(input_ids=valid_ids, attention_mask=valid_mask)
                valid_vecs = out.last_hidden_state[:, 0, :].to(head_dtype)
                seg_vecs = valid_vecs.new_zeros((b * s, self.hidden_size))
                seg_vecs = seg_vecs.index_copy(0, valid_idx, valid_vecs)
            else:
                seg_vecs = torch.zeros(
                    (b * s, self.hidden_size),
                    dtype=head_dtype,
                    device=input_ids.device,
                )

            seg_vecs = seg_vecs.reshape(b, s, self.hidden_size)  # [B, S, hidden]
            # Fold degenerate chunks into the segment mask so the pool ignores
            # them (an absent segment as far as attention pooling is concerned).
            degenerate_bs = degenerate.reshape(b, s)  # [B, S]
            if segment_mask is not None:
                segment_mask = segment_mask.clone()
                segment_mask[degenerate_bs] = 0
            else:
                segment_mask = (~degenerate_bs).to(input_ids.dtype)

            # Absent (padded) segments carry no signal and must not reach the pool
            # or the gradient. Even with the all-pad guard above, a real encoder
            # (e.g. DeBERTa-v3 disentangled attention) can emit non-finite values
            # not only for degenerate PADDED segments but also for degenerate
            # PRESENT ones (a very short chunk padded to chunk_size drives the
            # disentangled position-bias path into a NaN even in fp32). A single
            # non-finite entry anywhere then poisons the pool and the backward
            # pass (NaN * 0 == NaN in the gradient).
            #
            # Policy: (1) record whether the RAW encoder output for PRESENT
            # segments was non-finite so the caller can decide to skip/recover the
            # batch (training loop) rather than silently learning from garbage;
            # (2) unconditionally scrub every non-finite entry with nan_to_num so
            # the tensor that flows forward is finite; (3) hard-zero absent
            # segments with a mask multiply (clean 0 gradient). This keeps the
            # forward pass finite for both training and validation while still
            # surfacing genuine problems via ``self.last_nonfinite_segments``.
            self.last_nonfinite_segments = 0
            if segment_mask is not None:
                keep = segment_mask.to(seg_vecs.dtype).unsqueeze(-1)  # [B, S, 1], 1.0/0.0
                present_mask = keep.bool().expand_as(seg_vecs)
                # Count non-finite entries in PRESENT segments (diagnostic only).
                present_nonfinite = (~torch.isfinite(seg_vecs)) & present_mask
                self.last_nonfinite_segments = int(present_nonfinite.sum().item())
                # Scrub ALL non-finite entries (present and absent) so nothing
                # non-finite reaches the pool or the gradient.
                seg_vecs = torch.nan_to_num(seg_vecs, nan=0.0, posinf=0.0, neginf=0.0)
                # Hard-zero absent segments with a multiply (clean 0 gradient).
                seg_vecs = seg_vecs * keep
            else:
                present_nonfinite = ~torch.isfinite(seg_vecs)
                self.last_nonfinite_segments = int(present_nonfinite.sum().item())
                seg_vecs = torch.nan_to_num(seg_vecs, nan=0.0, posinf=0.0, neginf=0.0)
            assert_finite("segment_embeddings", seg_vecs)
            if return_attention:
                # Ask the pool for the normalized per-segment attention weights
                # ([B, S], summing to 1 over present segments) so the caller can
                # attribute the document representation -- and thus the stance
                # score -- to individual segments (Req 7.4).
                doc, weights = self.pool(seg_vecs, mask=segment_mask,
                                         return_weights=True)  # [B, hidden], [B, S]
                return doc, weights
            doc = self.pool(seg_vecs, mask=segment_mask)  # [B, hidden]
            return doc

        def forward(self, batch):
            """Run the encoder + five heads; return the objective-input dict.

            ``batch`` is a mapping with ``input_ids``/``attention_mask`` (and, in
            hierarchical mode, ``segment_mask``) plus a ``target`` tensor. Returns
            a dict consumable by :func:`compute_losses`: ``regression`` (point
            prediction, scaled space), ``direction`` (sign logit),
            ``log_variance`` (predicted log-variance for the uncertainty NLL).
            The temporal/contrastive terms are supplied by the training loop.
            """
            doc = self._encode_document(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("segment_mask"),
            )
            reg = self.regression_head(doc).squeeze(-1)
            direction = self.direction_head(doc).squeeze(-1)
            log_var = self.uncertainty_head(doc).squeeze(-1)
            # Clamp log-variance so exp(-log_var) in the NLL stays finite AND the
            # heteroscedastic term cannot buy loss by inflating predicted variance
            # to the point that the regression-fit gradient vanishes (the observed
            # mean-collapse). The ceiling caps how far exp(-log_var) can shrink the
            # per-example fit weight: at the config default max=2.0,
            # exp(-2)~=0.135 keeps real fit pressure (vs exp(-8)~=3e-4 before).
            # Both bounds are config-driven (no hard-coded policy); the defaults
            # are read from model.deberta.uncertainty.{log_var_min,log_var_max}.
            log_var_min = float(_cfg_get(self._cfg, "model.deberta.uncertainty.log_var_min", -8.0))
            log_var_max = float(_cfg_get(self._cfg, "model.deberta.uncertainty.log_var_max", 2.0))
            log_var = torch.clamp(log_var, min=log_var_min, max=log_var_max)
            assert_finite("regression_preds", reg)
            assert_finite("log_variance", log_var)
            outputs = {
                "regression": reg,
                "direction": direction,
                "log_variance": log_var,
                "document": doc,
            }

            # Feature: continuous-stance-scoring -- additive stance outputs (Req
            # 6.1, 6.4, 6.5). When the stance head is attached, fuse the pooled
            # document vector with the per-example market-emotion and
            # market-expectation signals carried on the batch and append the
            # stance logits/probabilities/mask WITHOUT touching the existing
            # keys. The head is mode-agnostic (``doc`` is ``[B, hidden]`` in both
            # hierarchical and flat modes).
            if self.stance_head is not None:
                emotion = batch.get("stance_emotion")
                expectation = batch.get("stance_expectation")
                emotion_present = batch.get("stance_emotion_present")
                expectation_present = batch.get("stance_expectation_present")

                # Per-example supervision mask: which examples carry a defined
                # market-derived stance label. Fall back to an all-True mask
                # (one per example) when the batch does not supply one.
                stance_mask = batch.get("stance_mask")
                if stance_mask is None:
                    stance_mask = torch.ones(
                        doc.shape[0], dtype=torch.bool, device=doc.device
                    )

                # NaN-recovery policy for the stance path (Req 9.5), mirroring the
                # encoder path in ``_encode_document``. The StanceHead applies the
                # finite-value guard (``assert_finite``) to its fusion input,
                # logits, and probabilities (Req 9.1) and RAISES on a non-finite
                # value. During training that raise must not crash the run: a
                # detected non-finite value in the stance computation routes
                # through the recovery policy here -- scrub every non-finite entry
                # to a finite value (``nan_to_num``), hard-zero absent stance
                # components via a mask multiply (clean-zero gradient), record the
                # count of non-finite stance components for the batch, and let the
                # training loop append the offending batch to the debug log
                # (it reads ``self.last_nonfinite_stance``) rather than
                # propagating the non-finite value.
                try:
                    stance_probs, stance_logits = self.stance_head(
                        doc,
                        emotion,
                        expectation,
                        emotion_present,
                        expectation_present,
                    )
                    # Even when the head's guards pass, apply the same recovery
                    # policy as the encoder path so the outputs that leave
                    # forward are guaranteed finite and the count is recorded.
                    stance_logits, stance_probs = self._recover_stance(
                        stance_logits, stance_probs, stance_mask
                    )
                except ValueError:
                    # The head's finite-value guard fired: a non-finite value was
                    # detected somewhere in the stance computation. Recompute the
                    # stance outputs WITHOUT the raising guard so the recovery
                    # scrub + mask-zero can produce a finite forward tensor.
                    stance_logits, stance_probs = self._recover_stance_from_raise(
                        doc,
                        emotion,
                        expectation,
                        emotion_present,
                        expectation_present,
                        stance_mask,
                    )

                outputs["stance"] = stance_logits
                outputs["stance_probs"] = stance_probs
                outputs["stance_mask"] = stance_mask

            return outputs

        def _recover_stance(self, stance_logits, stance_probs, stance_mask):
            """Apply the NaN-recovery policy to stance outputs (Req 9.5).

            Mirrors the encoder-path policy in ``_encode_document``:
              1. Count the non-finite entries in PRESENT (supervised) rows for
                 the batch and record it on ``self.last_nonfinite_stance`` so the
                 training loop can append the offending batch to the debug log.
              2. Scrub EVERY non-finite entry (present and absent) with
                 ``nan_to_num`` so nothing non-finite flows forward or into the
                 gradient.
              3. Hard-zero absent stance components with a mask multiply (a clean
                 0 gradient) using the per-example ``stance_mask`` broadcast over
                 the three stance components.

            Returns the recovered ``(logits, probs)`` pair, both finite.
            """
            # keep: [B, 1] float mask (1.0 present / 0.0 absent) broadcastable
            # over the three stance components [B, 3].
            keep = stance_mask.to(stance_logits.dtype).reshape(-1, 1)
            present = keep.bool().expand_as(stance_logits)
            # Count non-finite entries in PRESENT rows across logits AND probs
            # (diagnostic only) BEFORE scrubbing.
            logits_nonfinite = (~torch.isfinite(stance_logits)) & present
            probs_nonfinite = (~torch.isfinite(stance_probs)) & present
            self.last_nonfinite_stance = int(
                logits_nonfinite.sum().item() + probs_nonfinite.sum().item()
            )
            # Scrub ALL non-finite entries so the forward stays finite.
            stance_logits = torch.nan_to_num(
                stance_logits, nan=0.0, posinf=0.0, neginf=0.0
            )
            stance_probs = torch.nan_to_num(
                stance_probs, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Hard-zero absent stance components with a mask multiply so an
            # excluded example carries a clean 0 gradient through the stance path.
            stance_logits = stance_logits * keep
            stance_probs = stance_probs * keep
            return stance_logits, stance_probs

        def _recover_stance_from_raise(
            self,
            doc,
            emotion,
            expectation,
            emotion_present,
            expectation_present,
            stance_mask,
        ):
            """Recover finite stance outputs after the head's guard raised.

            The head's ``assert_finite`` guard raises before returning when a
            non-finite value is detected. To keep the forward finite (Req 9.5),
            re-run the head fusion here WITHOUT the raising guards, then route the
            raw (possibly non-finite) tensors through the shared
            ``_recover_stance`` scrub + mask-zero + count policy. This is the
            training-time recovery branch; the head's guard remains intact for
            the finite-value-guard semantics (Req 9.1) on the normal path.
            """
            head = self.stance_head
            b = doc.shape[0] if hasattr(doc, "shape") else int(
                torch.as_tensor(doc).shape[0]
            )
            doc_t = doc if torch.is_tensor(doc) else torch.as_tensor(doc)
            doc_t = doc_t.to(torch.float32)
            if doc_t.dim() == 1:
                doc_t = doc_t.unsqueeze(0)

            def _as_col(x, default):
                if x is None:
                    return torch.full((b,), float(default), dtype=torch.float32)
                t = x if torch.is_tensor(x) else torch.as_tensor(x)
                return t.to(torch.float32).reshape(-1)

            def _as_bool(x):
                if x is None:
                    return torch.zeros(b, dtype=torch.bool)
                t = x if torch.is_tensor(x) else torch.as_tensor(x)
                return t.reshape(-1).to(torch.bool)

            emo = _as_col(emotion, head.default_emotion)
            exp = _as_col(expectation, head.default_expectation)
            emo_present = _as_bool(emotion_present)
            exp_present = _as_bool(expectation_present)
            emo = torch.where(
                emo_present, emo, torch.full_like(emo, float(head.default_emotion))
            )
            exp = torch.where(
                exp_present, exp, torch.full_like(exp, float(head.default_expectation))
            )
            head.emotion_default_applied = ~emo_present
            head.expectation_default_applied = ~exp_present

            fusion_input = torch.cat(
                [doc_t, emo.unsqueeze(-1), exp.unsqueeze(-1)], dim=-1
            )
            # Count non-finite fusion-input entries in PRESENT rows for the
            # diagnostic BEFORE scrubbing (mirrors _encode_document counting the
            # raw encoder output).
            keep_row = stance_mask.to(torch.bool).reshape(-1)
            present_in = keep_row.reshape(-1, 1).expand_as(fusion_input)
            n_input_nonfinite = int(
                ((~torch.isfinite(fusion_input)) & present_in).sum().item()
            )
            # Scrub the fusion input BEFORE it enters ``fuse`` so no non-finite
            # value ever enters the autograd graph. Scrubbing only the OUTPUT
            # would leave the gradient poisoned (NaN * 0 == NaN); scrubbing the
            # input keeps the whole backward pass finite -- exactly the discipline
            # the encoder path uses when it routes degenerate chunks around the
            # encoder before the NaN-producing op.
            fusion_input = torch.nan_to_num(
                fusion_input, nan=0.0, posinf=0.0, neginf=0.0
            )
            logits = head.fuse(fusion_input)
            shifted = (logits - logits.amax(dim=-1, keepdim=True)) / head.temperature
            probs = torch.softmax(shifted, dim=-1)
            logits, probs = self._recover_stance(logits, probs, stance_mask)
            # Fold the input-side non-finite count into the recorded total so the
            # training-loop debug log reflects non-finite entries detected
            # anywhere in the stance computation (inputs + outputs).
            self.last_nonfinite_stance += n_input_nonfinite
            return logits, probs

    # =========================================================================
    # Council-of-supervision heads (Req 1.1, 1.4, 7.1) -- Task 5.1
    #
    # Two new heads that read the SAME pooled ``[B, hidden]`` document vector the
    # five existing heads consume (``doc`` from ``_encode_document`` -- identical
    # in hierarchical and flat modes, so both heads are mode-agnostic, Req 1.4):
    #
    #   * ``CouncilHeads`` -- one ``nn.Linear(hidden, 1)`` per configured Council
    #     signal (generalizes the single ``regression_head`` to a bank of N
    #     per-signal regression heads, Req 1.1) plus one learned ``log_var``
    #     ``nn.Parameter`` per signal (the per-signal task-weight scalar the
    #     uncertainty-based ``SignalFusion`` consumes; mirrors the per-example
    #     ``uncertainty_head`` mechanism lifted to a per-signal task weight).
    #   * ``StanceScoreHead`` -- ``nn.Linear(hidden, 1)`` -> ``tanh`` producing a
    #     continuous scalar in ``[-1, 1]`` locating each Event on the
    #     hawkish--dovish spectrum (Req 7.1); the continuous scalar counterpart of
    #     the soft-distribution ``StanceHead``.
    #
    # Both guard their outputs with the shared ``assert_finite`` guard so the
    # Council path enforces the SAME non-finite policy as the rest of training.
    # These are defined here (Task 5.1) but NOT yet wired into a model forward --
    # ``CouncilModel`` (Task 8.1) will run them on the shared ``doc`` vector.
    # =========================================================================

    def _council_head_signal_names(cfg: Any) -> list[str]:
        """Return the configured Council signal names (``council.signals[i].name``).

        Reads the signal set straight from configuration (Req 1.1) and returns
        the per-signal ``name`` values in configured order. A malformed entry
        (not a mapping, or a missing / blank ``name``) is skipped rather than
        fabricated -- this mirrors ``training._council_signal_names`` so the head
        bank keys line up exactly with the columns the ``SignalNormalizer`` and
        the batch ``signal_targets``/``signal_present`` fields use.
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

    def _council_signal_weights(cfg: Any) -> dict:
        """Return the configured fixed-mode per-signal weights (``signals[i].weight``).

        Reads the Council signal set from configuration and maps each signal
        ``name`` to its configured ``weight`` (Req 2.2). This is the fixed-mode
        analogue of ``compute_losses``' ``w(name)`` lookup over
        ``model.deberta.loss_weights``: a signal with no explicit weight defaults
        to ``1.0`` (unit weight) rather than being fabricated or dropped, so the
        fixed fusion reproduces the current per-signal weighting behavior. A
        malformed / nameless entry is skipped, matching
        :func:`_council_head_signal_names`.
        """
        signals = _cfg_get(cfg, "council.signals", []) or []
        weights: dict[str, float] = {}
        for sig in signals:
            if not isinstance(sig, dict):
                continue
            name = sig.get("name")
            if name is None:
                continue
            name = str(name).strip()
            if not name:
                continue
            raw = sig.get("weight", 1.0)
            try:
                weights[name] = float(raw)
            except (TypeError, ValueError):
                weights[name] = 1.0
        return weights

    class _NamedParameterDict(nn.ParameterDict):
        """ParameterDict that accepts public signal names for internal keys."""

        def __init__(self, values: dict, aliases: dict[str, str]):
            self._aliases = dict(aliases)
            super().__init__(values)

        def _key(self, key):
            return self._aliases.get(key, key)

        def __contains__(self, key):
            return self._key(key) in self._parameters

        def __getitem__(self, key):
            return self._parameters[self._key(key)]

        def keys(self):
            return self._aliases.keys()

    class CouncilHeads(nn.Module):
        """Per-signal Signal_Head bank over the shared document vector (Req 1.1, 1.4).

        Generalizes the single ``regression_head`` into a bank of N per-signal
        regression heads -- one ``nn.Linear(hidden, 1)`` per configured Council
        signal (``council.signals[i].name``) -- plus one learned ``log_var``
        ``nn.Parameter`` per signal. Every head reads the SAME pooled ``[B,
        hidden]`` document vector produced by ``_encode_document`` (identical in
        hierarchical and flat modes, so the bank is mode-agnostic, Req 1.4).

        The per-signal ``log_var`` scalars are the learned task weights the
        uncertainty-based ``SignalFusion`` consumes (mirrors the per-example
        ``uncertainty_head`` mechanism, lifted to one scalar per signal). They
        are exposed as a ``ParameterDict`` keyed by signal name so ``SignalFusion``
        can look each up by name and clamp it to ``[log_var_min, log_var_max]``.

        The signal names are read from ``council.signals`` at init (Req 1.1);
        head keys and ``log_var`` keys use those names verbatim so a downstream
        consumer can map each prediction and task weight back to its signal.
        """

        def __init__(self, hidden_size: int, cfg: Any):
            super().__init__()
            self.hidden_size = int(hidden_size)
            # Read the configured Council signal set at init (Req 1.1). Order is
            # preserved so the head bank aligns with the batch signal columns.
            self.signal_names = _council_head_signal_names(cfg)
            # PyTorch module containers reserve attribute-like names (for
            # example ``eval``) and reject keys containing dots. Keep the
            # configured names as the public API, but use stable indexed keys
            # internally so arbitrary signal identifiers remain valid.
            self._head_keys = {
                name: f"signal_{index}" for index, name in enumerate(self.signal_names)
            }
            # One nn.Linear(hidden, 1) regression head per signal -- the bank that
            # generalizes the single regression_head (Req 1.1).
            self.heads = nn.ModuleDict(
                {
                    self._head_keys[name]: nn.Linear(self.hidden_size, 1)
                    for name in self.signal_names
                }
            )
            # One learned log-variance scalar per signal (the per-signal task
            # weight for uncertainty-based fusion). Initialized to 0.0 so the
            # initial effective weight exp(-log_var)=1.0 (unweighted) before the
            # optimizer learns per-signal uncertainty.
            self.log_var = _NamedParameterDict(
                {
                    self._head_keys[name]: nn.Parameter(torch.zeros(1))
                    for name in self.signal_names
                },
                aliases=self._head_keys,
            )

        def named_log_vars(self) -> dict:
            """Return learned log-variances keyed by configured signal name."""
            return {
                name: self.log_var[self._head_keys[name]]
                for name in self.signal_names
            }

        def forward(self, doc: "torch.Tensor") -> dict:
            """Predict every configured signal from the shared document vector.

            Args:
                doc: Pooled document representation ``[B, hidden]`` (or ``[hidden]``
                    for a single example -- treated as a batch of one). Identical
                    in hierarchical and flat modes, so the bank is mode-agnostic
                    (Req 1.4).

            Returns:
                A dict mapping each configured signal name to its per-example
                prediction tensor ``[B]``. Every prediction is guarded with
                :func:`assert_finite` so a non-finite head output surfaces
                immediately rather than poisoning downstream fusion.
            """
            doc = doc if torch.is_tensor(doc) else torch.as_tensor(doc)
            if doc.dim() == 1:
                doc = doc.unsqueeze(0)
            preds: dict[str, "torch.Tensor"] = {}
            for name in self.signal_names:
                head = self.heads[self._head_keys[name]]
                pred = head(doc).squeeze(-1)  # [B]
                assert_finite("council_signal_pred::" + name, pred)
                preds[name] = pred
            return preds

    class StanceScoreHead(nn.Module):
        """Continuous Stance_Score head over the shared document vector (Req 7.1, 1.4).

        A ``nn.Linear(hidden, 1)`` followed by ``tanh`` producing a continuous
        scalar in ``[-1, 1]`` that locates each Event on the hawkish--dovish
        spectrum (Req 7.1) -- the continuous scalar counterpart of the
        soft-distribution :class:`~src.stance.StanceHead`. The ``tanh`` bounds the
        score and preserves monotonic direction (a higher pre-activation maps to
        a higher, more-hawkish score).

        It reads the SAME pooled ``[B, hidden]`` document vector the five existing
        heads and the ``CouncilHeads`` bank consume (Req 1.4); because ``doc`` is
        ``[B, hidden]`` in both hierarchical and flat modes the head is
        mode-agnostic. The output is guarded with :func:`assert_finite`.
        """

        def __init__(self, hidden_size: int, cfg: Any = None):
            super().__init__()
            self.hidden_size = int(hidden_size)
            self.score = nn.Linear(self.hidden_size, 1)

        def forward(self, doc: "torch.Tensor") -> "torch.Tensor":
            """Map the shared document vector to a continuous stance score in [-1, 1].

            Args:
                doc: Pooled document representation ``[B, hidden]`` (or ``[hidden]``
                    for a single example -- treated as a batch of one).
                    Mode-agnostic across hierarchical / flat (Req 1.4).

            Returns:
                A per-example continuous stance score ``[B]`` in ``[-1, 1]``
                (Req 7.1), guarded finite via :func:`assert_finite`.
            """
            doc = doc if torch.is_tensor(doc) else torch.as_tensor(doc)
            if doc.dim() == 1:
                doc = doc.unsqueeze(0)
            score = torch.tanh(self.score(doc).squeeze(-1))  # [B] in [-1, 1]
            assert_finite("stance_score", score)
            return score

    class SignalFusion(nn.Module):
        """Fuse per-signal Council losses into one scalar loss (Req 2.1, 2.2, 2.4).

        Generalizes the fixed-weight ``Σ_name w(name)·loss_name`` composition in
        :func:`compute_losses` to a Council of N per-signal losses. Two modes,
        selected by ``council.fusion_mode``:

        * ``"uncertainty"`` (default, Kendall & Gal homoscedastic weighting,
          Req 2.1): each signal carries a learned log-variance scalar
          ``log_var_i`` (the per-signal ``nn.Parameter`` held on
          :class:`CouncilHeads`). The fused loss is::

              L_council = Σ_i [ exp(-log_var_i) · L_i + log_var_i ]

          The multiplicative ``exp(-log_var_i)`` down-weights noisy signals; the
          additive ``log_var_i`` regularizer blocks the trivial escape of sending
          every log-variance to +inf. Each ``log_var_i`` is clamped to
          ``[log_var_min, log_var_max]`` EXACTLY as
          ``MarketSupervisedModel.forward`` clamps the per-example log-variance
          (``torch.clamp(log_var, min=log_var_min, max=log_var_max)``), only here
          the bounds are read from ``council.log_var_min`` / ``council.log_var_max``.
          Clamping keeps every effective weight finite and bounded within
          ``[exp(-log_var_max), exp(-log_var_min)]`` (Req 2.4).

        * ``"fixed"`` (Req 2.2): applies the configured per-signal weight
          ``council.signals[i].weight`` with no learned log-variance, reproducing
          today's ``w(name)`` behavior::

              L_council = Σ_i council.signals[i].weight · L_i

        In both modes :meth:`fuse` also returns the effective per-signal weight
        applied at this step (``exp(-clamp(log_var_i))`` in uncertainty mode, the
        configured weight in fixed mode) so ``run_training_loop`` can log it
        (Req 2.3). The fused loss is guarded with :func:`assert_finite`.
        """

        # Bounds convention shared with MarketSupervisedModel.forward's per-example
        # log-variance clamp; here read from the Council namespace.
        _DEFAULT_LOG_VAR_MIN = -8.0
        _DEFAULT_LOG_VAR_MAX = 2.0

        def __init__(self, cfg: Any):
            super().__init__()
            self._cfg = cfg
            mode = _cfg_get(cfg, "council.fusion_mode", "uncertainty")
            self.fusion_mode = str(mode).strip().lower() if mode is not None else "uncertainty"
            # Fixed-mode per-signal weights (Req 2.2). Uncertainty mode ignores
            # these but they are read once so a switch to fixed needs no re-init.
            self.fixed_weights = _council_signal_weights(cfg)
            self.log_var_min = float(
                _cfg_get(cfg, "council.log_var_min", self._DEFAULT_LOG_VAR_MIN)
            )
            self.log_var_max = float(
                _cfg_get(cfg, "council.log_var_max", self._DEFAULT_LOG_VAR_MAX)
            )

        def fuse(
            self,
            signal_losses: dict,
            log_var: Any = None,
        ) -> tuple:
            """Fuse per-signal losses into one scalar Council loss.

            Args:
                signal_losses: Mapping ``{signal_name: L_i}`` of per-signal masked
                    losses (each a scalar ``torch.Tensor``), as produced by the
                    per-signal masked MSE. Each ``L_i`` is assumed already guarded
                    finite by the caller.
                log_var: The per-signal log-variance parameters, keyed by signal
                    name (typically ``CouncilHeads.log_var``, an ``nn.ParameterDict``,
                    or any mapping ``{name: scalar-tensor}``). Required in
                    ``"uncertainty"`` mode; ignored in ``"fixed"`` mode.

            Returns:
                ``(fused_loss, effective_weights)`` where ``fused_loss`` is a
                scalar ``torch.Tensor`` (guarded finite via :func:`assert_finite`)
                and ``effective_weights`` is a ``{signal_name: float}`` dict of the
                effective per-signal weight applied this step (Req 2.3):
                ``exp(-clamp(log_var_i))`` in uncertainty mode, the configured
                ``weight`` in fixed mode.

            Raises:
                ValueError: If uncertainty mode is selected but ``log_var`` lacks a
                    parameter for a signal that carries a loss, or if the fused
                    loss is non-finite.
            """
            _require_torch("SignalFusion.fuse")
            if not signal_losses:
                raise ValueError("SignalFusion.fuse requires at least one per-signal loss")

            effective_weights: dict[str, float] = {}
            terms: list["torch.Tensor"] = []

            if self.fusion_mode == "fixed":
                for name, loss_i in signal_losses.items():
                    loss_i = loss_i if torch.is_tensor(loss_i) else torch.as_tensor(
                        float(loss_i), dtype=torch.float32
                    )
                    # Configured weight; default to unit weight when unset so a
                    # signal is never silently dropped (mirrors compute_losses' w()).
                    w_i = float(self.fixed_weights.get(name, 1.0))
                    terms.append(w_i * loss_i)
                    effective_weights[name] = w_i
            else:
                # Uncertainty-based fusion (Req 2.1). Each signal needs its learned
                # log-variance; clamp EXACTLY like MarketSupervisedModel.forward.
                if log_var is None:
                    raise ValueError(
                        "SignalFusion.fuse requires per-signal log_var parameters "
                        "in 'uncertainty' fusion_mode"
                    )
                for name, loss_i in signal_losses.items():
                    loss_i = loss_i if torch.is_tensor(loss_i) else torch.as_tensor(
                        float(loss_i), dtype=torch.float32
                    )
                    if name not in log_var:
                        raise ValueError(
                            "SignalFusion.fuse missing log_var parameter for signal "
                            f"'{name}' in 'uncertainty' fusion_mode"
                        )
                    lv = log_var[name]
                    lv = lv if torch.is_tensor(lv) else torch.as_tensor(
                        float(lv), dtype=torch.float32
                    )
                    lv = lv.reshape(())  # scalar per signal
                    # Finite by construction (design principle): scrub non-finite
                    # log-variances BEFORE clamping. torch.clamp passes NaN through
                    # unchanged, so a NaN/±Inf log-var would otherwise survive the
                    # clamp and blow up exp()/assert_finite. Map NaN into the valid
                    # band (0.0, then re-clamped below) and ±Inf onto the bounds,
                    # mirroring _encode_document's nan_to_num discipline (Req 4.2).
                    lv = torch.nan_to_num(
                        lv,
                        nan=0.0,
                        posinf=self.log_var_max,
                        neginf=self.log_var_min,
                    )
                    # Same clamp policy as the per-example log-variance in
                    # MarketSupervisedModel.forward, here with council.log_var_*.
                    lv_c = torch.clamp(lv, min=self.log_var_min, max=self.log_var_max)
                    assert_finite("signal_log_var::" + name, lv_c)
                    weight_i = torch.exp(-lv_c)  # bounded in [e^-max, e^-min]
                    # L_council term: exp(-log_var)·L_i + log_var (Kendall & Gal).
                    terms.append(weight_i * loss_i + lv_c)
                    effective_weights[name] = float(weight_i.detach().reshape(()).item())

            fused_loss = terms[0]
            for extra in terms[1:]:
                fused_loss = fused_loss + extra
            assert_finite("council_fused_loss", fused_loss)
            return fused_loss, effective_weights

        # Convenience alias so callers can treat SignalFusion like a callable module.
        def forward(self, signal_losses: dict, log_var: Any = None) -> tuple:
            return self.fuse(signal_losses, log_var=log_var)

    class CouncilModel(MarketSupervisedModel):
        """Council-of-supervision model over the SHARED document vector (Task 8.1).

        Subclasses :class:`MarketSupervisedModel` so the entire encoder path --
        ``_encode_document`` (hierarchical segment encoding + attention pooling,
        degenerate-chunk routing, the ``nan_to_num`` scrub + mask-multiply
        discipline) -- is reused UNCHANGED (Req 1.4). The single ``regression_head``
        is generalized into the :class:`CouncilHeads` bank and a continuous
        :class:`StanceScoreHead`, both reading the ONE pooled ``[B, hidden]``
        ``doc`` vector the base heads read (Req 1.1). A :class:`SignalFusion`
        instance is attached so :func:`compute_council_losses` can fuse the
        per-signal masked MSEs into one scalar ``L_council``.

        ``forward`` extends the numerical-stability discipline the base class
        already applies to the encoder and stance paths to the two new Council
        paths (Req 4.2):

          * the batch ``signal_targets * signal_present`` supervision tensor is
            scrubbed with ``nan_to_num`` and hard-zeroed on absent signals with a
            mask multiply (a clean-zero gradient), exactly like ``seg_vecs * keep``
            in ``_encode_document``;
          * the continuous ``stance_score`` is scrubbed the same way;
          * per-signal non-finite counts are recorded on
            ``self.last_nonfinite_council`` (mirroring ``last_nonfinite_segments``
            / ``last_nonfinite_stance``) so ``run_training_loop`` can append the
            offending batch to the training debug log rather than crashing.

        The returned dict carries ``outputs["council"]`` -> ``{name: [B]}``,
        ``outputs["stance_score"]`` -> ``[B]``, ``outputs["document"]`` (the shared
        ``doc``), the scrubbed ``outputs["signal_targets"]`` / ``signal_present``,
        and ``outputs["council_signal_weights"]`` (the per-signal effective fusion
        weight this step, Req 2.3) -- everything :func:`compute_council_losses`
        consumes.
        """

        def __init__(self, cfg: Any, encoder: Optional["nn.Module"] = None):
            super().__init__(cfg, encoder=encoder)
            # The Council has an explicit three-way primary direction head
            # (down/neutral/up). The base binary head is replaced only for this
            # subclass; neutral labels use the raw-unit material threshold.
            self.direction_head = nn.Linear(self.hidden_size, 3)
            # Council stance is the continuous stance_score_head below. Remove
            # the inherited auxiliary StanceHead so no enabled-but-unused module
            # silently remains outside the active objective.
            self.stance_head = None
            # The Signal_Head bank (generalizes regression_head) + continuous
            # Stance_Score head, both over the shared [B, hidden] doc vector.
            self.council_heads = CouncilHeads(self.hidden_size, cfg)
            # Retain the former independent scalar head and its parameter keys
            # for explicit legacy analysis/warm starts. This does NOT make an
            # entire historical Council checkpoint inference-compatible: those
            # checkpoints used a one-logit direction layer and cannot supply a
            # calibrated neutral class. Ordinary loaders reject that schema.
            # Canonical continuous stance is derived from the trained three-way
            # probabilities: P(hawkish) - P(dovish).
            self.stance_score_head = StanceScoreHead(self.hidden_size, cfg)
            # Fusion (uncertainty / fixed) consuming the per-signal log-variances
            # held on ``council_heads.log_var``. Clamp bounds live in SignalFusion.
            self.signal_fusion = SignalFusion(cfg)
            # Configured signal order -- the canonical [B, N] column order for
            # signal_targets / signal_present and the head-bank keys (Req 1.1).
            self.council_signal_names = _council_head_signal_names(cfg)
            # Diagnostic (mirrors ``last_nonfinite_segments`` / ``last_nonfinite_stance``):
            # number of non-finite entries the Council supervision path emitted
            # for PRESENT signals on the most recent forward pass. Scrubbed before
            # the outputs leave forward so the forward stays finite; the training
            # loop reads this to append the offending batch to the debug log
            # (Req 4.2) rather than crashing.
            self.last_nonfinite_council = 0
            # Diagnostic: non-finite entries scrubbed from the continuous
            # stance-score path on the most recent forward pass.
            self.last_nonfinite_stance_score = 0

        def _scrub_council_targets(self, signal_targets, signal_present, batch_size):
            """Scrub + mask the Council supervision tensor (Req 4.2).

            Applies the SAME ``nan_to_num`` scrub + mask-multiply discipline as
            ``_encode_document`` (``seg_vecs * keep``) to ``signal_targets`` using
            ``signal_present`` as the keep mask:

              1. Coerce ``signal_targets`` -> ``[B, N]`` float and ``signal_present``
                 -> ``[B, N]`` bool (via :func:`_council_present_matrix`).
              2. Count non-finite entries in PRESENT signals (diagnostic only) on
                 ``self.last_nonfinite_council`` BEFORE scrubbing.
              3. Scrub EVERY non-finite target with ``nan_to_num`` so nothing
                 non-finite reaches the loss or the gradient.
              4. Hard-zero absent signals with the presence mask multiply
                 (``targets * present``) -- a clean-zero gradient for masked terms.

            Returns the recovered ``(targets, present)`` pair, both finite.
            """
            if signal_present is None:
                # No presence mask supplied -> treat every configured signal as
                # present (all-ones), so the scrub still runs and nothing is
                # silently fabricated as absent.
                n = len(self.council_signal_names) or (
                    signal_targets.shape[-1]
                    if (torch.is_tensor(signal_targets) and signal_targets.dim() >= 1)
                    else 1
                )
                present = torch.ones(batch_size, n, dtype=torch.bool)
            else:
                present = _council_present_matrix(signal_present)

            if signal_targets is None:
                targets = torch.zeros(
                    present.shape[0], present.shape[1], dtype=torch.float32
                )
            else:
                targets = (
                    signal_targets
                    if torch.is_tensor(signal_targets)
                    else torch.as_tensor(signal_targets, dtype=torch.float32)
                )
                targets = targets.to(torch.float32)
                if targets.dim() == 1:
                    targets = targets.unsqueeze(0)

            # Align the presence mask dtype/shape with the targets for the multiply.
            keep = present.to(targets.dtype)
            present_bool = present.bool()
            # Count non-finite entries in PRESENT signals (diagnostic only) before
            # scrubbing -- mirrors ``last_nonfinite_segments`` in _encode_document.
            present_nonfinite = (~torch.isfinite(targets)) & present_bool
            self.last_nonfinite_council = int(present_nonfinite.sum().item())
            # Scrub ALL non-finite entries so nothing non-finite flows forward.
            targets = torch.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            # Hard-zero absent signals with the presence mask multiply (clean-0
            # gradient) -- the ``seg_vecs * keep`` discipline applied to
            # ``signal_targets * signal_present`` (Req 4.2).
            targets = targets * keep
            assert_finite("council_signal_targets", targets)
            return targets, present_bool

        def forward(self, batch):
            """Encode once, run the Council + Stance_Score heads on the shared doc.

            Reuses ``_encode_document`` UNCHANGED for the shared ``doc`` vector,
            then runs :class:`CouncilHeads` and :class:`StanceScoreHead` on it.
            Extends the base ``nan_to_num`` scrub + mask-multiply discipline to
            ``signal_targets * signal_present`` and the continuous stance path,
            records per-signal non-finite counts on ``self.last_nonfinite_council``
            (Req 4.2), and returns a dict consumable by
            :func:`compute_council_losses`.
            """
            doc = self._encode_document(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("segment_mask"),
            )  # [B, hidden], already finite (base scrub + mask-multiply)
            batch_size = doc.shape[0]

            # Dedicated primary outputs keep evaluation/training API-compatible
            # while the Council auxiliary heads learn the full signal vector.
            primary_prediction = self.regression_head(doc).squeeze(-1)
            primary_direction_logits = self.direction_head(doc)  # [B, 3]
            primary_log_variance = self.uncertainty_head(doc).squeeze(-1)
            log_var_min = float(
                _cfg_get(self._cfg, "model.deberta.uncertainty.log_var_min", -8.0)
            )
            log_var_max = float(
                _cfg_get(self._cfg, "model.deberta.uncertainty.log_var_max", 2.0)
            )
            primary_log_variance = torch.clamp(
                primary_log_variance, min=log_var_min, max=log_var_max
            )
            assert_finite("primary_prediction", primary_prediction)
            assert_finite("primary_direction_logits", primary_direction_logits)
            assert_finite("primary_log_variance", primary_log_variance)

            # Signal_Head bank: one [B] prediction per configured signal. Each is
            # guarded finite inside CouncilHeads.forward.
            council_preds = self.council_heads(doc)  # {name: [B]}

            # The trained three-way direction classifier is the canonical stance
            # classifier. Its class order is explicitly
            # (dovish, neutral, hawkish), matching direction targets
            # (down=0, neutral=1, up=2). Softmax in FP32 for numerical stability,
            # then derive the continuous stance deterministically so categorical
            # prediction always precedes continuous scoring.
            primary_direction_probs = torch.softmax(
                primary_direction_logits.to(torch.float32), dim=-1
            )
            stance_score = (
                primary_direction_probs[:, 2] - primary_direction_probs[:, 0]
            )
            assert_finite("primary_direction_probs", primary_direction_probs)
            assert_finite("stance_score", stance_score)

            # Compatibility-only diagnostic from the former independent score
            # head. Preserve this individual parameter block for explicit legacy
            # analysis, but do not imply that a full one-logit historical Council
            # checkpoint is a valid three-way model. This disconnected value is
            # never the canonical score or training target.
            legacy_stance_score = self.stance_score_head(doc)
            score_nonfinite = ~torch.isfinite(legacy_stance_score)
            self.last_nonfinite_stance_score = int(score_nonfinite.sum().item())
            legacy_stance_score = torch.nan_to_num(
                legacy_stance_score, nan=0.0, posinf=0.0, neginf=0.0
            )
            assert_finite("legacy_stance_score", legacy_stance_score)

            # Scrub + mask the Council supervision targets with the shared
            # discipline (signal_targets * signal_present), recording the
            # per-signal non-finite count on ``last_nonfinite_council`` (Req 4.2).
            signal_targets = (
                batch.get("signal_targets") if isinstance(batch, dict) else None
            )
            signal_present = (
                batch.get("signal_present") if isinstance(batch, dict) else None
            )
            scrubbed_targets, scrubbed_present = self._scrub_council_targets(
                signal_targets, signal_present, batch_size
            )

            # Effective per-signal fusion weights for THIS step (Req 2.3). Read
            # them off the fusion + head log-variances without building the loss
            # here (loss composition stays in compute_council_losses). In fixed
            # mode these are the configured weights; in uncertainty mode they are
            # ``exp(-clamp(log_var_i))`` with the clamp bounds SignalFusion holds.
            council_signal_weights = self._effective_signal_weights(council_preds)

            outputs = {
                # Canonical primary API plus legacy aliases consumed by generic
                # prediction/temporal helpers.
                "primary_prediction": primary_prediction,
                "regression": primary_prediction,
                "primary_direction_logits": primary_direction_logits,
                "direction": primary_direction_logits,
                # Canonical Council stance outputs. All probability tensors use
                # COUNCIL_STANCE_CLASS_ORDER = (dovish, neutral, hawkish).
                "stance_logits": primary_direction_logits,
                "primary_direction_probs": primary_direction_probs,
                "direction_probs": primary_direction_probs,
                "stance_probs": primary_direction_probs,
                "stance_prob_dovish": primary_direction_probs[:, 0],
                "stance_prob_neutral": primary_direction_probs[:, 1],
                "stance_prob_hawkish": primary_direction_probs[:, 2],
                "stance_class_order": COUNCIL_STANCE_CLASS_ORDER,
                "primary_log_variance": primary_log_variance,
                "log_variance": primary_log_variance,
                "council": council_preds,          # {name: [B]}
                # Canonical continuous stance is derived from the categorical
                # distribution; the old head output is diagnostic-only.
                "stance_score": stance_score,
                "probability_stance_score": stance_score,
                "legacy_stance_score": legacy_stance_score,
                "document": doc,                   # shared [B, hidden] doc vector
                "signal_targets": scrubbed_targets,   # [B, N] scrubbed + masked
                "signal_present": scrubbed_present,   # [B, N] bool
                "council_log_var": self.council_heads.named_log_vars(),  # for fusion
                "council_signal_weights": council_signal_weights,  # {name: float}
            }
            return outputs

        def _effective_signal_weights(self, council_preds) -> dict:
            """Effective per-signal fusion weight applied this step (Req 2.3).

            Mirrors :meth:`SignalFusion.fuse`'s weight computation without running
            the loss: fixed mode returns the configured weights; uncertainty mode
            returns ``exp(-clamp(log_var_i))`` with the SAME scrub + clamp bounds
            SignalFusion applies, so the logged weight matches the one actually
            used in the fused loss.
            """
            fusion = self.signal_fusion
            names = [n for n in self.council_signal_names if n in council_preds] or list(
                council_preds.keys()
            )
            weights: dict[str, float] = {}
            if getattr(fusion, "fusion_mode", "uncertainty") == "fixed":
                for name in names:
                    weights[name] = float(fusion.fixed_weights.get(name, 1.0))
                return weights
            log_var = self.council_heads.named_log_vars()
            for name in names:
                if name not in log_var:
                    continue
                lv = log_var[name]
                lv = lv if torch.is_tensor(lv) else torch.as_tensor(
                    float(lv), dtype=torch.float32
                )
                lv = lv.reshape(())
                lv = torch.nan_to_num(
                    lv,
                    nan=0.0,
                    posinf=fusion.log_var_max,
                    neginf=fusion.log_var_min,
                )
                lv_c = torch.clamp(lv, min=fusion.log_var_min, max=fusion.log_var_max)
                weights[name] = float(torch.exp(-lv_c).detach().reshape(()).item())
            return weights

    def council_enabled(cfg: Any) -> bool:
        """Return whether the Council pipeline is selected for this config.

        The selector is the presence of a non-empty resolved signal set
        (``council.signals`` via :func:`_council_head_signal_names`), matching
        how ``attach_council_fields`` and the ``SignalNormalizer`` already gate
        themselves. No separate top-level flag is introduced, so a non-Council
        run stays byte-identical to today (Req 1.1, 1.2).
        """
        return len(_council_head_signal_names(cfg)) > 0

    def build_model(cfg: Any) -> "nn.Module":
        """Build the market-supervised model; hierarchical-by-default.

        Reads ``cfg.model.deberta.encoder_mode``: ``"hierarchical"`` (default)
        wires per-segment encoding + attention pooling; ``"flat"`` selects the
        single-truncated-segment ablation variant. A flat truncated encoder is
        NEVER the default (protects hypothesis H3).

        The DeBERTa encoder body is constructed here from
        ``model.deberta.base_model`` via HuggingFace ``AutoModel``. Set
        ``model.deberta.encoder_override`` to a name/path to swap in a smaller
        encoder (e.g. for a CPU smoke test); when transformers is unavailable or
        the encoder cannot be loaded and ``model.deberta.allow_stub_encoder`` is
        true, a tiny randomly-initialised stub encoder is used so the wiring is
        runnable without network access.
        """
        # Fewer-than-two-signals guard (Req 1.4). This runs BEFORE any encoder is
        # built so a mis-specified Council config halts initialization without
        # wasting the (potentially expensive) encoder construction. "Council is
        # configured" is signalled by the presence of a ``council.signals`` list;
        # when that list is present but resolves to fewer than two usable signal
        # names, halt with a descriptive error naming the count deficiency.
        _council_signals_cfg = _cfg_get(cfg, "council.signals", None)
        if _council_signals_cfg is not None:
            _resolved_signal_names = _council_head_signal_names(cfg)
            _n_signals = len(_resolved_signal_names)
            if _n_signals < 2:
                raise ValueError(
                    "Council pipeline is configured but defines fewer than two "
                    f"signals: found {_n_signals} usable signal(s) "
                    f"({_resolved_signal_names!r}); the Council model requires at "
                    "least 2 signals. Add signals to council.signals or remove the "
                    "Council configuration to use the standard model."
                )
        # _build_encoder resolves model.deberta.encoder_override (DAPT) before
        # base_model, so the encoder is built the same way for either model type.
        encoder = _build_encoder(cfg)
        # Council path is selected purely by the presence of a configured signal
        # set (Req 1.1); otherwise construct the standard model (Req 1.2).
        # CouncilModel.__init__ already reads fusion_mode/log_var_min/log_var_max
        # into SignalFusion and sets council_signal_names (Req 1.3, 1.5).
        if council_enabled(cfg):
            model = CouncilModel(cfg, encoder=encoder)
        else:
            model = MarketSupervisedModel(cfg, encoder=encoder)
        if bool(_cfg_get(cfg, "model.deberta.gradient_checkpointing", False)):
            enc = model.encoder
            if enc is not None and hasattr(enc, "gradient_checkpointing_enable"):
                try:
                    enc.gradient_checkpointing_enable()
                except Exception:  # pragma: no cover - optional optimisation
                    pass
        return model

    def _build_encoder(cfg: Any) -> "nn.Module":
        """Construct the shared segment encoder (DeBERTa body) from config.

        Order of resolution: ``encoder_override`` if set, else ``base_model``.
        Falls back to :class:`_StubEncoder` when transformers/weights are
        unavailable and ``allow_stub_encoder`` is true (used for CPU smoke
        tests); otherwise the underlying load error is raised.
        """
        override = _cfg_get(cfg, "model.deberta.encoder_override", None)
        allow_stub = bool(_cfg_get(cfg, "model.deberta.allow_stub_encoder", False))
        # Explicit sentinel: force the tiny stub encoder without any network I/O.
        if override == "__stub__" or _cfg_get(cfg, "model.deberta.force_stub", False):
            hidden = int(_cfg_get(cfg, "model.deberta.hidden_size", 768))
            vocab = int(_cfg_get(cfg, "model.deberta.stub_vocab_size", 4096))
            return _StubEncoder(vocab_size=vocab, hidden_size=hidden)
        base_model = override or _cfg_get(
            cfg, "model.deberta.base_model", "microsoft/deberta-v3-base"
        )
        try:
            from transformers import AutoModel

            # Transformers 5 may honor checkpoint dtype metadata. Force FP32 at
            # load time so a half-precision checkpoint cannot create FP16 AdamW
            # master parameters; run_training_loop asserts the invariant again
            # immediately before optimizer construction.
            return AutoModel.from_pretrained(base_model, dtype=torch.float32)
        except Exception as exc:  # network/weights/transformers unavailable
            if allow_stub:
                hidden = int(_cfg_get(cfg, "model.deberta.hidden_size", 768))
                vocab = int(_cfg_get(cfg, "model.deberta.stub_vocab_size", 4096))
                return _StubEncoder(vocab_size=vocab, hidden_size=hidden)
            raise RuntimeError(
                f"Could not load encoder '{base_model}': {exc}. Set "
                "model.deberta.allow_stub_encoder=true to use the stub encoder "
                "for a CPU smoke test."
            ) from exc

    class _StubEncoder(nn.Module):
        """Tiny randomly-initialised encoder mimicking a HF encoder interface.

        Returns an object exposing ``last_hidden_state`` shaped ``[B, L, hidden]``
        so the model's forward pass runs identically to a real DeBERTa body. For
        smoke testing the wiring only - it learns nothing meaningful.
        """

        class _Cfg:
            def __init__(self, hidden_size: int):
                self.hidden_size = hidden_size

        class _Out:
            def __init__(self, last_hidden_state):
                self.last_hidden_state = last_hidden_state

        def __init__(self, vocab_size: int = 4096, hidden_size: int = 32):
            super().__init__()
            self.config = _StubEncoder._Cfg(hidden_size)
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layer = nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=2, dim_feedforward=hidden_size * 2,
                batch_first=True,
            )

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids.clamp(min=0, max=self.embedding.num_embeddings - 1))
            pad_mask = None
            if attention_mask is not None:
                pad_mask = attention_mask == 0  # True where padded
            h = self.layer(x, src_key_padding_mask=pad_mask)
            return _StubEncoder._Out(h)

else:  # pragma: no cover - torch-free fallback keeps the module importable

    def build_model(cfg: Any):
        """Build the market-supervised model (requires torch)."""
        _require_torch("build_model")


def compute_losses(outputs: Any, batch: Any, cfg: Any) -> dict:
    """Compose the five training objectives into a named loss dict.

    Objectives (Requirement 23.4): ``regression`` (MSE on standardized target),
    ``direction`` (BCE on sign), ``uncertainty`` (Gaussian NLL using predicted
    log-variance), ``temporal`` (near/far margin), and ``contrastive`` (positive
    vs negative groups from :func:`make_contrastive_groups`). Each component is
    weighted by ``model.deberta.loss_weights`` and every component plus the total
    is guarded with :func:`assert_finite`.

    This is loss-composition CODE. It is not invoked during agent execution
    (training is owner-run, Requirement 33); it requires torch to run.
    """
    _require_torch("compute_losses")

    weights = _cfg_get(cfg, "model.deberta.loss_weights", {}) or {}

    def w(name: str) -> float:
        return float(weights.get(name, 0.0)) if isinstance(weights, dict) else 0.0

    losses: dict[str, "torch.Tensor"] = {}

    preds = outputs.get("regression") if isinstance(outputs, dict) else None
    target = batch.get("target") if isinstance(batch, dict) else None
    if preds is not None and target is not None:
        assert_finite("regression_preds", preds)
        assert_finite("targets", target)
        losses["regression"] = w("regression") * F.mse_loss(preds, target)

    dir_logits = outputs.get("direction") if isinstance(outputs, dict) else None
    raw_target = batch.get("target_raw", target) if isinstance(batch, dict) else target
    if dir_logits is not None and raw_target is not None and dir_logits.dim() == 1:
        raw_target = raw_target.reshape(-1).to(dir_logits.dtype)
        # Direction is defined in raw market units. The old standardized sign
        # classified "above/below the train mean", which is not a market move.
        dir_target = (raw_target > 0).float()
        threshold = batch.get("material_threshold") if isinstance(batch, dict) else None
        if threshold is None:
            threshold = torch.full_like(
                raw_target,
                float(_cfg_get(cfg, "market_windows.material_move_bp", 0.0)),
            )
        elif not torch.is_tensor(threshold):
            threshold = torch.as_tensor(
                threshold, dtype=raw_target.dtype, device=raw_target.device
            )
        threshold = threshold.reshape(-1).to(raw_target.device, raw_target.dtype)
        if threshold.numel() == 1:
            threshold = threshold.expand_as(raw_target)
        direction_mask = raw_target.abs() >= threshold
        if isinstance(batch, dict) and batch.get("primary_present") is not None:
            direction_mask = direction_mask & batch["primary_present"].reshape(-1).bool()
        per_example = F.binary_cross_entropy_with_logits(
            dir_logits, dir_target, reduction="none"
        )
        mask_f = direction_mask.to(per_example.dtype)
        denom = mask_f.sum()
        direction_loss = (
            (per_example * mask_f).sum() / denom
            if bool(direction_mask.any())
            else dir_logits.sum() * 0.0
        )
        losses["direction"] = w("direction") * direction_loss

    log_var = outputs.get("log_variance") if isinstance(outputs, dict) else None
    if log_var is not None and preds is not None and target is not None:
        # Gaussian negative log-likelihood with predicted log-variance.
        nll = 0.5 * (torch.exp(-log_var) * (preds - target) ** 2 + log_var)
        losses["uncertainty"] = w("uncertainty") * nll.mean()

    temporal_term = outputs.get("temporal") if isinstance(outputs, dict) else None
    if temporal_term is not None:
        assert_finite("temporal_term", temporal_term)
        losses["temporal"] = w("temporal") * temporal_term.mean()

    contrastive_term = outputs.get("contrastive") if isinstance(outputs, dict) else None
    if contrastive_term is not None:
        assert_finite("contrastive_term", contrastive_term)
        losses["contrastive"] = w("contrastive") * contrastive_term.mean()

    # Sixth objective: soft-cross-entropy stance term against the market-derived
    # soft label, masked per-example (Req 6.2, 6.6, 6.7, 9.1, 9.6). Only computed
    # when the stance logits (from the head, exposed in ``outputs``), the soft
    # label (supplied by the batch), and the supervision mask are all present.
    stance_logits = outputs.get("stance") if isinstance(outputs, dict) else None
    soft_label = batch.get("stance_soft_label") if isinstance(batch, dict) else None
    # The forward pass exposes ``stance_mask`` in ``outputs``; fall back to the
    # batch-supplied mask when the model output does not carry one.
    stance_mask = outputs.get("stance_mask") if isinstance(outputs, dict) else None
    if stance_mask is None and isinstance(batch, dict):
        stance_mask = batch.get("stance_mask")
    stance_excluded_count: Optional[int] = None
    if stance_logits is not None and soft_label is not None and stance_mask is not None:
        assert_finite("stance_logits", stance_logits)
        assert_finite("stance_soft_label", soft_label)
        # Stable log-softmax: temperature-scaled, max-subtracted (Req 9.2), matching
        # the softmax used inside the head.
        temperature = float(_cfg_get(cfg, "model.deberta.stance.temperature", 1.0))
        log_probs = F.log_softmax(
            (stance_logits - stance_logits.amax(dim=-1, keepdim=True)) / temperature,
            dim=-1,
        )
        # Soft cross-entropy per example; mask out undefined labels (Req 6.7).
        per_example = -(soft_label * log_probs).sum(dim=-1)  # [B]
        mask_f = stance_mask.to(per_example.dtype)  # 1.0 where the label is defined
        # ``clamp(min=1.0)`` keeps the gradient finite even when every example is
        # masked (the masked term contributes 0 * log_probs = 0) (Req 9.6).
        denom = mask_f.sum().clamp(min=1.0)
        stance_loss = (per_example * mask_f).sum() / denom
        assert_finite("stance_loss", stance_loss)
        losses["stance"] = w("stance") * stance_loss
        # Record the count of examples excluded from the stance term so the
        # exclusion is observable at the training interface (Req 6.7).
        stance_excluded_count = int((mask_f <= 0).sum().item())

    if losses:
        # The total stays a sum of the objective keys (five, or six when the
        # stance term is present); ``stance_excluded_count`` is attached only
        # after the sum so it never counts as an objective.
        total = sum(losses.values())
        assert_finite("total_loss", total)
        losses["total"] = total
    if stance_excluded_count is not None:
        losses["stance_excluded_count"] = stance_excluded_count
    return losses


# =============================================================================
# Council loss composition (Task 7.1). Generalizes ``compute_losses`` to a bank
# of per-signal masked MSEs fused by :class:`SignalFusion`. Requires torch.
# =============================================================================

#: Enumerated missing reason recorded for an event whose ``signal_present`` row
#: is all-false (no configured signal present). Reuses the enumerated
#: ``target_construction`` reason codes rather than inventing a new taxonomy: an
#: all-missing council event is, by construction, one for which no signal
#: produced a within-window observation (Req 3.3).
try:  # pragma: no cover - import guarded so the module stays importable alone.
    from .target_construction import REASON_WINDOW_MISSING as _REASON_ALL_SIGNALS_MISSING
except Exception:  # pragma: no cover
    _REASON_ALL_SIGNALS_MISSING = "market_window_missing"


def _council_present_matrix(signal_present: Any) -> Any:
    """Return ``signal_present`` as a 2-D boolean ``[B, N]`` torch tensor."""
    if not (_HAS_TORCH and torch.is_tensor(signal_present)):
        signal_present = torch.as_tensor(signal_present)
    present = signal_present.to(torch.bool)
    if present.dim() == 1:
        present = present.unsqueeze(0)
    return present


def _append_jsonl(path: str, records: Sequence[dict]) -> None:
    """Append ``records`` as JSON lines to ``path`` (create parent dirs).

    Follows the ``_log_offending_batch`` append pattern in ``src/training.py``:
    one compact JSON object per line, parent directory created on demand, silent
    no-op when ``path`` is falsy so the loss path never crashes on an audit I/O
    problem.
    """
    if not path or not records:
        return
    import json
    import os

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, default=str) + "\n")
    except OSError:  # pragma: no cover - audit logging must never crash training
        pass


def _persist_presence_matrix(path: str, present: Any, event_ids: Any = None) -> None:
    """Persist the full per-event, per-signal presence matrix to parquet (Req 3.4).

    Writes one row per event with a boolean column per configured signal plus an
    ``event_id`` column (positional index when the batch carries no ids) and an
    ``all_absent`` flag. Silent no-op when ``path`` is falsy or pandas is
    unavailable so the loss path never crashes on an audit I/O problem.
    """
    if not path:
        return
    try:
        import os

        import pandas as pd
    except Exception:  # pragma: no cover - pandas is a project dep; guard anyway
        return

    present_arr = present.detach().cpu().numpy() if _HAS_TORCH and torch.is_tensor(present) else np.asarray(present)
    if present_arr.ndim == 1:
        present_arr = present_arr.reshape(1, -1)
    b, n = present_arr.shape

    if event_ids is None:
        ids = list(range(b))
    else:
        if _HAS_TORCH and torch.is_tensor(event_ids):
            ids = [x for x in event_ids.detach().cpu().reshape(-1).tolist()]
        elif _HAS_NUMPY and isinstance(event_ids, (np.ndarray, np.generic)):
            ids = [x for x in np.asarray(event_ids).reshape(-1).tolist()]
        else:
            ids = list(event_ids)
        if len(ids) != b:
            ids = list(range(b))

    data: dict[str, Any] = {"event_id": ids}
    # One boolean column per signal, in configured column order.
    for j in range(n):
        col = f"signal_{j}"
        data[col] = [bool(present_arr[i, j]) for i in range(b)]
    data["all_absent"] = [not bool(present_arr[i].any()) for i in range(b)]

    frame = pd.DataFrame(data)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        frame.to_parquet(path, index=False)
    except Exception:  # pragma: no cover - audit persistence must never crash
        pass


def compute_council_losses(
    outputs: Any,
    batch: Any,
    cfg: Any,
    fusion: "SignalFusion",
    log_var: Any = None,
) -> dict:
    """Compose the Council multi-signal loss from per-signal masked MSEs.

    Generalizes :func:`compute_losses`' single masked term into a bank of N
    per-signal masked MSEs, one per configured Council signal, and fuses them
    into one scalar ``L_council`` via :class:`SignalFusion` (Req 1.1, 3.1).

    For a configured signal set ``S = {s_1..s_N}`` with per-event prediction
    ``ŷ_i`` (``outputs["council"][s_i]``, ``[B]``), normalized target ``y_i``
    (column ``i`` of ``batch["signal_targets"]``, ``[B, N]``) and presence mask
    ``m_i`` (column ``i`` of ``batch["signal_present"]``, ``[B, N]`` bool), each
    signal contributes a masked MSE using the SAME divide-by-zero guard as the
    stance term in :func:`compute_losses`::

        denom_i = m_i.sum().clamp(min=1.0)
        L_i     = Σ_b m_{i,b} · (ŷ_{i,b} − y_{i,b})²  /  denom_i

    An event whose whole ``signal_present`` row is false
    (``signal_present.all(dim=1) == False`` -> no configured signal present) is
    excluded from Council supervision: its row is masked out of every ``L_i`` so
    it contributes exactly ``0`` and a clean-zero gradient, and the exclusion is
    recorded with an enumerated missing reason to
    ``outputs/audit/council_exclusions.jsonl`` (Req 3.3). An event with at least
    one present signal is retained and contributes loss only for its present
    signals (Req 3.1, 3.2). The full presence matrix is persisted to
    ``outputs/audit/council_presence.parquet`` (Req 3.4).

    Each ``L_i`` is guarded with ``assert_finite("signal_loss::"+s_i, L_i)`` and
    the fused ``L_council`` is guarded via :meth:`SignalFusion.fuse` (Req 4.2).

    Args:
        outputs: Model outputs; must carry ``outputs["council"]`` -> ``{name:
            [B] prediction}`` (as produced by :class:`CouncilHeads`).
        batch: Batch dict carrying ``signal_targets`` ``[B, N]`` float and
            ``signal_present`` ``[B, N]`` bool (Task 4.2), optionally
            ``event_id``.
        cfg: The resolved config (read for the configured signal names / order
            and the audit-artifacts directory).
        fusion: The :class:`SignalFusion` instance combining the per-signal
            losses (uncertainty or fixed mode).
        log_var: Per-signal log-variance parameters keyed by signal name
            (typically ``CouncilHeads.log_var``). Required by ``fusion`` in
            ``uncertainty`` mode; ignored in ``fixed`` mode.

    Returns:
        A dict with:

        * ``signal_losses``: ``{name: L_i}`` per-signal masked MSE tensors,
        * ``effective_weights``: ``{name: float}`` fusion weight applied per
          signal this step (Req 2.3),
        * ``council``: the fused scalar ``L_council`` (guarded finite),
        * ``total``: alias of ``council`` so the trainer can sum it uniformly,
        * ``council_excluded_count``: number of all-missing events excluded from
          Council supervision this batch (Req 3.3).

    This is loss-composition CODE; like :func:`compute_losses` it is not invoked
    during agent execution (training is owner-run, Requirement 33) and requires
    torch to run.
    """
    _require_torch("compute_council_losses")

    if not isinstance(outputs, dict) or "council" not in outputs:
        raise ValueError(
            "compute_council_losses requires outputs['council'] -> {signal: [B] pred}"
        )
    council_preds = outputs["council"]
    if not isinstance(council_preds, dict) or not council_preds:
        raise ValueError("outputs['council'] must be a non-empty {signal: [B]} mapping")

    signal_targets = batch.get("signal_targets") if isinstance(batch, dict) else None
    signal_present = batch.get("signal_present") if isinstance(batch, dict) else None
    if signal_targets is None or signal_present is None:
        raise ValueError(
            "compute_council_losses requires batch['signal_targets'] and "
            "batch['signal_present'] ([B, N] tensors from attach_council_fields)"
        )

    # Configured signal names give the canonical column order for the [B, N]
    # signal_targets / signal_present tensors and the head-bank keys (Req 1.1).
    signal_names = _council_head_signal_names(cfg)
    if not signal_names:
        # Fall back to the prediction keys when no config signal set is present
        # (keeps the function usable in isolated unit tests).
        signal_names = list(council_preds.keys())

    targets = signal_targets if torch.is_tensor(signal_targets) else torch.as_tensor(
        signal_targets, dtype=torch.float32
    )
    targets = targets.to(torch.float32)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    present = _council_present_matrix(signal_present)

    b, n = present.shape
    assert_finite("signal_targets", targets)

    # An event is excluded from Council supervision when NO configured signal is
    # present for it (signal_present.all(dim=1) == False -> the whole row is
    # false). ``keep_event`` retains rows with >= 1 present signal (Req 3.2/3.3).
    any_present = present.any(dim=1)  # [B] True where >= 1 signal present
    if not bool(any_present.any()):
        raise ValueError(
            "all configured Council signals are absent for this batch; refusing "
            "to optimize placeholders or uncertainty regularizers"
        )
    keep_event = any_present.to(torch.float32)  # [B] 1.0 retained, 0.0 excluded
    excluded_rows = (~any_present)
    council_excluded_count = int(excluded_rows.sum().item())

    signal_losses: dict[str, "torch.Tensor"] = {}
    active_signal_losses: dict[str, "torch.Tensor"] = {}
    signal_coverage: dict[str, int] = {}
    for i, name in enumerate(signal_names):
        if name not in council_preds:
            # A configured signal with no matching head prediction is skipped
            # rather than fabricated (mirrors the head-bank keying discipline).
            continue
        pred_i = council_preds[name]
        pred_i = pred_i if torch.is_tensor(pred_i) else torch.as_tensor(
            pred_i, dtype=torch.float32
        )
        pred_i = pred_i.reshape(-1)
        assert_finite("council_signal_pred::" + name, pred_i)

        y_i = targets[:, i] if i < targets.shape[1] else torch.zeros_like(pred_i)
        # Per-signal presence mask, AND-ed with the retain-event mask so an
        # all-missing (excluded) event never contributes to any signal loss.
        m_i = present[:, i].to(pred_i.dtype) * keep_event
        coverage = int(m_i.sum().item())
        signal_coverage[name] = coverage
        if coverage == 0:
            # Retain a diagnostic zero for API compatibility, but do NOT pass
            # this signal to uncertainty fusion: otherwise its log-variance
            # regularizer is optimized despite having no observed target.
            L_i = pred_i.sum() * 0.0
        else:
            denom = m_i.sum()
            sq_err = (pred_i - y_i) ** 2  # [B]
            L_i = (sq_err * m_i).sum() / denom
            active_signal_losses[name] = L_i
        assert_finite("signal_loss::" + name, L_i)
        signal_losses[name] = L_i

    if not active_signal_losses:
        raise ValueError(
            "compute_council_losses found zero observed targets across every "
            "configured Council signal"
        )

    # Fuse observed signals only. Zero-coverage heads and their log-variance
    # parameters are completely absent from this step's optimization.
    fused_loss, effective_weights = fusion.fuse(
        active_signal_losses, log_var=log_var
    )
    assert_finite("council_loss", fused_loss)

    # Composite objective: the Council is auxiliary supervision around an
    # explicit primary regression/direction task. Legacy isolated loss tests may
    # provide only ``outputs['council']``; in that case this cleanly reduces to
    # the original fused Council term.
    configured_weights = _cfg_get(cfg, "model.deberta.loss_weights", {}) or {}

    def _weight(name: str, default: float = 0.0) -> float:
        if not isinstance(configured_weights, dict):
            return default
        return float(configured_weights.get(name, default))

    weighted_council = _weight("council", 1.0) * fused_loss
    objective_losses: dict[str, "torch.Tensor"] = {
        "council_weighted": weighted_council
    }
    total_loss = weighted_council

    primary_pred = outputs.get("primary_prediction", outputs.get("regression"))
    target_scaled = batch.get("target") if isinstance(batch, dict) else None
    target_raw = batch.get("target_raw", target_scaled) if isinstance(batch, dict) else None
    if primary_pred is not None and target_scaled is not None:
        primary_pred = primary_pred.reshape(-1)
        target_scaled = target_scaled.reshape(-1).to(primary_pred.dtype)
        if isinstance(batch, dict) and batch.get("primary_present") is not None:
            primary_mask = batch["primary_present"].reshape(-1).bool()
        else:
            primary_mask = torch.ones_like(primary_pred, dtype=torch.bool)
        mask_f = primary_mask.to(primary_pred.dtype)
        primary_count = int(primary_mask.sum().item())
        if primary_count > 0:
            denom = mask_f.sum()
            regression_loss = (
                ((primary_pred - target_scaled) ** 2) * mask_f
            ).sum() / denom
            objective_losses["regression"] = _weight("regression", 1.0) * regression_loss
            total_loss = total_loss + objective_losses["regression"]

            log_variance = outputs.get(
                "primary_log_variance", outputs.get("log_variance")
            )
            if log_variance is not None and _weight("uncertainty", 0.0) != 0.0:
                log_variance = log_variance.reshape(-1)
                nll = 0.5 * (
                    torch.exp(-log_variance) * (primary_pred - target_scaled) ** 2
                    + log_variance
                )
                uncertainty_loss = (nll * mask_f).sum() / denom
                objective_losses["uncertainty"] = (
                    _weight("uncertainty", 0.0) * uncertainty_loss
                )
                total_loss = total_loss + objective_losses["uncertainty"]

            direction_logits = outputs.get("primary_direction_logits")
            if direction_logits is not None and target_raw is not None:
                raw = target_raw.reshape(-1).to(primary_pred.dtype)
                threshold = batch.get("material_threshold") if isinstance(batch, dict) else None
                if threshold is None:
                    threshold = torch.full_like(
                        raw,
                        float(_cfg_get(cfg, "market_windows.material_move_bp", 0.0)),
                    )
                elif not torch.is_tensor(threshold):
                    threshold = torch.as_tensor(
                        threshold, dtype=raw.dtype, device=raw.device
                    )
                threshold = threshold.reshape(-1).to(raw.device, raw.dtype)
                if threshold.numel() == 1:
                    threshold = threshold.expand_as(raw)
                # Class order: down=0, neutral=1, up=2. Labels are based on the
                # raw move, never on the train-mean-centered regression target.
                direction_target = torch.ones_like(raw, dtype=torch.long)
                direction_target = torch.where(
                    raw < -threshold,
                    torch.zeros_like(direction_target),
                    direction_target,
                )
                direction_target = torch.where(
                    raw > threshold,
                    torch.full_like(direction_target, 2),
                    direction_target,
                )
                per_example = F.cross_entropy(
                    direction_logits, direction_target, reduction="none"
                )
                direction_loss = (per_example * mask_f).sum() / denom
                objective_losses["direction"] = (
                    _weight("direction", 0.0) * direction_loss
                )
                total_loss = total_loss + objective_losses["direction"]

            # Continuous stance supervision flows through the categorical
            # distribution: score = P(hawkish) - P(dovish). The fallback keeps
            # isolated legacy callers compatible when they provide only
            # ``stance_score``.
            stance_score = outputs.get(
                "probability_stance_score", outputs.get("stance_score")
            )
            stance_target = batch.get("stance_target") if isinstance(batch, dict) else None
            if stance_score is not None and stance_target is not None:
                stance_target = stance_target.reshape(-1).to(stance_score.dtype)
                stance_loss = (
                    ((stance_score.reshape(-1) - stance_target) ** 2) * mask_f
                ).sum() / denom
                objective_losses["stance"] = _weight("stance", 0.0) * stance_loss
                total_loss = total_loss + objective_losses["stance"]

    for name in ("temporal", "contrastive"):
        term = outputs.get(name)
        if term is not None and _weight(name, 0.0) != 0.0:
            objective_losses[name] = _weight(name, 0.0) * term.mean()
            total_loss = total_loss + objective_losses[name]

    assert_finite("council_total_loss", total_loss)

    # --- Audit artifacts (Req 3.3, 3.4) -----------------------------------
    audit_dir = _cfg_get(cfg, "experiment.output_dir", None) or _cfg_get(
        cfg, "output_dir", None
    )
    if audit_dir:
        import os as _os

        exclusions_path = _os.path.join(audit_dir, "audit", "council_exclusions.jsonl")
        presence_path = _os.path.join(audit_dir, "audit", "council_presence.parquet")
    else:
        exclusions_path = "outputs/audit/council_exclusions.jsonl"
        presence_path = "outputs/audit/council_presence.parquet"

    event_ids = batch.get("event_id") if isinstance(batch, dict) else None

    if council_excluded_count > 0:
        # Resolve the excluded rows' event ids for the audit record.
        if event_ids is not None:
            if torch.is_tensor(event_ids):
                id_list = event_ids.detach().cpu().reshape(-1).tolist()
            elif _HAS_NUMPY and isinstance(event_ids, (np.ndarray, np.generic)):
                id_list = np.asarray(event_ids).reshape(-1).tolist()
            else:
                id_list = list(event_ids)
        else:
            id_list = list(range(b))
        records = []
        excluded_idx = excluded_rows.nonzero(as_tuple=False).reshape(-1).tolist()
        for row in excluded_idx:
            eid = id_list[row] if row < len(id_list) else row
            records.append(
                {
                    "event_id": eid,
                    "reason": _REASON_ALL_SIGNALS_MISSING,
                    "n_signals": int(n),
                    "n_present": 0,
                }
            )
        _append_jsonl(exclusions_path, records)

    # Persist the full presence matrix for every event in the batch (Req 3.4).
    _persist_presence_matrix(presence_path, present, event_ids)

    return {
        "signal_losses": signal_losses,
        "signal_coverage": signal_coverage,
        "active_signals": list(active_signal_losses),
        "effective_weights": effective_weights,
        "council": fused_loss,
        "objective_losses": objective_losses,
        "total": total_loss,
        "council_excluded_count": council_excluded_count,
    }


@dataclass
class TrainResult:
    """Outcome of a training run (populated by the owner-run trainer).

    ``status`` is one of ``"OK"`` or ``"INSUFFICIENT_SAMPLE"``; when a split/target
    has fewer than ``min_observations`` samples the trainer records
    ``INSUFFICIENT_SAMPLE`` and does not report metrics as if valid.
    """

    status: str = "OK"
    best_epoch: int = -1
    best_metric: float = float("nan")
    history: list = field(default_factory=list)
    used_amp: bool = False
    n_samples: int = 0
    # Number of real optimiser steps that landed (a run with 0 means the model
    # is still at init and any predictions are the constant-bias collapse).
    optim_steps: int = 0
    # Cumulative non-finite gradient/loss events observed during the run.
    nonfinite_events: int = 0


def train(model: Any, loaders: Any, cfg: Any, debug_log_path: str) -> "TrainResult":
    """Training interface - NOT executed by the agent (Requirement 33).

    Wires the numerical-stability policy from the design's *Training
    Numerical-Stability Design*:

    - **Two-phase precision** (17.2/17.3): the initial debug run forces FP32 with
      AMP/BF16 disabled (``use_amp=False``) regardless of the config default;
      BF16 AMP is re-enabled only after a finite-value validation pass.
    - **Gradient clipping** every step (17.3).
    - **Finite assertions** via :func:`assert_finite` on inputs, attention masks,
      targets, outputs, loss, and gradients (17.1); a non-finite detection appends
      the offending batch to ``debug_log_path`` /
      ``outputs/audit/training_nan_debug.jsonl`` (17.4).
    - **Minimum sample-size gates**: fewer than ``min_observations`` samples records
      ``status='INSUFFICIENT_SAMPLE'`` (17.5).
    - **Contrastive-group thresholds** via :func:`make_contrastive_groups`.

    Execution is owner-run (Requirement 33): the agent authored this wiring and
    verified it with a CPU smoke test, but full-scale training on the real corpus
    is run by the owner on a GPU. The loop body lives in
    :func:`src.training.run_training_loop`; this thin wrapper delegates to it so
    the documented ``train(model, loaders, cfg, debug_log_path)`` signature keeps
    working.
    """
    from .training import run_training_loop

    return run_training_loop(model, loaders, cfg, debug_log_path)
