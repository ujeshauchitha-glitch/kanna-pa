"""Stance inference on new / unseen central-bank communication.

This is the payoff of the market-supervised framework: given a trained model
checkpoint (and the train-fit target scaler), infer the IMPLICIT monetary-policy
stance latent in an arbitrary speech / statement - the model's prediction of how
the market would have moved in response to it.

``StanceInferencer`` loads a checkpoint produced by
:func:`src.training.train_target` (``best.pt``) plus the saved scaler and exposes:

* :meth:`predict` - stance score in the target's natural units (e.g. bp of OIS
  change), the implied direction (hawkish>0 / dovish<0 for a rate target), and a
  model-derived uncertainty (from the predicted log-variance head).
* :meth:`stance_vector` - the pooled document latent (the "stance-space"
  embedding) for visualization / clustering.

Everything torch is resolved at run time; the module stays importable without
torch so the pipeline can be inspected in a torch-free environment.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .market_supervised import (
    COUNCIL_STANCE_CLASS_ORDER,
    CouncilModel,
    _cfg_get,
    build_model,
)

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


@dataclass
class StancePrediction:
    """One document's inferred monetary-policy stance.

    Attributes:
        text_preview: The first ~120 chars of the input (for logging).
        stance_score: Predicted market response in the target's natural units
            (inverse-scaled when a scaler is supplied). For a rate target a
            positive score is a yields-up reaction and negative is yields-down.
        stance_scaled: The corresponding regression output in standardized space.
        direction: Backward-compatible market direction obtained by thresholding
            ``stance_score``: ``"hawkish"``, ``"dovish"``, or ``"neutral"``.
        direction_prob: Probability of an up/hawkish class. For Council models
            this is the three-way hawkish softmax component; for binary models it
            is the sigmoid direction probability.
        uncertainty_std: Model-derived predictive std-dev from the log-variance
            head (standardized space), or ``nan`` when the head is absent.
        dovish_probability, neutral_probability, hawkish_probability: Council
            three-way stance probabilities in explicit Dovish/Neutral/Hawkish
            order, or ``nan`` for a binary non-Council model.
        probability_score: Canonical dimensionless continuous stance,
            ``P(hawkish) - P(dovish)``, in ``[-1, 1]`` when Council probabilities
            are available.
        stance_dominant: Argmax of the three Council probabilities; falls back to
            ``direction`` for a non-Council model.
        market_prediction, market_prediction_scaled: Explicit aliases for the
            backward-compatible natural-unit/standardized regression pair.
    """

    text_preview: str = ""
    stance_score: float = float("nan")
    stance_scaled: float = float("nan")
    direction: str = "neutral"
    direction_prob: float = float("nan")
    uncertainty_std: float = float("nan")
    dovish_probability: float = float("nan")
    neutral_probability: float = float("nan")
    hawkish_probability: float = float("nan")
    probability_score: float = float("nan")
    market_prediction: float = float("nan")
    market_prediction_scaled: float = float("nan")
    stance_dominant: str = "neutral"


class StanceInferencer:
    """Load a trained checkpoint and infer stance for new text.

    Args:
        cfg: The resolved configuration used to build the model architecture.
        checkpoint_path: Path to a ``best.pt`` / ``latest.pt`` state dict (a raw
            ``state_dict`` or a dict with a ``"model"`` key, as saved by the
            training loop). When ``None`` an untrained model is built (useful only
            for smoke tests).
        scaler: ``{"mean","std"}`` train-fit scaler to inverse-transform the
            standardized prediction back to target units. Optional.
        target_name: Name of the target the checkpoint was trained on (metadata).
    """

    def __init__(self, cfg: Any, checkpoint_path: Optional[str] = None,
                 scaler: Optional[dict] = None, target_name: Optional[str] = None):
        if not _HAS_TORCH:
            raise RuntimeError("StanceInferencer requires PyTorch.")
        from .training import build_tokenizer, _segment_text  # local import

        self.cfg = cfg
        self.target_name = target_name
        self.scaler = scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = _cfg_get(cfg, "model.deberta.encoder_mode", "hierarchical")
        self.max_len = int(_cfg_get(cfg, "model.deberta.chunk_size", 512))
        self.max_chunks = int(_cfg_get(cfg, "model.deberta.max_chunks", 4))
        self.overlap = int(_cfg_get(cfg, "model.deberta.chunk_overlap", 64))
        self.chunk_words = max(16, int(self.max_len * 0.75))
        self.overlap_words = max(0, int(self.overlap * 0.75))
        self._segment_text = _segment_text
        self.material_bp = float(_cfg_get(cfg, "market_windows.material_move_bp", 0.0))

        self.tokenizer = build_tokenizer(cfg)
        self.model = build_model(cfg).to(self.device)
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        self.model.eval()

    # ------------------------------------------------------------------
    @classmethod
    def from_target_dir(cls, cfg: Any, output_dir: str, target_name: str,
                        scaler: Optional[dict] = None) -> "StanceInferencer":
        """Build an inferencer from a ``train_target`` output directory.

        Loads ``<output_dir>/models/checkpoints/<target_name>/best.pt`` and, when
        no ``scaler`` is passed, the saved scaler at
        ``<output_dir>/models/target_scaler_<target_name>.json`` if present.
        """
        ckpt = os.path.join(output_dir, "models", "checkpoints", target_name, "best.pt")
        if scaler is None:
            safe_target = "".join(
                char if char.isalnum() or char in ("-", "_") else "_"
                for char in str(target_name)
            )
            for cand in (
                os.path.join(output_dir, "models", "scalers", f"{safe_target}.json"),
                os.path.join(output_dir, "models", f"target_scaler_{target_name}.json"),
                os.path.join(output_dir, "models", "target_scaler.json"),
            ):
                if os.path.exists(cand):
                    with open(cand, encoding="utf-8") as fh:
                        scaler = json.load(fh)
                    break
        return cls(cfg, checkpoint_path=ckpt if os.path.exists(ckpt) else None,
                   scaler=scaler, target_name=target_name)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a schema-compatible state dict without silent legacy migration.

        Historical Council checkpoints used an untrained one-logit direction
        layer and cannot represent a calibrated neutral class. They are rejected
        explicitly instead of being reshaped into a misleading three-way model.
        Current raw state dicts and wrapped training checkpoints remain accepted.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if not hasattr(state, "items"):
            raise TypeError("checkpoint state must be a tensor mapping")

        for name, value in state.items():
            if (
                torch.is_tensor(value)
                and value.is_floating_point()
                and not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"checkpoint contains non-finite tensor: {name}")

        if isinstance(self.model, CouncilModel):
            expected_weight = self.model.direction_head.weight
            found_weight = state.get("direction_head.weight")
            found_bias = state.get("direction_head.bias")
            found_shape = tuple(found_weight.shape) if torch.is_tensor(found_weight) else None
            found_bias_shape = tuple(found_bias.shape) if torch.is_tensor(found_bias) else None
            expected_shape = tuple(expected_weight.shape)
            expected_bias_shape = tuple(self.model.direction_head.bias.shape)
            if found_shape != expected_shape or found_bias_shape != expected_bias_shape:
                raise ValueError(
                    "legacy/incompatible Council checkpoint: the current model "
                    "requires a three-logit direction head in "
                    f"{COUNCIL_STANCE_CLASS_ORDER} order with shapes "
                    f"{expected_shape}/{expected_bias_shape}, found "
                    f"{found_shape}/{found_bias_shape}. Retrain or use an "
                    "explicit legacy architecture; a one-logit head cannot be "
                    "safely migrated to a calibrated neutral class."
                )
            contract = ckpt.get("checkpoint_contract") if isinstance(ckpt, dict) else None
            if isinstance(contract, dict):
                family = contract.get("model_family")
                semantics = contract.get("direction_semantics")
                class_order = contract.get("stance_class_order")
                if family not in (None, "council_3way"):
                    raise ValueError(
                        f"checkpoint model_family={family!r} is not council_3way"
                    )
                if semantics not in (None, "dovish_neutral_hawkish_v1"):
                    raise ValueError(
                        f"unsupported Council direction semantics: {semantics!r}"
                    )
                if class_order not in (None, list(COUNCIL_STANCE_CLASS_ORDER)):
                    raise ValueError(
                        f"unsupported Council class order: {class_order!r}"
                    )

        try:
            self.model.load_state_dict(state)
        except RuntimeError as exc:
            raise ValueError(
                "checkpoint schema is incompatible with the resolved model; "
                "ordinary inference requires an exact current-schema checkpoint"
            ) from exc

    # ------------------------------------------------------------------
    def _segments(self, text: str) -> list:
        """Segment one document EXACTLY as :meth:`_encode` does (hierarchical).

        Returns the ordered list of segment texts the shared encoder sees, so an
        attention weight at index ``i`` (from ``_AttentionPool``) attributes to
        ``segments[i]``. In flat mode there is a single (truncated) segment.
        """
        if self.mode == "flat":
            return [text or ""]
        return self._segment_text(text or "", self.chunk_words, self.max_chunks,
                                  self.overlap_words)

    def _encode(self, text: str) -> dict:
        """Tokenize one document into a single-item batch on the model device."""
        def enc_one(span: str):
            e = self.tokenizer(span, max_length=self.max_len, truncation=True,
                               padding="max_length")
            return e["input_ids"], e["attention_mask"]

        if self.mode == "flat":
            ids, mask = enc_one(text or "")
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long, device=self.device),
                "attention_mask": torch.tensor([mask], dtype=torch.long, device=self.device),
            }

        segs = self._segments(text)
        seg_ids, seg_masks = [], []
        for s in segs:
            ids, mask = enc_one(s)
            seg_ids.append(ids)
            seg_masks.append(mask)
        present = len(seg_ids)
        pad_id = getattr(self.tokenizer, "pad_id", 0)
        while len(seg_ids) < self.max_chunks:
            seg_ids.append([pad_id] * self.max_len)
            seg_masks.append([0] * self.max_len)
        segment_mask = [1] * present + [0] * (self.max_chunks - present)
        return {
            "input_ids": torch.tensor([seg_ids[: self.max_chunks]], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([seg_masks[: self.max_chunks]], dtype=torch.long, device=self.device),
            "segment_mask": torch.tensor([segment_mask], dtype=torch.long, device=self.device),
        }

    def _inverse_scale(self, scaled: float) -> float:
        if self.scaler is None:
            return scaled
        mean = float(self.scaler.get("mean", 0.0))
        std = float(self.scaler.get("std", 1.0)) or 1.0
        return scaled * std + mean

    def _direction_label(self, score: float) -> str:
        thr = self.material_bp if self.material_bp and self.material_bp > 0 else 0.0
        if score > thr:
            return "hawkish"
        if score < -thr:
            return "dovish"
        return "neutral"

    def predict(self, text: str) -> "StancePrediction":
        """Infer categorical stance first, then its continuous probability score."""
        batch = self._encode(text)
        with torch.no_grad():
            out = self.model(batch)
        scaled = float(out["regression"].detach().float().cpu().reshape(-1)[0])
        market_prediction = self._inverse_scale(scaled)

        dovish = neutral = hawkish = probability_score = float("nan")
        direction = self._direction_label(market_prediction)
        stance_dominant = direction
        direction_prob = float("nan")
        direction_tensor = out.get("primary_direction_logits", out.get("direction"))
        if torch.is_tensor(direction_tensor):
            logits = direction_tensor.detach().float()
            if logits.ndim >= 2 and logits.shape[-1] == 3:
                probs_tensor = out.get("primary_direction_probs")
                if not torch.is_tensor(probs_tensor):
                    probs_tensor = torch.softmax(logits, dim=-1)
                probs = probs_tensor.detach().float().cpu().reshape(-1, 3)[0]
                dovish, neutral, hawkish = (float(value) for value in probs)
                probability_score = hawkish - dovish
                dominant_index = int(torch.argmax(probs).item())
                stance_dominant = COUNCIL_STANCE_CLASS_ORDER[dominant_index]
                direction_prob = hawkish  # legacy field means P(up/hawkish)
            else:
                # Legacy non-Council binary direction head.
                direction_prob = float(
                    torch.sigmoid(logits).cpu().reshape(-1)[0]
                )

        unc = float("nan")
        if "log_variance" in out:
            log_var = float(out["log_variance"].detach().float().cpu().reshape(-1)[0])
            unc = float(math.exp(0.5 * log_var))

        # Keep the historical regression pair unit-consistent. The canonical
        # Council continuous stance is exposed separately as probability_score;
        # it must never masquerade as a basis-point/return prediction.
        stance_score = market_prediction
        return StancePrediction(
            text_preview=(text or "")[:120],
            stance_score=stance_score,
            stance_scaled=scaled,
            direction=direction,
            direction_prob=direction_prob,
            uncertainty_std=unc,
            dovish_probability=dovish,
            neutral_probability=neutral,
            hawkish_probability=hawkish,
            probability_score=probability_score,
            stance_dominant=stance_dominant,
            market_prediction=market_prediction,
            market_prediction_scaled=scaled,
        )

    def predict_batch(self, texts) -> list:
        """Infer stance for a list of documents (returns list of predictions)."""
        return [self.predict(t) for t in texts]

    def stance_vector(self, text: str) -> "np.ndarray":
        """Return the pooled document latent (stance-space embedding)."""
        batch = self._encode(text)
        with torch.no_grad():
            out = self.model(batch)
        return out["document"].detach().float().cpu().numpy().reshape(-1)

    def segment_attributions(self, text: str, top_k: Optional[int] = None) -> list:
        """Attention-based per-segment attribution for one document (Req 7.4).

        Runs the SAME segmentation + shared-encoder + ``_AttentionPool`` path as
        :meth:`predict` (via ``_encode_document(..., return_attention=True)``) and
        pairs each PRESENT segment's text with the normalized attention weight the
        pool assigned it. The weights over present segments sum to 1 (that is the
        pool's contract), so a higher weight means the segment contributed more to
        the pooled document vector the ``Stance_Score`` head reads.

        Args:
            text: The raw document text.
            top_k: When given, return only the ``top_k`` highest-weighted present
                segments (sorted by descending weight). ``None`` returns every
                present segment in descending-weight order.

        Returns:
            A list of ``{"segment_index", "segment_text", "attention_weight"}``
            dicts, sorted by descending ``attention_weight``.
        """
        segments = self._segments(text)
        batch = self._encode(text)
        with torch.no_grad():
            _doc, weights = self.model._encode_document(
                batch["input_ids"],
                batch["attention_mask"],
                batch.get("segment_mask"),
                return_attention=True,
            )
        # weights: [1, S] -> [S]; keep only the PRESENT segments (a padded/absent
        # slot carries no real text and the pool's mask drives its weight to ~0).
        w = weights.detach().float().cpu().reshape(-1).tolist()
        present = len(segments)
        rows = [
            {
                "segment_index": i,
                "segment_text": segments[i],
                "attention_weight": float(w[i]) if i < len(w) else 0.0,
            }
            for i in range(present)
        ]
        rows.sort(key=lambda r: r["attention_weight"], reverse=True)
        if top_k is not None:
            rows = rows[: max(0, int(top_k))]
        return rows


class StanceSpace:
    """Interpretable continuous Stance_Space over an ordered Event timeline (Req 7).

    Wraps a :class:`StanceInferencer` and turns per-event documents into the two
    interpretability artifacts the design's *Interpretable Continuous Stance
    Space* section specifies:

    * :meth:`series` (Req 7.3) - the event-ordered ``(event_date, stance_score)``
      sequence written to ``outputs/results/stance_series.csv``.
    * :meth:`attributions` (Req 7.4) - the top-k contributing communication
      segments per event (segment text + ``_AttentionPool`` attention weight)
      written to ``outputs/results/stance_attributions.jsonl``.

    Args:
        inferencer: A ready :class:`StanceInferencer` (a loaded checkpoint).
        output_dir: Base run directory; artifacts land under
            ``<output_dir>/results/``. Defaults to ``"outputs"`` so the paths
            match the design (``outputs/results/...``).
    """

    def __init__(self, inferencer: "StanceInferencer", output_dir: str = "outputs"):
        self.inferencer = inferencer
        self.output_dir = output_dir

    # ------------------------------------------------------------------
    @staticmethod
    def _results_path(output_dir: str, filename: str) -> str:
        return os.path.join(output_dir, "results", filename)

    def _order_events(self, events):
        """Return ``[(event_date, text), ...]`` sorted by event date (Req 7.3).

        ``events`` may be a mapping ``{event_date: text}`` or an iterable of
        ``(event_date, text)`` pairs. Ordering is chronological by the (stringly
        comparable ISO) event date so the emitted series is a time-ordered
        sequence over the Event timeline.
        """
        if isinstance(events, dict):
            items = list(events.items())
        else:
            items = [(d, t) for d, t in events]
        return sorted(items, key=lambda pair: pair[0])

    def series(
        self,
        events,
        output_path: Optional[str] = None,
        include_probabilities: bool = False,
    ) -> str:
        """Emit the event-ordered ``(event_date, stance_score)`` series (Req 7.3).

        Runs the inferencer's :meth:`~StanceInferencer.predict` on each event's
        document, in chronological event order, and writes the time-ordered
        ``event_date,stance_score`` rows to ``outputs/results/stance_series.csv``
        (or ``output_path`` when supplied).

        Args:
            events: ``{event_date: text}`` mapping or iterable of
                ``(event_date, text)`` pairs.
            output_path: Optional override for the CSV destination.

        Returns:
            The path the CSV was written to.
        """
        import csv

        ordered = self._order_events(events)
        path = output_path or self._results_path(self.output_dir, "stance_series.csv")
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            if include_probabilities:
                writer.writerow([
                    "event_date",
                    "stance_score",
                    "probability_score",
                    "stance_prob_dovish",
                    "stance_prob_neutral",
                    "stance_prob_hawkish",
                    "stance_dominant",
                ])
            else:
                # Preserve the historical exact two-column contract by default.
                writer.writerow(["event_date", "stance_score"])
            for event_date, text in ordered:
                pred = self.inferencer.predict(text)
                if include_probabilities:
                    writer.writerow([
                        event_date,
                        pred.stance_score,
                        pred.probability_score,
                        pred.dovish_probability,
                        pred.neutral_probability,
                        pred.hawkish_probability,
                        pred.stance_dominant,
                    ])
                else:
                    writer.writerow([event_date, pred.stance_score])
        return path

    def attributions(self, events, top_k: int = 5,
                     output_path: Optional[str] = None) -> str:
        """Export top-k contributing segments per event (Req 7.4).

        For each event (in chronological order) records the ``stance_score`` and
        the ``top_k`` highest-weighted communication segments -- each segment's
        text paired with its ``_AttentionPool`` attention weight -- as one JSON
        object per line in ``outputs/results/stance_attributions.jsonl`` (or
        ``output_path`` when supplied).

        Args:
            events: ``{event_date: text}`` mapping or iterable of
                ``(event_date, text)`` pairs.
            top_k: Number of highest-weighted segments to record per event.
            output_path: Optional override for the JSONL destination.

        Returns:
            The path the JSONL was written to.
        """
        ordered = self._order_events(events)
        path = output_path or self._results_path(
            self.output_dir, "stance_attributions.jsonl"
        )
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            for event_date, text in ordered:
                pred = self.inferencer.predict(text)
                segments = self.inferencer.segment_attributions(text, top_k=top_k)
                record = {
                    "event_date": event_date,
                    "stance_score": pred.stance_score,
                    "top_segments": segments,
                }
                fh.write(json.dumps(record, default=str) + "\n")
        return path
