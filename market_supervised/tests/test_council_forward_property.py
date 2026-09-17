"""Property-based test for the finite Council forward under injected non-finite inputs.

Property 9: Forward pass stays finite under injected non-finite inputs.

The Council forward pass reuses ``_encode_document`` (the base ``nan_to_num``
scrub + mask-multiply discipline over the shared encoder), then runs
:class:`CouncilHeads` and :class:`StanceScoreHead` on the ONE pooled ``[B,
hidden]`` document vector. It extends that same discipline to the two new Council
supervision paths (Requirement 4.2):

  * the batch ``signal_targets * signal_present`` tensor is scrubbed with
    ``nan_to_num`` and hard-zeroed on absent signals with a presence mask
    multiply (a clean-zero gradient), exactly like ``seg_vecs * keep`` in
    ``_encode_document``;
  * the continuous ``stance_score`` is scrubbed the same way;
  * the count of non-finite entries in the PRESENT signals is recorded on
    ``self.last_nonfinite_council`` (and the stance-score path on
    ``self.last_nonfinite_stance_score``) BEFORE scrubbing, so the training loop
    can append the offending batch to the debug log rather than crashing
    (Requirements 4.2, 4.4).

This property asserts, for any batch in which arbitrary NaN/+/-Inf values are
injected into the batch ``signal_targets`` (and, where applicable, the shared
encoder inputs / stance path):

  * EVERY forward output is finite -- the Council signal predictions, the
    continuous ``stance_score``, the shared ``document`` vector, and the scrubbed
    ``signal_targets`` (Requirement 4.2 -- the scrub + mask-multiply policy keeps
    the forward finite);
  * the batch non-finite count is RECORDED: whenever at least one NaN/+/-Inf was
    injected into a PRESENT signal cell, ``last_nonfinite_council`` is > 0 and
    equals the exact number of non-finite entries injected into present cells;
    injecting only into ABSENT cells records 0 present non-finite entries (they
    are masked out) while the forward still stays finite.

The Shared_Encoder is the pipeline's ``_StubEncoder`` (built via
``model.deberta.force_stub`` on :func:`build_model`'s resolution path), so no
DeBERTa weights and no network access are needed. ``torch`` is an optional
project dependency, so the module skips cleanly when it is unavailable.

Validates: Requirements 4.2, 4.4.

# Feature: council-of-supervision, Property 9
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.market_supervised import CouncilModel, _StubEncoder


# The stub encoder uses a 2-head TransformerEncoderLayer, so the hidden width
# must be divisible by 2 (an encoder-construction constraint, not a property
# constraint). Keep it small and even for a fast CPU forward.
_HIDDEN = 8
_VOCAB = 64

# The non-finite values injected into signal_targets / stance path. Covers NaN
# and both infinities -- "arbitrary NaN/+/-Inf" per the property statement.
_NONFINITE_VALUES = (float("nan"), float("inf"), float("-inf"))


def _council_cfg(signal_names):
    """Config selecting the CPU stub encoder + the configured Council signal set.

    ``force_stub`` + ``allow_stub_encoder`` route ``_build_encoder`` to the tiny
    random :class:`_StubEncoder` (no network, no DeBERTa weights). ``council.signals``
    drives ``CouncilHeads`` / ``StanceScoreHead`` (hence ``N``) and the canonical
    ``[B, N]`` column order. Uncertainty fusion is the default headline mode.
    """
    return {
        "model": {
            "deberta": {
                "encoder_mode": "hierarchical",
                "force_stub": True,
                "allow_stub_encoder": True,
                "hidden_size": _HIDDEN,
                "stub_vocab_size": _VOCAB,
                "min_chunk_tokens": 1,
            }
        },
        "council": {
            "fusion_mode": "uncertainty",
            "log_var_min": -8.0,
            "log_var_max": 2.0,
            "signals": [
                {"name": name, "source": "src", "identifier": name}
                for name in signal_names
            ],
        },
    }


def _build_council_model(signal_names):
    """Build a :class:`CouncilModel` on the forced CPU stub-encoder path."""
    cfg = _council_cfg(signal_names)
    encoder = _StubEncoder(vocab_size=_VOCAB, hidden_size=_HIDDEN)
    model = CouncilModel(cfg, encoder=encoder)
    model.eval()  # deterministic forward; the property is about numerical scrub
    return model


@st.composite
def _nonfinite_forward_case(draw):
    """Draw a batch + an arbitrary NaN/+/-Inf injection pattern for the forward.

    Returns a dict describing:
      * ``B`` events, each with ``S >= 1`` hierarchical segments of ``L >= 1``
        tokens; ``segment_mask`` marks which segments are present (>= 1 present
        per event so every event encodes to a real doc vector);
      * ``N >= 2`` configured signals (Req 1.2 -- two or more heterogeneous
        signals), a ``[B, N]`` boolean ``signal_present`` grid, and finite base
        ``signal_targets``;
      * an injection map placing an arbitrary non-finite value at a chosen set of
        ``(row, col)`` cells of ``signal_targets`` -- some in PRESENT cells, some
        (optionally) in ABSENT cells -- so both the "recorded > 0 for present"
        and "masked absent contributes 0" branches are exercised across examples.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    n_segments = draw(st.integers(min_value=1, max_value=4))
    seq_len = draw(st.integers(min_value=1, max_value=5))
    n_signals = draw(st.integers(min_value=2, max_value=5))
    signal_names = [f"sig_{i}" for i in range(n_signals)]

    # Token ids in-range for the stub embedding.
    input_ids = [
        [
            [draw(st.integers(min_value=0, max_value=_VOCAB - 1)) for _ in range(seq_len)]
            for _ in range(n_segments)
        ]
        for _ in range(batch_size)
    ]

    # Per-event segment presence: at least one present segment per event so the
    # doc vector is real; the attention/token mask matches the segment mask.
    segment_present = []
    for _ in range(batch_size):
        flags = [draw(st.booleans()) for _ in range(n_segments)]
        if not any(flags):
            flags[draw(st.integers(min_value=0, max_value=n_segments - 1))] = True
        segment_present.append(flags)

    # Per-signal presence grid: each event turns on an arbitrary subset. Ensure
    # at least one event has at least one PRESENT signal so a present injection
    # is possible; individual rows may be all-absent (masked-out) to exercise the
    # "absent contributes 0 and stays finite" branch.
    signal_present = [
        [draw(st.booleans()) for _ in range(n_signals)] for _ in range(batch_size)
    ]
    if not any(any(row) for row in signal_present):
        r = draw(st.integers(min_value=0, max_value=batch_size - 1))
        c = draw(st.integers(min_value=0, max_value=n_signals - 1))
        signal_present[r][c] = True

    # Finite base signal targets.
    finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
    signal_targets = [
        [draw(finite) for _ in range(n_signals)] for _ in range(batch_size)
    ]

    # Injection: choose a non-empty set of cells and drop an arbitrary non-finite
    # value into each. Bias toward including present cells so the "recorded > 0"
    # assertion is meaningfully exercised.
    all_cells = [(r, c) for r in range(batch_size) for c in range(n_signals)]
    present_cells = [
        (r, c) for (r, c) in all_cells if signal_present[r][c]
    ]
    # Always inject into at least one present cell when any exists.
    injected = {}
    if present_cells:
        chosen_present = draw(
            st.lists(st.sampled_from(present_cells), min_size=1, max_size=len(present_cells), unique=True)
        )
        for cell in chosen_present:
            injected[cell] = draw(st.sampled_from(_NONFINITE_VALUES))
    # Optionally also inject into absent cells (masked out; must not be counted).
    absent_cells = [(r, c) for (r, c) in all_cells if not signal_present[r][c]]
    if absent_cells and draw(st.booleans()):
        chosen_absent = draw(
            st.lists(st.sampled_from(absent_cells), min_size=1, max_size=len(absent_cells), unique=True)
        )
        for cell in chosen_absent:
            injected[cell] = draw(st.sampled_from(_NONFINITE_VALUES))

    return {
        "input_ids": input_ids,
        "segment_present": segment_present,
        "signal_present": signal_present,
        "signal_targets": signal_targets,
        "injected": injected,
        "signal_names": signal_names,
    }


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=_nonfinite_forward_case())
def test_council_forward_finite_under_injected_nonfinite(case):
    # Feature: council-of-supervision, Property 9
    signal_names = case["signal_names"]
    n_signals = len(signal_names)
    input_ids_list = case["input_ids"]
    batch_size = len(input_ids_list)

    model = _build_council_model(signal_names)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)  # [B, S, L]
    segment_mask = torch.tensor(case["segment_present"], dtype=torch.long)  # [B, S]
    # Attention over tokens follows the segment presence (present segments attend
    # all tokens; absent segments attend none). The encoder path guards all-pad
    # rows internally so the forward stays finite regardless.
    b, s, length = input_ids.shape
    attention_mask = segment_mask.unsqueeze(-1).expand(b, s, length).to(torch.long)

    signal_present = torch.tensor(case["signal_present"], dtype=torch.bool)  # [B, N]

    # Build signal_targets and inject the arbitrary non-finite values.
    signal_targets = torch.tensor(case["signal_targets"], dtype=torch.float32)  # [B, N]
    for (r, c), val in case["injected"].items():
        signal_targets[r, c] = val

    # The exact number of non-finite entries injected into PRESENT cells -- the
    # value the forward must record on last_nonfinite_council (Req 4.2, 4.4).
    expected_present_nonfinite = sum(
        1
        for (r, c) in case["injected"]
        if bool(signal_present[r, c].item())
    )

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_mask": segment_mask,
        "signal_targets": signal_targets,
        "signal_present": signal_present,
    }

    with torch.no_grad():
        outputs = model(batch)

    # ---- (Req 4.2) EVERY forward output is finite. --------------------------
    council_preds = outputs["council"]
    assert set(council_preds.keys()) == set(signal_names)
    for name in signal_names:
        pred = council_preds[name]
        assert tuple(pred.shape) == (batch_size,)
        assert bool(torch.isfinite(pred).all()), f"non-finite council pred for {name}"

    stance_score = outputs["stance_score"]
    assert tuple(stance_score.shape) == (batch_size,)
    assert bool(torch.isfinite(stance_score).all()), "non-finite stance_score"
    assert bool((stance_score >= -1.0).all() and (stance_score <= 1.0).all())

    # Council class order is explicit and the continuous score is downstream of
    # the categorical distribution, never a disconnected scalar prediction.
    assert outputs["stance_class_order"] == ("dovish", "neutral", "hawkish")
    probs = outputs["stance_probs"]
    assert tuple(probs.shape) == (batch_size, 3)
    assert bool(torch.isfinite(probs).all())
    assert torch.allclose(
        probs.sum(dim=-1), torch.ones(batch_size), atol=1e-6, rtol=1e-6
    )
    assert torch.allclose(
        stance_score, probs[:, 2] - probs[:, 0], atol=1e-7, rtol=1e-7
    )
    assert torch.equal(outputs["stance_logits"], outputs["primary_direction_logits"])
    assert torch.equal(outputs["primary_direction_probs"], probs)

    doc = outputs["document"]
    assert tuple(doc.shape) == (batch_size, model.hidden_size)
    assert bool(torch.isfinite(doc).all()), "non-finite shared document vector"

    scrubbed_targets = outputs["signal_targets"]
    assert tuple(scrubbed_targets.shape) == (batch_size, n_signals)
    assert bool(torch.isfinite(scrubbed_targets).all()), "non-finite scrubbed targets"

    # Absent signals are hard-zeroed by the presence mask multiply (clean-0).
    absent = ~signal_present
    if bool(absent.any()):
        assert bool((scrubbed_targets[absent] == 0.0).all()), (
            "absent signal cells must be masked to exactly 0.0"
        )

    # ---- (Req 4.2, 4.4) The non-finite count is RECORDED. -------------------
    # last_nonfinite_council counts non-finite entries in PRESENT signals BEFORE
    # scrubbing -- exactly the number injected into present cells.
    assert model.last_nonfinite_council == expected_present_nonfinite, (
        f"recorded {model.last_nonfinite_council} present non-finite entries, "
        f"expected {expected_present_nonfinite}"
    )
    # Whenever a NaN/+/-Inf was injected into a present signal, the count is > 0.
    if expected_present_nonfinite > 0:
        assert model.last_nonfinite_council > 0

    # The recorded count is always a non-negative integer (a valid batch record).
    assert isinstance(model.last_nonfinite_council, int)
    assert model.last_nonfinite_council >= 0
    assert isinstance(model.last_nonfinite_stance_score, int)
    assert model.last_nonfinite_stance_score >= 0
