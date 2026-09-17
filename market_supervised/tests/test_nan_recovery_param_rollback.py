"""Regression tests for the FP32 AdamW numerical contract.

The failed DeBERTa run reached its first optimizer boundary with finite loss and
finite gradients, then produced non-finite parameters.  FP16 AdamW with the
usual ``eps=1e-8`` reproduces that behavior because the epsilon rounds to zero.
The production loop now keeps parameters and moments in FP32, uses
``eps>=1e-6`` with scalar AdamW, and fails closed if a post-step invariant is
ever violated.  It intentionally does not clone the full model and optimizer
state before every healthy update.
"""

from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

import src.training as training


def _base_cfg() -> dict:
    return {"model": {"deberta": {}}}


def _build_cpu_run_inputs(tmp_path):
    cfg = training.smoke_config(_base_cfg())
    cfg["model"]["deberta"]["max_epochs"] = 1
    cfg["model"]["deberta"]["gradient_accumulation_steps"] = 1
    task_path = training._synthetic_task_parquet(str(tmp_path / "task.parquet"))
    tokenizer = training.build_tokenizer(cfg)
    loaders = training.build_dataloaders(task_path, tokenizer, cfg)
    model = training.build_model(cfg).to(torch.device("cpu"))
    dbg = str(tmp_path / "audit" / "nan_debug.jsonl")
    return cfg, loaders, model, dbg


def _read_log_records(debug_log_path: str) -> list:
    if not os.path.exists(debug_log_path):
        return []
    records = []
    with open(debug_log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def test_post_step_parameter_corruption_fails_closed_with_diagnostics(
    tmp_path, monkeypatch
):
    """An irreversible post-step mutation aborts instead of retrying bad state."""
    torch.manual_seed(0)
    cfg, loaders, model, dbg = _build_cpu_run_inputs(tmp_path)

    real_adamw = torch.optim.AdamW
    anchor = next(p for p in model.parameters() if p.requires_grad)

    class _CorruptingAdamW(real_adamw):  # type: ignore[misc, valid-type]
        _calls = 0

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            type(self)._calls += 1
            if type(self)._calls == 1:
                with torch.no_grad():
                    anchor.view(-1)[0] = float("inf")
            return result

    monkeypatch.setattr(torch.optim, "AdamW", _CorruptingAdamW)

    result = training.run_training_loop(model, loaders, cfg, dbg, resume=False)

    assert result.status == "NAN_ABORTED"
    assert result.optim_steps == 0
    records = _read_log_records(dbg)
    parameter_records = [
        record
        for record in records
        if record.get("stage") == "parameters"
        and record.get("reason") == "non_finite_parameters_after_step"
    ]
    assert parameter_records
    assert parameter_records[0].get("parameter_name")
    assert parameter_records[0].get("loss_component")
    assert any(
        record.get("stage") == "nan_recovery"
        and record.get("reason") == "post_step_state_corrupted"
        for record in records
    )


def test_post_step_optimizer_state_corruption_fails_closed(
    tmp_path, monkeypatch
):
    """The first Adam moment audit catches poisoned state with finite weights."""
    torch.manual_seed(0)
    cfg, loaders, model, dbg = _build_cpu_run_inputs(tmp_path)

    real_adamw = torch.optim.AdamW

    class _StateCorruptingAdamW(real_adamw):  # type: ignore[misc, valid-type]
        _calls = 0

        def step(self, *args, **kwargs):
            result = super().step(*args, **kwargs)
            type(self)._calls += 1
            if type(self)._calls == 1:
                for state in self.state.values():
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].view(-1)[0] = float("inf")
                        break
            return result

    monkeypatch.setattr(torch.optim, "AdamW", _StateCorruptingAdamW)

    result = training.run_training_loop(model, loaders, cfg, dbg, resume=False)

    assert result.status == "NAN_ABORTED"
    assert result.optim_steps == 0
    assert all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
    records = _read_log_records(dbg)
    state_records = [
        record for record in records if record.get("stage") == "optimizer_state"
    ]
    assert state_records
    assert state_records[0]["reason"] == "non_finite_optimizer_state_after_step"
    assert state_records[0]["optimizer_state_key"] == "exp_avg_sq"


def test_half_input_model_is_promoted_and_healthy_adamw_state_is_fp32(
    tmp_path, monkeypatch
):
    """The loop converts half checkpoints before constructing safe AdamW."""
    torch.manual_seed(0)
    cfg, loaders, model, dbg = _build_cpu_run_inputs(tmp_path)
    model = model.half()
    assert any(parameter.dtype == torch.float16 for parameter in model.parameters())

    real_adamw = torch.optim.AdamW
    created = []
    captured_kwargs = {}

    class _CapturingAdamW(real_adamw):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(torch.optim, "AdamW", _CapturingAdamW)

    result = training.run_training_loop(model, loaders, cfg, dbg, resume=False)

    assert result.status == "OK", result.status
    assert captured_kwargs["eps"] >= 1.0e-6
    assert captured_kwargs["foreach"] is False
    assert captured_kwargs["fused"] is False
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert created
    for state in created[0].state.values():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state:
                assert state[key].dtype == torch.float32
                assert bool(torch.isfinite(state[key]).all())
    records = _read_log_records(dbg)
    assert not [
        record
        for record in records
        if record.get("stage") in {"parameters", "optimizer_state"}
    ]
