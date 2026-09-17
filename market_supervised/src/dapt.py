"""Domain-adaptive pre-training (DAPT) over the full central-bank corpus.

DAPT continues masked-language-model pretraining of the base encoder on the ECB
speech corpus so the encoder's representations are adapted to central-bank
language before the market-supervised fine-tuning stage. ``build_dapt_corpus``
collects every document (including pre-2004 events that carry no supervised
target); ``run_dapt`` runs the MLM objective and saves the adapted encoder.

Everything requiring torch/transformers is resolved at RUN time; the module
stays importable without them and ``run_dapt`` degrades to a recorded no-op when
those libraries (or the base weights) are unavailable, so the pipeline never
fabricates a trained encoder.

Satisfies: 23 (support).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd

# Candidate text columns in the master event table, in priority order. The full
# DAPT corpus includes every event (including pre-2004) regardless of whether it
# carries a supervised target.
_TEXT_COLUMN_CANDIDATES: tuple[str, ...] = ("text", "document_text", "raw_text", "body")

# Default location (relative to the outputs dir) of the DAPT corpus audit that
# records which test-split documents were excluded from pretraining (Req 5.5).
_DAPT_CORPUS_AUDIT_RELPATH: tuple[str, ...] = ("audit", "dapt_corpus_audit.json")


def build_dapt_corpus(master: "pd.DataFrame") -> "Iterable[str]":
    """Build the full DAPT corpus (including pre-2004 events).

    Yields the non-empty document text of every row in the master event table,
    with no date filtering, so domain-adaptive pre-training sees the entire
    corpus. Rows whose text is missing or blank are skipped.

    Args:
        master: The master event table.

    Returns:
        A list of document strings (materialised so it can be reused/sized).

    Raises:
        ValueError: If no recognised text column is present in ``master``.
    """
    text_col = next((c for c in _TEXT_COLUMN_CANDIDATES if c in master.columns), None)
    if text_col is None:
        raise ValueError(
            f"master event table has no text column; expected one of "
            f"{_TEXT_COLUMN_CANDIDATES}, got {list(master.columns)}"
        )

    corpus: list[str] = []
    for value in master[text_col].tolist():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            corpus.append(text)
    return corpus


def _load_external_corpus(path: str) -> list[str]:
    """Load documents from one external corpus file (Req 5.2).

    Supports the corpus file formats an external large-corpus release commonly
    ships in, resolved by extension:

    * ``.txt`` / ``.text``: the whole file is one document.
    * ``.jsonl`` / ``.ndjson``: one JSON record per line; a ``text``-like field is
      taken when the record is an object, otherwise the raw line.
    * ``.json``: a JSON list of strings/objects, or a single string/object.
    * ``.csv`` / ``.tsv`` / ``.parquet``: a tabular file; the first recognised
      text column (see ``_TEXT_COLUMN_CANDIDATES``) is used, falling back to the
      first column.

    Missing files are skipped (returning no documents) rather than raising, so a
    stale configured path never aborts the whole pretraining stage. Blank
    documents are dropped.
    """
    if not path or not os.path.exists(path):
        return []

    ext = os.path.splitext(path)[1].lower()
    docs: list[str] = []

    if ext in (".txt", ".text", ""):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read().strip()
        if text:
            docs.append(text)
        return docs

    if ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    docs.append(line)
                    continue
                docs.append(_document_from_record(rec))
        return [d for d in (str(x).strip() for x in docs) if d]

    if ext == ".json":
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
        records = data if isinstance(data, list) else [data]
        for rec in records:
            docs.append(_document_from_record(rec))
        return [d for d in (str(x).strip() for x in docs) if d]

    if ext in (".csv", ".tsv", ".parquet"):
        if ext == ".parquet":
            frame = pd.read_parquet(path)
        else:
            sep = "\t" if ext == ".tsv" else ","
            frame = pd.read_csv(path, sep=sep)
        text_col = next(
            (c for c in _TEXT_COLUMN_CANDIDATES if c in frame.columns), None
        )
        if text_col is None and len(frame.columns):
            text_col = frame.columns[0]
        if text_col is not None:
            for value in frame[text_col].tolist():
                if value is None:
                    continue
                s = str(value).strip()
                if s:
                    docs.append(s)
        return docs

    # Unknown extension: treat as a plain-text document.
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read().strip()
    if text:
        docs.append(text)
    return docs


def _document_from_record(rec: Any) -> str:
    """Extract a document string from a JSON record (object/str/other)."""
    if isinstance(rec, dict):
        for key in _TEXT_COLUMN_CANDIDATES:
            if key in rec and rec[key] is not None:
                return str(rec[key])
        # Fall back to the first non-empty string value.
        for value in rec.values():
            if isinstance(value, str) and value.strip():
                return value
        return ""
    return str(rec)


def build_council_dapt_corpus(
    master: "pd.DataFrame",
    cfg: Any,
    splits: "Optional[pd.DataFrame]" = None,
    output_dir: Optional[str] = None,
    audit_path: Optional[str] = None,
) -> "tuple[list[str], dict]":
    """Build the Council DAPT corpus with external docs and test-text exclusion.

    Assembles the pretraining corpus for the Council pipeline (Req 5.1, 5.2, 5.5):

    1. Builds the full ECB corpus via :func:`build_dapt_corpus` (every event's
       text, including pre-2004), keyed to each row's ``event_id`` so test-split
       documents can be excluded.
    2. Excludes every document whose ``event_id`` is in the test split. The test
       ids are taken from ``splits`` (rows where ``split == "test"``) when
       provided, otherwise from a ``split`` column already on ``master``. The
       excluded ids are recorded to the corpus audit (Req 5.5).
    3. Concatenates documents loaded from every path in
       ``model.dapt.external_corpus_paths`` onto the ECB corpus (Req 5.2).

    The audit JSON (``<output_dir>/audit/dapt_corpus_audit.json`` by default) is
    written whenever an ``output_dir`` or explicit ``audit_path`` is given, and
    records the excluded test-split ids plus corpus sizing so the leakage audit
    can cross-check that no test id survived into the corpus (Design: DAPT
    test-text exclusion).

    Args:
        master: The master event table (carries text and ``event_id``).
        cfg: The resolved configuration mapping (read for
            ``model.dapt.external_corpus_paths``).
        splits: Optional split-assigned frame (from ``temporal_split``) whose
            ``split == "test"`` rows define the ids to exclude. When ``None`` a
            ``split`` column on ``master`` is used if present.
        output_dir: Base outputs directory; the audit is written under
            ``<output_dir>/audit/dapt_corpus_audit.json`` unless ``audit_path`` is
            given.
        audit_path: Explicit path for the corpus audit JSON (overrides the
            ``output_dir`` default).

    Returns:
        A ``(corpus, audit)`` tuple: ``corpus`` is the list of document strings to
        pass to :func:`run_dapt`; ``audit`` is the recorded audit dict.
    """
    test_ids = _test_split_event_ids(master, splits)

    text_col = next((c for c in _TEXT_COLUMN_CANDIDATES if c in master.columns), None)
    if text_col is None:
        raise ValueError(
            f"master event table has no text column; expected one of "
            f"{_TEXT_COLUMN_CANDIDATES}, got {list(master.columns)}"
        )
    has_event_id = "event_id" in master.columns

    ecb_corpus: list[str] = []
    excluded_ids: list[str] = []
    for _, row in master.iterrows():
        value = row[text_col]
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if has_event_id:
            eid = row["event_id"]
            if eid is not None and str(eid) in test_ids:
                excluded_ids.append(str(eid))
                continue
        ecb_corpus.append(text)

    # Req 5.2: concatenate every configured external corpus onto the ECB corpus.
    external_paths = _cfg_get(cfg, "model.dapt.external_corpus_paths", []) or []
    external_corpus: list[str] = []
    external_sources: list[dict] = []
    for path in external_paths:
        docs = _load_external_corpus(str(path))
        external_corpus.extend(docs)
        external_sources.append({"path": str(path), "n_documents": len(docs)})

    corpus = ecb_corpus + external_corpus

    excluded_unique = sorted(set(excluded_ids))
    audit = {
        "n_ecb_documents": len(ecb_corpus),
        "n_external_documents": len(external_corpus),
        "n_total_documents": len(corpus),
        "n_excluded_test_documents": len(excluded_unique),
        "excluded_test_event_ids": excluded_unique,
        "external_corpus_sources": external_sources,
    }

    resolved_audit_path = _resolve_audit_path(output_dir, audit_path)
    if resolved_audit_path is not None:
        os.makedirs(os.path.dirname(resolved_audit_path), exist_ok=True)
        with open(resolved_audit_path, "w", encoding="utf-8") as fh:
            json.dump(audit, fh, indent=2)
        audit["audit_path"] = resolved_audit_path

    return corpus, audit


def _test_split_event_ids(
    master: "pd.DataFrame", splits: "Optional[pd.DataFrame]"
) -> set[str]:
    """Return the set of ``event_id`` values assigned to the test split.

    Prefers an explicit ``splits`` frame (its ``split == "test"`` rows); falls
    back to a ``split`` column on ``master``. Returns an empty set when neither
    source carries split membership or ``event_id``.
    """
    source = splits if splits is not None else master
    if source is None or "split" not in source.columns or "event_id" not in source.columns:
        return set()
    test_rows = source[source["split"] == "test"]
    return {
        str(eid) for eid in test_rows["event_id"].tolist() if eid is not None
    }


def _resolve_audit_path(
    output_dir: Optional[str], audit_path: Optional[str]
) -> Optional[str]:
    """Resolve the corpus-audit output path from ``audit_path`` / ``output_dir``."""
    if audit_path:
        return audit_path
    if output_dir:
        return os.path.join(output_dir, *_DAPT_CORPUS_AUDIT_RELPATH)
    return None


def run_dapt_with_external_corpus(
    master: "pd.DataFrame",
    cfg: Any,
    output_dir: Optional[str] = None,
    splits: "Optional[pd.DataFrame]" = None,
    audit_path: Optional[str] = None,
) -> "DaptResult":
    """Build the Council DAPT corpus and run domain-adaptive pretraining (Req 5).

    Thin wrapper that ties the corpus assembly to :func:`run_dapt`:

    1. Assembles the corpus via :func:`build_council_dapt_corpus` - the full ECB
       corpus with test-split documents excluded and the configured external
       corpora concatenated on, recording the excluded ids to the corpus audit
       (Req 5.1, 5.2, 5.5).
    2. Delegates to :func:`run_dapt`, which applies the ``model.dapt.min_corpus_docs``
       gate returning ``DaptResult(status="INSUFFICIENT_CORPUS")`` when the corpus
       is too small (Req 5.3) and otherwise checkpoints the adapted encoder to
       ``<output_dir>/models/dapt_encoder`` with ``dapt_meta.json`` (Req 5.4). The
       supervision stage loads that path via ``model.deberta.encoder_override``.

    Args:
        master: The master event table.
        cfg: The resolved configuration mapping.
        output_dir: Base outputs directory for the encoder checkpoint and audit.
        splits: Optional ``temporal_split`` frame defining the test-split ids to
            exclude (falls back to a ``split`` column on ``master``).
        audit_path: Explicit corpus-audit path (overrides the ``output_dir`` default).

    Returns:
        The :class:`DaptResult` from :func:`run_dapt`.
    """
    corpus, _audit = build_council_dapt_corpus(
        master, cfg, splits=splits, output_dir=output_dir, audit_path=audit_path
    )
    return run_dapt(corpus, cfg, output_dir=output_dir)


@dataclass
class DaptResult:
    """Outcome of a domain-adaptive pre-training run.

    ``status`` is one of ``"OK"``, ``"INSUFFICIENT_CORPUS"`` (fewer than
    ``min_corpus_docs`` documents), ``"SKIPPED"`` (torch/transformers/base
    weights unavailable), or ``"HEALTH_GATE_FAILED"`` (a non-finite update or
    too few validated optimizer steps). ``encoder_path`` is published only for
    ``OK``.
    """

    status: str = "OK"
    n_documents: int = 0
    n_train_docs: int = 0
    steps: int = 0
    attempted_steps: int = 0
    nonfinite_loss_steps: int = 0
    nonfinite_gradient_steps: int = 0
    nonfinite_parameter_steps: int = 0
    final_loss: float = float("nan")
    encoder_path: Optional[str] = None
    reason: Optional[str] = None
    history: list = field(default_factory=list)


def _cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    node = cfg
    for part in dotted.split("."):
        if node is None:
            return default
        if isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        else:
            node = getattr(node, part, None)
            if node is None:
                return default
    return node if node is not None else default


def run_dapt(corpus, cfg: dict, output_dir: Optional[str] = None) -> "DaptResult":
    """Run masked-LM domain-adaptive pre-training over ``corpus``.

    Continues MLM pretraining of ``model.deberta.base_model`` (or the configured
    ``encoder_override``) on the corpus with dynamic whole-token masking at
    ``model.dapt.masking_probability`` and saves the adapted encoder to
    ``<output_dir>/models/dapt_encoder``. Honours a minimum corpus size
    (``model.dapt.min_corpus_docs``) and degrades to a recorded ``SKIPPED`` result
    when torch/transformers/weights are unavailable rather than fabricating a
    trained encoder.

    Args:
        corpus: An iterable of document strings (e.g. from :func:`build_dapt_corpus`).
        cfg: The resolved configuration mapping.
        output_dir: Base outputs directory; the adapted encoder is written under
            ``<output_dir>/models/dapt_encoder``. When ``None`` the encoder is
            trained but not persisted.

    Returns:
        A :class:`DaptResult` describing the run.
    """
    docs = [str(d).strip() for d in (corpus or []) if str(d).strip()]
    n_docs = len(docs)
    min_docs = int(_cfg_get(cfg, "model.dapt.min_corpus_docs", 100))

    if n_docs < min_docs:
        return DaptResult(
            status="INSUFFICIENT_CORPUS", n_documents=n_docs,
            reason=f"corpus has {n_docs} docs (< min_corpus_docs={min_docs})",
        )

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
        )
    except Exception as exc:  # torch/transformers unavailable
        return DaptResult(
            status="SKIPPED", n_documents=n_docs,
            reason=f"transformers/torch unavailable: {type(exc).__name__}: {exc}",
        )

    override = _cfg_get(cfg, "model.deberta.encoder_override", None)
    base_model = override or _cfg_get(
        cfg, "model.deberta.base_model", "microsoft/deberta-v3-base"
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        # Transformers 5 may honor checkpoint dtype metadata. Force FP32 master
        # parameters at load time; the device move below asserts this invariant
        # again before AdamW creates its moment buffers.
        model = AutoModelForMaskedLM.from_pretrained(
            base_model, dtype=torch.float32
        )
    except Exception as exc:  # weights/network unavailable
        return DaptResult(
            status="SKIPPED", n_documents=n_docs,
            reason=f"could not load base model '{base_model}': "
                   f"{type(exc).__name__}: {exc}",
        )

    mlm_prob = float(_cfg_get(cfg, "model.dapt.masking_probability", 0.15))
    max_len = int(_cfg_get(cfg, "model.deberta.chunk_size", 512))
    batch_size = int(_cfg_get(cfg, "model.dapt.batch_size",
                              _cfg_get(cfg, "model.deberta.batch_size", 4)))
    epochs = int(_cfg_get(cfg, "model.dapt.epochs", 1))
    lr = float(_cfg_get(cfg, "model.dapt.learning_rate", 5e-5))
    max_steps = int(_cfg_get(cfg, "model.dapt.max_steps", 0))  # 0 = epoch bound only
    min_successful_steps = max(
        1, int(_cfg_get(cfg, "model.dapt.min_successful_steps", 25))
    )

    class _MLMDataset(Dataset):
        def __init__(self, texts):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(self.texts[idx], truncation=True, max_length=max_len)
            return {"input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"]}

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob
    )
    loader = DataLoader(_MLMDataset(docs), batch_size=batch_size, shuffle=True,
                        collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parameter_dtype = str(
        _cfg_get(cfg, "model.deberta.optimizer.parameter_dtype", "float32")
    ).lower()
    if parameter_dtype not in {"float32", "fp32", "torch.float32"}:
        raise ValueError(
            "model.deberta.optimizer.parameter_dtype must be 'float32'; "
            f"got {parameter_dtype!r}"
        )
    adamw_eps = float(
        _cfg_get(cfg, "model.deberta.optimizer.eps", 1.0e-6)
    )
    if not math.isfinite(adamw_eps) or adamw_eps < 1.0e-6:
        raise ValueError(
            "model.deberta.optimizer.eps must be finite and >= 1e-6; "
            f"got {adamw_eps!r}"
        )
    if bool(_cfg_get(cfg, "model.deberta.optimizer.foreach", False)) or bool(
        _cfg_get(cfg, "model.deberta.optimizer.fused", False)
    ):
        raise ValueError(
            "stable DAPT requires optimizer.foreach=false and optimizer.fused=false"
        )

    # Disabling autocast is not enough when checkpoint parameters were loaded in
    # FP16. Keep model parameters and Adam moments in FP32; mixed precision is a
    # supervised-forward autocast choice only.
    model = model.to(device=device, dtype=torch.float32)
    first_non_fp32 = next(
        (
            (name, parameter.dtype)
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ),
        None,
    )
    if first_non_fp32 is not None:
        raise RuntimeError(
            "DAPT FP32 optimizer invariant failed: "
            f"parameter {first_non_fp32[0]!r} has dtype {first_non_fp32[1]}"
        )
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, eps=adamw_eps, foreach=False, fused=False
    )
    # Linear LR warmup. The DeBERTa-v3 base ships its MLM head under non-standard
    # weight names, so AutoModelForMaskedLM re-initialises cls.predictions.* at
    # random (the "checkpoint seem corrupted" load warning). A randomly-init MLM
    # decoder over a large vocab at full LR from step 1 can emit an unstable, even
    # non-finite, loss and burn every step (successful_steps=0 -> HEALTH_GATE_FAILED).
    # Warming the LR from ~0 over the first steps lets the head settle before the
    # full learning rate applies. Warmup is config-driven and defaults to a small
    # fraction of the step budget.
    _bounded_total = max_steps if max_steps else max(1, len(loader) * max(1, epochs))
    warmup_steps = int(_cfg_get(cfg, "model.dapt.warmup_steps", 0)) or max(
        1, min(50, _bounded_total // 10)
    )

    def _lr_lambda(step_index: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step_index + 1) / float(warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    history: list[dict] = []
    successful_steps = 0
    attempted_steps = 0
    nonfinite_loss_steps = 0
    nonfinite_gradient_steps = 0
    nonfinite_parameter_steps = 0
    final_loss = float("nan")
    fatal_reason: Optional[str] = None

    for epoch in range(max(1, epochs)):
        for batch in loader:
            # max_steps bounds ATTEMPTS, not only successful updates, so repeated
            # unhealthy batches cannot silently extend runtime.
            if max_steps and attempted_steps >= max_steps:
                break
            attempted_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out.loss
            if not bool(torch.isfinite(loss).all()):
                nonfinite_loss_steps += 1
                history.append({
                    "epoch": epoch,
                    "attempt": attempted_steps,
                    "status": "NONFINITE_LOSS",
                })
                continue

            loss.backward()
            gradient_checks = [
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            saw_gradient = bool(gradient_checks)
            gradients_finite = bool(
                torch.stack(gradient_checks).all().item()
            ) if gradient_checks else False
            if not saw_gradient or not gradients_finite:
                nonfinite_gradient_steps += 1
                history.append({
                    "epoch": epoch,
                    "attempt": attempted_steps,
                    "status": "NONFINITE_OR_MISSING_GRADIENT",
                })
                optimizer.zero_grad(set_to_none=True)
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm).all()):
                nonfinite_gradient_steps += 1
                history.append({
                    "epoch": epoch,
                    "attempt": attempted_steps,
                    "status": "NONFINITE_CLIPPED_GRADIENT_NORM",
                })
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            parameter_checks = [
                torch.isfinite(parameter).all() for parameter in model.parameters()
            ]
            parameters_finite = bool(
                torch.stack(parameter_checks).all().item()
            ) if parameter_checks else False
            if not parameters_finite:
                nonfinite_parameter_steps += 1
                fatal_reason = "model parameters became non-finite after optimizer step"
                history.append({
                    "epoch": epoch,
                    "attempt": attempted_steps,
                    "status": "NONFINITE_PARAMETERS",
                })
                break

            # Audit the newly-created Adam moments on the first update. With
            # FP32 parameters they must be finite FP32 tensors; checking once
            # proves the invariant without scanning two model-sized buffers on
            # every subsequent step.
            if successful_steps == 0:
                bad_state = None
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if not torch.is_tensor(value) or not value.is_floating_point():
                            continue
                        if not bool(torch.isfinite(value).all()):
                            bad_state = (str(key), str(value.dtype), "non-finite")
                            break
                        if key in {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"} \
                                and value.dtype != torch.float32:
                            bad_state = (str(key), str(value.dtype), "non-FP32")
                            break
                    if bad_state is not None:
                        break
                if bad_state is not None:
                    nonfinite_parameter_steps += 1
                    fatal_reason = (
                        "AdamW state violated FP32/finite invariant after first step: "
                        f"key={bad_state[0]} dtype={bad_state[1]} ({bad_state[2]})"
                    )
                    history.append({
                        "epoch": epoch,
                        "attempt": attempted_steps,
                        "status": "INVALID_OPTIMIZER_STATE",
                        "state_key": bad_state[0],
                        "state_dtype": bad_state[1],
                    })
                    break

            scheduler.step()
            successful_steps += 1
            final_loss = float(loss.detach().cpu())
            if successful_steps == 1 or successful_steps % 50 == 0:
                history.append({
                    "epoch": epoch,
                    "attempt": attempted_steps,
                    "step": successful_steps,
                    "status": "OK",
                    "loss": final_loss,
                })
        if fatal_reason or (max_steps and attempted_steps >= max_steps):
            break

    health_reasons: list[str] = []
    if fatal_reason:
        health_reasons.append(fatal_reason)
    if nonfinite_loss_steps:
        health_reasons.append(f"{nonfinite_loss_steps} non-finite loss step(s)")
    if nonfinite_gradient_steps:
        health_reasons.append(
            f"{nonfinite_gradient_steps} non-finite/missing gradient step(s)"
        )
    if nonfinite_parameter_steps:
        health_reasons.append(
            f"{nonfinite_parameter_steps} non-finite parameter step(s)"
        )
    if successful_steps < min_successful_steps:
        health_reasons.append(
            f"only {successful_steps} successful steps (< min_successful_steps="
            f"{min_successful_steps})"
        )
    if not math.isfinite(final_loss):
        health_reasons.append("final successful-step loss is not finite")

    health_payload = {
        "status": "HEALTH_GATE_FAILED" if health_reasons else "OK",
        "n_documents": n_docs,
        "attempted_steps": attempted_steps,
        "successful_steps": successful_steps,
        "min_successful_steps": min_successful_steps,
        "nonfinite_loss_steps": nonfinite_loss_steps,
        "nonfinite_gradient_steps": nonfinite_gradient_steps,
        "nonfinite_parameter_steps": nonfinite_parameter_steps,
        "final_loss": final_loss,
        "max_steps": max_steps,
        "optimizer_numerics": {
            "parameter_dtype": "float32",
            "eps": adamw_eps,
            "foreach": False,
            "fused": False,
        },
        "reasons": health_reasons,
    }
    if output_dir is not None:
        health_path = os.path.join(output_dir, "audit", "dapt_health.json")
        os.makedirs(os.path.dirname(health_path), exist_ok=True)
        with open(health_path, "w", encoding="utf-8") as fh:
            json.dump(health_payload, fh, indent=2)

    # Never publish an encoder from a run that skipped a non-finite update or
    # failed to meet the minimum amount of validated adaptation.
    if health_reasons:
        return DaptResult(
            status="HEALTH_GATE_FAILED",
            n_documents=n_docs,
            n_train_docs=n_docs,
            steps=successful_steps,
            attempted_steps=attempted_steps,
            nonfinite_loss_steps=nonfinite_loss_steps,
            nonfinite_gradient_steps=nonfinite_gradient_steps,
            nonfinite_parameter_steps=nonfinite_parameter_steps,
            final_loss=final_loss,
            encoder_path=None,
            reason="; ".join(health_reasons),
            history=history,
        )

    encoder_path = None
    if output_dir is not None:
        encoder_path = os.path.join(output_dir, "models", "dapt_encoder")
        os.makedirs(encoder_path, exist_ok=True)
        # Persist the adapted base encoder + tokenizer only after all health
        # gates pass, so supervised training cannot consume a partial model.
        try:
            base = getattr(model, model.base_model_prefix, model)
            base.save_pretrained(encoder_path)
        except Exception:
            model.save_pretrained(encoder_path)
        tokenizer.save_pretrained(encoder_path)
        with open(os.path.join(encoder_path, "dapt_meta.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "n_documents": n_docs,
                "steps": successful_steps,
                "attempted_steps": attempted_steps,
                "nonfinite_loss_steps": nonfinite_loss_steps,
                "nonfinite_gradient_steps": nonfinite_gradient_steps,
                "nonfinite_parameter_steps": nonfinite_parameter_steps,
                "min_successful_steps": min_successful_steps,
                "final_loss": final_loss,
                "mlm_probability": mlm_prob,
                "base_model": base_model,
            }, fh, indent=2)

    return DaptResult(
        status="OK",
        n_documents=n_docs,
        n_train_docs=n_docs,
        steps=successful_steps,
        attempted_steps=attempted_steps,
        nonfinite_loss_steps=nonfinite_loss_steps,
        nonfinite_gradient_steps=nonfinite_gradient_steps,
        nonfinite_parameter_steps=nonfinite_parameter_steps,
        final_loss=final_loss,
        encoder_path=encoder_path,
        history=history,
    )
