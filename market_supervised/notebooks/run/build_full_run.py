"""Generator for the L4-ready results notebook.

Run:  python notebooks/run/build_full_run.py

Produces ``notebooks/run/full_run.ipynb`` -- a single Colab
notebook that, on a 20 GB L4 GPU, runs the SAME validated ``src/`` pipeline as the
publication notebook but with two differences designed to fill the paper's
"[YET TO OBTAIN]" cells:

  1. A numerically-stable MAIN/FULL profile so the end-to-end fine-tuned Council
     CONVERGES instead of diverging to NaN (the previous run aborted with
     NAN_ABORTED). This un-gates the ``council_finetune`` comparison row and its
     test predictions, because ``train_target`` only writes them when the run
     returns status OK.

  2. Extra orchestration cells that emit artifacts the base ``run_council_pipeline``
     does not: per-signal frozen+finetune scoring for every linked signal, the
     full ablation matrix, and per-fold cross-validation predictive metrics
     (rolling-origin AND leave-one-meeting-out) via ``CrossValidator.aggregate``.

The notebook is orchestration-only: every heavy operation calls the tested ``src/``
package. Nothing is reimplemented in cells and no numbers are fabricated -- a stage
that cannot run records NOT_RUN / SKIPPED / INSUFFICIENT_SAMPLE.
"""

from __future__ import annotations

import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "full_run.ipynb")

_CELL_N = [0]


def _cid() -> str:
    _CELL_N[0] += 1
    return f"cell-{_CELL_N[0]:02d}"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": _cid(), "metadata": {},
            "source": text.strip("\n") + "\n"}


def code(text: str) -> dict:
    return {"cell_type": "code", "id": _cid(), "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.strip("\n") + "\n"}


CELLS: list = []
A = CELLS.append

# ---------------------------------------------------------------------------
A(md(r"""
# Market-Supervised ECB Council — L4 Results Notebook

This notebook produces the results the manuscript still marks **[YET TO OBTAIN]**.
It runs the **same tested `src/` pipeline** as the publication notebook, but:

1. Uses a **numerically-stable MAIN/FULL profile** so the end-to-end **fine-tuned
   Council converges** instead of aborting with `NAN_ABORTED`. This un-gates the
   `council_finetune` row + its predictions (they are only written on a status-OK run).
2. Adds cells that emit what `run_council_pipeline` does not: **per-signal**
   frozen+finetune scoring, the full **ablation matrix**, and **per-fold
   cross-validation** metrics (rolling-origin and leave-one-meeting-out).

**Target hardware:** one **NVIDIA L4 (20–24 GB)** Colab runtime.
`Runtime > Change runtime type > L4 GPU`.

**Scientific invariants:** chronological splits, no test-set tuning, no fabricated
numbers (a stage that can't run records `NOT_RUN`/`SKIPPED`/`INSUFFICIENT_SAMPLE`).
"""))

# ---------------------------------------------------------------------------
A(md("## [1] Install dependencies"))
A(code(r"""
INSTALL_DEPS = True
import os, sys, subprocess

_here = os.getcwd()
REPO_ROOT = _here if os.path.isfile(os.path.join(_here, "requirements.txt")) \
    else os.path.abspath(os.path.join(_here, ".."))
_REQ = os.path.join(REPO_ROOT, "requirements.txt")

if INSTALL_DEPS:
    if os.path.isfile(_REQ):
        print("Installing from", _REQ)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", _REQ], check=False)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch", "transformers", "pandas", "pyarrow", "openpyxl",
                        "scikit-learn", "scipy", "statsmodels", "PyYAML",
                        "matplotlib", "numpy"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentencepiece"], check=False)
else:
    print("Skipping dependency install (INSTALL_DEPS=False).")
"""))

# ---------------------------------------------------------------------------
A(md("## [2] Mount Drive + resolve project root"))
A(code(r"""
DRIVE_PROJECT = "/content/drive/MyDrive/market_supervised_ecb"
REPO_COLAB_BASE = "/content/drive/MyDrive/market_supervised"  # committed config base

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive")
    IN_COLAB = True
except Exception:
    IN_COLAB = False
print("Colab:", IN_COLAB)

PROJECT_ROOT = REPO_COLAB_BASE if os.path.isdir(REPO_COLAB_BASE) else REPO_ROOT
assert os.path.isdir(PROJECT_ROOT), f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ARTIFACT_ROOT = DRIVE_PROJECT if IN_COLAB else os.path.join(PROJECT_ROOT, "outputs_L4")
os.makedirs(ARTIFACT_ROOT, exist_ok=True)
print("PROJECT_ROOT  =", PROJECT_ROOT)
print("ARTIFACT_ROOT =", ARTIFACT_ROOT)
"""))

A(code(r"""
_SUBDIRS = ["config", "data/raw", "data/processed", "data/cached", "data/manifests",
    "checkpoints", "models", "predictions", "metrics", "tables", "figures", "logs",
    "leakage_audit", "cross_validation", "ablations", "regime_analysis", "stance",
    "gpu_benchmark", "experiments", "manifests", "final_report", "per_signal"]
for _sub in _SUBDIRS:
    os.makedirs(os.path.join(ARTIFACT_ROOT, _sub), exist_ok=True)
print("Created", len(_SUBDIRS), "artifact subdirectories.")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [3] Imports + **live console logging** + reproducibility (seed 42)

Everything this notebook does is streamed to the console — it is not a black box.
This cell installs a single stdout logging handler on the **root** logger so **every**
`src/` module surfaces its progress: the `pipeline.*` loggers (`pipeline.training`,
`pipeline.market_data`, `pipeline.checkpoint`, `pipeline.data_io`) AND the orchestrator
`src.run_council`. It also defines two small helpers reused by every heavy cell:

* `banner(title)` — prints a labelled `>>> START` / `<<< DONE (elapsed)` boundary so you
  always know which stage is running and how long it took;
* `step(msg)` — timestamped one-line progress notes.

Combined with `verbose=True` and `progress_every_steps` on the encoder (set in `[4]`), the
long DeBERTa fine-tune prints a per-epoch summary and a per-N-step heartbeat, so a running
cell shows movement instead of looking hung.
"""))
A(code(r"""
import json, gc, logging, time
import numpy as np
import pandas as pd

from src.config import ConfigManager, config_hash
from src.reporting import set_seeds, create_outputs_manifest, write_run_metadata
from src import notebook_support as ns

# --- ONE stdout handler on the ROOT logger => every src/ module is visible ----
# Line-buffered, flushing StreamHandler so notebook output streams live rather
# than arriving in a buffered burst at the end of a long cell.
class _FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass

_ROOT = logging.getLogger()
for _old in list(_ROOT.handlers):
    _ROOT.removeHandler(_old)
_console = _FlushStreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter(
    "%(asctime)s | %(name)-22s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
_ROOT.addHandler(_console)
_ROOT.setLevel(logging.INFO)
# Make sure the pipeline + orchestrator loggers propagate to the root handler
# (do not attach private handlers that would swallow their records).
for _name in ("pipeline", "pipeline.training", "pipeline.market_data",
              "pipeline.checkpoint", "pipeline.data_io", "src.run_council"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.INFO)
    _lg.propagate = True
    for _old in list(_lg.handlers):
        _lg.removeHandler(_old)
# Keep third-party noise down so the pipeline story stays readable.
for _noisy in ("transformers", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

_PIPELINE_T0 = time.time()

def _elapsed(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return (f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s"))

class banner:
    'Context manager that brackets a heavy stage with START/DONE + elapsed.'
    def __init__(self, title):
        self.title = title
    def __enter__(self):
        self.t0 = time.time()
        print("\n" + "=" * 78, flush=True)
        print(f">>> START  {self.title}", flush=True)
        print(f"    (t+{_elapsed(time.time() - _PIPELINE_T0)} since notebook start)", flush=True)
        print("=" * 78, flush=True)
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        status = "DONE " if exc_type is None else f"FAILED ({exc_type.__name__})"
        print("-" * 78, flush=True)
        print(f"<<< {status} {self.title}  |  took {_elapsed(dt)}  |  "
              f"t+{_elapsed(time.time() - _PIPELINE_T0)} total", flush=True)
        print("-" * 78 + "\n", flush=True)
        return False  # never suppress exceptions

def step(msg):
    'Timestamped one-line progress note (also mirrors to stdout with flush).'
    print(f"    [{time.strftime('%H:%M:%S')}] {msg}", flush=True)

SEED = 42
set_seeds(SEED)
step(f"Seeded all RNGs with {SEED}")
step("Root logger wired to stdout at INFO -- all src/ stages will stream live.")
print("Console logging is ACTIVE. This run is fully traced, not a black box.")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [4] MODE + config + **first-step AdamW stability contract**

`MODE = "MAIN"` fetches the real DeBERTa-v3-base and fine-tunes on the GPU (no FinBERT
download); `"FULL"` additionally runs FinBERT and every baseline.

The failed run had a finite loss and finite gradients immediately before the first
`optimizer.step()`, then non-finite parameters immediately afterward. A minimal
reproducer identified the root cause: half-precision AdamW parameters/moments combined
with the default `eps=1e-8`. That epsilon underflows in FP16, so a zero-gradient update
can divide `0/0` and write NaNs. AMP/autocast settings control compute precision; they do
not promote half-precision parameter storage or Adam moments.

The primary fix is enforced in `src/` and repeated here as an explicit run contract:

* load/promote all trainable parameters to **FP32 before constructing AdamW**;
* keep Adam moments in **FP32**;
* use `eps: 1e-6`, `foreach: false`, and `fused: false` for deterministic scalar AdamW;
* audit the first landed optimizer step and fail closed if parameters or moments become
  non-finite.

The lower head LR, uncertainty weight, tighter clipping, wider minimum chunk length, and
log-variance floor below are retained as **secondary stability mitigations**. They were
not the demonstrated cause of the first-step corruption and LR backoff alone did not
repair it.
"""))
A(code(r"""
MODE = "MAIN"        # USER-EDITABLE: MAIN | FULL
RESUME_EXISTING = True
FORCE_RERUN = False

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "default.yaml")
cfg_mgr = ConfigManager()
config = cfg_mgr.load_config(CONFIG_PATH, project_root=PROJECT_ROOT)
config["data"]["outputs_dir"] = os.path.join(ARTIFACT_ROOT, "experiments")

_deb = config["model"]["deberta"]
config["model"].setdefault("baselines", {})

# --- Real encoder, real fine-tune -----------------------------------------
_deb["force_stub"] = False
_deb["allow_stub_encoder"] = False
_deb["verbose"] = True
_deb["progress_every_steps"] = 25
_deb["frozen_encoder"] = False        # end-to-end fine-tune is the headline here
_deb["use_amp"] = True                # epoch 0 FP32 compute; then BF16 autocast with FP32 parameters

# --- PRIMARY OPTIMIZER FIX (the first-step AdamW numerical contract) -------
_opt = _deb.setdefault("optimizer", {})
_opt.update({"parameter_dtype": "float32", "eps": 1.0e-6,
             "foreach": False, "fused": False})

# --- SECONDARY STABILITY MITIGATIONS (not the identified root cause) -------
_deb["head_lr_multiplier"] = 2.0      # extra margin vs the former 10.0
_deb["grad_clip_norm"] = 0.5
_deb["min_chunk_tokens"] = 8
_deb.setdefault("loss_weights", {})["uncertainty"] = 0.05
_deb.setdefault("uncertainty", {})["log_var_min"] = -4.0
_deb["uncertainty"]["log_var_max"] = 2.0
config.setdefault("council", {})["log_var_min"] = -4.0
config["council"]["log_var_max"] = 2.0
# Gradient-side recovery remains active. Post-step state corruption fails closed.
_deb.setdefault("nan_recovery", {})
_deb["nan_recovery"].update({"lr_backoff_after_steps": 10, "abort_after_events": 200,
                             "lr_reduction_factor": 0.5, "min_lr": 1e-8})

# Requested complete run: retain per-signal and ablation stages.
_runtime = config.setdefault("runtime", {})
_runtime["run_per_signal"] = True
_runtime["run_ablations"] = True

# --- Baselines -------------------------------------------------------------
if MODE == "MAIN":
    config["model"]["baselines"]["exclude"] = ["finbert"]
elif MODE == "FULL":
    config["model"]["baselines"]["exclude"] = []
else:
    raise ValueError(f"Unknown MODE={MODE!r}; use MAIN or FULL")

CFG_HASH = config_hash(config)
print("MODE          =", MODE)
print("config_hash   =", CFG_HASH)
print("primary target=", _deb["stance"]["primary_target"])
print("encoder       =", _deb.get("base_model"))
print("optimizer     =", _opt)
print("head_lr_mult  =", _deb["head_lr_multiplier"],
      "| uncertainty w =", _deb["loss_weights"]["uncertainty"],
      "| min_chunk_tokens =", _deb["min_chunk_tokens"])
print("run_per_signal=", _runtime["run_per_signal"],
      "| run_ablations =", _runtime["run_ablations"])
"""))

A(code(r"""
registry = ns.ExperimentRegistry(os.path.join(ARTIFACT_ROOT, "experiments"))
RUN_ID = f"{MODE.lower()}_seed{SEED}_{CFG_HASH[:12]}"
registry.register(RUN_ID, experiment_name=f"council_{MODE.lower()}",
    model_name="council_finetune", configuration_hash=CFG_HASH, random_seed=SEED,
    status=ns.STATE_RUNNING)
write_run_metadata(cfg_mgr.record_environment(), config, CFG_HASH,
                   os.path.join(ARTIFACT_ROOT, "manifests"))
print("RUN_ID =", RUN_ID, "| status =", registry.status_of(RUN_ID))
"""))

# ---------------------------------------------------------------------------
A(md("## [5] GPU diagnostics — must be a real CUDA device"))
A(code(r"""
gpu = ns.gpu_diagnostics()
print(json.dumps(gpu, indent=2, default=str))
with open(os.path.join(ARTIFACT_ROOT, "gpu_benchmark", "gpu_diagnostics.json"), "w") as fh:
    json.dump(gpu, fh, indent=2, default=str)
if not gpu.get("cuda_available"):
    raise RuntimeError("No CUDA GPU detected. Set Runtime > Change runtime type > L4 GPU.")
print("GPU:", gpu.get("gpu_name"), "| VRAM(GB):", gpu.get("total_vram_gb"))
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [6] Build the data foundation (full corpus)

The tested `src/` chain: AUDIT → DATE_VALIDATION → DEDUP → EVENT_CLASSIFICATION →
MARKET_ALIGNMENT → TARGET_CONSTRUCTION → MASTER_PARQUET, with `[-15,+60]` minute windows.
No DEBUG subsampling — the full corpus is used.
"""))
A(code(r"""
from src.data_io import locate_datasets
from src.market_data import assemble_inputs, attach_speech_text
from src.corpus_audit import inventory_datasets, audit_market_series, write_inventory
from src.date_validation import build_report as build_date_report, write_reports as write_date_reports
from src.deduplication import dedup_speeches, write_report as write_dedup_report
from src.event_classification import classify_events
from src.event_linking import link_events_to_market, write_report as write_link_report
from src.target_construction import build_targets, write_definition_report
from src.dataset_builder import build_master, build_master_schema, write_master
from src.confounders import build_confounders

with banner("[6] Data foundation: locate datasets + assemble inputs + audit"):
    run = cfg_mgr.create_experiment(config)
    cfg_mgr.save_resolved_config(run)
    OUTPUT_DIR = run.output_dir
    AUDIT_DIR = os.path.join(OUTPUT_DIR, "audit")
    INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, "intermediate")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
    create_outputs_manifest(OUTPUT_DIR)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    registry.update(RUN_ID, stage="data_foundation", checkpoint_path=OUTPUT_DIR)
    step(f"OUTPUT_DIR = {OUTPUT_DIR}")

    step("Locating datasets on disk ...")
    datasets = locate_datasets(config)
    registry_specs = cfg_mgr.target_registry(config)
    step(f"Assembling corpus + market series ({len(datasets)} dataset entries) ...")
    corpus, series, imc_events = assemble_inputs(config, datasets)
    write_inventory(inventory_datasets({"corpus": corpus, **series}), AUDIT_DIR)
    _ = audit_market_series(series)
    step(f"AUDIT -> corpus rows={len(corpus)}; series={ {k: len(v) for k, v in series.items()} }")
    step(f"IMC events loaded = {len(imc_events)}")
"""))

A(code(r"""
with banner("[6b] Date validation -> dedup -> classify -> link -> targets -> MASTER"):
    step("Date-range validation + coverage matrix ...")
    date_report, coverage_matrix = build_date_report(corpus, registry_specs, series)
    write_date_reports(date_report, coverage_matrix, AUDIT_DIR)

    step("Deduplicating speeches ...")
    deduped, dedup_report = dedup_speeches({"all_ECB_speeches": corpus})
    write_dedup_report(dedup_report, AUDIT_DIR)
    step(f"dedup -> {len(deduped.get('all_ECB_speeches', deduped))} rows retained"
         if isinstance(deduped, dict) else f"dedup complete")

    step("Attaching speech text to events + classifying ...")
    events_with_text, text_match_stats = attach_speech_text(imc_events, deduped)
    classified = classify_events(events_with_text, imc_events)
    step(f"CLASSIFY -> {len(classified)} events; {text_match_stats['n_matched']} with text")

    step("Linking events to market windows ...")
    _linked, link_diag = link_events_to_market(classified, registry_specs, series)
    write_link_report(link_diag, AUDIT_DIR)

    step("Constructing supervised targets ...")
    targets, conversion_log = build_targets(classified, registry_specs, series)
    write_definition_report(registry_specs, conversion_log, AUDIT_DIR)

    step("Building master table + confounders ...")
    master = build_master(classified, targets)
    _conf = (config.get("data", {}).get("confounder_files", {}) or {})
    _dsdir = config["data"]["datasets_dir"]
    _vix = os.path.join(_dsdir, _conf["vix_indices"]) if _conf.get("vix_indices") else None
    _mps = os.path.join(_dsdir, _conf["us_mps"]) if _conf.get("us_mps") else None
    confounders = build_confounders(master, indices_path=_vix, us_mps_path=_mps)
    master = build_master(classified, targets, provenance=confounders)
    master_schema = build_master_schema(master, registry_specs, CFG_HASH)
    write_master(master, master_schema, INTERMEDIATE_DIR)
    step(f"MASTER_PARQUET -> {len(master)} rows, {master['event_id'].nunique()} unique events")
"""))

# ---------------------------------------------------------------------------
A(md("## [7] Task parquets → temporal split → train-only scaler (validation gates)"))
A(code(r"""
from src.splitting import (temporal_split, primary_test_ids,
                           build_integrity_report, write_integrity_report)
from src.dataset_builder import build_task_parquets, write_exclusion_log, save_target_scaler
from src.validation import master_validate, task_validate, split_validate, assert_gate
from sklearn.preprocessing import StandardScaler

splits = temporal_split(master, config["splits"])
task_paths, exclusions = build_task_parquets(master, registry_specs, splits, models_dir=MODELS_DIR)
write_exclusion_log(exclusions, AUDIT_DIR)

primary_col = registry_specs[0].target_column if registry_specs else None
train_ids = splits.loc[splits["split"] == "train", "event_id"]
train_rows = master[master["event_id"].isin(train_ids)]
_fit_vals = (train_rows[primary_col].dropna().to_numpy().reshape(-1, 1)
             if primary_col and primary_col in train_rows.columns and len(train_rows)
             else np.zeros((1, 1)))
scaler = StandardScaler().fit(_fit_vals)
save_target_scaler({"mean": float(scaler.mean_[0]), "std": float(scaler.scale_[0]),
                    "n": int(len(_fit_vals)), "fit_on": "train"}, MODELS_DIR)
SCALER_META = {"mean": float(scaler.mean_[0]), "std": float(scaler.scale_[0])}

primary_ids = primary_test_ids(splits[splits["split"] == "test"])
write_integrity_report(build_integrity_report(splits), AUDIT_DIR)
print("splits:", splits["split"].value_counts().to_dict())
print("primary test ids:", len(primary_ids))
print("task parquets:", sorted(task_paths.keys()))
"""))

A(code(r"""
canonical_event_count = master["event_id"].nunique() if len(master) else 0
assert_gate(master_validate(master, master_schema, canonical_event_count))
assert_gate(task_validate(task_paths, exclusions))
assert_gate(split_validate(splits, scaler, primary_ids))
print("GATES passed (MASTER_VALIDATE, TASK_VALIDATE, SPLIT_VALIDATE).")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [8] GPU auto-batch probe — fill the L4's ~20 GB

Probes real forward+backward passes through the production Council loss dispatcher and
applies the largest safe physical batch (+ gradient accumulation to a target effective
batch) with BF16 compute. Synthetic loss-audit writes are isolated in an automatically
deleted probe-only directory, and the resolved gradient-checkpointing policy is preserved
on both probe success and fallback; on a smaller GPU the probe simply backs off.
"""))
A(code(r"""
BATCH_CANDIDATES = (4, 8, 12, 16, 24, 32)
TARGET_EFFECTIVE_BATCH = 32
VRAM_SAFETY_FRACTION = 0.85

# USER-EDITABLE: safe manual physical batch for a 24 GB L4 at the configured
# max_chunks x chunk_size. Used when the auto-probe is skipped or fails.
L4_FALLBACK_BATCH = 16
RUN_GPU_PROBE = True   # set False to skip the probe entirely and use L4_FALLBACK_BATCH
_resolved_gradient_checkpointing = bool(_deb.get("gradient_checkpointing", False))

GPU_BENCH = {"status": ns.NOT_RUN, "reason": "probe skipped"}
if RUN_GPU_PROBE and gpu.get("cuda_available"):
    import tempfile
    import torch
    from src.market_supervised import build_model
    from src.training import _move_batch, _select_and_compute_loss
    _probe_audit_tmp = tempfile.TemporaryDirectory(prefix="council_gpu_probe_")
    _probe_config = json.loads(json.dumps(config))
    _probe_config.setdefault("experiment", {})["output_dir"] = _probe_audit_tmp.name
    _probe_model = None
    try:
        _probe_model = build_model(_probe_config).cuda().train()
        _seq = int(_deb.get("chunk_size", 512)); _nseg = int(_deb.get("max_chunks", 4))
        _vocab = 30000; _n_sig = len(_probe_config["council"]["signals"])
        def _build_batch(bs):
            # Build on CPU, then move the WHOLE batch to CUDA via the training helper so
            # every tensor lands on the same device.
            target_raw = torch.randn(bs)
            batch = {"input_ids": torch.randint(0, _vocab, (bs, _nseg, _seq)),
                     "attention_mask": torch.ones((bs, _nseg, _seq), dtype=torch.long),
                     "segment_mask": torch.ones((bs, _nseg), dtype=torch.long),
                     "signal_targets": torch.randn(bs, _n_sig),
                     "signal_present": torch.ones(bs, _n_sig, dtype=torch.bool),
                     "target": target_raw.clone(), "target_raw": target_raw,
                     "primary_present": torch.ones(bs, dtype=torch.bool),
                     "material_threshold": torch.ones(bs)}
            return _move_batch(batch, torch.device("cuda"))
        def _fwd_bwd(batch):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = _probe_model(batch)
                loss = _select_and_compute_loss(
                    _probe_model, out, batch, _probe_config
                )["total"]
            loss.backward(); _probe_model.zero_grad(set_to_none=True)
        _res = ns.probe_max_batch_size(_build_batch, _fwd_bwd, candidate_sizes=BATCH_CANDIDATES,
                                       safety_fraction=VRAM_SAFETY_FRACTION)
        GPU_BENCH = _res.to_dict()
    except Exception as _e:
        GPU_BENCH = {"status": ns.FAILED, "reason": f"{type(_e).__name__}: {_e}"}
    finally:
        if _probe_model is not None:
            del _probe_model
        gc.collect(); torch.cuda.empty_cache()
        # compute_council_losses emits presence audits. Probe batches are
        # synthetic, so route those writes to this disposable directory and
        # delete it before any real experiment stage begins.
        _probe_audit_tmp.cleanup()

# Apply a successful recommendation; otherwise use the manual L4 batch. Never
# change the resolved gradient-checkpointing stability policy as a side effect.
if isinstance(GPU_BENCH, dict) and GPU_BENCH.get("status") == "OK" and GPU_BENCH.get("recommended_batch_size"):
    ns.apply_gpu_benchmark_to_config(config, GPU_BENCH, target_effective_batch=TARGET_EFFECTIVE_BATCH)
else:
    print("GPU probe not applied (", GPU_BENCH.get("reason"), ") -> manual L4 batch:", L4_FALLBACK_BATCH)
    _deb["batch_size"] = int(L4_FALLBACK_BATCH)
    _deb["gradient_accumulation_steps"] = max(1, -(-int(TARGET_EFFECTIVE_BATCH) // int(L4_FALLBACK_BATCH)))
    _deb["use_amp"] = True
_deb["gradient_checkpointing"] = _resolved_gradient_checkpointing
CFG_HASH = config_hash(config); run.config = config; cfg_mgr.save_resolved_config(run)

with open(os.path.join(ARTIFACT_ROOT, "gpu_benchmark", "gpu_benchmark.json"), "w") as fh:
    json.dump({"probe": GPU_BENCH, "applied": _deb.get("gpu_autobatch", {})}, fh, indent=2, default=str)
print("GPU probe:", json.dumps(GPU_BENCH, indent=2, default=str))
print("batch_size =", _deb.get("batch_size"), "| grad_accum =", _deb.get("gradient_accumulation_steps"),
      "| gradient_checkpointing =", _deb.get("gradient_checkpointing"))
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [9] Run the validated pipeline (DAPT → **frozen + fine-tune** → eval → CV plan → regime → leakage → baselines)

`run_council_pipeline` trains BOTH the frozen headline and the end-to-end fine-tuned
Council on the primary target, then scores baselines on the same split. With the stable
config the fine-tune should converge and produce a `council_finetune` row in
`results/model_comparison.csv` (the earlier run left it empty after `NAN_ABORTED`).
"""))
A(code(r"""
from src.run_council import run_council_pipeline

with banner("[9] run_council_pipeline: DAPT -> frozen + FINE-TUNE -> eval -> CV plan -> regime -> leakage -> baselines"):
    step("This is the long stage. Live logs below stream from the src.* loggers:")
    step("  - pipeline.training  : per-epoch train_loss / val_metric + per-step heartbeat")
    step("  - src.run_council    : stage-by-stage orchestration transitions")
    step("  - pipeline.checkpoint: checkpoint saves")
    result = run_council_pipeline(config, OUTPUT_DIR, master=master, splits=splits, config_hash=CFG_HASH)
    step(f"Council run complete -> {result.output_dir}")
    print("\n  Stage-by-stage outcome:")
    for name, stage in result.to_dict()["stages"].items():
        print(f"    {name:<18} {stage['status']:<9} {(stage.get('detail') or '')[:72]}", flush=True)
    registry.update(RUN_ID, stage="pipeline_complete")

    # Report the fine-tune training outcome explicitly so a divergence is visible.
    _training_stage = result.stages.get("training_eval")
    _te = _training_stage.data if _training_stage is not None else {}
    print("\n  finetune summary:", _te.get("finetune_summary"), flush=True)
    print("  frozen   summary:", _te.get("frozen_summary"), flush=True)
"""))

A(code(r"""
# Show the main comparison table straight from the run (numbers come from artifacts).
_cmp = os.path.join(OUTPUT_DIR, "results", "model_comparison.csv")
if os.path.exists(_cmp):
    comparison_df = pd.read_csv(_cmp)
    cols = [c for c in ("model","n","r_squared","directional_accuracy","spearman","mse","mae") if c in comparison_df.columns]
    print(comparison_df[cols].to_string(index=False))
    if "council_finetune" in set(comparison_df["model"]):
        print("\nOK: council_finetune row PRESENT -> fine-tune converged.")
    else:
        print("\nWARNING: council_finetune row ABSENT -> fine-tune did not produce OK predictions.")
        print("Check", os.path.join(OUTPUT_DIR, "finetune", "audit", "training_nan_debug.jsonl"))
else:
    print(ns.NOT_RUN, "- no model_comparison.csv")
    comparison_df = None
"""))

A(md(r"""
### [9a] Held-out categorical stance → continuous stance

The fine-tuned Council first predicts a three-class distribution in the fixed order
**Dovish / Neutral / Hawkish** from the shared market-supervised DeBERTa representation.
Its continuous stance is then derived—not independently regressed—as
`P(Hawkish) - P(Dovish)`, bounded to `[-1, 1]`.

This cell accepts paths only from a current `COMPLETED` manifest whose stance stage is
`OK`, then reads the persisted held-out artifacts, verifies the probability simplex and
score identity, and prints mean probability mass separately from dominant-class event
share. If the fine-tune did not finish successfully, it reports `NOT_RUN` rather than
showing frozen, random, or stale logits.
"""))
A(code(r"""
_stance_facts = ns.gather_run_facts(
    OUTPUT_DIR, registry=registry, artifact_root=ARTIFACT_ROOT
)
_stance_info = _stance_facts.get("stance") or {}

if not _stance_info.get("available"):
    print(ns.NOT_RUN, "- current held-out stance artifacts are unavailable.")
    print("  stage status:", _stance_info.get("status", ns.NOT_RUN))
    print("  detail:", _stance_info.get("detail") or _stance_info.get("availability_reason") or ns.NOT_RUN)
    stance_summary = None
    stance_events = None
else:
    from src.training import print_stance_summary, summarize_stance_predictions

    _stance_summary_path = _stance_info["summary_path"]
    _stance_events_path = _stance_info["event_predictions_path"]
    _stance_series_path = _stance_info["series_path"]
    with open(_stance_summary_path, encoding="utf-8") as _fh:
        stance_summary = json.load(_fh)
    stance_events = pd.read_parquet(_stance_events_path)
    _verified_stance = summarize_stance_predictions(stance_events)

    _expected_order = ["dovish", "neutral", "hawkish"]
    if stance_summary.get("class_order") != _expected_order:
        raise RuntimeError(f"Unexpected stance class order: {stance_summary.get('class_order')}")
    if int(stance_summary.get("n_events", -1)) != len(stance_events):
        raise RuntimeError("stance_summary.json event count does not match the event parquet")

    for _section in ("mean_probability_mass", "dominant_event_share"):
        _saved = stance_summary.get(_section, {})
        _checked = _verified_stance[_section]
        if any(not np.isclose(float(_saved.get(_label, np.nan)), _checked[_label], atol=1e-10)
               for _label in _expected_order):
            raise RuntimeError(f"{_section} in stance_summary.json does not match the event parquet")
    for _metric in ("mean", "median", "std", "min", "max"):
        if not np.isclose(float(stance_summary.get("continuous_stance", {}).get(_metric, np.nan)),
                          _verified_stance["continuous_stance"][_metric], atol=1e-10):
            raise RuntimeError(f"continuous_stance.{_metric} does not match the event parquet")

    print_stance_summary(stance_summary)
    _checks = _verified_stance["identity_checks"]
    print("  Formula:", stance_summary["continuous_stance"]["formula"])
    print("  Identity checks: PASSED | max |sum(probabilities)-1| ="
          f" {_checks['probability_sum_max_abs_error']:.3e} | max score error ="
          f" {_checks['score_formula_max_abs_error']:.3e}")
    print("  Scope:", stance_summary.get("scope", "held_out_test_events"),
          "| events:", stance_summary["n_events"])
    print("  Artifacts:", _stance_summary_path, _stance_events_path, _stance_series_path,
          sep="\n    ")
    print("\nEvent-level preview:")
    _preview_cols = ["event_id", "event_date", "stance_prob_dovish", "stance_prob_neutral",
                     "stance_prob_hawkish", "stance_score", "stance_dominant"]
    print(stance_events[_preview_cols].head(10).to_string(index=False))
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [10] PER-SIGNAL scoring (Table: Per-Signal Council Performance)

The base pipeline scores only the primary target. This cell trains the **frozen** and
**fine-tuned** Council for **every linked signal** (a signal with no coverage — e.g.
`fx` — records `INSUFFICIENT_SAMPLE`) and records R², directional accuracy, and the
learned effective fusion weight `exp(-log_var_i)` when the fine-tune exposes it.
Results → `per_signal/per_signal_metrics.csv`.
"""))
A(code(r"""
from src.training import train_target_frozen, train_target
from src.evaluation import evaluate_model

PER_SIGNAL_DIR = os.path.join(ARTIFACT_ROOT, "per_signal")
os.makedirs(PER_SIGNAL_DIR, exist_ok=True)

# Effective fusion weights logged during the primary fine-tune (Req 2.3), if present.
_weights_path = os.path.join(AUDIT_DIR, "council_signal_weights.jsonl")
_eff_weight = {}
if os.path.exists(_weights_path):
    try:
        _last = None
        for _line in open(_weights_path):
            _line = _line.strip()
            if _line: _last = json.loads(_line)
        if _last and isinstance(_last.get("weights"), dict):
            _eff_weight = {k: float(v) for k, v in _last["weights"].items()}
    except Exception as _e:
        print("could not parse council_signal_weights.jsonl:", _e)

rows = []
with banner("[10] Per-signal scoring: frozen + fine-tune for every linked signal"):
    _signals = sorted(task_paths.items())
    step(f"{len(_signals)} signals x 2 modes = {len(_signals) * 2} trainings to run.")
    for _i, (model_name, path) in enumerate(_signals, start=1):
        target = model_name.replace("model_", "")
        sig_out = os.path.join(OUTPUT_DIR, "per_signal", target)
        for mode_name, frozen in (("frozen", True), ("finetune", False)):
            step(f"({_i}/{len(_signals)}) signal={target} mode={mode_name} -> training ...")
            _t0 = time.time()
            vcfg = json.loads(json.dumps(config))    # deep copy via JSON (config is plain dict)
            vcfg["model"]["deberta"]["frozen_encoder"] = frozen
            try:
                fn = train_target_frozen if frozen else train_target
                res = fn(path, vcfg, os.path.join(sig_out, mode_name), target, scaler=SCALER_META)
            except Exception as _e:
                step(f"    signal={target} mode={mode_name} ERROR:{type(_e).__name__}: {_e}")
                rows.append({"signal": target, "mode": mode_name, "status": f"ERROR:{type(_e).__name__}"})
                continue
            if res.get("status") != "OK" or res.get("y_true") is None:
                step(f"    signal={target} mode={mode_name} status={res.get('status')} "
                     f"(n_test={int(res.get('n_test', 0) or 0)}) [{_elapsed(time.time() - _t0)}]")
                rows.append({"signal": target, "mode": mode_name, "status": res.get("status"),
                             "n": int(res.get("n_test", 0) or 0)})
                continue
            rec = evaluate_model(res["y_true"], res["y_pred"], cfg=vcfg, model=f"council_{mode_name}")
            step(f"    signal={target} mode={mode_name} OK n={rec.n} "
                 f"R2={rec.r_squared:.4f} dir_acc={rec.directional_accuracy:.4f} "
                 f"[{_elapsed(time.time() - _t0)}]")
            rows.append({"signal": target, "mode": mode_name, "status": "OK", "n": rec.n,
                         "r_squared": rec.r_squared, "directional_accuracy": rec.directional_accuracy,
                         "spearman": rec.spearman, "mse": rec.mse, "mae": rec.mae,
                         "eff_fusion_weight": _eff_weight.get(target)})

    per_signal_df = pd.DataFrame(rows)
    per_signal_df.to_csv(os.path.join(PER_SIGNAL_DIR, "per_signal_metrics.csv"), index=False)
    step(f"wrote {os.path.join(PER_SIGNAL_DIR, 'per_signal_metrics.csv')}")
    print(per_signal_df.to_string(index=False))
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [11] ABLATION matrix (Table: Ablation Study + flat-DeBERTa contrast)

`run_ablations` trains the full variant first (for the Diebold–Mariano comparison) then
each single-factor variant: removed objectives, **`flat_encoder`** (the flat single-span
DeBERTa contrast that the paper's "Flat DeBERTa baseline"/"− hierarchical" rows need),
and `daily_labels`. Results → `results/ablation_results.csv`.
"""))
A(code(r"""
from src.ablation import run_ablations, ABLATION_VARIANTS

RUN_ABLATIONS = True   # USER-EDITABLE: set False to skip (long on a real encoder).
ablation_df = None
with banner("[11] Ablation matrix: full + single-factor variants"):
    if RUN_ABLATIONS:
        step(f"variants = {list(ABLATION_VARIANTS)}")
        step("Each variant trains on the primary target; live training logs stream below.")
        try:
            ablation_df = run_ablations(task_paths, cfg=config, output_dir=OUTPUT_DIR,
                variants=ABLATION_VARIANTS,
                targets=(config["model"]["deberta"]["stance"]["primary_target"],))
            step(f"Ablation rows: {len(ablation_df)}")
            cols = [c for c in ("ablation","status","n","r_squared","directional_accuracy","spearman","dm_pvalue") if c in ablation_df.columns]
            print(ablation_df[cols].to_string(index=False))
        except Exception as _exc:
            step(f"Ablation stage: {ns.FAILED} - {type(_exc).__name__}: {_exc}")
    else:
        step(f"Ablation stage: {ns.NOT_RUN} (RUN_ABLATIONS=False)")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [12] PER-FOLD cross-validation (Table: Cross-Validation)

The base pipeline writes only the fold *plan*. This cell trains the **frozen** Council
per fold (fast + numerically stable) and feeds per-fold `(y_true, y_pred)` into
`CrossValidator.aggregate`, which reports mean ± std of R² / directional accuracy /
Spearman across usable folds. Runs both **rolling-origin** and **leave-one-meeting-out**.
`aggregate` requires ≥2 usable folds each with ≥`min_fold_events` eval events; a scheme
that cannot meet that records its reason instead of a fabricated number.
Results → `cross_validation/cv_summary.json`.
"""))
A(code(r"""
from src.cross_validation import build_folds, CrossValidator
from src.training import train_target_frozen
from src.dataset_builder import build_task_parquets as _btp

PRIMARY = config["model"]["deberta"]["stance"]["primary_target"]
CV_DIR = os.path.join(ARTIFACT_ROOT, "cross_validation")
os.makedirs(CV_DIR, exist_ok=True)

# Event identity/date (and meeting identity when available) drive each fold.
_date_col = "event_date" if "event_date" in master.columns else "event_timestamp"
_event_cols = ["event_id", _date_col]
if "meeting_id" in master.columns:
    _event_cols.append("meeting_id")
_events = master[_event_cols].drop_duplicates("event_id").copy()
_events["event_date"] = pd.to_datetime(_events[_date_col])
if _date_col != "event_date":
    _events = _events.drop(columns=[_date_col])
_min_fold = int(config.get("cross_validation", {}).get("min_fold_events", 5))

def _fold_task_parquet(train_ids, eval_ids, out_path):
    # Build a canonical fold-local master and split map. Every selected event is
    # assigned exactly once, and scaling is re-fit from this fold's train rows.
    train_ids, eval_ids = set(train_ids), set(eval_ids)
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"fold train/eval overlap: {sorted(overlap)[:5]}")
    fold_ids = train_ids | eval_ids
    sub = master[master["event_id"].isin(fold_ids)].copy()
    sub_splits = sub[["event_id"]].drop_duplicates().copy()
    sub_splits["split"] = np.where(
        sub_splits["event_id"].isin(eval_ids), "test", "train"
    )
    sub_splits["is_primary_test"] = sub_splits["split"].eq("test")
    paths, _exc = _btp(
        sub, registry_specs, sub_splits,
        models_dir=os.path.join(out_path, "models"), config=config
    )
    return paths.get(f"model_{PRIMARY}")

def _predict_fold(fold):
    fdir = os.path.join(OUTPUT_DIR, "cv", fold.fold_id)
    os.makedirs(fdir, exist_ok=True)
    tp = _fold_task_parquet(fold.train_ids, fold.eval_ids, fdir)
    if tp is None:
        return np.array([]), np.array([])
    res = train_target_frozen(tp, config, fdir, PRIMARY)
    if res.get("status") != "OK" or res.get("y_true") is None:
        return np.array([]), np.array([])
    return np.asarray(res["y_true"]), np.asarray(res["y_pred"])

cv_summary = {}
with banner("[12] Cross-validation: rolling-origin + leave-one-meeting-out (frozen Council per fold)"):
    for scheme in ("rolling_origin", "leave_one_meeting_out"):
        step(f"scheme={scheme}: building folds ...")
        cfg_cv = dict(config.get("cross_validation", {}))
        cfg_cv["scheme"] = scheme
        try:
            folds = build_folds(_events, cfg_cv)
            step(f"scheme={scheme}: {len(folds)} candidate folds (min_fold_events={_min_fold})")
            validator = CrossValidator(scheme=scheme, min_fold_events=_min_fold)
            preds = {}
            for _j, f in enumerate(folds, start=1):
                if (len(f.eval_ids) < _min_fold):
                    step(f"    fold {_j}/{len(folds)} {f.fold_id}: SKIP (eval={len(f.eval_ids)} < {_min_fold})")
                    continue
                step(f"    fold {_j}/{len(folds)} {f.fold_id}: training frozen Council "
                     f"(train={len(f.train_ids)} eval={len(f.eval_ids)}) ...")
                _tf = time.time()
                yt, yp = _predict_fold(f)
                if len(yt) >= _min_fold:
                    preds[f.fold_id] = (yt, yp)
                    step(f"    fold {_j}/{len(folds)} {f.fold_id}: OK n_eval={len(yt)} "
                         f"[{_elapsed(time.time() - _tf)}]")
                else:
                    step(f"    fold {_j}/{len(folds)} {f.fold_id}: insufficient predictions "
                         f"(n={len(yt)}) [{_elapsed(time.time() - _tf)}]")
            usable = [f for f in folds if f.fold_id in preds]
            agg = validator.aggregate(usable, preds, cfg=config, model="council_frozen")
            cv_summary[scheme] = {"n_usable_folds": agg.n_usable_folds,
                "mean": agg.mean, "std": agg.std,
                "usable_fold_ids": agg.usable_fold_ids, "excluded_fold_ids": agg.excluded_fold_ids}
            step(f"[{scheme}] usable folds={agg.n_usable_folds} mean={agg.mean}")
        except Exception as _exc:
            cv_summary[scheme] = {"status": "NOT_RUN", "reason": f"{type(_exc).__name__}: {_exc}"}
            step(f"[{scheme}] NOT_RUN - {_exc}")

    with open(os.path.join(CV_DIR, "cv_summary.json"), "w") as fh:
        json.dump(cv_summary, fh, indent=2, default=str)
    step(f"wrote {os.path.join(CV_DIR, 'cv_summary.json')}")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [12b] EXTRA ablation variants the manuscript needs (fixed fusion, − stance, − DAPT)

`src.ablation.ABLATION_VARIANTS` covers removed objectives, `flat_encoder`, and
`daily_labels` but NOT three rows the paper's ablation table lists:
`− uncertainty fusion (+ fixed weights)`, `− stance objective`, and
`− domain-adaptive pretraining`. Each is a single config toggle, so this cell trains
them with the same `train_target` + `evaluate_model` path `run_ablations` uses and appends
them to `results/ablation_results.csv`. Toggles: `council.fusion_mode="fixed"`;
`model.deberta.stance.enabled=False` (+ zero `loss_weights.stance`); clear
`model.deberta.encoder_override` so the base (non-DAPT) encoder is used.
"""))
A(code(r"""
from src.training import train_target
from src.evaluation import evaluate_model, diebold_mariano_test

PRIMARY = config["model"]["deberta"]["stance"]["primary_target"]
_primary_task = task_paths.get(f"model_{PRIMARY}")

def _deepcfg():
    return json.loads(json.dumps(config))

# Reference: the FULL fine-tuned variant's test errors, for the DM p-value column.
_full_dir = os.path.join(OUTPUT_DIR, "ablation", "full", PRIMARY)
_full_pred = os.path.join(_full_dir, "evaluation", f"predictions_{PRIMARY}.parquet")
_ref_true = _ref_pred = None
if os.path.exists(_full_pred):
    _fp = pd.read_parquet(_full_pred); _ref_true = _fp["y_true"].to_numpy(); _ref_pred = _fp["y_pred"].to_numpy()

_extra_specs = {
    "minus_fixed_fusion": lambda c: c.setdefault("council", {}).update({"fusion_mode": "fixed"}),
    "minus_stance": lambda c: (c["model"]["deberta"].setdefault("stance", {}).update({"enabled": False}),
                               c["model"]["deberta"].setdefault("loss_weights", {}).update({"stance": 0.0})),
    "minus_dapt": lambda c: c["model"]["deberta"].update({"encoder_override": None}),
}

_extra_rows = []
if _primary_task is not None:
    for vname, mutate in _extra_specs.items():
        vcfg = _deepcfg(); vcfg["model"]["deberta"]["frozen_encoder"] = False
        mutate(vcfg)
        vout = os.path.join(OUTPUT_DIR, "ablation", vname, PRIMARY)
        try:
            res = train_target(_primary_task, vcfg, vout, PRIMARY, scaler=SCALER_META)
        except Exception as _e:
            _extra_rows.append({"ablation": vname, "target": PRIMARY, "status": f"ERROR:{type(_e).__name__}"}); continue
        if res.get("status") != "OK" or res.get("y_true") is None:
            _extra_rows.append({"ablation": vname, "target": PRIMARY, "status": res.get("status"),
                                "n": int(res.get("n_test", 0) or 0)}); continue
        rec = evaluate_model(res["y_true"], res["y_pred"], cfg=vcfg, model=vname)
        dm = float("nan")
        if _ref_true is not None and len(_ref_pred) == len(res["y_pred"]):
            dm = diebold_mariano_test(_ref_true - _ref_pred,
                                      np.asarray(res["y_true"]) - np.asarray(res["y_pred"]))
        _extra_rows.append({"ablation": vname, "target": PRIMARY, "status": "OK", "n": rec.n,
                            "mse": rec.mse, "mae": rec.mae, "r_squared": rec.r_squared,
                            "directional_accuracy": rec.directional_accuracy,
                            "spearman": rec.spearman, "dm_pvalue": dm})

# Append to the ablation CSV so the paper's ablation table has every row in one file.
_abl_csv = os.path.join(OUTPUT_DIR, "results", "ablation_results.csv")
if _extra_rows:
    _extra_df = pd.DataFrame(_extra_rows)
    if os.path.exists(_abl_csv):
        _extra_df = pd.concat([pd.read_csv(_abl_csv), _extra_df], ignore_index=True)
    _extra_df.to_csv(_abl_csv, index=False)
    print("extra ablation variants appended:", [r["ablation"] for r in _extra_rows])
    print(pd.DataFrame(_extra_rows).to_string(index=False))
else:
    print(ns.NOT_RUN, "- primary task parquet unavailable for extra ablations")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [12c] Diebold–Mariano column for the main table + full-metric flat-DeBERTa row

The paper's main comparison table has a DM column (each model vs the Council) and a
Flat-DeBERTa row with the full metric set. `compare_models` does not emit either, so this
cell (1) reads each model's saved test predictions, (2) computes a DM p-value of every
model against the fine-tuned Council (falling back to the frozen Council if the fine-tune
did not converge), and (3) scores the `flat_encoder` ablation with the full metric set as
a `flat_deberta` row. Output → `results/model_comparison_dm.csv`.
"""))
A(code(r"""
from src.evaluation import diebold_mariano_test, evaluate_model

# Collect every model's (y_true, y_pred): baselines/council live in results + frozen/finetune dirs.
def _load_preds(path):
    if path and os.path.exists(path):
        d = pd.read_parquet(path); return d["y_true"].to_numpy(), d["y_pred"].to_numpy()
    return None

_pred_sources = {
    "council_frozen": os.path.join(OUTPUT_DIR, "frozen", "evaluation", f"predictions_{PRIMARY}.parquet"),
    "council_finetune": os.path.join(OUTPUT_DIR, "finetune", "evaluation", f"predictions_{PRIMARY}.parquet"),
    "flat_deberta": os.path.join(OUTPUT_DIR, "ablation", "flat_encoder", PRIMARY, "evaluation", f"predictions_{PRIMARY}.parquet"),
}
_preds = {k: _load_preds(p) for k, p in _pred_sources.items()}
_preds = {k: v for k, v in _preds.items() if v is not None}

# Reference for DM: prefer the fine-tuned Council, else frozen.
_ref_key = "council_finetune" if "council_finetune" in _preds else ("council_frozen" if "council_frozen" in _preds else None)

dm_rows = []
if _ref_key is not None:
    rt, rp = _preds[_ref_key]
    # Baseline predictions are not persisted per-model by compare_models; where they are
    # unavailable the DM cell records NaN (honest) rather than fabricating a value.
    for name, (yt, yp) in _preds.items():
        dm = float("nan")
        if name != _ref_key and len(yp) == len(rp):
            dm = diebold_mariano_test(rt - rp, np.asarray(yt) - np.asarray(yp))
        rec = evaluate_model(yt, yp, cfg=config, model=name)
        dm_rows.append({"model": name, "n": rec.n, "r_squared": rec.r_squared,
                        "directional_accuracy": rec.directional_accuracy, "spearman": rec.spearman,
                        "mse": rec.mse, "mae": rec.mae, "dm_vs_council": dm})
dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv(os.path.join(OUTPUT_DIR, "results", "model_comparison_dm.csv"), index=False)
print("DM reference model:", _ref_key)
print(dm_df.to_string(index=False) if len(dm_df) else (ns.NOT_RUN + " - no council predictions to compare against"))
print("\nNOTE: DM against the sklearn/lexicon/FinBERT baselines requires their per-model "
      "test predictions; those rows stay NaN unless baseline predictions are persisted.")
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [12d] Corpus statistics — mean tokens & segments per event per split

Fills the coverage table's `Mean tokens / event` and `Mean segments / event` rows using
the SAME tokenizer + chunking the model uses (`chunk_size`, `max_chunks`, `chunk_overlap`).
Output → `tables/corpus_stats.csv`.
"""))
A(code(r"""
from src.training import build_tokenizer

_tok = build_tokenizer(config)
_chunk = int(_deb.get("chunk_size", 512)); _overlap = int(_deb.get("chunk_overlap", 64))
_maxseg = int(_deb.get("max_chunks", 4))
_stride = max(1, _chunk - _overlap)

def _seg_count(n_tokens):
    if n_tokens <= _chunk: return 1
    return min(_maxseg, 1 + int(np.ceil((n_tokens - _chunk) / _stride)))

_split_of = splits.set_index("event_id")["split"].to_dict()
_text_col = "text" if "text" in master.columns else ("segment_text" if "segment_text" in master.columns else None)
corpus_rows = []
if _text_col is not None:
    _mm = master[["event_id", _text_col]].dropna(subset=[_text_col]).copy()
    _mm["split"] = _mm["event_id"].map(_split_of)
    for split in ("train", "val", "test"):
        _s = _mm[_mm["split"] == split]
        if not len(_s):
            corpus_rows.append({"split": split, "n_docs": 0, "mean_tokens": None, "mean_segments": None}); continue
        _tok_counts = _s[_text_col].astype(str).map(lambda t: len(_tok.encode(t, add_special_tokens=False)))
        corpus_rows.append({"split": split, "n_docs": int(len(_s)),
                            "mean_tokens": float(_tok_counts.mean()),
                            "mean_segments": float(_tok_counts.map(_seg_count).mean())})
corpus_stats_df = pd.DataFrame(corpus_rows)
os.makedirs(os.path.join(ARTIFACT_ROOT, "tables"), exist_ok=True)
corpus_stats_df.to_csv(os.path.join(ARTIFACT_ROOT, "tables", "corpus_stats.csv"), index=False)
print(corpus_stats_df.to_string(index=False) if len(corpus_stats_df) else (ns.NOT_RUN + " - no text column in master"))
"""))

# ---------------------------------------------------------------------------
A(md("## [13] Regime audit + corrected leakage classification (unchanged protocol)"))
A(code(r"""
from src.leakage_audit import audit_leakage, write_report as write_leak_report

_leak_frame = splits.copy()
if primary_col and primary_col in master.columns and "target" not in _leak_frame.columns:
    _leak_frame = _leak_frame.merge(
        master[["event_id", primary_col]].rename(columns={primary_col: "target"}),
        on="event_id", how="left")
_scaler_meta_path = os.path.join(AUDIT_DIR, "signal_scalers.json")
_scaler_meta = None
if os.path.exists(_scaler_meta_path):
    with open(_scaler_meta_path, encoding="utf-8") as _fh:
        _scaler_meta = json.load(_fh)
_dapt_audit = os.path.join(AUDIT_DIR, "dapt_corpus_audit.json")
raw_leak = audit_leakage(_leak_frame, scaler_meta=_scaler_meta,
    run_meta={"selection_split": "val", "hyperparameter_search_split": "val"},
    dapt_corpus_audit=_dapt_audit if os.path.exists(_dapt_audit) else None)
write_leak_report(raw_leak, AUDIT_DIR)
leak_class = ns.classify_leakage(raw_leak, splits=_leak_frame)
with open(os.path.join(ARTIFACT_ROOT, "leakage_audit", "final_leakage_audit.json"), "w") as fh:
    json.dump({"raw": raw_leak, "classified": leak_class.to_dict()}, fh, indent=2, default=str)
print("Raw leaking checks:", raw_leak["leaking_checks"])
print("CORRECTED VERDICT:", leak_class.verdict)
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [14] Publication tables + figures + final report

Copies the run-generated comparison table, per-signal metrics, ablation table, and CV
summary into `tables/`, draws the comparison figure, and writes the final report +
readiness checklist. Every number is read from a saved artifact.
"""))
A(code(r"""
import shutil
TABLES_DIR = os.path.join(ARTIFACT_ROOT, "tables"); FIGURES_DIR = os.path.join(ARTIFACT_ROOT, "figures")
os.makedirs(TABLES_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

for src_path, dst_name in [
    (os.path.join(OUTPUT_DIR, "results", "model_comparison.csv"), "main_comparison.csv"),
    (os.path.join(OUTPUT_DIR, "results", "model_comparison.tex"), "main_comparison.tex"),
    (os.path.join(OUTPUT_DIR, "results", "ablation_results.csv"), "ablation_results.csv"),
    (os.path.join(ARTIFACT_ROOT, "per_signal", "per_signal_metrics.csv"), "per_signal_metrics.csv"),
    (os.path.join(ARTIFACT_ROOT, "cross_validation", "cv_summary.json"), "cv_summary.json"),
]:
    if os.path.exists(src_path):
        shutil.copy(src_path, os.path.join(TABLES_DIR, dst_name)); print("copied", dst_name)

if comparison_df is not None and len(comparison_df):
    from src.visualization import plot_model_comparison
    _cols = [c for c in ("model","r_squared","directional_accuracy","spearman") if c in comparison_df.columns]
    print("figures:", plot_model_comparison(comparison_df[_cols], FIGURES_DIR, cfg=config) or ns.NOT_RUN)
"""))

A(code(r"""
with banner("[14] Final report + readiness checklist + artifact index"):
    facts = ns.gather_run_facts(OUTPUT_DIR, registry=registry, artifact_root=ARTIFACT_ROOT)
    md_path, json_path = ns.build_final_report(
        OUTPUT_DIR, os.path.join(ARTIFACT_ROOT, "final_report"),
        facts=facts, leakage_classification=leak_class,
        extra={"mode": MODE, "config_hash": CFG_HASH,
               "limitations": ["Single seed; multi-seed variance not characterized."]})
    checklist = ns.readiness_checklist(facts, leakage_classification=leak_class)
    step(f"final report -> {md_path}")
    print("\nPUBLICATION STATUS:", "READY" if checklist["ready"] else "NOT READY", flush=True)
    for item in checklist.get("blocking", []):
        print("  -", item, flush=True)
    registry.set_status(RUN_ID, ns.STATE_COMPLETED, stage="final_report",
                        best_checkpoint_path=os.path.join(OUTPUT_DIR, "finetune"))
    print("\n  Artifacts under:", ARTIFACT_ROOT, flush=True)
    print("    main comparison :", os.path.join(OUTPUT_DIR, "results", "model_comparison.csv"))
    print("    main + DM       :", os.path.join(OUTPUT_DIR, "results", "model_comparison_dm.csv"))
    _reported_stance = facts.get("stance") or {}
    if _reported_stance.get("available"):
        print("    stance summary  :", _reported_stance["summary_path"])
        print("    stance events   :", _reported_stance["event_predictions_path"])
        print("    stance series   :", _reported_stance["series_path"])
    else:
        print("    stance          :", ns.NOT_RUN, "-",
              _reported_stance.get("detail") or _reported_stance.get("availability_reason"))
    print("    ablation        :", os.path.join(OUTPUT_DIR, "results", "ablation_results.csv"))
    print("    per-signal      :", os.path.join(ARTIFACT_ROOT, "per_signal", "per_signal_metrics.csv"))
    print("    cross-validation:", os.path.join(ARTIFACT_ROOT, "cross_validation", "cv_summary.json"))
    print("    corpus stats    :", os.path.join(ARTIFACT_ROOT, "tables", "corpus_stats.csv"))
    print(f"\n  PIPELINE COMPLETE. Total wall time: {_elapsed(time.time() - _PIPELINE_T0)}", flush=True)
"""))

# ---------------------------------------------------------------------------
A(md(r"""
## [15] Paste-ready LaTeX rows for the manuscript's pending tables

Prints each `\pending{}` table's rows already formatted so you can paste the numbers
straight into `paper/sections/results.tex`. Every value is read from the artifacts this
run produced; a cell with no data prints `\pending` so nothing is fabricated.
"""))
A(code(r"""
def _f(x, nd=4):
    try:
        import math
        if x is None or (isinstance(x, float) and math.isnan(x)): return r"\pending"
        return f"${float(x):.{nd}f}$"
    except Exception:
        return r"\pending"

def _row(cells):
    return " & ".join(cells) + r" \\"

print("="*72); print("tab:coverage  (mean tokens / segments per event)"); print("="*72)
if len(corpus_stats_df):
    _cs = corpus_stats_df.set_index("split")
    def _g(split, col):
        try: return _f(_cs.loc[split, col], 1)
        except Exception: return r"\pending"
    print(_row(["Mean tokens / event", _g("train","mean_tokens"), _g("val","mean_tokens"), _g("test","mean_tokens")]))
    print(_row(["Mean segments / event", _g("train","mean_segments"), _g("val","mean_segments"), _g("test","mean_segments")]))

print("\n" + "="*72); print("tab:main  (Council fine-tuned + flat DeBERTa + DM column)"); print("="*72)
if len(dm_df):
    _d = dm_df.set_index("model")
    def _mrow(label, key):
        if key not in _d.index: 
            return _row([label] + [r"\pending"]*6)
        r = _d.loc[key]
        return _row([label, _f(r.r_squared), _f(r.directional_accuracy), _f(r.spearman),
                     _f(r.mse), _f(r.mae), _f(r.get("dm_vs_council"))])
    print(_mrow("Flat DeBERTa baseline", "flat_deberta"))
    print(_mrow(r"\textbf{Council (fine-tuned)}", "council_finetune"))

print("\n" + "="*72); print("tab:ablation  (all variants)"); print("="*72)
_abl = os.path.join(OUTPUT_DIR, "results", "ablation_results.csv")
if os.path.exists(_abl):
    _ab = pd.read_csv(_abl)
    _label = {"full":"Full Council (hierarchical, uncertainty)","flat_encoder":"$-$ hierarchical (flat encoder)",
              "minus_fixed_fusion":"$-$ uncertainty fusion ($+$ fixed weights)","minus_direction":"$-$ direction objective",
              "minus_uncertainty":"$-$ heteroscedastic uncertainty","minus_temporal":"$-$ temporal objective",
              "minus_contrastive":"$-$ contrastive objective","minus_stance":"$-$ stance objective",
              "minus_dapt":"$-$ domain-adaptive pretraining"}
    for key, label in _label.items():
        _r = _ab[_ab["ablation"] == key]
        if len(_r):
            r = _r.iloc[0]
            print(_row([label, _f(r.get("r_squared")), _f(r.get("directional_accuracy")), _f(r.get("spearman"))]))
        else:
            print(_row([label, r"\pending", r"\pending", r"\pending"]))

print("\n" + "="*72); print("tab:persignal  (per-signal R2 / dir / fusion weight)"); print("="*72)
if len(per_signal_df):
    _ps = per_signal_df[per_signal_df["mode"] == "finetune"].set_index("signal")
    for sig in ("ois2y","ois5y","ois10y","equity"):
        if sig in _ps.index and _ps.loc[sig, "status"] == "OK":
            r = _ps.loc[sig]
            print(_row([f"\\texttt{{{sig}}}", _f(r.get("r_squared")), _f(r.get("directional_accuracy")), _f(r.get("eff_fusion_weight"))]))
        else:
            print(_row([f"\\texttt{{{sig}}}", r"\pending", r"\pending", r"\pending"]))

print("\n" + "="*72); print("tab:cv  (rolling-origin + LOMO mean, std)"); print("="*72)
for scheme, label in (("rolling_origin","Rolling-origin"),("leave_one_meeting_out","Leave-one-meeting-out")):
    s = cv_summary.get(scheme, {})
    if s.get("mean"):
        m, sd = s["mean"], s.get("std", {})
        def _ms(k):
            try: return f"${m[k]:.4f}\\pm{sd.get(k, float('nan')):.4f}$"
            except Exception: return r"\pending"
        print(_row([label, _ms("r_squared"), _ms("directional_accuracy"), str(s.get("n_usable_folds", r"\pending"))]))
    else:
        print(_row([label, r"\pending", r"\pending", r"\pending"]))
"""))


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "L4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(NB_PATH, "w") as fh:
        json.dump(nb, fh, indent=1)
    print("wrote", NB_PATH, "with", len(CELLS), "cells")


if __name__ == "__main__":
    build()
