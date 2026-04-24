"""
DETECTive — central configuration.

All paths and hyperparameters live here so every module (training, evaluation,
visualization) reads the same source of truth. Override at runtime by setting
the matching environment variable (e.g. DETECTIVE_DATA_DIR=...).
"""

from __future__ import annotations
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────
# The submission directory is self-contained, but large pickles + model
# checkpoints are expected in the parent folder (the "messy" dev folder)
# to avoid duplicating 120+ MB of data. Override with DETECTIVE_DATA_DIR.

SUBMISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT   = SUBMISSION_DIR.parent

DATA_DIR       = Path(os.environ.get("DETECTIVE_DATA_DIR", PROJECT_ROOT))
RESULTS_DIR    = SUBMISSION_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_PKL       = DATA_DIR / "train_dataset.pkl"
VAL_PKL         = DATA_DIR / "val_dataset.pkl"
BEST_MODEL      = DATA_DIR / "best_detective_model.pt"
CHECKPOINT      = DATA_DIR / "checkpoint_last.pt"

# Benchmark directories (populated by benchmarks.py)
BENCH_DIR_RAW   = DATA_DIR / "benchmarks_bench"      # ISCAS-85 .bench files
BENCH_DIR_SYNTH = DATA_DIR / "benchmarks"             # Yosys NAND+NOT Verilog

# ── Gate vocabulary ──────────────────────────────────────────────
GATE_TYPES   = ["INPUT", "AND", "NAND", "OR", "NOR", "XOR", "XNOR", "NOT", "BUF"]
GATE_TO_IDX  = {g: i for i, g in enumerate(GATE_TYPES)}

# Feature vector layout per node:
#   [one-hot gate type (9 bits)] + [is_faulty flag] + [log1p(fanout)]
FEATURE_DIM  = len(GATE_TYPES) + 2         # = 11

# ── Model hyperparameters (match paper section 5) ─────────────────
HIDDEN_DIM   = 32                          # GNN hidden + LSTM hidden
P_PATHS      = 10                          # max activation/propagation paths
LR           = 1e-3
EPOCHS       = 250

# ── Evaluation defaults ───────────────────────────────────────────
K_FAULTS_PER_BENCHMARK = 30
ATPG_TIMEOUT_SEC       = 60
