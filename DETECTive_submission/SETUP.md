# DETECTive — Ultimate Setup & Usage Guide

This is the **only** document you need to reproduce every result in the
DETECTive paper (GLSVLSI '24) from this submission. Follow it top to bottom.

> **TL;DR — the one-command run.** Upload `train_dataset.pkl` and
> `val_dataset.pkl` into the parent folder, then:
> ```
> cd DETECTive_submission
> python run_all.py
> ```
> That trains the model to completion, runs every evaluation stage, and
> drops all plots + CSVs + the 90% verification report into `results/`.
>
> **Just want the plots from the existing weights?** Skip training with
> `python pipeline.py --skip-benchmarks` (~2 minutes).

---

## 0. What's in this folder

| File                            | Purpose                                                              |
| ------------------------------- | -------------------------------------------------------------------- |
| `run_all.py`                    | **one-shot runner** — train + evaluate + plot in a single command    |
| `pipeline.py`                   | orchestrator (training optional via `--with-training`)               |
| `config.py`                     | central paths + hyperparameters (one source of truth)                |
| `circuits.py`                   | Verilog parser + graph builder + path extractor                      |
| `models.py`                     | GNN + LSTM modules + full DETECTiveModel                             |
| `training.py`                   | training loop, runnable as `python training.py`                      |
| `evaluation.py`                 | accuracy metrics used by everything else                             |
| `atalanta.py`                   | subprocess wrapper around the ATALANTA ATPG tool                     |
| `analysis.py`                   | synthetic-data accuracy breakdowns + 90% verification report         |
| `benchmarks.py`                 | ISCAS-85 download, synthesis, DETECTive vs ATALANTA evaluation       |
| `visualization.py`              | reproduces paper figures 5b, 6a, 6b/c, 7 from the CSVs + history     |
| `fault_sim.py`                  | clean 2-valued gate-level fault simulator (`parse_netlist`, `simulate_fault_detected`) — used by `demo.ipynb` for honest fault-coverage numbers |
| `demo.ipynb`                    | 4-way ATPG comparison (PODEM / D-Alg / FAN / ATPP) on c17, c432, c499, c880, c1908 — bit-accuracy, fault-sim coverage, runtime, paper Fig 5a/6a replications |
| `requirements.txt`              | pip dependencies                                                     |
| `results/`                      | all generated output (CSVs, PNGs, text reports)                      |
| `docs/paper_comparison.md`      | our numbers vs the paper's claims, side by side                      |

**Data and weights** live in the parent directory by default
(`../train_dataset.pkl`, `../val_dataset.pkl`, `../best_detective_model.pt`,
`../checkpoint_last.pt`). Override with `DETECTIVE_DATA_DIR=...` if your
data is elsewhere.

---

## 1. Prerequisites

| Tool        | Needed for                 | Install                                                                     |
| ----------- | -------------------------- | --------------------------------------------------------------------------- |
| Python 3.10+| everything                 | <https://www.python.org/>                                                   |
| PyTorch     | training + inference       | `pip install torch --index-url https://download.pytorch.org/whl/cu121`      |
| CUDA 12.x   | GPU training (optional)    | comes with NVIDIA drivers                                                   |
| **Yosys**   | benchmark synthesis        | <https://yosyshq.net/yosys/>                                                |
| **ABC**     | benchmark synthesis        | <https://github.com/berkeley-abc/abc>                                       |
| **ATALANTA**| benchmark ground truth + runtime baseline | `git clone https://github.com/hsluoyz/Atalanta && cd Atalanta && make` |

Yosys, ABC, and ATALANTA are **only** required for `benchmarks.py`. You can
reproduce the paper's synthetic-data results (Sections 6.1, 6.2, 6.3) and all
the plots without any of them — use `pipeline.py --skip-benchmarks`.

---

## 2. One-time environment setup

```bash
# 1. Create a venv (keep the project isolated)
python -m venv venv

# 2. Activate it
#    Windows PowerShell:  venv\Scripts\Activate.ps1
#    Windows CMD:         venv\Scripts\activate.bat
#    Linux / macOS:       source venv/bin/activate

# 3. Install PyTorch with CUDA (RTX 30xx needs cu121 or cu118)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install -r requirements.txt

# 5. Sanity check
python -c "import torch; print('CUDA OK:', torch.cuda.is_available())"
python -c "import config, circuits, models, training, evaluation, atalanta, benchmarks, visualization, analysis; print('all 9 modules import OK')"
```

### Optional external tools

```bash
# ATALANTA (needed only for ISCAS benchmarks stage)
git clone https://github.com/hsluoyz/Atalanta.git
cd Atalanta && make
export ATALANTA_BIN=$(pwd)/atalanta         # Linux / macOS
$env:ATALANTA_BIN = "$(pwd)\atalanta.exe"    # Windows PowerShell
```

---

## 3. The script cheat sheet

Every script is both **importable** (from other scripts) and **runnable**
(`python <name>.py --help` shows its flags). Here's the full list of CLI
options, grouped by purpose.

### 3.1  `training.py` — train or resume

```bash
python training.py                      # resume from checkpoint_last.pt
python training.py --fresh              # ignore any existing checkpoint
python training.py --epochs 250         # run until epoch 250 (default)
python training.py --lr 5e-4            # override learning rate (default 1e-3)
```

Checkpoints save every 10 epochs to `../checkpoint_last.pt`. The best
validation-accuracy weights are saved to `../best_detective_model.pt`. You
can Ctrl-C any time — an emergency checkpoint is written.

### 3.2  `analysis.py` — synthetic-data breakdowns + 90% verdict

```bash
python analysis.py                      # uses best_detective_model.pt + val_dataset.pkl
python analysis.py --model ../checkpoint_last.pt
python analysis.py --out-dir ./results
```

Outputs:
- `results/results_reconvergence.csv`   (Fig 5b)
- `results/results_fault_depth.csv`     (Fig 6a)
- `results/results_accuracy_summary.csv` (headline metrics for the 90% question)
- `results/accuracy_report.txt`          (human-readable verdict + gap analysis)

### 3.3  `benchmarks.py` — ISCAS-85 evaluation

```bash
python benchmarks.py                                      # full pipeline (download + synth + eval)
python benchmarks.py --skip-download                      # reuse benchmarks_bench/
python benchmarks.py --skip-synth                         # reuse benchmarks/ (must have .v files)
python benchmarks.py --designs c17 c432 c880              # only run these
python benchmarks.py --faults 50                          # 50 sampled faults per design (default 30)
python benchmarks.py --atalanta /path/to/atalanta         # override binary location
```

Output: `results/results_benchmarks.csv` with per-design bit_accuracy,
detective_ms, atalanta_ms, and speedup.

### 3.4  `visualization.py` — reproduce paper figures

```bash
python visualization.py                                   # read all CSVs + checkpoint
python visualization.py --checkpoint ../checkpoint_last.pt
```

Writes PNGs into `results/`:
- `training_curves.png`      (loss + train/val acc vs epoch)
- `fig5b_reconvergence.png`  (paper Fig 5b heatmap)
- `fig6a_fault_depth.png`    (paper Fig 6a line plot)
- `fig6_runtime.png`         (paper Fig 6b/c runtime comparison)
- `fig7_benchmarks.png`      (paper Fig 7 benchmark accuracy)
- `fig5a_size_depth.png`     (paper Fig 5a — only if the size/depth sweep CSV exists)

### 3.5  `pipeline.py` — evaluation orchestrator

```bash
python pipeline.py                      # analysis + benchmarks + plots
python pipeline.py --with-training      # train first, then all the above
python pipeline.py --skip-benchmarks    # no external tools required
python pipeline.py --skip-plots         # just numbers
python pipeline.py --faults 50          # deeper benchmark eval
python pipeline.py --fresh --with-training --epochs 300    # train from scratch
```

### 3.6  `run_all.py` — **zero-flags one-shot runner**

```bash
python run_all.py                       # train + evaluate + plot, everything
python run_all.py --fresh               # from-scratch training
python run_all.py --skip-benchmarks     # skip ISCAS stage (no external tools)
```

This is the single command to hand a new machine. It imports and calls
`pipeline.py --with-training`, forwarding any additional flags. If the
datasets aren't present yet, it prints a clear message and exits.

---

## 4. Typical workflows

### 4.1  "I just want the plots from the current weights"

```bash
cd DETECTive_submission
python pipeline.py --skip-benchmarks
open results/        # or: explorer results
```

Time: ~2 minutes on CPU, <1 minute on GPU.

### 4.2  "I want to verify the 90% claim myself"

```bash
cd DETECTive_submission
python analysis.py
cat results/accuracy_report.txt
```

The report prints a PAPER vs OURS table, a verdict (yes/no on the 90%
target), a gap analysis pointing at the hardest sample buckets, and a
roadmap of what to try next.

### 4.3  "I want to train from scratch and reproduce end-to-end"

```bash
cd DETECTive_submission

# One command (needs Yosys + ABC + ATALANTA for the benchmark stage)
python run_all.py --fresh --epochs 250

# Or equivalently:
python pipeline.py --with-training --fresh --epochs 250
```

If you don't have the external tools yet, add `--skip-benchmarks`:

```bash
python run_all.py --fresh --epochs 250 --skip-benchmarks
```

### 4.4  "I want to resume partially-trained weights"

```bash
cd DETECTive_submission
python training.py                      # picks up checkpoint_last.pt automatically
```

### 4.5  "I want benchmark numbers but I don't have ATALANTA"

```bash
cd DETECTive_submission
python benchmarks.py                    # will still time DETECTive, just no ground-truth comparison
python visualization.py                 # plots what it can
```

The `results_benchmarks.csv` will have `N/A` in the accuracy + atalanta
columns, but `detective_ms` is filled in for every design.

---

## 5. Troubleshooting

| Symptom                                                       | Fix                                                                                           |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: torch_geometric`                        | `pip install torch_geometric`                                                                 |
| `UnicodeEncodeError: 'charmap' codec can't encode...`         | Already handled — every CLI script does `sys.stdout.reconfigure(encoding='utf-8')` at import. |
| `CUDA out of memory`                                          | Lower `HIDDEN_DIM` in `config.py` from 32 to 16, or switch to CPU.                            |
| `[benchmarks] yosys not found on PATH`                        | Install Yosys (section 1). Or run `pipeline.py --skip-benchmarks`.                            |
| `[benchmarks] atalanta not found`                             | Install ATALANTA (section 1) or set `ATALANTA_BIN`. Benchmarks still record DETECTive runtime. |
| Training per-epoch time keeps growing                         | Already handled by periodic `torch.cuda.empty_cache()` every 500 samples. If it recurs, reduce dataset size. |
| `checkpoint_last.pt not found` when running `training.py`     | Start fresh: `python training.py --fresh`.                                                    |
| Pandas complains about CSV dtype                              | Delete stale CSVs in `results/` and rerun `analysis.py`.                                      |

---

## 6. Understanding the 90% question

The paper claims **"average >90%, up to 100% pattern accuracy on synthetic
circuits"** (Abstract + Section 6.1). Important nuances:

- **Pattern accuracy** = bit-wise match to the closest ground-truth test
  pattern — *not* full-pattern equality. Most test patterns have many
  don't-care bits, making per-bit match easier than it sounds.
- The paper's Fig 5a averages are per-**circuit-configuration** (fixed
  input size + depth). Their highest cell is 100%, and their lowest is 79%.
  "Average >90%" is the mean across configurations, not across individual
  samples.
- Our val_dataset.pkl mixes depths 4–30 with 4 PIs — heterogeneous. The
  overall mean here will be lower than any one of the paper's Fig 5a
  cells in isolation.

`analysis.py` reports **both** the overall mean AND the per-bucket means
(shallow / medium / deep, low / high reconvergence). Compare bucket-wise
to the paper's figures — that's the apples-to-apples comparison.

Verdict criteria:
- **Overall 90% target**: `mean(bit_accuracy) >= 0.9` on the full val set.
- **Bucket 90% target**: any bucket in the Fig 5b or Fig 6a CSV has
  `bit_accuracy >= 0.9`.
- **Shallow 90% target**: `mean_acc_shallow >= 0.9` (samples with max
  activation-path length ≤ 5).

`accuracy_report.txt` answers each of these explicitly.

---

## 7. Command quick-reference

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Train
python training.py                        # resume
python training.py --fresh --epochs 250   # from scratch

# Evaluate
python analysis.py                        # synthetic + 90% report
python benchmarks.py                      # ISCAS + ATALANTA
python visualization.py                   # plots

# Everything
python pipeline.py                        # all three stages
python pipeline.py --skip-benchmarks      # no external tools
```

---

## 8. Where to look next

- `results/accuracy_report.txt`           — the 90% verdict
- `results/results_accuracy_summary.csv`  — headline metrics in CSV form
- `results/*.png`                         — paper-figure reproductions
- `docs/paper_comparison.md`              — our numbers vs the paper's, side by side

---

## 9. 4-way ATPG comparison demo (`demo.ipynb`)

`demo.ipynb` runs PODEM, D-Algorithm, FAN, and ATPP head-to-head on five
ISCAS-85 circuits (c17, c432, c499, c880, c1908) and renders every figure
needed for the writeup. Two metrics are reported side by side:

- **Bit-accuracy** — the paper-faithful metric (Section 5 of the paper).
  Per-bit match against the closest classical-ATPG ground-truth pattern.
  This is what the bar charts and Fig 5a-style heatmap report.
- **Fault-sim coverage** — a stricter optional metric computed via
  `fault_sim.py` (clean 2-valued simulator with `parse_netlist()` and
  `simulate_fault_detected()`). Reports the fraction of injected stuck-at
  faults each predicted pattern actually detects at a primary output.
  Included as an honesty addendum next to the bit-accuracy bars.

Sections 7 and 8 of the notebook reproduce paper Fig 5a (size × depth
heatmap) and Fig 6a (accuracy vs fault depth curve) from this run.
