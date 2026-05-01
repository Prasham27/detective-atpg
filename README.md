# DETECTive — Learned ATPG vs Classical (PODEM / D-Algorithm / FAN)

End-to-end replication of **DETECTive: Machine Learning-driven Automatic Test
Pattern Prediction for Faults in Digital Circuits** (Petrolo, Medya, Graziano,
Pal — GLSVLSI '24), benchmarked head-to-head against the three classical
ATPG algorithms — **PODEM** (Goel 1981), the **D-algorithm** (Roth 1966),
and **FAN** (Fujiwara–Shimono 1983) — on the full ISCAS-85 suite.

The learned model is a Graph Attention Network + two LSTMs + an MLP that
takes a faulted gate-level netlist and predicts a stuck-at test pattern in
**one forward pass**, no backtracking. We trained it on **15 000
(circuit, fault) samples generated from 2 000 random 4-input circuits**
(depth 4–30, 8 faults each — same setup as paper Section 5) for 100
epochs and then measured fault coverage, bit-accuracy, and ms/fault
across all three classical baselines plus the learned model.

**Headline result on c432** (160 gates, contains a redundant fault that
exhausts PODEM's search tree): **DETECTive is 17.6× faster than PODEM** with
85.5 % bit-accuracy vs PODEM's full ground truth.

---

## Directory guide — what each folder is for

| Path | What's in it | When to open |
|---|---|---|
| [`DETECTive_submission/`](DETECTive_submission/) | **Main learned-ATPG submission.** Modular PyTorch code, training script, full evaluation pipeline, 4-way comparison notebook, full setup guide. | Always — start here. See its [README](DETECTive_submission/README.md) and [SETUP.md](DETECTive_submission/SETUP.md). |
| [`100_epoch_run/`](100_epoch_run/) | Trained weights + training summary + ISCAS-85 benchmark CSV from the 100-epoch run (best val 0.8358 @ ep 40). | When you want inference without re-training. |
| [`netlists/`](netlists/) | 11 ISCAS-85 gate-level Verilog files: c17, c432, c499, c880, c1908, c1355, c2670, c3540, c5315, c6288, c7552. | Shared by all 4 notebooks. |
| [`PODEM/`](PODEM/) | Clean PODEM notebook with theory cells (D-algebra, singular cover, PDCF, PDCs) + full ATPG engine + ISCAS-85 benchmark sweep. | Classical baseline #1. |
| [`D_Algorithm/`](D_Algorithm/) | Roth's D-algorithm (line-by-line decision making, exponential worst case). ISCAS-85 sweep appended. | Classical baseline #2. |
| [`FAN_Algorithm/`](FAN_Algorithm/) | FAN (multiple backtrace, headlines, unique sensitisation). ISCAS-85 sweep appended. | Classical baseline #3. |
| [`ppt/`](ppt/) | Project presentation — `Presentation.pdf` and `Presentation.pptx`. | Open to review the slide deck. |
| [`report/`](report/) | Final BTech Mini Project Report — `report.pdf` (the deliverable) and `report.tex` (LaTeX source). | Read `report.pdf` for the full architecture walkthrough, results, and four-way comparison. |
| [`DETECTive_paper.pdf`](DETECTive_paper.pdf) | The original GLSVLSI '24 paper this work reproduces. | Reference. |
| [`full_pipeline.ipynb`](full_pipeline.ipynb) | Complete end-to-end walkthrough notebook (7 phases): circuit graph builder, GNN representation, LSTM path encoders, cone embedding, master model, dataset generation, and training loop. Produces `train_dataset.pkl` / `val_dataset.pkl`. | Read this if you want the full pipeline narrated cell by cell, or to regenerate the training data. |
| `best_detective_model.pt` | Active best-val weights (261 KB). | Automatically loaded by the four-way comparison notebook. |
| `checkpoint_last.pt` | Resume-from-epoch-100 checkpoint (760 KB). | Run `python DETECTive_submission/training.py --epochs 300` to continue. |

### Datasets included in the repo

| Path | Size | Needed for |
|---|---|---|
| `train_dataset.pkl` | 91 MB | Training + `analysis.py` |
| `val_dataset.pkl` | 23 MB | Validation + accuracy breakdown |

Both pickles are committed so the repo is self-contained — no extra
download step. If you want to regenerate them from scratch, open
`full_pipeline.ipynb` and run the Phase 6 cells.

### Not in the repo (regenerate locally)

| Path | Size | How to regenerate |
|---|---|---|
| `venv/` | 5.6 GB | `python -m venv venv && venv\Scripts\activate && pip install -r DETECTive_submission/requirements.txt` |

---

## Quick start (after clone)

```bash
# 1. Clone
git clone https://github.com/Prasham27/detective-atpg.git
cd detective-atpg

# 2. Python environment
python -m venv venv
# Windows:      venv\Scripts\activate
# Linux/macOS:  source venv/bin/activate

# 3. Install PyTorch (pick the CUDA version matching your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install -r DETECTive_submission/requirements.txt

# 5. Open the four-way ATPG comparison notebook (plots + ISCAS-85 sweep)
jupyter notebook DETECTive_submission/four_way_comparison.ipynb
```

`four_way_comparison.ipynb` is the one-stop deliverable — it loads the 100-epoch weights,
runs DETECTive on any ISCAS-85 netlist you point at, and renders a 4-way
comparison (PODEM | D-algorithm | FAN | DETECTive) with all plots embedded.

### Just want to see the numbers without running anything?

Every notebook in the repo is committed with its outputs already populated —
open any of them on GitHub and scroll through.

### Run the classical ATPG notebooks

```bash
jupyter notebook PODEM/PODEM.ipynb                      # PODEM
jupyter notebook D_Algorithm/D_Algorithm.ipynb          # D-algorithm
jupyter notebook FAN_Algorithm/FAN_Algorithm.ipynb      # FAN
```

Each has an **ISCAS-85 Benchmark Sweep** section near the bottom that
iterates over the shared `netlists/` folder.

### Retrain from scratch / resume training

```bash
# Regenerate datasets first (if pickles aren't present)
jupyter notebook full_pipeline.ipynb                     # Run All

# Fresh training from epoch 0
python DETECTive_submission/training.py --fresh --epochs 100

# Or resume from the provided checkpoint_last.pt (currently at ep 100)
python DETECTive_submission/training.py --epochs 300
```

---

## What each classical algorithm gives you (one-line summary)

- **PODEM** — decisions on primary inputs only; complete; exponential
  worst-case but usually fine on non-redundant circuits.
- **D-algorithm** — decisions on any internal line; complete; worst
  backtracking behaviour of the three classical methods.
- **FAN** — PODEM + testability heuristics (multiple backtrace, headlines,
  unique sensitisation) — faster than PODEM in practice on most circuits.
- **DETECTive** — one forward pass of a trained GNN+LSTM; heuristic (may
  miss faults); runtime is gate-count-dependent but backtracking-free.

---

## Paper citation

```
Petrolo, V., Medya, S., Graziano, M., Pal, D.
DETECTive: Machine Learning-driven Automatic Test Pattern Prediction for
Faults in Digital Circuits.
Great Lakes Symposium on VLSI 2024 (GLSVLSI '24).
DOI: 10.1145/3649476.3658696
```

## License

MIT. See [`LICENSE`](LICENSE).
