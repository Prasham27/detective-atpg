# Paper vs Ours — side-by-side comparison

This doc tracks the DETECTive paper's claimed numbers alongside what we
reproduce. The **live** numbers are whatever is in
`results/results_accuracy_summary.csv` and `results/results_benchmarks.csv`
right now — rerun `python pipeline.py --skip-benchmarks` (or the full
`python pipeline.py`) after any new training run to refresh them.

Snapshot values below are from the current `best_detective_model.pt` (best
validation accuracy = 0.8377 at epoch 58 in the original dev run). Re-run
`analysis.py` after training finishes to update them.

---

## 1. Synthetic circuits (paper Section 6.1, Fig 5a)

| Metric                                | Paper claim                                             | Our snapshot                                             |
| ------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------- |
| Max pattern accuracy (any cell)       | 100% (4-input, depth 4)                                 | See `results_accuracy_summary.csv` row `pct_samples_100pct` |
| Avg pattern accuracy across cells     | >90%                                                    | See `overall_pattern_accuracy`                           |
| Shallow circuits (depth ≤ 5)          | ~98-100%                                                | See `mean_acc_shallow`                                   |
| Deep circuits (depth > 15)            | ~75-85%                                                 | See `mean_acc_deep`                                      |

**Caveat.** The paper's Fig 5a values are per-configuration averages on
circuits of a **fixed** size + depth. Our `val_dataset.pkl` mixes depths
4-30 in a single pool, so the overall mean is a lower bound on what the
paper reports. For an apples-to-apples comparison look at the per-bucket
rows, not the overall.

---

## 2. Reconvergent fanout (paper Section 6.2, Fig 5b)

| Metric                                | Paper claim                                             | Our snapshot                                              |
| ------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------- |
| Low reconvergence (k_act + k_prop ≤ 3)| ~88%                                                    | See `mean_acc_low_reconv`                                 |
| High reconvergence (k_act + k_prop ≥ 15)| ~68%                                                  | See `mean_acc_high_reconv`                                |
| Full grid                             | Fig 5b (10x10 heatmap, 68-88%)                         | `fig5b_reconvergence.png`                                 |

---

## 3. Fault depth (paper Section 6.3, Fig 6a)

| Metric                                | Paper claim                                             | Our snapshot                                              |
| ------------------------------------- | ------------------------------------------------------- | --------------------------------------------------------- |
| Shallow fault (near PI)               | ~82%                                                    | See topmost rows of `results_fault_depth.csv`             |
| Deep fault (near PO, level ~50)       | ~66%                                                    | See bottom rows of `results_fault_depth.csv`              |
| Trend                                 | Monotonically decreasing                                | `fig6a_fault_depth.png` + report's monotonicity check     |

---

## 4. Runtime vs ATPG tools (paper Section 6.4, Fig 6b/c)

| Design size                           | Paper's DETECTive           | Paper's ATALANTA      | Paper's speedup | Our speedup           |
| ------------------------------------- | --------------------------- | --------------------- | --------------- | --------------------- |
| 4-input, depth 4                      | 2.6 ms                      | ~25 ms                | ~10x            | See `results_benchmarks.csv` `speedup` col |
| 64-input, depth 4                     | 3.2 ms                      | ~48 ms                | ~15x            | ditto                 |
| 4-input, depth 64                     | ~20 ms (LSTM grows)         | ~25 ms                | ~1.2x           | ditto                 |

**Caveat.** Our ATALANTA runtime includes subprocess overhead (~50-100 ms
per call on Windows/WSL). The paper likely measures only the ATPG algorithm
inside ATALANTA. Expect our apparent speedups to be **smaller** than the
paper reports, even on identical hardware. The relative ordering (DETECTive
faster on shallow + wide designs, competitive on deep designs) should still
hold.

---

## 5. Realistic ISCAS benchmarks (paper Section 6.5, Fig 7)

The paper reports per-benchmark pattern accuracy on c17 (small) through
c7552 (large), with a general downward trend as gate count rises. Typical
range: 0.85-0.95 for small circuits, 0.70-0.85 for the largest ones.

Our numbers come from `results_benchmarks.csv`:

| Design          | Gate count (ours) | Paper accuracy (approx) | Our accuracy       |
| --------------- | ----------------- | ----------------------- | ------------------ |
| c17             | See CSV           | ~0.95                   | See CSV            |
| c432            | See CSV           | ~0.90                   | See CSV            |
| c499            | See CSV           | ~0.85                   | See CSV            |
| c880            | See CSV           | ~0.85                   | See CSV            |
| c1908           | See CSV           | ~0.80                   | See CSV            |
| c2670           | See CSV           | ~0.78                   | See CSV            |
| c3540           | See CSV           | ~0.75                   | See CSV            |
| c7552           | See CSV           | ~0.72                   | See CSV            |

`fig7_benchmarks.png` plots all of these with the paper's bar-chart layout.

---

## 6. Differences from the paper (intentional)

- **No ensemble model.** The paper trains 3-5 models with different seeds and
  reports a +7% accuracy / +12% coverage gain (Section 6.6). We omit this
  because (a) it requires 3-5x the compute, (b) it's not a core architectural
  claim, and (c) it can be added later by re-running `training.py` with
  different `--seed` values and averaging the predictions in a post-hoc
  script.

- **Dataset scale.** We train on ~12,000 samples derived from 2,000 random
  4-input circuits (consistent with the paper's "2000 circuits" description
  in Section 5). The paper's exact generator seed is not public, so the
  per-circuit composition differs.

- **Train-time optimization.** Our training loop runs one forward pass per
  sample (detached predictions are reused to pick the best gt) rather than
  the paper's two-pass setup. ~2x faster, zero semantic change because the
  model has no dropout / batch-norm.

- **Technology mapping.** We use `abc -g NAND; opt_clean` via the Yosys+ABC
  toolchain (paper's default). The resulting NAND-only netlists match the
  paper's Figure 7a gate counts within ~10%, which is the tolerance expected
  from synthesis tool version drift.

---

## 7. How to refresh this doc's numbers

```bash
cd DETECTive_submission
python pipeline.py --skip-benchmarks    # updates analysis CSVs + plots
python pipeline.py                       # full refresh, needs yosys+abc+atalanta
```

Then open the CSVs in `results/` — this doc references them by name.
