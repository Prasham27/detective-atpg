"""
visualization.py
================

Reproduce the key figures from the DETECTive paper (GLSVLSI '24) using
matplotlib. Every plot is saved as a PNG into ``RESULTS_DIR``.

This module is both importable (used from ``pipeline.py``) and runnable
as a CLI entry point.

CLI usage
---------
    python visualization.py
    python visualization.py --checkpoint path/to/ckpt.pt
    python visualization.py --results  path/to/results_dir

Figures produced
----------------
    training_curves.png      - loss / train-acc / val-acc vs epoch
    fig5a_size_depth.png     - Fig 5a: input_size vs depth heatmap (optional)
    fig5b_reconvergence.png  - Fig 5b: num_act_paths vs num_prop_paths heatmap
    fig6a_fault_depth.png    - Fig 6a: bit accuracy vs fault depth
    fig6_runtime.png         - Fig 6b/c inspired runtime comparison
    fig7_benchmarks.png      - Fig 7b: per-design bit accuracy + gate count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from config import CHECKPOINT, BEST_MODEL, RESULTS_DIR  # noqa: F401

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    # Older matplotlib versions (or missing style) -> fall back to default.
    pass

_LINE_FIGSIZE = (10, 6)
_HEATMAP_FIGSIZE = (8, 7)
_SAVE_KWARGS = dict(dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _ensure_parent(out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def _save_and_close(fig: plt.Figure, out_path: Path) -> Path:
    out_path = _ensure_parent(out_path)
    fig.tight_layout()
    fig.savefig(out_path, **_SAVE_KWARGS)
    plt.close(fig)
    return out_path


def _load_checkpoint_history(checkpoint_path: Path) -> Optional[Mapping[str, Any]]:
    """Load just the ``history`` dict from a torch checkpoint, if possible."""
    try:
        import torch  # type: ignore
    except Exception as exc:
        print(f"  -> could not import torch to load checkpoint ({exc})")
        return None

    try:
        # map_location=cpu so we never need a GPU just to plot.
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    except Exception as exc:
        print(f"  -> failed to load checkpoint {checkpoint_path}: {exc}")
        return None

    if not isinstance(ckpt, dict):
        print(f"  -> checkpoint at {checkpoint_path} is not a dict; no history")
        return None

    history = ckpt.get("history")
    if not isinstance(history, dict):
        print(f"  -> checkpoint at {checkpoint_path} has no 'history' dict")
        return None
    return history


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(history: Mapping[str, Any], out_path: Path) -> Path:
    """Three stacked subplots: training loss, train accuracy, val accuracy.

    A vertical line marks ``best_epoch`` with a short annotation that also
    prints ``best_val`` if present.
    """
    train_loss = list(history.get("train_loss", []))
    train_acc = list(history.get("train_accuracy", []))
    val_acc = list(history.get("val_accuracy", []))
    best_epoch = history.get("best_epoch", None)
    best_val = history.get("best_val", None)

    n = max(len(train_loss), len(train_acc), len(val_acc))
    if n == 0:
        raise ValueError("history contains no per-epoch curves to plot")
    epochs = np.arange(1, n + 1)

    fig, axes = plt.subplots(3, 1, figsize=_LINE_FIGSIZE, sharex=True)

    ax_loss, ax_tr, ax_va = axes

    if train_loss:
        ax_loss.plot(epochs[: len(train_loss)], train_loss,
                     color="#C0392B", label="train loss")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training curves")
    ax_loss.legend(loc="upper right")

    if train_acc:
        ax_tr.plot(epochs[: len(train_acc)], train_acc,
                   color="#2980B9", label="train accuracy")
    ax_tr.set_ylabel("Train acc")
    ax_tr.set_ylim(0.0, 1.0)
    ax_tr.legend(loc="lower right")

    if val_acc:
        ax_va.plot(epochs[: len(val_acc)], val_acc,
                   color="#27AE60", label="val accuracy")
    ax_va.set_ylabel("Val acc")
    ax_va.set_xlabel("Epoch")
    ax_va.set_ylim(0.0, 1.0)
    ax_va.legend(loc="lower right")

    # Best-epoch marker across all three subplots.
    if isinstance(best_epoch, (int, float)) and best_epoch >= 0:
        be = int(best_epoch)
        # history might be 0-indexed; be tolerant.
        x = be + 1 if be < n else be
        annotation = f"best epoch {be}"
        if isinstance(best_val, (int, float)):
            annotation += f"\nbest val = {best_val:.4f}"
        for ax in axes:
            ax.axvline(x=x, color="#7F8C8D", linestyle="--", linewidth=1)
        ax_va.annotate(
            annotation,
            xy=(x, ax_va.get_ylim()[0]),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=9,
            color="#34495E",
        )

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Reconvergence heatmap (Fig 5b)
# ---------------------------------------------------------------------------
def plot_reconvergence_heatmap(csv_path: Path, out_path: Path) -> Path:
    """10x10 heatmap. X = num_prop_paths, Y = num_act_paths (inverted).

    Colour = bit_accuracy. Each cell is annotated with the accuracy value.
    """
    df = pd.read_csv(csv_path)

    max_axis = 10
    grid = np.full((max_axis, max_axis), np.nan, dtype=float)

    for _, row in df.iterrows():
        try:
            a = int(row["num_act_paths"])
            p = int(row["num_prop_paths"])
        except Exception:
            continue
        if not (1 <= a <= max_axis and 1 <= p <= max_axis):
            continue
        # Y axis runs 1..10 but inverted so 1 is at the bottom.
        # We store by (act-1, prop-1) then flip vertically at plot time.
        grid[a - 1, p - 1] = float(row["bit_accuracy"])

    # Flip so that num_act_paths = 1 sits at the bottom of the image.
    display = np.flipud(grid)

    fig, ax = plt.subplots(figsize=_HEATMAP_FIGSIZE)
    im = ax.imshow(
        display,
        cmap="viridis",
        aspect="auto",
        vmin=np.nanmin(display) if np.isfinite(np.nanmin(display)) else 0.0,
        vmax=np.nanmax(display) if np.isfinite(np.nanmax(display)) else 1.0,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Bit accuracy")

    ax.set_xticks(np.arange(max_axis))
    ax.set_xticklabels([str(i) for i in range(1, max_axis + 1)])
    ax.set_yticks(np.arange(max_axis))
    # After flipud, row 0 corresponds to num_act_paths = max_axis.
    ax.set_yticklabels([str(i) for i in range(max_axis, 0, -1)])

    ax.set_xlabel("num_prop_paths")
    ax.set_ylabel("num_act_paths")
    ax.set_title("Fig 5b: Accuracy under Reconvergent Fanout")

    # Cell annotations.
    for i in range(max_axis):
        for j in range(max_axis):
            v = display[i, j]
            if not np.isfinite(v):
                continue
            # Pick text colour for contrast with viridis.
            txt_color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Fault depth line plot (Fig 6a)
# ---------------------------------------------------------------------------
def plot_fault_depth(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path).sort_values("max_act_path_len")

    fig, ax = plt.subplots(figsize=_LINE_FIGSIZE)
    ax.plot(
        df["max_act_path_len"].values,
        df["bit_accuracy"].values,
        marker="o",
        color="#2C3E50",
        linewidth=2,
        label="bit accuracy",
    )
    ax.axhline(y=0.9, color="#C0392B", linestyle="--",
               linewidth=1.2, label="90% threshold")

    ax.set_xlabel("Fault depth (max activation-path length)")
    ax.set_ylabel("Bit accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Fig 6a: Bit Accuracy vs Fault Depth")
    ax.legend(loc="lower left")

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Benchmark accuracy (Fig 7b)
# ---------------------------------------------------------------------------
def plot_benchmark_accuracy(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path).sort_values("gate_count", ascending=True)

    designs = df["design"].astype(str).tolist()
    acc = df["bit_accuracy"].astype(float).to_numpy()
    gc = df["gate_count"].astype(float).to_numpy()
    x = np.arange(len(designs))

    fig, ax = plt.subplots(figsize=_LINE_FIGSIZE)
    bars = ax.bar(x, acc, color="#2980B9", label="bit accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(designs, rotation=30, ha="right")
    ax.set_ylabel("Bit accuracy", color="#2980B9")
    ax.set_ylim(0.0, 1.02)
    ax.tick_params(axis="y", labelcolor="#2980B9")
    ax.set_title("Fig 7b: Per-Design Bit Accuracy (sorted by gate count)")

    for xi, a in zip(x, acc):
        ax.text(xi, a + 0.01, f"{a:.2f}", ha="center", va="bottom", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, gc, color="#E67E22", marker="o", linewidth=1.8,
             label="gate count")
    ax2.set_ylabel("Gate count", color="#E67E22")
    ax2.tick_params(axis="y", labelcolor="#E67E22")
    ax2.grid(False)

    # Combined legend.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # Silence an unused-variable warning.
    _ = bars
    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Runtime comparison (Fig 6b/c inspired)
# ---------------------------------------------------------------------------
def plot_runtime_comparison(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)

    df = df[np.isfinite(pd.to_numeric(df["atalanta_ms"], errors="coerce"))]
    df = df.sort_values("gate_count", ascending=True).reset_index(drop=True)

    if df.empty:
        # Save a placeholder so callers still get a file if they want.
        fig, ax = plt.subplots(figsize=_LINE_FIGSIZE)
        ax.text(0.5, 0.5, "No benchmark rows with finite atalanta_ms",
                ha="center", va="center")
        ax.set_axis_off()
        ax.set_title("Fig 6b/c: Runtime Comparison (no data)")
        return _save_and_close(fig, out_path)

    designs = df["design"].astype(str).tolist()
    det = df["detective_ms"].astype(float).to_numpy()
    ata = df["atalanta_ms"].astype(float).to_numpy()
    speedup = df["speedup"].astype(float).to_numpy()

    x = np.arange(len(designs))
    width = 0.38

    fig, ax = plt.subplots(figsize=_LINE_FIGSIZE)
    b1 = ax.bar(x - width / 2, det, width, color="#27AE60", label="DETECTive")
    b2 = ax.bar(x + width / 2, ata, width, color="#C0392B", label="ATALANTA")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(designs, rotation=30, ha="right")
    ax.set_ylabel("Runtime (ms, log scale)")
    ax.set_title("Fig 6b/c: Runtime Comparison (DETECTive vs ATALANTA)")
    ax.legend(loc="upper left")

    # Annotate speedup above each pair.
    for xi, d, a, s in zip(x, det, ata, speedup):
        top = max(d, a)
        if not np.isfinite(top) or top <= 0:
            continue
        label = f"{s:.1f}x" if np.isfinite(s) else "n/a"
        ax.text(xi, top * 1.25, label, ha="center", va="bottom",
                fontsize=8, color="#34495E")

    _ = (b1, b2)
    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Size-depth heatmap (Fig 5a) -- optional
# ---------------------------------------------------------------------------
def plot_size_depth_heatmap(csv_path: Path, out_path: Path) -> Optional[Path]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"  -> {csv_path.name} not found; skipping Fig 5a")
        return None

    df = pd.read_csv(csv_path)

    sizes = sorted(df["input_size"].dropna().unique().tolist())
    depths = sorted(df["depth"].dropna().unique().tolist())

    # Paper Fig 5a is 5x5. If we have more we still render what's there;
    # if fewer we just use what's available.
    sizes = sizes[:5] if len(sizes) >= 5 else sizes
    depths = depths[:5] if len(depths) >= 5 else depths

    grid = np.full((len(sizes), len(depths)), np.nan, dtype=float)
    size_idx = {s: i for i, s in enumerate(sizes)}
    depth_idx = {d: j for j, d in enumerate(depths)}

    for _, row in df.iterrows():
        s = row.get("input_size")
        d = row.get("depth")
        if s not in size_idx or d not in depth_idx:
            continue
        grid[size_idx[s], depth_idx[d]] = float(row["pattern_accuracy"])

    fig, ax = plt.subplots(figsize=_HEATMAP_FIGSIZE)
    vmin = np.nanmin(grid) if np.isfinite(np.nanmin(grid)) else 0.0
    vmax = np.nanmax(grid) if np.isfinite(np.nanmax(grid)) else 1.0
    im = ax.imshow(grid, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pattern accuracy")

    ax.set_xticks(np.arange(len(depths)))
    ax.set_xticklabels([str(int(d)) if float(d).is_integer() else str(d)
                        for d in depths])
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_yticklabels([str(int(s)) if float(s).is_integer() else str(s)
                        for s in sizes])
    ax.set_xlabel("depth")
    ax.set_ylabel("input_size")
    ax.set_title("Fig 5a: Pattern Accuracy by Size and Depth")

    for i in range(len(sizes)):
        for j in range(len(depths)):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            txt_color = "white" if v < 0.55 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=txt_color)

    return _save_and_close(fig, out_path)


# ---------------------------------------------------------------------------
# generate_all
# ---------------------------------------------------------------------------
def generate_all(
    checkpoint_path: Optional[Path] = None,
    results_dir: Optional[Path] = None,
) -> dict:
    """Generate every plot whose inputs are available.

    Returns a dict with ``generated`` (list of PNG paths) and ``skipped``
    (list of reasons for each skipped figure).
    """
    ckpt_path = Path(checkpoint_path) if checkpoint_path else Path(CHECKPOINT)
    res_dir = Path(results_dir) if results_dir else Path(RESULTS_DIR)
    res_dir.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    skipped: list[str] = []

    print(f"visualization.generate_all -> results_dir = {res_dir}")
    print(f"visualization.generate_all -> checkpoint  = {ckpt_path}")

    # 1. Training curves
    if ckpt_path.exists():
        history = _load_checkpoint_history(ckpt_path)
        if history:
            try:
                p = plot_training_curves(history, res_dir / "training_curves.png")
                generated.append(p)
                print(f"  -> wrote {p.name}")
            except Exception as exc:
                skipped.append(f"training_curves.png: {exc}")
                print(f"  -> FAILED training_curves.png: {exc}")
        else:
            skipped.append("training_curves.png: checkpoint has no history")
    else:
        skipped.append(f"training_curves.png: checkpoint not found at {ckpt_path}")
        print(f"  -> skip training_curves.png (checkpoint missing)")

    # 2. Fig 5b reconvergence
    rec_csv = res_dir / "results_reconvergence.csv"
    if rec_csv.exists():
        try:
            p = plot_reconvergence_heatmap(rec_csv, res_dir / "fig5b_reconvergence.png")
            generated.append(p)
            print(f"  -> wrote {p.name}")
        except Exception as exc:
            skipped.append(f"fig5b_reconvergence.png: {exc}")
            print(f"  -> FAILED fig5b_reconvergence.png: {exc}")
    else:
        skipped.append(f"fig5b_reconvergence.png: {rec_csv.name} missing")
        print(f"  -> skip fig5b_reconvergence.png ({rec_csv.name} missing)")

    # 3. Fig 6a fault depth
    fd_csv = res_dir / "results_fault_depth.csv"
    if fd_csv.exists():
        try:
            p = plot_fault_depth(fd_csv, res_dir / "fig6a_fault_depth.png")
            generated.append(p)
            print(f"  -> wrote {p.name}")
        except Exception as exc:
            skipped.append(f"fig6a_fault_depth.png: {exc}")
            print(f"  -> FAILED fig6a_fault_depth.png: {exc}")
    else:
        skipped.append(f"fig6a_fault_depth.png: {fd_csv.name} missing")
        print(f"  -> skip fig6a_fault_depth.png ({fd_csv.name} missing)")

    # 4/5. Fig 7b + Fig 6b/c (share a CSV)
    bench_csv = res_dir / "results_benchmarks.csv"
    if bench_csv.exists():
        try:
            p = plot_benchmark_accuracy(bench_csv, res_dir / "fig7_benchmarks.png")
            generated.append(p)
            print(f"  -> wrote {p.name}")
        except Exception as exc:
            skipped.append(f"fig7_benchmarks.png: {exc}")
            print(f"  -> FAILED fig7_benchmarks.png: {exc}")

        try:
            p = plot_runtime_comparison(bench_csv, res_dir / "fig6_runtime.png")
            generated.append(p)
            print(f"  -> wrote {p.name}")
        except Exception as exc:
            skipped.append(f"fig6_runtime.png: {exc}")
            print(f"  -> FAILED fig6_runtime.png: {exc}")
    else:
        skipped.append(f"fig7_benchmarks.png: {bench_csv.name} missing")
        skipped.append(f"fig6_runtime.png: {bench_csv.name} missing")
        print(f"  -> skip fig7/fig6 ({bench_csv.name} missing)")

    # 6. Fig 5a size-depth (optional)
    sd_csv = res_dir / "results_size_depth.csv"
    try:
        p = plot_size_depth_heatmap(sd_csv, res_dir / "fig5a_size_depth.png")
        if p is not None:
            generated.append(p)
            print(f"  -> wrote {p.name}")
        else:
            skipped.append(f"fig5a_size_depth.png: {sd_csv.name} missing")
    except Exception as exc:
        skipped.append(f"fig5a_size_depth.png: {exc}")
        print(f"  -> FAILED fig5a_size_depth.png: {exc}")

    # Summary
    print("")
    print("visualization summary")
    print(f"  generated : {len(generated)}")
    for g in generated:
        print(f"    - {g}")
    print(f"  skipped   : {len(skipped)}")
    for s in skipped:
        print(f"    - {s}")

    return {"generated": generated, "skipped": skipped}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate DETECTive paper figures from result CSVs + checkpoint."
    )
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="Override path to the training checkpoint (.pt).")
    ap.add_argument("--results", type=Path, default=None,
                    help="Override results directory (defaults to config.RESULTS_DIR).")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    # Prevent interactive backends from trying to pop up windows during a
    # headless run.
    matplotlib.use(matplotlib.get_backend(), force=False)

    generate_all(checkpoint_path=args.checkpoint, results_dir=args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
