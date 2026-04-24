"""
DETECTive — end-to-end orchestrator.

One command runs every stage the paper reproduction needs:

    python pipeline.py                          # eval + plots from existing weights
    python pipeline.py --with-training          # TRAIN first, then eval + plots
    python pipeline.py --with-training --fresh  # from-scratch training + full eval
    python pipeline.py --skip-benchmarks        # eval + plots, no external tools
    python pipeline.py --skip-plots             # just numbers

Stages (run in this order, any can be skipped):

  0. training     -> trains (or resumes) the DETECTiveModel; produces
                     checkpoint_last.pt + best_detective_model.pt in DATA_DIR.
                     Off by default because it takes hours. Enable with
                     `--with-training`. Needs train_dataset.pkl +
                     val_dataset.pkl in DATA_DIR (parent folder by default).
  1. analysis     -> results/results_reconvergence.csv,
                     results/results_fault_depth.csv,
                     results/results_accuracy_summary.csv,
                     results/accuracy_report.txt
  2. benchmarks   -> results/results_benchmarks.csv
                     (downloads ISCAS-85, synthesizes with yosys+abc,
                      runs DETECTive + ATALANTA for K sampled faults each)
  3. visualization-> results/*.png (Fig 5b, 6a, 6/runtime, 7 benchmarks,
                                   training curves)

Each stage is a thin wrapper around the corresponding module. If a stage
fails, later stages run anyway (and skip inputs they don't have).
"""

from __future__ import annotations
import sys, argparse, traceback
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import BEST_MODEL, CHECKPOINT, RESULTS_DIR, TRAIN_PKL, VAL_PKL, EPOCHS, LR


def _banner(msg: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {msg}\n{bar}\n")


def stage_training(epochs: int, lr: float, fresh: bool) -> bool:
    _banner(f"STAGE 0/3: training ({epochs} epochs{' from scratch' if fresh else ''})")
    if not TRAIN_PKL.exists() or not VAL_PKL.exists():
        print(f"[training] missing dataset pickles:")
        print(f"           {TRAIN_PKL}  exists={TRAIN_PKL.exists()}")
        print(f"           {VAL_PKL}   exists={VAL_PKL.exists()}")
        print(f"           Upload them to the parent directory (or set DETECTIVE_DATA_DIR).")
        return False
    try:
        import training, pickle
        with open(TRAIN_PKL, "rb") as f: td = pickle.load(f)
        with open(VAL_PKL,   "rb") as f: vd = pickle.load(f)
        print(f"[training] train={len(td)}  val={len(vd)} samples")
        training.train(td, vd, num_epochs=epochs, lr=lr, resume=not fresh)
        return True
    except Exception as e:
        print(f"[training] FAILED: {e}")
        traceback.print_exc()
        return False


def stage_analysis(model: Path) -> bool:
    _banner("STAGE 1/3: analysis (synthetic accuracy breakdowns + 90% report)")
    try:
        import analysis
        analysis.analyze(model_path=model)
        return True
    except Exception as e:
        print(f"[analysis] FAILED: {e}")
        traceback.print_exc()
        return False


def stage_benchmarks(model: Path, faults: int) -> bool:
    _banner(f"STAGE 2/3: benchmarks (ISCAS-85, {faults} faults per design)")
    try:
        import benchmarks
        benchmarks.run_benchmarks(model_path=model, faults=faults)
        return True
    except Exception as e:
        print(f"[benchmarks] FAILED: {e}")
        traceback.print_exc()
        return False


def stage_visualization(checkpoint: Path) -> bool:
    _banner("STAGE 3/3: visualization (reproduce paper figures)")
    try:
        import visualization
        visualization.generate_all(checkpoint_path=checkpoint)
        return True
    except Exception as e:
        print(f"[visualization] FAILED: {e}")
        traceback.print_exc()
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default=str(BEST_MODEL),
                   help="weights used by analysis + benchmarks (default: best_detective_model.pt)")
    p.add_argument("--checkpoint", default=str(CHECKPOINT),
                   help="checkpoint file used to extract training history for the curves plot")
    p.add_argument("--faults",     type=int, default=30,
                   help="faults per ISCAS design (default: 30)")
    p.add_argument("--with-training",    action="store_true",
                   help="run training as stage 0 (off by default)")
    p.add_argument("--fresh",            action="store_true",
                   help="with --with-training, ignore any existing checkpoint")
    p.add_argument("--epochs",           type=int, default=EPOCHS,
                   help=f"total epochs for training stage (default {EPOCHS})")
    p.add_argument("--lr",               type=float, default=LR,
                   help=f"learning rate for training stage (default {LR})")
    p.add_argument("--skip-analysis",    action="store_true")
    p.add_argument("--skip-benchmarks",  action="store_true")
    p.add_argument("--skip-plots",       action="store_true")
    args = p.parse_args()

    model      = Path(args.model)
    checkpoint = Path(args.checkpoint)
    results    = {}

    if args.with_training:
        results["training"] = stage_training(args.epochs, args.lr, args.fresh)

    if not model.exists():
        if args.with_training and not results.get("training"):
            sys.exit("[ERROR] training stage failed and no existing weights to evaluate.")
        elif not args.with_training:
            sys.exit(f"[ERROR] model weights not found: {model}\n"
                     f"        Run `python pipeline.py --with-training` or "
                     f"`python training.py` first.")

    if not args.skip_analysis:
        results["analysis"] = stage_analysis(model)
    if not args.skip_benchmarks:
        results["benchmarks"] = stage_benchmarks(model, args.faults)
    if not args.skip_plots:
        results["visualization"] = stage_visualization(checkpoint)

    _banner("SUMMARY")
    for stage, ok in results.items():
        tag = "OK " if ok else "FAIL"
        print(f"  [{tag}] {stage}")
    print(f"\n  All outputs -> {RESULTS_DIR}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
