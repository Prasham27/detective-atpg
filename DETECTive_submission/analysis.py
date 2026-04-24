"""
DETECTive -- synthetic accuracy analysis (paper Sections 6.1, 6.2, 6.3).

Replicates (as far as the existing val_dataset.pkl allows) the three headline
accuracy breakdowns from the DETECTive paper and answers the bigger question
that drives this submission: "can our trained model reach the paper's claimed
>=90% accuracy?"

The existing val_dataset.pkl holds 3000 4-input circuits with depth 4..30, so
we cannot resize inputs on the fly -- Fig 5a (accuracy per input_size x depth)
can only be partially reproduced via the depth axis.  Fig 5b (reconvergence)
and Fig 6a (fault depth) are fully reproducible.

Outputs written to RESULTS_DIR:
  results_reconvergence.csv   Fig 5b -- bit-acc by (# act paths, # prop paths)
  results_fault_depth.csv     Fig 6a -- bit-acc by max activation-path length
  results_accuracy_summary.csv summary statistics that answer the 90% question
  accuracy_report.txt         human-readable report (paper vs ours + verdict)

CLI:
  python analysis.py                            # BEST_MODEL + VAL_PKL + RESULTS_DIR
  python analysis.py --model checkpoint_last.pt
  python analysis.py --val path/to/other.pkl --out-dir ./somewhere
"""

from __future__ import annotations

# Ensure stdout/stderr don't crash on Windows consoles when we print ASCII
# arrows etc. in an older CP1252 codepage. Best-effort only.
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import csv
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from config import (
    BEST_MODEL,
    CHECKPOINT,
    FEATURE_DIM,
    HIDDEN_DIM,
    P_PATHS,
    RESULTS_DIR,
    VAL_PKL,
)
from evaluation import compute_pattern_accuracy, to_device
from models import DETECTiveModel


# ----------------------------------------------------------------------
#  Model loading
# ----------------------------------------------------------------------
def load_detective_model(model_path: Path, device: torch.device) -> DETECTiveModel:
    """Instantiate DETECTiveModel and load weights from ``model_path``.

    Handles both file formats we produce during training:
      * bare state_dict  (best_detective_model.pt)
      * checkpoint dict  (checkpoint_last.pt -- has 'model_state_dict')
    """
    model = DETECTiveModel(
        in_channels=FEATURE_DIM,
        hidden_channels=HIDDEN_DIM,
        p=P_PATHS,
    ).to(device)
    # weights_only=False because our checkpoints may contain optimizer state
    # alongside the model state; torch >=2.4 defaults to True and would fail.
    state = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------------------------------------------------
#  CSV helpers
# ----------------------------------------------------------------------
def _write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Thin wrapper so we consistently log what we wrote."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}  ({len(rows)} rows)")


# ----------------------------------------------------------------------
#  Main analysis
# ----------------------------------------------------------------------
def _checkpoint_info(model_path: Path, device: torch.device) -> Dict[str, Any]:
    """Peek at the checkpoint to surface epoch/val_acc in the text report."""
    info: Dict[str, Any] = {"path": str(model_path), "epoch": None, "best_val_acc": None}
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            for k in ("epoch", "global_epoch", "step"):
                if k in state and isinstance(state[k], (int, float)):
                    info["epoch"] = int(state[k])
                    break
            for k in ("best_val_acc", "val_acc", "best_acc"):
                if k in state and isinstance(state[k], (int, float)):
                    info["best_val_acc"] = float(state[k])
                    break
    except Exception:
        # Best-effort metadata only -- the report will simply omit missing
        # fields rather than blowing up the analysis.
        pass
    return info


def analyze(model_path: Optional[Path] = None,
            val_path:   Optional[Path] = None,
            results_dir: Optional[Path] = None,
            device:     Optional[torch.device] = None) -> Dict[str, Any]:
    """Run the full synthetic-accuracy breakdown.

    Writes three CSVs and one text report into ``results_dir`` and returns a
    summary dict that pipeline.py can consume.
    """
    model_path  = Path(model_path   or BEST_MODEL)
    val_path    = Path(val_path     or VAL_PKL)
    results_dir = Path(results_dir  or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"model weights not found: {model_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val dataset not found: {val_path}")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device -> {device}")

    print(f"[INFO] loading val dataset from {val_path}")
    with open(val_path, "rb") as f:
        val = pickle.load(f)
    print(f"       {len(val)} samples loaded")

    print(f"[INFO] loading model from {model_path}")
    model = load_detective_model(model_path, device)
    ckpt_info = _checkpoint_info(model_path, device)

    # ------------------------------------------------------------------
    #  Per-sample inference. We keep a flat list (one record per sample)
    #  so the summary metrics can bucket the same data many ways without
    #  re-running inference.
    # ------------------------------------------------------------------
    records: List[Dict[str, Any]] = []
    overall_bit_sum     = 0.0
    overall_pattern_sum = 0.0
    skipped             = 0

    with torch.no_grad():
        for i, s in enumerate(val):
            # Samples without gt_patterns or PIs carry no learning signal --
            # treat them as missing rather than zero so they don't drag the
            # mean down artificially.
            if not s.get("gt_patterns") or not s.get("pi_indices"):
                skipped += 1
                continue
            sd = to_device(s, device)
            preds = model(
                sd["graph"], sd["pi_indices"], sd["fault_type"],
                sd.get("act_paths"), sd.get("prop_paths"),
            ).view(-1).tolist()

            bit_acc     = compute_pattern_accuracy(preds, s["gt_patterns"])
            # "Strict" pattern accuracy -- 1.0 only if every bit matches some
            # ground-truth pattern. This is how training.py scores.
            pattern_acc = 1.0 if bit_acc == 1.0 else 0.0

            k_act  = len(s.get("act_paths")  or [])
            k_prop = len(s.get("prop_paths") or [])
            ap_lens = [len(p) for p in (s.get("act_paths") or []) if p]
            max_ap_len = max(ap_lens) if ap_lens else 0

            records.append({
                "bit_acc":      bit_acc,
                "pattern_acc":  pattern_acc,
                "k_act":        min(max(k_act,  1), 10),
                "k_prop":       min(max(k_prop, 1), 10),
                "max_ap_len":   max_ap_len,
            })
            overall_bit_sum     += bit_acc
            overall_pattern_sum += pattern_acc

            if (i + 1) % 500 == 0:
                running = overall_bit_sum / max(len(records), 1)
                print(f"   processed {i+1}/{len(val)}  running bit-acc={running:.4f}")

    n = len(records)
    if n == 0:
        raise RuntimeError("no evaluable samples in val set")
    print(f"[INFO] evaluated {n} samples  (skipped {skipped})")

    overall_bit_acc     = overall_bit_sum     / n
    overall_pattern_acc = overall_pattern_sum / n
    print(f"[RESULT] overall bit accuracy     = {overall_bit_acc:.4f}")
    print(f"[RESULT] overall pattern accuracy = {overall_pattern_acc:.4f}")

    # ------------------------------------------------------------------
    #  CSV 1 -- Fig 5b reconvergence heatmap
    # ------------------------------------------------------------------
    reconv_acc: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for r in records:
        reconv_acc[(r["k_act"], r["k_prop"])].append(r["bit_acc"])

    reconv_rows: List[List[Any]] = []
    for k_act in range(1, 11):
        for k_prop in range(1, 11):
            accs = reconv_acc.get((k_act, k_prop), [])
            c    = len(accs)
            mean = (sum(accs) / c) if c else 0.0
            reconv_rows.append([k_act, k_prop, c, f"{mean:.4f}"])
    _write_csv(
        results_dir / "results_reconvergence.csv",
        ["num_act_paths", "num_prop_paths", "count", "bit_accuracy"],
        reconv_rows,
    )

    # ------------------------------------------------------------------
    #  CSV 2 -- Fig 6a fault depth
    # ------------------------------------------------------------------
    depth_acc: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        if r["max_ap_len"] > 0:
            depth_acc[r["max_ap_len"]].append(r["bit_acc"])

    depth_rows: List[List[Any]] = []
    for d in sorted(depth_acc):
        accs = depth_acc[d]
        mean = sum(accs) / len(accs)
        depth_rows.append([d, len(accs), f"{mean:.4f}"])
    _write_csv(
        results_dir / "results_fault_depth.csv",
        ["max_act_path_len", "count", "bit_accuracy"],
        depth_rows,
    )

    # ------------------------------------------------------------------
    #  CSV 3 -- summary statistics
    # ------------------------------------------------------------------
    bits = [r["bit_acc"] for r in records]

    def _frac(pred) -> float:
        return sum(1 for x in bits if pred(x)) / n

    def _bucket_mean(pred) -> Tuple[float, int]:
        vals = [r["bit_acc"] for r in records if pred(r)]
        if not vals:
            return 0.0, 0
        return sum(vals) / len(vals), len(vals)

    pct_100    = _frac(lambda x: x == 1.0)
    pct_ge_90  = _frac(lambda x: x >= 0.9)
    pct_ge_75  = _frac(lambda x: x >= 0.75)
    pct_lt_50  = _frac(lambda x: x < 0.5)

    shallow_mean,  shallow_n  = _bucket_mean(lambda r: 0 < r["max_ap_len"] <= 5)
    medium_mean,   medium_n   = _bucket_mean(lambda r: 6 <= r["max_ap_len"] <= 15)
    deep_mean,     deep_n     = _bucket_mean(lambda r: r["max_ap_len"] > 15)
    low_re_mean,   low_re_n   = _bucket_mean(lambda r: (r["k_act"] + r["k_prop"]) <= 3)
    high_re_mean,  high_re_n  = _bucket_mean(lambda r: (r["k_act"] + r["k_prop"]) >= 15)

    summary_rows: List[List[Any]] = [
        ["overall_bit_accuracy",     f"{overall_bit_acc:.4f}",     n,
         "Mean per-sample bit accuracy against best-matching GT pattern"],
        ["overall_pattern_accuracy", f"{overall_pattern_acc:.4f}", n,
         "Strict pattern accuracy (1.0 only if every bit matches)"],
        ["pct_samples_100pct",       f"{pct_100:.4f}",             n,
         "Fraction with bit_accuracy == 1.0"],
        ["pct_samples_ge_90pct",     f"{pct_ge_90:.4f}",           n,
         "Fraction with bit_accuracy >= 0.9  (paper's headline target)"],
        ["pct_samples_ge_75pct",     f"{pct_ge_75:.4f}",           n,
         "Fraction with bit_accuracy >= 0.75"],
        ["pct_samples_lt_50pct",     f"{pct_lt_50:.4f}",           n,
         "Fraction below 0.5 -- below-chance samples"],
        ["mean_acc_shallow",         f"{shallow_mean:.4f}",        shallow_n,
         "max act-path len <= 5 (small shallow circuits -- Fig 5a top-left)"],
        ["mean_acc_medium",          f"{medium_mean:.4f}",         medium_n,
         "6 <= max act-path len <= 15"],
        ["mean_acc_deep",            f"{deep_mean:.4f}",           deep_n,
         "max act-path len > 15 (Fig 6a right tail)"],
        ["mean_acc_low_reconv",      f"{low_re_mean:.4f}",         low_re_n,
         "k_act + k_prop <= 3 (Fig 5b top-left, minimal reconvergence)"],
        ["mean_acc_high_reconv",     f"{high_re_mean:.4f}",        high_re_n,
         "k_act + k_prop >= 15 (Fig 5b bottom-right, max reconvergence)"],
    ]
    _write_csv(
        results_dir / "results_accuracy_summary.csv",
        ["metric", "value", "n_samples", "notes"],
        summary_rows,
    )

    # ------------------------------------------------------------------
    #  Text report -- the deliverable that answers the 90% question
    # ------------------------------------------------------------------
    # Any bucket >= 0.9 means "there EXISTS a sub-regime where the model hits
    # paper-grade accuracy" -- a weaker but still informative claim than a
    # global 0.9 mean.
    best_reconv_bucket   = max(reconv_acc.items(),
                               key=lambda kv: sum(kv[1]) / len(kv[1])) if reconv_acc else None
    best_depth_bucket    = max(depth_acc.items(),
                               key=lambda kv: sum(kv[1]) / len(kv[1])) if depth_acc  else None
    worst_depth_bucket   = min(depth_acc.items(),
                               key=lambda kv: sum(kv[1]) / len(kv[1])) if depth_acc  else None

    any_reconv_ge_90 = any(
        (sum(a) / len(a)) >= 0.9 for a in reconv_acc.values() if len(a) >= 5
    )
    any_depth_ge_90 = any(
        (sum(a) / len(a)) >= 0.9 for a in depth_acc.values() if len(a) >= 5
    )

    report_path = results_dir / "accuracy_report.txt"
    _write_report(
        report_path,
        ckpt_info         = ckpt_info,
        val_path          = val_path,
        n                 = n,
        overall_bit_acc   = overall_bit_acc,
        overall_pattern   = overall_pattern_acc,
        pct_100           = pct_100,
        pct_ge_90         = pct_ge_90,
        pct_ge_75         = pct_ge_75,
        pct_lt_50         = pct_lt_50,
        shallow           = (shallow_mean, shallow_n),
        medium            = (medium_mean,  medium_n),
        deep              = (deep_mean,    deep_n),
        low_re            = (low_re_mean,  low_re_n),
        high_re           = (high_re_mean, high_re_n),
        best_reconv       = best_reconv_bucket,
        best_depth        = best_depth_bucket,
        worst_depth       = worst_depth_bucket,
        any_reconv_ge_90  = any_reconv_ge_90,
        any_depth_ge_90   = any_depth_ge_90,
        depth_rows        = depth_rows,
    )
    print(f"  wrote {report_path}")

    return {
        "n_samples":             n,
        "overall_bit_accuracy":  overall_bit_acc,
        "overall_pattern_accuracy": overall_pattern_acc,
        "pct_samples_ge_90pct":  pct_ge_90,
        "pct_samples_100pct":    pct_100,
        "mean_acc_shallow":      shallow_mean,
        "mean_acc_deep":         deep_mean,
        "mean_acc_low_reconv":   low_re_mean,
        "mean_acc_high_reconv":  high_re_mean,
        "any_bucket_ge_90":      any_reconv_ge_90 or any_depth_ge_90,
        "results_dir":           str(results_dir),
    }


# ----------------------------------------------------------------------
#  Report rendering
# ----------------------------------------------------------------------
def _write_report(path: Path, **kw: Any) -> None:
    """Render accuracy_report.txt.

    Split out of analyze() so the numerical logic and the presentation logic
    don't get tangled -- also makes the template easier to tweak.
    """
    ckpt     = kw["ckpt_info"]
    bit      = kw["overall_bit_acc"]
    pattern  = kw["overall_pattern"]
    shallow  = kw["shallow"]
    medium   = kw["medium"]
    deep     = kw["deep"]
    low_re   = kw["low_re"]
    high_re  = kw["high_re"]

    # Verdict booleans -- the "did we hit 90%?" answer has three granularities.
    verdict_overall  = bit >= 0.9
    verdict_shallow  = shallow[0] >= 0.9
    verdict_anywhere = kw["any_reconv_ge_90"] or kw["any_depth_ge_90"]

    # Paper numbers sourced from the DETECTive paper (Sec 6.1--6.3) to keep
    # the comparison table self-contained -- no external look-up needed.
    lines: List[str] = []
    W = 78
    hr = "=" * W
    sub = "-" * W

    lines.append(hr)
    lines.append("DETECTive -- Synthetic Accuracy Report".center(W))
    lines.append(hr)
    lines.append(f"Generated          : {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Checkpoint         : {ckpt.get('path')}")
    if ckpt.get("epoch") is not None:
        lines.append(f"Checkpoint epoch   : {ckpt['epoch']}")
    if ckpt.get("best_val_acc") is not None:
        lines.append(f"Checkpoint val_acc : {ckpt['best_val_acc']:.4f}")
    lines.append(f"Validation set     : {kw['val_path']}  ({kw['n']} samples)")
    lines.append("")

    # ── Paper vs ours table ──────────────────────────────────────────
    lines.append(sub)
    lines.append("PAPER CLAIMS vs THIS MODEL")
    lines.append(sub)
    lines.append(f"{'Metric':<42} {'Paper':>16} {'Ours':>16}")
    lines.append(f"{'-'*42} {'-'*16} {'-'*16}")
    rows: List[Tuple[str, str, str]] = [
        ("Fig 5a: best-case small+shallow bit-acc", ">= 0.95  (up to 1.00)", f"{shallow[0]:.4f}"),
        ("Fig 5a: typical mean bit-acc",            ">= 0.90",               f"{bit:.4f}"),
        ("Fig 5b: best-case reconvergence bit-acc", "~0.88",                 _fmt_bucket(kw["best_reconv"])),
        ("Fig 5b: max-reconvergence bit-acc",       "~0.68",                 f"{high_re[0]:.4f}"),
        ("Fig 6a: shallow-fault bit-acc",           "~0.82",                 f"{shallow[0]:.4f}"),
        ("Fig 6a: deep-fault bit-acc",              "~0.66",                 f"{deep[0]:.4f}"),
        ("Pattern-level (strict) accuracy",         "not reported",          f"{pattern:.4f}"),
    ]
    for label, paper, ours in rows:
        lines.append(f"{label:<42} {paper:>16} {ours:>16}")
    lines.append("")

    # ── Distribution of bit-accuracies ───────────────────────────────
    lines.append(sub)
    lines.append("BIT-ACCURACY DISTRIBUTION")
    lines.append(sub)
    lines.append(f"  bit_acc == 1.00 : {kw['pct_100']   *100:6.2f}%   "
                 f"(perfect predictions)")
    lines.append(f"  bit_acc >= 0.90 : {kw['pct_ge_90'] *100:6.2f}%   "
                 f"(paper-grade samples)")
    lines.append(f"  bit_acc >= 0.75 : {kw['pct_ge_75'] *100:6.2f}%")
    lines.append(f"  bit_acc <  0.50 : {kw['pct_lt_50'] *100:6.2f}%   "
                 f"(below-chance failures)")
    lines.append("")
    lines.append("  Shallow  (depth <= 5 ) : "
                 f"{shallow[0]:.4f}  over {shallow[1]} samples")
    lines.append("  Medium   (6..15)       : "
                 f"{medium[0]:.4f}  over {medium[1]}  samples")
    lines.append("  Deep     (> 15)        : "
                 f"{deep[0]:.4f}  over {deep[1]}  samples")
    lines.append("  Low  reconv (<= 3 paths): "
                 f"{low_re[0]:.4f}  over {low_re[1]} samples")
    lines.append("  High reconv (>=15 paths): "
                 f"{high_re[0]:.4f}  over {high_re[1]} samples")
    lines.append("")

    # ── Verdict ──────────────────────────────────────────────────────
    lines.append(sub)
    lines.append("VERDICT -- 90% TARGET REACHED?")
    lines.append(sub)
    lines.append(f"  Overall mean bit-acc >= 0.90 ?            "
                 f"{_yn(verdict_overall)}  ({bit:.4f})")
    lines.append(f"  Shallow-circuit mean  >= 0.90 ?           "
                 f"{_yn(verdict_shallow)}  ({shallow[0]:.4f})")
    lines.append(f"  ANY bucket     hits >= 0.90 (n>=5) ?      "
                 f"{_yn(verdict_anywhere)}")
    lines.append("")
    if verdict_overall:
        lines.append("  -> The model meets the paper's global accuracy target.")
    elif verdict_shallow or verdict_anywhere:
        lines.append("  -> The model meets the paper's target on a subset of")
        lines.append("     regimes but not on the full validation distribution.")
    else:
        lines.append("  -> The model does NOT reach the paper's 90% target on")
        lines.append("     any regime we can measure.")
    lines.append("")

    # ── Gap analysis ─────────────────────────────────────────────────
    lines.append(sub)
    lines.append("GAP ANALYSIS -- WHERE IS THE ACCURACY LOST?")
    lines.append(sub)
    gap_overall = max(0.0, 0.90 - bit)
    lines.append(f"  Global gap to 0.90        : {gap_overall:+.4f}")

    if kw["worst_depth"] is not None:
        d, accs = kw["worst_depth"]
        lines.append(f"  Hardest fault-depth bucket: max_act_path_len = {d}"
                     f"  ->  bit_acc = {sum(accs)/len(accs):.4f}  (n={len(accs)})")
    if kw["best_depth"] is not None:
        d, accs = kw["best_depth"]
        lines.append(f"  Easiest fault-depth bucket: max_act_path_len = {d}"
                     f"  ->  bit_acc = {sum(accs)/len(accs):.4f}  (n={len(accs)})")
    if kw["best_reconv"] is not None:
        (ka, kp), accs = kw["best_reconv"]
        lines.append(f"  Best reconvergence bucket : k_act={ka}, k_prop={kp}"
                     f"  ->  bit_acc = {sum(accs)/len(accs):.4f}  (n={len(accs)})")

    # Rough monotonicity check on fault depth -- the paper's Fig 6a claim.
    depth_rows = kw["depth_rows"]
    if len(depth_rows) >= 3:
        first_mean = float(depth_rows[0][2])
        last_mean  = float(depth_rows[-1][2])
        trend = last_mean - first_mean
        direction = "decreasing" if trend < -0.02 else ("increasing" if trend > 0.02 else "flat")
        lines.append(f"  Fault-depth trend (first->last bucket): {trend:+.4f}  "
                     f"({direction})")
    lines.append("")

    # ── Roadmap ──────────────────────────────────────────────────────
    lines.append(sub)
    lines.append("ROADMAP -- WHAT TO TRY NEXT")
    lines.append(sub)
    roadmap: List[str] = []
    if bit < 0.9:
        roadmap.append("Train longer: if val accuracy was still rising at the "
                       "last checkpoint, extend training and lower LR after "
                       "the first plateau.")
        roadmap.append("Warm restart with a lower LR (e.g. 1e-4 cosine) for the "
                       "final 50-100 epochs -- often recovers the last few pp.")
    if deep[0] + 0.05 < shallow[0]:
        roadmap.append("Deep-circuit gap: add more deep samples (max act-path "
                       "len > 15) to the training set, or upweight their loss "
                       "contribution -- Fig 6a shows this regime is hardest.")
    if high_re[0] + 0.05 < low_re[0]:
        roadmap.append("Reconvergence gap: regenerate training data with more "
                       "high-fanout reconvergent topologies; the current trainset "
                       "is under-represented at k_act+k_prop >= 15.")
    if kw["pct_lt_50"] > 0.10:
        roadmap.append("A meaningful tail of samples sits below 0.5 -- inspect "
                       "those individually, they often reveal a data bug "
                       "(mis-labelled PI order or missing path coverage).")
    if pattern < 0.5 and bit > 0.8:
        roadmap.append("Strict-pattern accuracy lags bit accuracy by a wide "
                       "margin: consider a post-processing step that snaps "
                       "each PI to the closest valid GT pattern.")
    if not roadmap:
        roadmap.append("No obvious gaps -- consider larger benchmark circuits "
                       "(ISCAS-85) as the next evaluation step.")
    for i, r in enumerate(roadmap, 1):
        # Hand-wrap at ~70 chars so the report stays legible in plain text.
        lines.extend(_wrap_numbered(i, r))
    lines.append("")
    lines.append(hr)

    path.write_text("\n".join(lines), encoding="utf-8")


def _yn(b: bool) -> str:
    return "YES" if b else "NO "


def _fmt_bucket(bucket: Optional[Tuple[Any, List[float]]]) -> str:
    if not bucket:
        return "n/a"
    _, accs = bucket
    return f"{sum(accs)/len(accs):.4f}"


def _wrap_numbered(idx: int, text: str, width: int = 72) -> List[str]:
    """Naive word-wrap with a '  N. ' prefix on the first line, indented
    continuation on the rest. Avoids pulling in ``textwrap`` for one use."""
    prefix = f"  {idx}. "
    cont   = " " * len(prefix)
    words  = text.split()
    lines, cur = [], prefix
    for w in words:
        if len(cur) + 1 + len(w) > width:
            lines.append(cur.rstrip())
            cur = cont + w
        else:
            cur = cur + (" " if cur[-1] != " " else "") + w
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--model",   type=Path, default=BEST_MODEL,
                    help=f"model weights .pt  (default: {BEST_MODEL})")
    ap.add_argument("--val",     type=Path, default=VAL_PKL,
                    help=f"validation pickle  (default: {VAL_PKL})")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR,
                    help=f"output directory   (default: {RESULTS_DIR})")
    args = ap.parse_args()

    try:
        summary = analyze(
            model_path  = args.model,
            val_path    = args.val,
            results_dir = args.out_dir,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print("")
    print("[SUMMARY]")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<28} -> {v:.4f}")
        else:
            print(f"  {k:<28} -> {v}")


if __name__ == "__main__":
    _cli()
