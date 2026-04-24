"""
DETECTive - realistic benchmark evaluation (paper sections 6.4 and 6.5).

Evaluates the trained DETECTive model on ISCAS-85 circuits (c17, c432, c499,
c880, c1355, c1908, c2670, c3540, c5315, c6288, c7552) and compares its
per-fault test pattern accuracy + runtime against the classical ATPG tool
ATALANTA. This reproduces the paper's "realistic benchmark" numbers and the
~15x speedup claim.

Pipeline:
  1. Download ISCAS-85 .bench files into BENCH_DIR_RAW if they are missing.
  2. Synthesize each .bench to NAND+NOT Verilog using ABC + Yosys, stored
     under BENCH_DIR_SYNTH (so the CircuitGraphBuilder sees a netlist whose
     gate library matches the training distribution).
  3. For each benchmark, sample K internal gates as stuck-at fault sites.
     For each fault:
       - run DETECTive inference, record runtime and predicted pattern
       - (if ATALANTA is available) target the same fault on the raw .bench
         and record ATALANTA's patterns + runtime
       - score DETECTive's prediction against the best matching ATALANTA
         pattern (treating 'X' don't-cares as free matches)
  4. Aggregate per-design statistics into
     RESULTS_DIR / "results_benchmarks.csv".

CLI:
  python benchmarks.py                                  # full pipeline
  python benchmarks.py --skip-download --skip-synth     # use existing files
  python benchmarks.py --designs c17 c432 --faults 10
  python benchmarks.py --model path/to/checkpoint.pt
  python benchmarks.py --atalanta /opt/atalanta/atalanta

External tools required (only for the synth step):
  - abc    (https://github.com/berkeley-abc/abc)   on PATH
  - yosys  (https://yosyshq.net/yosys/)            on PATH
ATALANTA is optional; without it, accuracy + speedup columns are NaN.
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows console defaults to cp1252, which trips on any non-ASCII byte that
# might sneak into a stderr message from yosys/abc. Keep this best-effort.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch

from config import (
    BEST_MODEL,
    BENCH_DIR_RAW,
    BENCH_DIR_SYNTH,
    RESULTS_DIR,
    K_FAULTS_PER_BENCHMARK,
    ATPG_TIMEOUT_SEC,
    FEATURE_DIM,
    HIDDEN_DIM,
    P_PATHS,
)
from circuits import CircuitGraphBuilder, PathExtractor
from models import DETECTiveModel
from evaluation import to_device  # noqa: F401 -- re-exported for pipeline.py consumers
from atalanta import (
    is_available as atalanta_available,
    install_instructions,
    run_fault,
)


# ══════════════════════════════════════════════════════════════════
#  Defaults
# ══════════════════════════════════════════════════════════════════
DEFAULT_DESIGNS: List[str] = [
    "c17", "c432", "c499", "c880", "c1355",
    "c1908", "c2670", "c3540", "c5315", "c6288", "c7552",
]

# Public mirror of the ISCAS-85 set. If this URL ever 404s, the download
# failure message below tells the user how to recover.
BENCH_ZIP_URL = "https://github.com/circuitminer/iscas85/archive/refs/heads/master.zip"

DEFAULT_CSV = RESULTS_DIR / "results_benchmarks.csv"


# ══════════════════════════════════════════════════════════════════
#  Step 1 - download ISCAS-85 .bench files
# ══════════════════════════════════════════════════════════════════
def download_benchmarks(bench_dir: Path, designs: List[str]) -> None:
    """Ensure every requested .bench file exists under `bench_dir`."""
    bench_dir.mkdir(parents=True, exist_ok=True)
    missing = [d for d in designs if not (bench_dir / f"{d}.bench").is_file()]
    if not missing:
        print(f"[bench] all {len(designs)} files present in {bench_dir}")
        return

    print(f"[bench] downloading archive for: {', '.join(missing)}")
    try:
        # Short timeout - if the user is offline we want to fail fast rather
        # than stall the whole evaluation.
        with urllib.request.urlopen(BENCH_ZIP_URL, timeout=60) as resp:
            buf = resp.read()
    except Exception as e:
        print(
            f"[ERROR] could not download {BENCH_ZIP_URL}: {e}\n"
            f"        Manually place the .bench files into:\n"
            f"          {bench_dir}\n"
            f"        then rerun with --skip-download.",
            file=sys.stderr,
        )
        sys.exit(1)

    wanted = {f"{d}.bench".lower() for d in missing}
    copied = 0
    with zipfile.ZipFile(io.BytesIO(buf)) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name.lower()
            if base in wanted:
                target = bench_dir / f"{Path(base).stem}.bench"
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                copied += 1
    print(f"[bench] extracted {copied} files -> {bench_dir}")


# ══════════════════════════════════════════════════════════════════
#  Step 2 - synthesize .bench -> NAND+NOT Verilog via ABC + Yosys
# ══════════════════════════════════════════════════════════════════
def _check_synth_tools() -> Tuple[str, str]:
    """Locate abc + yosys on PATH or fail with install instructions."""
    abc = shutil.which("abc")
    ys  = shutil.which("yosys")
    if abc is None or ys is None:
        missing = []
        if abc is None: missing.append("abc   (https://github.com/berkeley-abc/abc)")
        if ys  is None: missing.append("yosys (https://yosyshq.net/yosys/)")
        print(
            "[ERROR] required synthesis tools not on PATH: "
            + ", ".join(missing) + "\n"
            "        Install them and rerun, or pass --skip-synth if the "
            ".v files already exist in BENCH_DIR_SYNTH.",
            file=sys.stderr,
        )
        sys.exit(1)
    return abc, ys


def _synthesize_one(bench_file: Path, out_v: Path, abc_bin: str, yosys_bin: str) -> bool:
    """ABC maps .bench -> BLIF, yosys rewrites to NAND+NOT Verilog.

    Two-stage because yosys does not ingest .bench natively while ABC does,
    and we still want yosys' clean NAND-only rewriter + -noattr Verilog.
    """
    tmp_blif = out_v.with_suffix(".blif")
    abc_script = f"read_bench {bench_file}; strash; map; write_blif {tmp_blif}"
    try:
        subprocess.run(
            [abc_bin, "-q", abc_script],
            check=True, capture_output=True, text=True, timeout=120,
        )
    except subprocess.CalledProcessError as e:
        print(f"    [ABC] failed on {bench_file.name}: {e.stderr[-400:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"    [ABC] timeout on {bench_file.name}")
        return False

    yosys_script = (
        f"read_blif {tmp_blif}; "
        "hierarchy -auto-top; proc; opt; techmap; opt; "
        "abc -g NAND; opt_clean; "
        f"write_verilog -noattr {out_v}"
    )
    try:
        subprocess.run(
            [yosys_bin, "-q", "-p", yosys_script],
            check=True, capture_output=True, text=True, timeout=180,
        )
    except subprocess.CalledProcessError as e:
        print(f"    [yosys] failed on {tmp_blif.name}: {e.stderr[-400:]}")
        return False
    except subprocess.TimeoutExpired:
        print(f"    [yosys] timeout on {tmp_blif.name}")
        return False
    finally:
        if tmp_blif.is_file():
            tmp_blif.unlink()
    return True


def synthesize_benchmarks(bench_dir: Path, synth_dir: Path,
                          designs: List[str]) -> List[str]:
    """Synthesize every requested design; return the list that succeeded."""
    synth_dir.mkdir(parents=True, exist_ok=True)
    abc_bin, yosys_bin = _check_synth_tools()

    ok: List[str] = []
    for d in designs:
        bench = bench_dir  / f"{d}.bench"
        vfile = synth_dir  / f"{d}.v"
        if not bench.is_file():
            print(f"[synth] {d}: no .bench file -> skipping")
            continue
        if vfile.is_file():
            ok.append(d)
            continue
        print(f"[synth] {d}: synthesizing ...")
        if _synthesize_one(bench, vfile, abc_bin, yosys_bin):
            ok.append(d)
            print(f"        -> {vfile}")
    return ok


# ══════════════════════════════════════════════════════════════════
#  Step 3 - inference + scoring helpers
# ══════════════════════════════════════════════════════════════════
def load_detective_model(model_path: Path, device: torch.device) -> DETECTiveModel:
    """Load weights from either a bare state_dict or a training checkpoint.

    `best_detective_model.pt` is a raw state_dict (torch.save(model.state_dict())).
    `checkpoint_last.pt` is a training dict with {"model_state_dict": ..., ...}.
    We auto-detect by checking for the canonical checkpoint key.
    """
    if not model_path.is_file():
        print(f"[ERROR] model weights not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # weights_only=False because our checkpoint dicts carry optimizer state
    # and epoch counters, which torch.load rejects under the default setting.
    obj = torch.load(model_path, map_location=device, weights_only=False)
    state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj

    model = DETECTiveModel(
        in_channels=FEATURE_DIM, hidden_channels=HIDDEN_DIM, p=P_PATHS,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _sample_fault_nodes(builder: CircuitGraphBuilder,
                        k: int,
                        seed: int) -> List[Tuple[str, int]]:
    """Pick up to k internal (non-INPUT) gates with a random stuck value."""
    rng = random.Random(seed)
    candidates = [n for n, d in builder.gates.items() if d["type"] != "INPUT"]
    rng.shuffle(candidates)
    return [(n, rng.randint(0, 1)) for n in candidates[:k]]


def _detective_predict(model: DETECTiveModel,
                       builder: CircuitGraphBuilder,
                       fault_node: str,
                       stuck_value: int,
                       device: torch.device) -> Tuple[str, float]:
    """Run DETECTive on one fault; return (bit_string, runtime_ms)."""
    graph, name_to_id = builder.get_pyg_graph(faulty_node=fault_node)
    graph = graph.to(device)

    fault_idx = name_to_id.get(fault_node)
    fault_type = torch.tensor([[float(stuck_value)]], device=device)

    pe = PathExtractor(graph.edge_index, graph.x.shape[0])
    act_paths  = pe.get_activation_paths(fault_idx,  max_paths=P_PATHS)
    prop_paths = pe.get_propagation_paths(fault_idx, max_paths=P_PATHS)

    # PI ordering matches builder.gates iteration order, which aligns with
    # how ATALANTA lists inputs (left-to-right declaration in the .bench).
    pi_indices = [name_to_id[n] for n, d in builder.gates.items()
                  if d["type"] == "INPUT" and n in name_to_id]

    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(graph, pi_indices, fault_type, act_paths, prop_paths)
    rt_ms = (time.perf_counter() - t0) * 1000.0

    bits = "".join("1" if p.item() > 0.5 else "0" for p in preds.view(-1))
    return bits, rt_ms


def _best_bit_accuracy(pred: str, refs: List[str]) -> Optional[float]:
    """Bit accuracy against the best matching ATALANTA pattern.

    ATALANTA may emit 'X' (don't care) for undecided PIs - those bits count
    as correct regardless of DETECTive's guess, matching the paper's metric.
    Patterns with a different length (e.g. ATALANTA expanded fanouts that
    our parser didn't) are skipped rather than silently truncating.
    """
    if not refs:
        return None
    n = len(pred)
    best = 0.0
    matched_any = False
    for r in refs:
        if len(r) != n:
            continue
        matched_any = True
        hit = sum(1 for p, g in zip(pred, r) if g.upper() == "X" or p == g)
        best = max(best, hit / n)
    return best if matched_any else None


# ══════════════════════════════════════════════════════════════════
#  Step 4 - main orchestration
# ══════════════════════════════════════════════════════════════════
def run_benchmarks(model_path: Path = BEST_MODEL,
                   designs: Optional[List[str]] = None,
                   faults: int = K_FAULTS_PER_BENCHMARK,
                   seed: int = 42,
                   skip_download: bool = False,
                   skip_synth: bool = False,
                   atalanta_binary: Optional[str] = None,
                   out_csv: Optional[Path] = None,
                   bench_dir_raw: Path = BENCH_DIR_RAW,
                   bench_dir_synth: Path = BENCH_DIR_SYNTH,
                   atpg_timeout_s: int = ATPG_TIMEOUT_SEC,
                   ) -> List[Dict]:
    """Programmatic entrypoint used by pipeline.py.

    Returns a list of per-design result dicts (same columns as the CSV).
    """
    designs = list(designs) if designs else list(DEFAULT_DESIGNS)
    out_csv = Path(out_csv) if out_csv else DEFAULT_CSV
    model_path = Path(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # 1. bench files
    if not skip_download:
        download_benchmarks(bench_dir_raw, designs)

    # 2. synthesis
    if skip_synth:
        available = [d for d in designs if (bench_dir_synth / f"{d}.v").is_file()]
    else:
        available = synthesize_benchmarks(bench_dir_raw, bench_dir_synth, designs)
    if not available:
        print("[ERROR] no benchmarks available to evaluate", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] evaluating on: {', '.join(available)}")

    # 3. model
    model = load_detective_model(model_path, device)

    # 4. ATALANTA availability
    have_atalanta = atalanta_available(atalanta_binary)
    if have_atalanta:
        print("[INFO] ATALANTA available -> accuracy + speedup enabled")
    else:
        print("[WARN] ATALANTA not available -> DETECTive runtime only")
        print(install_instructions())

    # 5. per-design eval
    rows: List[Dict] = []
    for d in available:
        vfile     = bench_dir_synth / f"{d}.v"
        benchfile = bench_dir_raw   / f"{d}.bench"
        try:
            netlist = vfile.read_text()
            builder = CircuitGraphBuilder(netlist)
        except Exception as e:
            print(f"[{d}] parse failed: {e} -> skipping")
            continue

        gate_count = sum(1 for g in builder.gates.values() if g["type"] != "INPUT")
        fault_sites = _sample_fault_nodes(builder, faults, seed=seed)

        det_accs: List[float] = []
        det_rts:  List[float] = []
        atal_rts: List[float] = []

        for i, (node, sv) in enumerate(fault_sites):
            # DETECTive inference
            try:
                pred, rt = _detective_predict(model, builder, node, sv, device)
            except Exception as e:
                print(f"    [{d}] fault {i} ({node}/{sv}): DETECTive error: {e}")
                continue
            det_rts.append(rt)

            # ATALANTA reference (optional, one fault at a time). We keep
            # going on any failure - a stuck-at fault may legitimately be
            # untestable, or ATALANTA may time out on pathological cases.
            if have_atalanta and benchfile.is_file():
                try:
                    ref = run_fault(
                        str(benchfile), node, sv,
                        binary=atalanta_binary, timeout_s=atpg_timeout_s,
                    )
                except subprocess.TimeoutExpired:
                    print(f"    [{d}] fault {i} ({node}/{sv}): ATALANTA timeout")
                    continue
                except Exception as e:
                    print(f"    [{d}] fault {i} ({node}/{sv}): ATALANTA error: {e}")
                    continue

                atal_rts.append(ref.runtime_ms)
                acc = _best_bit_accuracy(pred, ref.patterns)
                if acc is not None:
                    det_accs.append(acc)

        mean_acc     = sum(det_accs) / len(det_accs) if det_accs else float("nan")
        mean_det_rt  = sum(det_rts)  / len(det_rts)  if det_rts  else float("nan")
        mean_atal_rt = sum(atal_rts) / len(atal_rts) if atal_rts else float("nan")
        speedup = (mean_atal_rt / mean_det_rt) if det_rts and atal_rts else float("nan")

        print(
            f"[{d:>6}] gates={gate_count:>5} | "
            f"acc={mean_acc:.3f} (n={len(det_accs)}) | "
            f"det_rt={mean_det_rt:.2f}ms | "
            f"atal_rt={mean_atal_rt:.2f}ms | "
            f"speedup={speedup:.2f}x"
        )

        rows.append({
            "design":            d,
            "gate_count":        gate_count,
            "num_scored_faults": len(det_accs),
            "bit_accuracy":      mean_acc,
            "detective_ms":      mean_det_rt,
            "atalanta_ms":       mean_atal_rt,
            "speedup":           speedup,
        })

    # 6. CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "design", "gate_count", "num_scored_faults",
            "bit_accuracy", "detective_ms", "atalanta_ms", "speedup",
        ])
        for r in rows:
            w.writerow([
                r["design"],
                r["gate_count"],
                r["num_scored_faults"],
                f"{r['bit_accuracy']:.4f}",
                f"{r['detective_ms']:.3f}",
                f"{r['atalanta_ms']:.3f}",
                f"{r['speedup']:.3f}",
            ])
    print(f"\n[DONE] wrote {out_csv}")
    return rows


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DETECTive benchmark evaluation on ISCAS-85.",
    )
    p.add_argument("--model", type=Path, default=BEST_MODEL,
                   help="model weights (state_dict or training checkpoint)")
    p.add_argument("--designs", nargs="+", default=DEFAULT_DESIGNS,
                   help="benchmark names (default: full ISCAS-85 set)")
    p.add_argument("--faults", type=int, default=K_FAULTS_PER_BENCHMARK,
                   help="faults sampled per design")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducible fault sampling")
    p.add_argument("--skip-download", action="store_true",
                   help="assume .bench files already exist in BENCH_DIR_RAW")
    p.add_argument("--skip-synth", action="store_true",
                   help="assume .v files already exist in BENCH_DIR_SYNTH")
    p.add_argument("--atalanta", default=None,
                   help="override ATALANTA binary path (else $ATALANTA_BIN / PATH)")
    p.add_argument("--out", type=Path, default=DEFAULT_CSV,
                   help="CSV output path")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_benchmarks(
        model_path      = args.model,
        designs         = args.designs,
        faults          = args.faults,
        seed            = args.seed,
        skip_download   = args.skip_download,
        skip_synth      = args.skip_synth,
        atalanta_binary = args.atalanta,
        out_csv         = args.out,
    )
