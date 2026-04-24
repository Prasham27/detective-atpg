"""
DETECTive — ATALANTA subprocess wrapper.

ATALANTA is the classical academic ATPG tool the paper uses as ground truth
source and as the runtime baseline for the 15x speedup claim (Section 6.4).
This module only invokes the binary; the user installs it separately.

Install:
    git clone https://github.com/hsluoyz/Atalanta.git
    cd Atalanta && make
    export ATALANTA_BIN=$(pwd)/atalanta    # or put it on PATH

Public API:
    is_available(binary=None)                    -> bool
    install_instructions()                       -> str
    run_full(bench_file, ...)                    -> AtalantaResult
    run_fault(bench_file, node, stuck_value, ...)-> AtalantaResult

Both run_* functions time the subprocess internally, so you don't need to
wrap them in your own timer.
"""

from __future__ import annotations
import os, re, shutil, subprocess, tempfile, time
from dataclasses import dataclass
from typing import List, Optional


# ══════════════════════════════════════════════════════════════════
#  Binary discovery
# ══════════════════════════════════════════════════════════════════
def _find_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Look for the atalanta binary. Priority: explicit arg -> env -> PATH."""
    if explicit and os.path.isfile(explicit):
        return explicit
    env = os.environ.get("ATALANTA_BIN")
    if env and os.path.isfile(env):
        return env
    return shutil.which("atalanta")


def is_available(binary: Optional[str] = None) -> bool:
    return _find_binary(binary) is not None


def install_instructions() -> str:
    return (
        "\nATALANTA not found. To install:\n"
        "  git clone https://github.com/hsluoyz/Atalanta.git\n"
        "  cd Atalanta && make\n"
        "Then add the resulting `atalanta` binary to PATH or set\n"
        "ATALANTA_BIN to its absolute path.\n"
    )


# ══════════════════════════════════════════════════════════════════
#  Result type + .test parser
# ══════════════════════════════════════════════════════════════════
@dataclass
class AtalantaResult:
    patterns:   List[str]           # "01XD..." strings, MSB = first PI
    runtime_ms: float
    stdout:     str
    stderr:     str


def _parse_test_file(text: str) -> List[str]:
    """Extract the binary pattern column from ATALANTA's .test file."""
    patterns: List[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+\s*:\s*([01XxDd]+)", line)
        if m:
            patterns.append(m.group(1).upper())
    return patterns


# ══════════════════════════════════════════════════════════════════
#  Subprocess runners
# ══════════════════════════════════════════════════════════════════
def run_full(bench_file:  str,
             binary:      Optional[str] = None,
             extra_args:  Optional[List[str]] = None,
             timeout_s:   float = 120.0) -> AtalantaResult:
    """Run ATALANTA on an entire .bench file."""
    atal = _find_binary(binary)
    if atal is None:
        raise FileNotFoundError(install_instructions())
    if not os.path.isfile(bench_file):
        raise FileNotFoundError(f"bench file not found: {bench_file}")

    with tempfile.TemporaryDirectory() as tmp:
        local  = os.path.join(tmp, os.path.basename(bench_file))
        shutil.copy(bench_file, local)
        tfile  = os.path.splitext(local)[0] + ".test"
        cmd    = [atal, local] + list(extra_args or [])

        t0   = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, cwd=tmp)
        rt   = (time.perf_counter() - t0) * 1000.0

        text = ""
        if os.path.isfile(tfile):
            with open(tfile) as f: text = f.read()

    return AtalantaResult(_parse_test_file(text), rt, proc.stdout, proc.stderr)


def run_fault(bench_file:  str,
              fault_node:  str,
              stuck_value: int,
              binary:      Optional[str] = None,
              timeout_s:   float = 120.0) -> AtalantaResult:
    """Target a single stuck-at-<value> fault on <node>."""
    if stuck_value not in (0, 1):
        raise ValueError("stuck_value must be 0 or 1")
    atal = _find_binary(binary)
    if atal is None:
        raise FileNotFoundError(install_instructions())

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, os.path.basename(bench_file))
        shutil.copy(bench_file, local)
        flt   = os.path.join(tmp, "target.flt")
        with open(flt, "w") as f:
            f.write(f"{fault_node} /{stuck_value}\n")
        tfile = os.path.splitext(local)[0] + ".test"

        cmd   = [atal, "-f", flt, local]
        t0    = time.perf_counter()
        proc  = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=timeout_s, cwd=tmp)
        rt    = (time.perf_counter() - t0) * 1000.0

        text = ""
        if os.path.isfile(tfile):
            with open(tfile) as f: text = f.read()

    return AtalantaResult(_parse_test_file(text), rt, proc.stdout, proc.stderr)


# ══════════════════════════════════════════════════════════════════
#  CLI smoke test
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("bench")
    p.add_argument("--fault", help="optional 'node /value' (e.g. 'G1 /0')")
    p.add_argument("--binary")
    a = p.parse_args()
    if a.fault:
        node, sv = a.fault.split()
        r = run_fault(a.bench, node, int(sv.lstrip("/")), binary=a.binary)
    else:
        r = run_full(a.bench, binary=a.binary)
    print(f"runtime: {r.runtime_ms:.1f} ms")
    print(f"patterns ({len(r.patterns)}):")
    for pat in r.patterns[:10]:
        print(f"  {pat}")
