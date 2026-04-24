"""
Evaluate the trained DETECTive model (best_detective_model.pt, 100-epoch run,
best val 0.8358 @ ep 40) on the 11 ISCAS-85 circuits that PODEM runs on.

For each design, sample K faults uniformly at random (same faults for both
DETECTive and PODEM so the comparison is apples-to-apples), then measure:

  * DETECTive inference time per fault (wall-clock, ms)
  * Bit-accuracy of DETECTive's predicted pattern vs PODEM's ground-truth
    pattern on the same fault (only counted when PODEM found a valid pattern)

Outputs a CSV `results_benchmarks.csv` plus a printed table.

Notes
-----
- DETECTive was trained on NAND+NOT-only technology-mapped circuits; the
  ISCAS-85 Verilog netlists from ./netlists/ use a mixed gate library
  (AND/OR/NAND/XOR/BUF/NOT/...). The model's feature vector can still encode
  all those gate types, but untrained one-hot dimensions may degrade
  accuracy. This is called out explicitly in the summary below; the raw
  inference-time numbers are valid regardless.
"""

from __future__ import annotations
import sys, time, csv, random, re, math
from pathlib import Path
from collections import deque

import torch

HERE       = Path(__file__).parent
ROOT       = HERE.parent
NETLISTS   = ROOT / 'PODEM' / 'netlists'
SUBMISSION = ROOT / 'DETECTive_submission'

sys.path.insert(0, str(SUBMISSION))
from circuits import CircuitGraphBuilder, PathExtractor        # noqa: E402
from models   import DETECTiveModel                             # noqa: E402

# ── config ────────────────────────────────────────────────────────
DESIGNS_ASSIGNED = ['c17', 'c432', 'c499', 'c880', 'c1908']
DESIGNS_EXTRAS   = ['c1355', 'c2670', 'c3540', 'c5315', 'c6288', 'c7552']
K_PER_DESIGN     = 20        # faults sampled per design
PODEM_TIMEOUT_S  = 3.0        # per-fault PODEM cap (for ground truth only)
SEED             = 42

WEIGHTS_PATH = HERE / 'best_detective_model.pt'
OUT_CSV      = HERE / 'results_benchmarks.csv'


# ── inline PODEM for ground-truth test patterns ──────────────────
# Small re-implementation so we don't need to extract/import from the notebook.
sys.setrecursionlimit(10000)
ZERO, ONE, X, D, DBAR = '0', '1', 'X', 'D', 'Dbar'


def _NOT(a):
    return {ZERO: ONE, ONE: ZERO, D: DBAR, DBAR: D, X: X}.get(a, X)


def _AND2(a, b):
    if a == ZERO or b == ZERO: return ZERO
    if a == X or b == X: return X
    if a == ONE: return b
    if b == ONE: return a
    if a == b:   return a
    if (a == D and b == DBAR) or (a == DBAR and b == D): return ZERO
    return X


def _OR2(a, b):
    if a == ONE or b == ONE: return ONE
    if a == X or b == X: return X
    if a == ZERO: return b
    if b == ZERO: return a
    if a == b:    return a
    if (a == D and b == DBAR) or (a == DBAR and b == D): return ONE
    return X


def _XOR2(a, b):
    if a == X or b == X: return X
    if a == ZERO: return b
    if b == ZERO: return a
    if a == ONE:  return _NOT(b)
    if b == ONE:  return _NOT(a)
    if a == b:    return ZERO
    return ONE


def _reduce(op, args):
    r = args[0] if args else X
    for v in args[1:]: r = op(r, v)
    return r


_AND  = lambda *a: _reduce(_AND2, a)
_OR   = lambda *a: _reduce(_OR2, a)
_XOR  = lambda *a: _reduce(_XOR2, a)
_NAND = lambda *a: _NOT(_AND(*a))
_NOR  = lambda *a: _NOT(_OR(*a))
_XNOR = lambda *a: _NOT(_XOR(*a))
_BUF  = lambda a: a

_EVAL = {'AND': _AND, 'OR': _OR, 'NAND': _NAND, 'NOR': _NOR,
         'XOR': _XOR, 'XNOR': _XNOR, 'NOT': _NOT, 'BUF': _BUF}


class _Gate:
    __slots__ = ('name', 'type', 'inputs', 'value')
    def __init__(self, name, gt, inp):
        self.name, self.type, self.inputs, self.value = name, gt, inp, X


class _Circuit:
    def __init__(self):
        self.gates = {}
        self.fault_node = None

    def evaluate(self):
        for name in self._topo():
            g = self.gates[name]
            if g.type == 'INPUT': continue
            comp = _EVAL[g.type](*(self.gates[i].value for i in g.inputs))
            if name == self.fault_node:
                if g.value in (D, DBAR):
                    ff = ONE if g.value == D else ZERO
                    if comp != X and comp != ff: return False
                continue
            if comp != X:
                if g.value != X and g.value != comp: return False
                g.value = comp
        return True

    def _topo(self):
        visited, order = set(), []
        def visit(n):
            if n in visited: return
            g = self.gates.get(n)
            if g and g.type != 'INPUT':
                for i in g.inputs: visit(i)
            visited.add(n); order.append(n)
        for n in self.gates: visit(n)
        return order

    def get_outputs(self):
        driven = {i for g in self.gates.values() for i in g.inputs}
        return [n for n in self.gates if n not in driven]


def _parse_verilog(text):
    c = _Circuit()
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)
    text = re.sub(r'//.*', '', text).replace('\n', ' ')
    for grp in re.findall(r'input\s+([^;]+);', text):
        for inp in grp.split(','):
            n = inp.strip()
            if n: c.gates[n] = _Gate(n, 'INPUT', [])
    for gt, args in re.findall(
            r'(and|or|nand|nor|xor|xnor|not|buf)\s+\w+\s*\(([^)]+)\);',
            text, re.IGNORECASE):
        toks = [t.strip() for t in args.split(',')]
        c.gates[toks[0]] = _Gate(toks[0], gt.upper(), toks[1:])
    return c


def _find_df(circ):
    return [n for n, g in circ.gates.items()
            if g.value == X and any(circ.gates[i].value in (D, DBAR) for i in g.inputs)]


def _xpath(circ, df):
    if not df: return False
    pos, q, seen = set(circ.get_outputs()), list(df), set(df)
    while q:
        cur = q.pop(0)
        if cur in pos: return True
        for nm, g in circ.gates.items():
            if cur in g.inputs and g.value == X and nm not in seen:
                seen.add(nm); q.append(nm)
    return False


def _objective(circ, sv):
    fn = circ.fault_node; g = circ.gates[fn]
    if g.value not in (D, DBAR):
        return (fn, ONE if sv == 0 else ZERO)
    df = _find_df(circ)
    if not df: return None
    gg = circ.gates[df[0]]
    nc = ONE if gg.type in ('AND', 'NAND') else ZERO
    for inp in gg.inputs:
        if circ.gates[inp].value == X: return (inp, nc)
    return None


def _backtrace(circ, node, val):
    cur_n, cur_v = node, val
    while circ.gates[cur_n].type != 'INPUT':
        g = circ.gates[cur_n]
        if g.type in ('NOT', 'NAND', 'NOR', 'XNOR'):
            cur_v = _NOT(cur_v)
        nxt = next((i for i in g.inputs if circ.gates[i].value == X), None)
        if nxt is None: return (None, None)
        cur_n = nxt
    return (cur_n, cur_v)


def _activate(circ, sv):
    fn = circ.fault_node; g = circ.gates[fn]
    if g.value in (D, DBAR): return
    comp = g.value if g.type == 'INPUT' else _EVAL[g.type](*(circ.gates[i].value for i in g.inputs))
    if comp == X: return
    req = ONE if sv == 0 else ZERO
    if comp == req:
        g.value = D if sv == 0 else DBAR
        circ.evaluate()
    else:
        g.value = comp


def _podem(circ, sv, deadline):
    if time.perf_counter() > deadline: return 'timeout'
    if any(circ.gates[o].value in (D, DBAR) for o in circ.get_outputs()):
        return True
    fnv = circ.gates[circ.fault_node].value
    if fnv in (ZERO, ONE): return False
    if fnv in (D, DBAR):
        df = _find_df(circ)
        if not df or not _xpath(circ, df): return False
    obj = _objective(circ, sv)
    if obj is None: return False
    pi_n, pi_v = _backtrace(circ, obj[0], obj[1])
    if pi_n is None: return False
    for v in (pi_v, _NOT(pi_v)):
        snap = {n: g.value for n, g in circ.gates.items()}
        circ.gates[pi_n].value = v
        circ.evaluate(); _activate(circ, sv)
        r = _podem(circ, sv, deadline)
        if r is True: return True
        if r == 'timeout': return 'timeout'
        for n, val in snap.items(): circ.gates[n].value = val
    return False


def podem_pattern(circuit_text, fault_node, sv, timeout_s=PODEM_TIMEOUT_S):
    """Return a dict {pi_name: '0'|'1'|'X'} or None if undetectable/timeout."""
    c = _parse_verilog(circuit_text)
    for g in c.gates.values(): g.value = X
    c.fault_node = fault_node
    deadline = time.perf_counter() + timeout_s
    r = _podem(c, sv, deadline)
    if r is not True: return None
    return {n: g.value for n, g in c.gates.items() if g.type == 'INPUT'}


# ── DETECTive inference on one fault ──────────────────────────────
def detective_predict(model, device, verilog_text, fault_node, sv):
    builder = CircuitGraphBuilder(verilog_text)
    if fault_node not in builder.gates:
        return None, 0.0
    graph, name_to_id = builder.get_pyg_graph(faulty_node=fault_node)
    graph = graph.to(device)
    fault_idx  = name_to_id[fault_node]
    fault_type = torch.tensor([[float(sv)]], device=device)

    pe = PathExtractor(graph.edge_index, graph.x.shape[0])
    act  = pe.get_activation_paths(fault_idx)
    prop = pe.get_propagation_paths(fault_idx)
    pi_names   = [n for n, g in builder.gates.items() if g['type'] == 'INPUT']
    pi_indices = [name_to_id[n] for n in pi_names if n in name_to_id]

    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model(graph, pi_indices, fault_type, act, prop)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    pattern = {pi_names[i]: ('1' if preds[i].item() > 0.5 else '0')
               for i in range(len(pi_names))}
    return pattern, dt_ms


def _bit_acc(pred, gt):
    """Fraction of PI bits where pred matches gt (X in gt = free match)."""
    if pred is None or gt is None: return None
    hits = total = 0
    for k, v in pred.items():
        g = gt.get(k, X)
        if g == X:
            hits += 1           # don't-care matches anything
        elif g == v:
            hits += 1
        total += 1
    return hits / total if total else 0.0


# ── main ──────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[info] device: {device}")
    print(f"[info] netlists dir: {NETLISTS.resolve()}")
    print(f"[info] weights: {WEIGHTS_PATH.name}")

    model = DETECTiveModel(in_channels=11, hidden_channels=32, p=10).to(device)
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    rng = random.Random(SEED)
    rows = []
    all_designs = DESIGNS_ASSIGNED + DESIGNS_EXTRAS
    print(f"\nevaluating {len(all_designs)} designs, K={K_PER_DESIGN} faults each ...\n")
    print(f"{'design':<8} {'gates':>6} {'faults':>6} {'det_ms':>8} {'gt_found':>8} {'bit_acc':>8}")
    print('-' * 50)

    for d in all_designs:
        path = NETLISTS / f'{d}.v'
        if not path.exists():
            print(f"  SKIP {d}: {path} not found")
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        if text.startswith('﻿'): text = text[1:]

        builder = CircuitGraphBuilder(text)
        internal = [n for n, g in builder.gates.items() if g['type'] != 'INPUT']
        rng.shuffle(internal)
        faults = [(n, rng.randint(0, 1)) for n in internal[:K_PER_DESIGN]]
        gate_count = len(internal)

        det_times, accs = [], []
        gt_found = 0
        for fn, sv in faults:
            pat, dt = detective_predict(model, device, text, fn, sv)
            if pat is None:
                continue
            det_times.append(dt)

            gt = podem_pattern(text, fn, sv, timeout_s=PODEM_TIMEOUT_S)
            if gt is not None:
                gt_found += 1
                ba = _bit_acc(pat, gt)
                if ba is not None:
                    accs.append(ba)

        det_ms = sum(det_times) / len(det_times) if det_times else float('nan')
        bit_acc = sum(accs) / len(accs) if accs else float('nan')
        print(f"{d:<8} {gate_count:>6} {len(faults):>6} "
              f"{det_ms:>8.2f} {gt_found:>8} {bit_acc:>8.4f}")
        rows.append({
            'design':         d,
            'gate_count':     gate_count,
            'n_faults':       len(faults),
            'detective_ms':   f"{det_ms:.3f}",
            'podem_gt_found': gt_found,
            'bit_accuracy':   f"{bit_acc:.4f}" if not math.isnan(bit_acc) else 'N/A',
        })

    with OUT_CSV.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['design', 'gate_count', 'n_faults',
                                           'detective_ms', 'podem_gt_found',
                                           'bit_accuracy'])
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == '__main__':
    main()
