"""
DETECTive - minimal 2-valued gate-level fault simulator.

Used by the ATPP coverage evaluation script (`atpp_coverage.py`) to convert
the model's predicted test patterns into a real *fault coverage* number,
matching how the classical PODEM/D-Algorithm/FAN notebooks measure coverage.

A fault is "detected" by a pattern iff the fault-free circuit and the
single-stuck-at faulty circuit produce different values on at least one
primary output for that pattern.

Public API:
    parse_netlist(text)
        -> dict with keys: 'gates', 'inputs', 'outputs', 'topo'
    simulate_fault_detected(parsed, fault_node, fault_stuck_at, pattern)
        -> bool  (True iff pattern detects (fault_node s-a-fault_stuck_at))

Supported gates (case-insensitive in the netlist): AND, OR, NAND, NOR, NOT,
BUF, XOR, XNOR. Inputs are declared with `input ...;`, outputs with
`output ...;`. Comments and whitespace are stripped by the parser.

We deliberately keep this independent of the PODEM notebook's 5-valued
machinery: coverage simulation only needs 0/1 logic + a single point of
fault injection on the named line.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ============================================================================
#  Gate evaluation (2-valued: 0 or 1, ints)
# ============================================================================

def _eval_gate(gtype: str, inputs: List[int]) -> int:
    """Pure 2-valued gate evaluator. Inputs are ints 0 or 1."""
    t = gtype.upper()
    if t == "AND":
        out = 1
        for v in inputs:
            out &= v
        return out
    if t == "NAND":
        out = 1
        for v in inputs:
            out &= v
        return 1 - out
    if t == "OR":
        out = 0
        for v in inputs:
            out |= v
        return out
    if t == "NOR":
        out = 0
        for v in inputs:
            out |= v
        return 1 - out
    if t == "XOR":
        out = 0
        for v in inputs:
            out ^= v
        return out
    if t == "XNOR":
        out = 0
        for v in inputs:
            out ^= v
        return 1 - out
    if t == "NOT":
        return 1 - inputs[0]
    if t == "BUF":
        return inputs[0]
    raise ValueError(f"Unsupported gate type: {gtype}")


# ============================================================================
#  Verilog parser  (matches CircuitGraphBuilder / PODEM_final's subset)
# ============================================================================

_GATE_RE = re.compile(
    r"(and|or|nand|nor|xor|xnor|not|buf)\s+\w+\s*\(([^)]+)\);",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    text = re.sub(r"/\*[\s\S]*?\*/", "", text)
    text = re.sub(r"//.*", "", text)
    return text.replace("\n", " ")


def parse_netlist(text: str) -> dict:
    """Parse a flat gate-level Verilog netlist.

    Returns:
        {
          'gates'  : {name: {'type': 'AND'|'INPUT'|..., 'inputs': [...]}},
          'inputs' : [PI names in declaration order],
          'outputs': [explicit PO names],
          'topo'   : [topologically sorted gate names (excluding inputs)]
        }
    """
    if text.startswith("﻿"):
        text = text[1:]
    cleaned = _clean(text)

    gates: Dict[str, dict] = {}
    inputs: List[str] = []
    outputs: List[str] = []

    for grp in re.findall(r"input\s+([^;]+);", cleaned):
        for name in (n.strip() for n in grp.split(",")):
            if name and name not in gates:
                gates[name] = {"type": "INPUT", "inputs": []}
                inputs.append(name)

    for grp in re.findall(r"output\s+([^;]+);", cleaned):
        for name in (n.strip() for n in grp.split(",")):
            if name:
                outputs.append(name)

    for gtype, args in _GATE_RE.findall(cleaned):
        toks = [t.strip() for t in args.split(",")]
        out_node = toks[0]
        in_nodes = toks[1:]
        gates[out_node] = {"type": gtype.upper(), "inputs": in_nodes}

    # Topological sort (Kahn-ish via DFS post-order)
    topo: List[str] = []
    seen: set = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        g = gates.get(n)
        if g is None:
            seen.add(n)
            return
        if g["type"] != "INPUT":
            for inp in g["inputs"]:
                visit(inp)
        seen.add(n)
        if g["type"] != "INPUT":
            topo.append(n)

    for name in gates:
        visit(name)

    return {
        "gates": gates,
        "inputs": inputs,
        "outputs": outputs,
        "topo": topo,
    }


# ============================================================================
#  Fault-free + faulty simulation
# ============================================================================

def _simulate(parsed: dict,
              pattern: Dict[str, str],
              fault_node: Optional[str] = None,
              fault_value: Optional[int] = None) -> Dict[str, int]:
    """Simulate the circuit on `pattern`. If fault_node is given, the value
    of that line is forced to `fault_value` (0 or 1) AFTER its normal
    evaluation, so downstream gates see the stuck value.

    Returns a dict {node_name: int_value} for every gate / PI.
    """
    gates = parsed["gates"]
    values: Dict[str, int] = {}

    # Primary inputs from the supplied pattern.
    for pi in parsed["inputs"]:
        v = pattern.get(pi, "0")
        # Accept '0'/'1' or 0/1 ints.
        if isinstance(v, str):
            v = 1 if v == "1" else 0
        values[pi] = int(v) & 1

    # If the fault is on a primary input, override that PI's value.
    if fault_node is not None and fault_node in parsed["inputs"]:
        values[fault_node] = int(fault_value) & 1

    # Forward evaluation in topo order.
    for name in parsed["topo"]:
        g = gates[name]
        try:
            in_vals = [values[i] for i in g["inputs"]]
        except KeyError:
            # Wire referenced but not produced anywhere (e.g. stray name) -
            # treat as 0 to keep going. Should not happen on ISCAS-85.
            in_vals = [values.get(i, 0) for i in g["inputs"]]
        v = _eval_gate(g["type"], in_vals)
        if name == fault_node:
            v = int(fault_value) & 1
        values[name] = v

    return values


def _resolve_outputs(parsed: dict) -> List[str]:
    """Return the list of primary-output node names. Prefer the explicit
    `output ...;` declaration; fall back to gates with no fanout."""
    if parsed["outputs"]:
        return parsed["outputs"]
    driven = {i for g in parsed["gates"].values() for i in g["inputs"]}
    return [n for n, g in parsed["gates"].items()
            if g["type"] != "INPUT" and n not in driven]


def simulate_fault_detected(parsed: dict,
                            fault_node: str,
                            fault_stuck_at: int,
                            pattern: Dict[str, str]) -> bool:
    """Return True iff `pattern` distinguishes the fault-free circuit from
    the (fault_node stuck-at fault_stuck_at) faulty circuit.

    Parameters
    ----------
    parsed         : output of parse_netlist
    fault_node     : the name of the line carrying the stuck-at fault
    fault_stuck_at : 0 or 1
    pattern        : {PI_name: '0'|'1' or int} - missing PIs default to 0
    """
    if fault_node not in parsed["gates"]:
        # Fault on an unknown line - treat as undetectable rather than crash.
        return False

    pos = _resolve_outputs(parsed)
    good = _simulate(parsed, pattern)
    bad  = _simulate(parsed, pattern,
                     fault_node=fault_node,
                     fault_value=int(fault_stuck_at) & 1)

    for po in pos:
        if good.get(po, 0) != bad.get(po, 0):
            return True
    return False
