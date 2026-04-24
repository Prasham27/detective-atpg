"""
DETECTive — circuit graph utilities.

Two classes:
  * CircuitGraphBuilder  — parses a gate-level Verilog netlist and produces a
    PyG `Data(x, edge_index)` object. Each node is one gate (or primary input),
    encoded with a feature vector [gate_type_onehot | is_faulty | log1p(fanout)]
    (see config.FEATURE_DIM). Edges point from a gate's inputs to the gate.
  * PathExtractor        — computes activation paths (PI -> fault) and
    propagation paths (fault -> PO) using BFS + a capped DFS. These paths are
    what the paper's Activator/Propagator LSTMs consume.

Both classes mirror the paper's circuit modeling (Section 4.2). The DFS is
capped by `_MAX_NODES` to avoid exponential blow-up on circuits with many
reconvergent fanouts.
"""

from __future__ import annotations
import re, math
from collections import deque
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from config import GATE_TO_IDX, GATE_TYPES, FEATURE_DIM


# ══════════════════════════════════════════════════════════════════
#  CircuitGraphBuilder
# ══════════════════════════════════════════════════════════════════
class CircuitGraphBuilder:
    """Parse a flat gate-level Verilog string into a PyG graph.

    Supported gate primitives: and, or, nand, nor, xor, xnor, not, buf.
    Primary inputs are declared with `input` and become nodes of type INPUT.
    """

    _GATE_RE = r"(and|or|nand|nor|xor|xnor|not|buf)\s+\w+\s*\(([^)]+)\);"

    def __init__(self, netlist: str):
        self.raw_netlist      = netlist
        self.netlist          = self._clean_verilog(netlist)
        self.gates: Dict[str, dict] = {}
        self.explicit_outputs: List[str] = []
        self._parse_verilog()
        self._calculate_fanouts()

    @staticmethod
    def _clean_verilog(text: str) -> str:
        """Strip comments and flatten to one big line for easier regex."""
        text = re.sub(r"/\*[\s\S]*?\*/", "", text)
        text = re.sub(r"//.*", "", text)
        return text.replace("\n", " ")

    def _parse_verilog(self) -> None:
        # Inputs
        for group in re.findall(r"input\s+([^;]+);", self.netlist):
            for inp in group.split(","):
                name = inp.strip()
                if name:
                    self.gates[name] = {"type": "INPUT", "inputs": [], "fanout": 0}

        # Explicit outputs (so we can honor the design's output declaration
        # even when a PO has fanout > 0 inside the circuit).
        for group in re.findall(r"output\s+([^;]+);", self.netlist):
            for out in group.split(","):
                name = out.strip()
                if name:
                    self.explicit_outputs.append(name)

        # Gates
        for gtype, args in re.findall(self._GATE_RE, self.netlist, re.IGNORECASE):
            tokens   = [t.strip() for t in args.split(",")]
            out_node = tokens[0]
            in_nodes = tokens[1:]
            self.gates[out_node] = {
                "type":   gtype.upper(),
                "inputs": in_nodes,
                "fanout": 0,
            }

    def _calculate_fanouts(self) -> None:
        for node, data in self.gates.items():
            for inp in data["inputs"]:
                if inp in self.gates:
                    self.gates[inp]["fanout"] += 1

    def get_pyg_graph(self,
                      faulty_node: Optional[str] = None
                      ) -> Tuple[Data, Dict[str, int]]:
        """Return (PyG Data, name->index map).

        `faulty_node`, if given, sets the is_faulty flag on that node's feature
        vector (the paper's fault annotation).
        """
        node_names = list(self.gates.keys())
        name_to_id = {name: i for i, name in enumerate(node_names)}
        n = len(node_names)

        x = torch.zeros((n, FEATURE_DIM), dtype=torch.float)
        for i, name in enumerate(node_names):
            gdata = self.gates[name]
            x[i, GATE_TO_IDX.get(gdata["type"], 0)] = 1.0
            x[i, len(GATE_TYPES)]     = 1.0 if name == faulty_node else 0.0
            x[i, len(GATE_TYPES) + 1] = math.log1p(float(gdata["fanout"]))

        src: List[int] = []
        dst: List[int] = []
        for target_name, data in self.gates.items():
            target_id = name_to_id[target_name]
            # set() — ignore duplicate inputs to the same gate (e.g. `nand(y,a,a)`).
            for src_name in set(data["inputs"]):
                if src_name in name_to_id:
                    src.append(name_to_id[src_name])
                    dst.append(target_id)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return Data(x=x, edge_index=edge_index), name_to_id


# ══════════════════════════════════════════════════════════════════
#  PathExtractor
# ══════════════════════════════════════════════════════════════════
class PathExtractor:
    """Extracts fault activation and propagation paths from a graph.

    Activation paths run from each primary input to the fault site.
    Propagation paths run from the fault site to any primary output.

    Uses BFS to guarantee at least one (shortest) path when one exists, then
    tops up via a DFS capped by `_MAX_NODES` to bound runtime on densely-
    reconvergent circuits.
    """

    _MAX_NODES = 500        # cap for the DFS top-up to prevent explosion

    def __init__(self,
                 edge_index: torch.Tensor,
                 num_nodes: int,
                 explicit_pos: Optional[List[int]] = None):
        self.num_nodes = num_nodes
        self.adj:  Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        self.radj: Dict[int, List[int]] = {i: [] for i in range(num_nodes)}
        in_deg  = {i: 0 for i in range(num_nodes)}
        out_deg = {i: 0 for i in range(num_nodes)}

        ei = edge_index.cpu() if edge_index.device.type != "cpu" else edge_index
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            self.adj[u].append(v);  self.radj[v].append(u)
            out_deg[u] += 1;        in_deg[v]  += 1

        self.pis = [i for i in range(num_nodes) if in_deg[i] == 0]
        self.pos = (set(explicit_pos) if explicit_pos
                    else {i for i in range(num_nodes) if out_deg[i] == 0})

    # ── BFS / DFS primitives ──────────────────────────────────────
    def _bfs_one_path(self, start: int, target_or_pos, to_po: bool = False) -> List[int]:
        if to_po and start in self.pos: return [start]
        if not to_po and start == target_or_pos: return [start]
        parent = {start: None}
        q      = deque([start])
        while q:
            cur = q.popleft()
            for nb in self.adj.get(cur, []):
                if nb not in parent:
                    parent[nb] = cur
                    reached = (to_po and nb in self.pos) or (not to_po and nb == target_or_pos)
                    if reached:
                        path: List[int] = []
                        n: Optional[int] = nb
                        while n is not None:
                            path.append(n); n = parent[n]
                        return path[::-1]
                    q.append(nb)
        return []

    def _dfs_capped(self, start: int, target_or_pos,
                    max_paths: int, to_po: bool = False) -> List[List[int]]:
        paths: List[List[int]] = []
        nodes_seen = 0
        stack = [(start, [start], {start})]
        while stack and len(paths) < max_paths and nodes_seen < self._MAX_NODES:
            cur, path, visited = stack.pop()
            nodes_seen += 1
            reached = (to_po and cur in self.pos) or (not to_po and cur == target_or_pos)
            if reached:
                paths.append(list(path))
                continue
            for nb in self.adj.get(cur, []):
                if nb not in visited:
                    stack.append((nb, path + [nb], visited | {nb}))
        return paths

    # ── Public: activation + propagation paths ────────────────────
    def get_activation_paths(self, fault_idx: Optional[int],
                             max_paths: int = 10) -> List[List[int]]:
        if fault_idx is None or fault_idx not in self.adj:
            return []
        res: List[List[int]] = []
        for pi in self.pis:
            if len(res) >= max_paths: break
            shortest = self._bfs_one_path(pi, fault_idx)
            if not shortest: continue
            res.append(shortest)
            if len(res) >= max_paths: break
            extras = self._dfs_capped(pi, fault_idx, max_paths - len(res))
            seen   = {tuple(p) for p in res}
            for p in extras:
                if tuple(p) not in seen:
                    res.append(p); seen.add(tuple(p))
                if len(res) >= max_paths: break
        return res

    def get_propagation_paths(self, fault_idx: Optional[int],
                              max_paths: int = 10) -> List[List[int]]:
        if fault_idx is None or fault_idx not in self.adj:
            return []
        shortest = self._bfs_one_path(fault_idx, self.pos, to_po=True)
        res: List[List[int]] = [shortest] if shortest else []
        if len(res) >= max_paths: return res
        extras = self._dfs_capped(fault_idx, self.pos, max_paths - len(res), to_po=True)
        seen   = {tuple(p) for p in res}
        for p in extras:
            if tuple(p) not in seen:
                res.append(p); seen.add(tuple(p))
            if len(res) >= max_paths: break
        return res

    # ── Utilities (used by dataset sanity checks) ─────────────────
    def reachable_from_pis(self) -> set:
        vis = set(self.pis); q = deque(self.pis)
        while q:
            n = q.popleft()
            for nb in self.adj.get(n, []):
                if nb not in vis: vis.add(nb); q.append(nb)
        return vis

    def reachable_to_pos(self) -> set:
        vis = set(self.pos); q = deque(self.pos)
        while q:
            n = q.popleft()
            for nb in self.radj.get(n, []):
                if nb not in vis: vis.add(nb); q.append(nb)
        return vis
