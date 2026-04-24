"""
DETECTive — neural network modules.

Implements the three learnable blocks from paper Section 4 plus the master
DETECTiveModel that composes them:

  1. DETECTiveGNN     : GAT + 2 x GCN layers (Section 4.2) that enriches each
                        node's feature with its neighborhood. Operates on the
                        faulted circuit graph.
  2. Activator        : single-layer LSTM over an activation path + MLP head
                        conditioned on the stuck-at value. Predicts the PI
                        assignment that activates the fault (Section 4.3).
  3. Propagator       : LSTM over a propagation path + MLP head. Predicts the
                        PI assignment that propagates fault effect to a PO
                        (Section 4.3).
  4. InputPredictor   : assembles the "cone embedding" from up to P activation
                        and P propagation path encodings (zero-padded when
                        fewer paths exist) and emits the final Boolean
                        prediction for one PI (Section 4.4).
  5. DETECTiveModel   : glues it all together; exposes `forward(graph,
                        pi_indices, fault_type, act_paths, prop_paths)` which
                        predicts a probability for each PI in one go.

The paper's training loss (Binary Cross-Entropy) is computed externally in
training.py — the modules themselves emit sigmoid probabilities.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv

from config import FEATURE_DIM, HIDDEN_DIM, P_PATHS


# ══════════════════════════════════════════════════════════════════
#  GNN (Section 4.2)
# ══════════════════════════════════════════════════════════════════
class DETECTiveGNN(nn.Module):
    """GAT -> GCN -> GCN. Produces a [num_nodes, HIDDEN_DIM] embedding."""

    def __init__(self,
                 in_channels:     int = FEATURE_DIM,
                 hidden_channels: int = HIDDEN_DIM):
        super().__init__()
        self.gat  = GATConv(in_channels,     hidden_channels)
        self.gcn1 = GCNConv(hidden_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gat (x, edge_index))
        x = F.relu(self.gcn1(x, edge_index))
        x = self.gcn2(x, edge_index)
        return x


# ══════════════════════════════════════════════════════════════════
#  Path encoders (Section 4.3)
# ══════════════════════════════════════════════════════════════════
class Activator(nn.Module):
    """LSTM on an activation path, MLP head conditioned on stuck-at value."""

    def __init__(self, in_channels: int = HIDDEN_DIM,
                       hidden_channels: int = HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        # +1 accounts for the concatenated fault-type scalar (0.0 or 1.0).
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_channels + 1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self,
                path_seq:   torch.Tensor,    # [1, path_len, hidden]
                fault_type: torch.Tensor     # [1, 1]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.lstm(path_seq)
        enc  = h[-1]                                    # [1, hidden]
        prob = torch.sigmoid(self.mlp(torch.cat([enc, fault_type], dim=1)))
        return enc, prob


class Propagator(nn.Module):
    """LSTM on a propagation path + MLP head. Fault type is irrelevant here."""

    def __init__(self, in_channels: int = HIDDEN_DIM,
                       hidden_channels: int = HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_channels, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, path_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.lstm(path_seq)
        enc  = h[-1]
        prob = torch.sigmoid(self.mlp(enc))
        return enc, prob


# ══════════════════════════════════════════════════════════════════
#  Input predictor (Section 4.4)
# ══════════════════════════════════════════════════════════════════
class InputPredictor(nn.Module):
    """Combines up to P activation + P propagation path encodings + their
    scalar preds into a single "cone embedding" and predicts the PI's Boolean.
    Missing paths are zero-padded so the input dim is fixed."""

    def __init__(self, p: int = P_PATHS, path_encoding_dim: int = HIDDEN_DIM):
        super().__init__()
        self.p   = p
        self.dim = path_encoding_dim
        # For each of p slots, we pack [path_enc (dim) | scalar pred (1)] for
        # both activation and propagation streams.
        cone_dim = p * (path_encoding_dim + 1) * 2
        self.mlp = nn.Sequential(
            nn.Linear(cone_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # ── helper: zero-pad to p slots and flatten ───────────────────
    def _pack(self,
              encs:  Sequence[torch.Tensor],
              preds: Sequence[torch.Tensor],
              device: torch.device) -> torch.Tensor:
        encs  = list(encs [:self.p])
        preds = list(preds[:self.p])
        n     = len(encs)
        parts: List[torch.Tensor] = []
        for enc, pred in zip(encs, preds):
            parts.append(enc .to(device).view(1, -1))
            parts.append(pred.to(device).view(1, -1))
        # Pad the rest with zeros so the total dim is always p*(dim+1).
        for _ in range(self.p - n):
            parts.append(torch.zeros(1, self.dim, device=device))
            parts.append(torch.zeros(1, 1,        device=device))
        return torch.cat(parts, dim=1)

    def forward(self,
                act_enc:   Sequence[torch.Tensor],
                act_pred:  Sequence[torch.Tensor],
                prop_enc:  Sequence[torch.Tensor],
                prop_pred: Sequence[torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.mlp.parameters()).device
        cone = torch.cat([
            self._pack(act_enc,  act_pred,  device),
            self._pack(prop_enc, prop_pred, device),
        ], dim=1)
        return torch.sigmoid(self.mlp(cone)), cone


# ══════════════════════════════════════════════════════════════════
#  Master model (Section 4.1)
# ══════════════════════════════════════════════════════════════════
class DETECTiveModel(nn.Module):
    """Full DETECTive pipeline: graph -> node embeddings -> per-PI prediction.

    `forward` returns a tensor of shape [len(pi_indices), 1] with one
    sigmoid-probability per primary input.
    """

    def __init__(self,
                 in_channels:     int = FEATURE_DIM,
                 hidden_channels: int = HIDDEN_DIM,
                 p:               int = P_PATHS):
        super().__init__()
        self.gnn        = DETECTiveGNN (in_channels,     hidden_channels)
        self.activator  = Activator    (hidden_channels, hidden_channels)
        self.propagator = Propagator   (hidden_channels, hidden_channels)
        self.predictor  = InputPredictor(p=p, path_encoding_dim=hidden_channels)

    def encode(self, graph: Data) -> torch.Tensor:
        return self.gnn(graph.x, graph.edge_index)

    def forward(self,
                graph:        Data,
                pi_indices:   Sequence[int],
                fault_type:   torch.Tensor,
                act_paths:    List[List[int]],
                prop_paths:   List[List[int]]
                ) -> torch.Tensor:
        node_emb = self.encode(graph)
        if isinstance(pi_indices, int):
            pi_indices = [pi_indices]

        # Propagation paths are PI-independent; encode once.
        prop_enc, prop_pred = [], []
        for path in (prop_paths or []):
            e, y = self.propagator(node_emb[path].unsqueeze(0))
            prop_enc.append(e); prop_pred.append(y)

        preds: List[torch.Tensor] = []
        for pi in pi_indices:
            # Only activation paths that START at this PI are relevant.
            my_acts = [p for p in (act_paths or []) if p and p[0] == pi]
            act_enc, act_pred = [], []
            for path in my_acts:
                e, y = self.activator(node_emb[path].unsqueeze(0), fault_type)
                act_enc.append(e); act_pred.append(y)
            out, _ = self.predictor(act_enc, act_pred, prop_enc, prop_pred)
            preds.append(out)
        return torch.cat(preds, dim=0)
