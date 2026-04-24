"""
DETECTive — accuracy metrics and validation helpers.

Shared by training.py, analysis.py, and benchmarks.py so the three scripts
report comparable numbers.
"""

from __future__ import annotations
from typing import List, Optional, Sequence

import torch


def compute_pattern_accuracy(predicted_bits: Sequence[float],
                             gt_patterns:    Sequence[Sequence[int]]) -> float:
    """Bit-level match to the best-matching ground-truth pattern.

    Matches the paper's evaluation metric (Section 5, Figure 4). A sample with
    multiple valid test patterns is credited with the max match across them,
    since ATPG tools may return multiple equivalent patterns.
    """
    if not gt_patterns or not predicted_bits:
        return 0.0
    hard = [1 if p > 0.5 else 0 for p in predicted_bits]
    n    = len(hard)
    if n == 0:
        return 0.0
    return max(
        sum(1 for p, g in zip(hard, gt) if p == g) / n
        for gt in gt_patterns
    )


def select_best_gt_pattern(pred_probs: Sequence[float],
                           gt_patterns: Sequence[Sequence[int]]
                           ) -> Sequence[int]:
    """Pick the gt pattern most consistent with the current prediction.

    Resolves the ambiguity when multiple equally-valid gts exist (paper's
    "disambiguation" approach) without biasing training toward any one of
    them arbitrarily.
    """
    if len(gt_patterns) == 1:
        return gt_patterns[0]
    hard = [1 if p > 0.5 else 0 for p in pred_probs]
    best      = gt_patterns[0]
    best_hits = -1
    for gt in gt_patterns:
        hits = sum(1 for p, g in zip(hard, gt) if p == g)
        if hits > best_hits:
            best_hits = hits
            best      = gt
    return best


def evaluate_one(model, sample) -> float:
    """Forward `sample` through `model` (no gradients) and return pattern acc."""
    if not sample.get("gt_patterns") or not sample.get("pi_indices"):
        return 0.0
    preds = model(
        sample["graph"], sample["pi_indices"], sample["fault_type"],
        sample.get("act_paths"), sample.get("prop_paths"),
    )
    return compute_pattern_accuracy(preds.view(-1).tolist(), sample["gt_patterns"])


def to_device(sample: dict, device: torch.device) -> dict:
    """Move the graph and the fault-type tensor to the target device.
    (Paths are python lists of ints — no transfer needed.)
    """
    return {
        **sample,
        "graph":      sample["graph"].to(device),
        "fault_type": sample["fault_type"].to(device),
    }
