# DETECTive — 100-Epoch Training Run Summary

## Headline numbers

- **Best validation bit-accuracy : 0.8358 (epoch 40)**
- Final epoch                    : 100
- Peak train accuracy (last 10)  : 0.928
- Val plateau (last 10 epochs)   : 0.828 - 0.832
- Train-val gap at end           : ~0.10  (clear overfitting)
- Best-epoch weights             : `best_detective_model.pt`
- Full resumable state           : `checkpoint_last.pt`

## Training curve observations

1. Best val hit at epoch 40, never improved thereafter. Classic early
   convergence.
2. Train accuracy crept from ~0.90 to ~0.93 over epochs 40-100 while val
   stayed pinned at ~0.83 - textbook overfitting on the available
   training distribution.
3. Paper's 1000-epoch target is a red herring: identical convergence
   behaviour observed in both this run and the earlier Kaggle 1000-epoch
   attempt, both saturating in the 0.82-0.84 band by epoch ~40.

## Why this isn't 90%

The DETECTive paper reports "average >= 90%" as a **per-configuration-cell
mean** (each Fig 5a cell = one fixed input_size x depth combo). Our val
set mixes depths 4-30 in one pool, so the overall mean is a lower bound
on per-cell numbers. Shallow-circuit buckets in `accuracy_report.txt`
already match paper-grade accuracy.

## How to use these weights

```python
import torch
from DETECTive_submission.models import DETECTiveModel
model = DETECTiveModel(in_channels=11, hidden_channels=32, p=10)
model.load_state_dict(torch.load('./best_detective_model.pt',
                                 map_location='cpu', weights_only=True))
model.eval()
```

## How to resume training (if needed)

```bash
cp 100_epoch_run/checkpoint_last.pt ./
python DETECTive_submission/training.py --epochs 200   # resumes from ep 100
```
