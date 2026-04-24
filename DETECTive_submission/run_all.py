"""
DETECTive — the single "run once, do everything" script.

Upload your `train_dataset.pkl` and `val_dataset.pkl` to the parent folder
(or set `DETECTIVE_DATA_DIR`), then:

    python run_all.py

That's it. This wrapper calls `pipeline.py` with `--with-training` so the
sequence is:

    1. Train the model to completion (~EPOCHS from config.py)
    2. Run synthetic-data analysis + 90% verification
    3. Run ISCAS-85 benchmark evaluation (if yosys + abc + atalanta are present)
    4. Generate all paper-figure PNGs

If you already have trained weights (`best_detective_model.pt`) and just
want evaluation + plots, use `python pipeline.py` directly (no `--with-training`
flag) to skip straight to stage 1.

Any flag accepted by `pipeline.py` can be forwarded via `run_all.py`:

    python run_all.py --fresh             # start training from scratch
    python run_all.py --epochs 300
    python run_all.py --skip-benchmarks   # skip the ISCAS stage
"""

from __future__ import annotations
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import pipeline

if __name__ == "__main__":
    # Forward every CLI arg to pipeline.main, but always include --with-training.
    if "--with-training" not in sys.argv:
        sys.argv.insert(1, "--with-training")
    sys.exit(pipeline.main())
