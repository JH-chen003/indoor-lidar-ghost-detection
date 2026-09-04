# Reproduction Guide

**Reproducibility level**: fixed-seed, same-environment re-run.
Results are expected to match on the same GPU model and CUDA/PyTorch version.
Bit-exact reproduction across different hardware is not guaranteed.
See: https://pytorch.org/docs/stable/notes/randomness.html

---

## Tested Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA Quadro RTX 4000 (8 GB) |
| CUDA | 12.1 |
| Python | 3.10.x |
| PyTorch | 2.1.0+cu121 |
| NumPy | 1.26.4 |
| scikit-learn | 1.3.2 |

---

## Installation (using lockfile for exact environment)

```bash
# Install PyTorch with CUDA 12.1 wheel first
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Then install remaining packages from lockfile
pip install -r requirements.lock
```

If you cannot match the exact CUDA/GPU environment, results may differ slightly.
The `requirements.lock` file pins the exact tested versions.

---

## Data Preparation

1. Obtain **3DRef** from the original authors under their license.
2. Place under `data/3dref_proper_split/sequences/` with this structure:

```
data/3dref_proper_split/sequences/
    00/velodyne/*.bin   00/labels/*.label   (train, 2240 frames)
    08/velodyne/*.bin   08/labels/*.label   (val,    746 frames)
    09/velodyne/*.bin   09/labels/*.label   (test,   746 frames)
```

3. Label binarization (automatic in `GhostDataset`):
   - Label 1 (normal) → 0
   - Label 5 (ghost)  → 1
   - All other labels excluded during loading

---

## Step 1: Training (Focal-priority, n=3 seeds)

```bash
for SEED in 2026 2027 2028; do
    python ghost_detector.py \
        --synth_dir     data/3dref_proper_split \
        --out_dir       outputs/target_only_seed${SEED} \
        --epochs        100 \
        --batch_size    8 \
        --n_points      8192 \
        --lr            1e-3 \
        --seed          ${SEED} \
        --loss_type     focal \
        --focal_alpha   0.90 \
        --focal_gamma   2.0 \
        --sampling_mode priority \
        --skip_baseline
done
```

---

## Step 2: Frozen Evaluation (Table 1c)

Use `evaluate.py` which sets `num_workers=0` and fixed seed for deterministic evaluation:

```bash
for SEED in 2026 2027 2028; do
    python evaluate.py \
        --ckpt      outputs/target_only_seed${SEED}/best_model.pth \
        --data_dir  data/3dref_proper_split \
        --split     test \
        --seed      42 \
        --out       results_target_only_seed${SEED}_test.json
done
```

**Rule**: each checkpoint is evaluated on the test set exactly once.
Do not use test-set results to select or adjust the model.

---

## Expected Results (Table 1c)

| Seed | mIoU | F1 | Precision | Recall | AP |
|------|------|----|-----------|--------|-----|
| 2026 | 0.9615 | 0.9800 | 0.9633 | 0.9974 | 0.9920 |
| 2027 | 0.9631 | 0.9809 | 0.9643 | 0.9981 | 0.9937 |
| 2028 | 0.9624 | 0.9805 | 0.9638 | 0.9978 | 0.9924 |
| **Mean±SD** | **0.962±0.001** | **0.9805±0.0005** | **0.9638±0.0005** | **0.9978±0.0004** | **0.9927±0.0009** |

---

## Frozen Checkpoint SHA256

| Seed | SHA256 |
|------|--------|
| 2026 | `1493cf874397e51b1fdb3294d8505205ddb9ccf3c828a5abf99aa10a57233818` |
| 2027 | `b443602040c00eae40e01d7bd708bdecc48bc8e3a969322a3d0f0233c0fc200b` |
| 2028 | `4ab30f01dad59d0ffc398df18439dacce80119e22c96204dd7071eb11b9672f4` |

Verify before evaluation:
```bash
sha256sum outputs/target_only_seed2026/best_model.pth
```

---

## Step 3: Regenerate All Paper Tables

```bash
python build_tables.py \
    --master results_master_formal.csv \
    --out    tables/

# Output: tables/table_e0_val.csv, table_e0_test.csv,
#         table_e2.csv, table_e3.csv, table_e3_full.csv,
#         table_e4_miou.csv, table_e4_full.csv,
#         table_e6.csv, table_e6_full.csv,
#         build_tables.log
```

Every paper number should match the corresponding CSV output.
The log file records the exact values used for each table.

---

## Known Limitations

1. Results may differ slightly across GPU models or CUDA versions.
2. `torch.use_deterministic_algorithms(True)` may raise errors on some operations
   on older GPU/CUDA combinations; a warning is printed but evaluation continues.
3. This code reproduces the fixed 3DRef target-domain evaluation only.
   External generalization to other datasets or sensors is not evaluated here.
