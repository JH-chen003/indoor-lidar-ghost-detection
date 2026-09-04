# Reproduction Guide

This document provides step-by-step instructions to reproduce the main results
reported in Table 1c of the paper.

## Environment

```
Python      3.10.x
PyTorch     2.1.0+cu121
CUDA        12.1
GPU         NVIDIA Quadro RTX 4000 (8 GB)
OS          Ubuntu 22.04
```

## Installation

```bash
git clone https://anonymous.4open.science/r/indoor-lidar-ghost-detection-F44B/
cd indoor-lidar-ghost-detection
pip install -r requirements.txt
```

## Data

1. Obtain **3DRef** from the original authors under their license.
2. Place under: `data/3dref_proper_split/`
3. Structure required:
```
data/3dref_proper_split/sequences/
    00/velodyne/*.bin  00/labels/*.label   (train, 2240 frames)
    08/velodyne/*.bin  08/labels/*.label   (val,   746 frames)
    09/velodyne/*.bin  09/labels/*.label   (test,  746 frames)
```
4. Label binarization: Label 1 → 0 (normal), Label 5 → 1 (ghost).
   All other labels are excluded during dataset loading.

## Reproduce Table 1c (Focal-priority, frozen, n=3 seeds)

```bash
for SEED in 2026 2027 2028; do
    # Step 1: Train
    python ghost_detector.py \
        --synth_dir data/3dref_proper_split \
        --out_dir   outputs/target_only_seed${SEED} \
        --epochs    100 --batch_size 8 --n_points 8192 \
        --lr        1e-3 --seed ${SEED} \
        --loss_type focal --focal_alpha 0.90 --focal_gamma 2.0 \
        --sampling_mode priority --skip_baseline

    # Step 2: Evaluate (deterministic, eval_seed=42)
    python evaluate_v16.py \
        --ckpt      outputs/target_only_seed${SEED}/best_model.pth \
        --data_dir  data/3dref_proper_split \
        --split     test --eval_seed 42 \
        --out       results_target_only_seed${SEED}_test.json
done
```

Expected results (Table 1c):

| Seed | mIoU  | F1     | Precision | Recall | AP     |
|------|-------|--------|-----------|--------|--------|
| 2026 | 0.9615| 0.9800 | 0.9633    | 0.9974 | 0.9920 |
| 2027 | 0.9631| 0.9809 | 0.9643    | 0.9981 | 0.9937 |
| 2028 | 0.9624| 0.9805 | 0.9638    | 0.9978 | 0.9924 |
| **Mean±SD** | **0.962±0.001** | **0.9805±0.0005** | **0.9638±0.0005** | **0.9978±0.0004** | **0.9927±0.0009** |

## Frozen Checkpoint SHA256

| Seed | SHA256 |
|------|--------|
| 2026 | `1493cf874397e51b1fdb3294d8505205ddb9ccf3c828a5abf99aa10a57233818` |
| 2027 | `b443602040c00eae40e01d7bd708bdecc48bc8e3a969322a3d0f0233c0fc200b` |
| 2028 | `4ab30f01dad59d0ffc398df18439dacce80119e22c96204dd7071eb11b9672f4` |

## Evidence Package

The submission includes `results_master_formal.csv` (51 rows, all AP/mIoU filled)
and `results_coming_soon.csv` (5 voxel-sensitivity rows, Coming Soon).
Every paper claim maps to a `claim_id` in `claim_ledger.md`.
