# Indoor LiDAR Ghost Point Detection

> **Anonymous Submission** — Code for the paper:
> *"Indoor LiDAR Ghost Point Detection with Physics-Inspired Synthetic Data: A Controlled Multi-Factor Diagnosis"*
>
> Author information will be disclosed upon acceptance.

---

## Overview

LiDAR sensors in indoor environments produce **ghost points** — spurious returns caused by multipath propagation near glass and metallic surfaces. This repository provides the complete training and evaluation code used in the paper.

This work is positioned as a **controlled multi-factor diagnosis** within a fixed 3DRef indoor target domain, not a general-purpose domain adaptation method. Conclusions are limited to the specified target domain and protocol.

---

## Key Results

Focal-priority model (original setting), frozen after training, evaluated **once** on the independent 3DRef target test set (seq09, 746 frames, n=3 seeds):

| Metric | Mean ± SD |
|--------|-----------|
| mIoU | **0.962 ± 0.001** |
| F1 | 0.9805 ± 0.0005 |
| Precision | 0.9638 ± 0.0005 |
| Recall | 0.9978 ± 0.0004 |
| AP | 0.9927 ± 0.0009 |

All SDs < 0.001 — high reproducibility within the fixed target-domain setting.

---

## Files

| File | Purpose |
|------|---------|
| `ghost_detector.py` | Core training pipeline (PointNet2 + Focal Loss + priority sampling) |
| `evaluate.py` | Basic evaluation on a checkpoint |
| `evaluate.py` | **Deterministic** evaluation with fixed `--eval_seed` and `torch.use_deterministic_algorithms` |
| `build_tables.py` | Regenerate all paper tables from `results_master_formal.csv` |
| `requirements.lock` | Package dependencies |
| `requirements.lock` | Tested environment lockfile |
| `REPRODUCE.md` | Step-by-step reproduction guide with expected outputs |
| `LICENSE` | MIT License |

---

## Requirements

```bash
pip install -r requirements.lock
```

Tested environment:

| Component | Version |
|-----------|---------|
| Python | 3.10.x |
| PyTorch | 2.1.0+cu121 |
| CUDA | 12.1 |
| GPU | NVIDIA Quadro RTX 4000 (8 GB) |
| OS | Ubuntu 22.04 |

---

## Data Preparation

**3DRef** (target domain, real indoor):
- Obtain from the original authors under their license
- Place at: `data/3dref_proper_split/`

Required structure:
```
data/3dref_proper_split/sequences/
    00/velodyne/*.bin  00/labels/*.label   (train, 2240 frames)
    08/velodyne/*.bin  08/labels/*.label   (val,    746 frames)
    09/velodyne/*.bin  09/labels/*.label   (test,   746 frames)
```

**Label binarization** (applied automatically in `GhostDataset`):
- Label 1 (normal) → 0
- Label 5 (ghost)  → 1
- All other labels excluded

**Split design**: temporal-block split — no adjacent-frame overlap between train/val/test.

---

## Training

Reproduce the **Focal-priority** frozen model (seeds 2026/2027/2028):

```bash
for SEED in 2026 2027 2028; do
    python ghost_detector.py \
        --synth_dir  data/3dref_proper_split \
        --out_dir    outputs/target_only_seed${SEED} \
        --epochs     100 \
        --batch_size 8 \
        --n_points   8192 \
        --lr         1e-3 \
        --seed       ${SEED} \
        --loss_type  focal \
        --focal_alpha 0.90 \
        --focal_gamma 2.0 \
        --sampling_mode priority \
        --skip_baseline
done
```

---

## Evaluation (Deterministic)

Use `evaluate.py` for reproducible evaluation with fixed sampling:

```bash
for SEED in 2026 2027 2028; do
    python evaluate.py \
        --ckpt      outputs/target_only_seed${SEED}/best_model.pth \
        --data_dir  data/3dref_proper_split \
        --split     test \
        --eval_seed 42 \
        --out       results_seed${SEED}_test.json
done
```

**Important**: Each frozen checkpoint is evaluated **once** on the test set. Test results are never used to select or adjust the model.

---

## Frozen Checkpoint SHA256

| Seed | SHA256 |
|------|--------|
| 2026 | `1493cf874397e51b1fdb3294d8505205ddb9ccf3c828a5abf99aa10a57233818` |
| 2027 | `b443602040c00eae40e01d7bd708bdecc48bc8e3a969322a3d0f0233c0fc200b` |
| 2028 | `4ab30f01dad59d0ffc398df18439dacce80119e22c96204dd7071eb11b9672f4` |

---

## Reproducing All Paper Tables

All paper tables can be regenerated from the evidence package:

```bash
python build_tables.py \
    --master results_master_formal.csv \
    --out    tables/
```

`results_master_formal.csv` contains 51 formal experiment rows with all AP/mIoU fields filled. Every number in the paper maps to a row in this file via `claim_ledger.md`.

See `REPRODUCE.md` for full step-by-step instructions.

---

## Evidence Package

The submission includes the following auditable materials:

| File | Content |
|------|---------|
| `frozen_manifest.md` | Checkpoint SHA256, frozen rules, per-seed target-test metrics |
| `claim_ledger.md` | Paper claims → master table row mapping |
| `results_master_formal.csv` | 51 formal experiment records, all AP/mIoU filled |
| `Dataset_Card.md` | Data specifications, split rules, binarization, limitations |
| `model_selection_policy.md` | Selection rule, honest post-hoc timing disclosure |

---

## Experimental Design Summary

| Experiment | Purpose | n seeds |
|------------|---------|---------|
| E0 | Baseline: Target-only / Source-only / Direct-mixed on unified target test | 3 |
| E2 | Proportion-oriented compound rebalancing (Ghost Ratio: 0.895% → 10.83%) | 3 |
| E3 | Common-FOV cropping to isolate vertical FOV difference | 3 |
| E4 | Complete 3×2 ablation: Loss (CE/WCE/Focal) x Sampling (priority/random) | 3 each |
| E6 | Intensity calibration: percentile / z-score / histogram matching | 3 each |

**Scope**: All conclusions are limited to the fixed 3DRef target domain. No claim of external generalization, cross-sensor robustness, or sole causality.

---

## License

MIT License — see `LICENSE` for details.
3DRef data is subject to its original license and is not redistributed here.
