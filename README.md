# Indoor LiDAR Ghost Point Detection

Code for the paper:  
**"Indoor LiDAR Ghost Point Detection with Physics-Inspired Synthetic Data: A Controlled Multi-Factor Diagnosis"**

> **Anonymous submission.** Author information will be disclosed after the review process.

---

## Overview

Ghost points caused by multipath propagation arise in indoor LiDAR near glass and metallic surfaces. This repository provides the training and evaluation code used in the paper.

**Key results (Focal-priority, frozen model, independent 3DRef target test, n=3 seeds):**

| Metric | Mean ± SD |
|--------|-----------|
| mIoU   | 0.962 ± 0.001 |
| F1     | 0.9805 ± 0.0005 |
| Precision | 0.9638 ± 0.0005 |
| Recall | 0.9978 ± 0.0004 |
| AP     | 0.9927 ± 0.0009 |

All SDs < 0.001 across 3 seeds, demonstrating high reproducibility within the fixed target-domain setting.

---

## Requirements

```bash
pip install -r requirements.txt
```

- Python ≥ 3.8
- PyTorch ≥ 2.0 (CUDA recommended)
- scikit-learn ≥ 1.2
- numpy ≥ 1.24

---

## Data Preparation

This paper uses:
- **3DRef** (target domain, real indoor): obtain from the original authors under their license. Place under `data/3dref_proper_split/sequences/`.
- **synth_C_v5** (source domain, physics-inspired synthetic): generated from KITTI using the synthesis pipeline described in the paper. Contact authors after acceptance.

Dataset split (temporal-block, no adjacent-frame leakage):

| Split | Sequence | Frames |
|-------|----------|--------|
| Train | seq00    | 2,240  |
| Val   | seq08    | 746    |
| Test  | seq09    | 746    |

Label binarization: Label 1 (normal) → 0, Label 5 (ghost) → 1. All other labels are excluded during preprocessing.

---

## Training

**Reproduce the frozen Focal-priority model (original setting, seed=2026):**

```bash
python ghost_detector.py \
    --synth_dir  data/synth_C_v5 \
    --out_dir    outputs/focal_priority_seed2026 \
    --epochs     100 \
    --batch_size 8 \
    --n_points   8192 \
    --lr         1e-3 \
    --seed       2026 \
    --loss_type  focal \
    --focal_alpha 0.90 \
    --focal_gamma 2.0 \
    --sampling_mode priority \
    --skip_baseline
```

**For Target-only training (no synthetic data):**

```bash
python ghost_detector.py \
    --synth_dir  data/3dref_proper_split \
    --out_dir    outputs/target_only_seed2026 \
    --seed       2026 \
    --loss_type  focal \
    --sampling_mode priority \
    --skip_baseline
```

**For Direct-mixed training:**

```bash
python ghost_detector.py \
    --synth_dir  data/synth_C_v5 \
    --ref_dir    data/3dref_proper_split/sequences \
    --out_dir    outputs/direct_mixed_seed2026 \
    --seed       2026 \
    --loss_type  focal \
    --sampling_mode priority \
    --skip_baseline
```

---

## Evaluation (Frozen Checkpoint)

```bash
python evaluate.py \
    --ckpt      outputs/target_only_seed2026/best_model.pth \
    --data_dir  data/3dref_proper_split \
    --split     test \
    --out       results_seed2026.json
```

**Important:** The test set (seq09) should be evaluated only once per final frozen model. Do not use test results to iterate model selection.

---

## Reproducing Main Results

To reproduce Table 1c (Focal-priority, frozen, independent target test, n=3 seeds):

```bash
for SEED in 2026 2027 2028; do
    python ghost_detector.py \
        --synth_dir data/3dref_proper_split \
        --out_dir   outputs/focal_priority_seed${SEED} \
        --seed      ${SEED} --loss_type focal \
        --sampling_mode priority --skip_baseline
    python evaluate.py \
        --ckpt    outputs/focal_priority_seed${SEED}/best_model.pth \
        --data_dir data/3dref_proper_split \
        --split   test \
        --out     results_target_only_seed${SEED}.json
done
```

Expected output per seed: mIoU ≈ 0.962, F1 ≈ 0.980, AP ≈ 0.993.

---

## Evidence Package

The submission includes a read-only evidence package with:
- `frozen_manifest.md` — checkpoint SHA256, frozen rules
- `claim_ledger.md` — mapping of paper claims to master table rows
- `results_master_formal.csv` — 56 formal experiment records (all AP/mIoU fields filled)
- `Dataset_Card.md` — data specifications, split rules, known limitations
- `model_selection_policy.md` — selection rules and honest disclosure of post-hoc timing

---

## Code Availability

Code is provided under the MIT License. Data (3DRef) is subject to its original license. Trained checkpoints (10MB each) will be released via the same repository after acceptance.
