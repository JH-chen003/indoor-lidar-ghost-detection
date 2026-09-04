#!/usr/bin/env python3
"""
evaluate_v16.py — Deterministic evaluation on a frozen checkpoint
==================================================================
Changes from evaluate.py:
  - Fixed eval_seed for reproducible random sampling
  - Saves sampling indices to .npz for exact reproduction
  - Sets torch.use_deterministic_algorithms(True)
  - Records PyTorch/CUDA/GPU environment

Usage:
    python evaluate_v16.py \
        --ckpt outputs/target_only_seed2026/best_model.pth \
        --data_dir data/3dref_proper_split \
        --split test \
        --eval_seed 42 \
        --out results_seed2026_test.json
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from sklearn.metrics import average_precision_score, confusion_matrix

from ghost_detector import GhostDataset, PointNet2GhostDetector

def set_deterministic(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[WARN] use_deterministic_algorithms: {e}")

def get_env_info():
    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info

def run_eval(ckpt_path, data_dirs, split='test', n_points=8192,
             batch_size=4, eval_seed=42, indices_path=None):
    set_deterministic(eval_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PointNet2GhostDetector(in_dim=8).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']}, val F1={ckpt['metrics']['F1']:.4f}")

    ds = GhostDataset(data_dirs, split=split, n_points=n_points, augment=False,
                      sampling_mode='priority')

    # Save or load fixed sampling indices
    if indices_path and Path(indices_path).exists():
        saved = np.load(indices_path)
        print(f"Loaded fixed indices from {indices_path}")
    else:
        print(f"Using fresh sampling (eval_seed={eval_seed})")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for feat, xyz, labels in loader:
            logits = model(feat.to(device), xyz.to(device))
            probs  = torch.softmax(logits, -1)[..., 1].cpu().numpy()
            all_probs.append(probs.ravel())
            all_labels.append(labels.numpy().ravel())

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds  = (all_probs > 0.5).astype(int)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    precision  = float(tp / (tp+fp+1e-8))
    recall     = float(tp / (tp+fn+1e-8))
    f1         = float(2*precision*recall / (precision+recall+1e-8))
    iou_ghost  = float(tp / (tp+fp+fn+1e-8))
    iou_normal = float(tn / (tn+fp+fn+1e-8))
    miou       = (iou_ghost + iou_normal) / 2
    ap         = float(average_precision_score(all_labels, all_probs))

    results = {
        "split": split,
        "eval_seed": eval_seed,
        "n_frames": len(ds),
        "Precision": round(precision,4),
        "Recall":    round(recall,4),
        "F1":        round(f1,4),
        "IoU_ghost": round(iou_ghost,4),
        "mIoU":      round(miou,4),
        "AP":        round(ap,4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "environment": get_env_info(),
    }
    print(json.dumps({k:v for k,v in results.items() if k!='environment'}, indent=2))
    return results

def main():
    p = argparse.ArgumentParser(description='Deterministic frozen checkpoint evaluation')
    p.add_argument('--ckpt',       required=True)
    p.add_argument('--data_dir',   required=True)
    p.add_argument('--ref_dir',    default=None)
    p.add_argument('--split',      default='test', choices=['train','val','test'])
    p.add_argument('--n_points',   type=int, default=8192)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--eval_seed',  type=int, default=42,
                   help='Fixed random seed for evaluation sampling (default: 42)')
    p.add_argument('--indices',    default=None,
                   help='Path to pre-saved sampling indices .npz (optional)')
    p.add_argument('--out',        default=None)
    args = p.parse_args()

    data_dirs = [Path(args.data_dir) / 'sequences']
    if args.ref_dir:
        data_dirs.append(Path(args.ref_dir))

    results = run_eval(args.ckpt, data_dirs, args.split,
                       args.n_points, args.batch_size,
                       args.eval_seed, args.indices)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.out}")

if __name__ == '__main__':
    main()
