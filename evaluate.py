#!/usr/bin/env python3
"""
evaluate.py — Reproducibility-aware evaluation on a frozen checkpoint
==========================================================================
Reproducibility level: fixed-seed, same-environment re-run.

What this script controls:
  - torch.manual_seed / numpy.random.seed / torch.cuda.manual_seed_all
  - torch.backends.cudnn.deterministic = True, benchmark = False
  - torch.use_deterministic_algorithms(True) where supported
  - DataLoader worker_init_fn + generator to fix worker RNG state
  - num_workers=0 for frozen test evaluation (eliminates multiprocess RNG)

What this script does NOT guarantee:
  - Bit-exact results across different GPU models or driver versions
  - Exact reproduction on hardware that uses non-deterministic CUDA kernels
  - Reproduction without the same PyTorch+CUDA version (see requirements.lock)

For fully deterministic evaluation, use num_workers=0 (default here).
See: https://pytorch.org/docs/stable/notes/randomness.html

Usage:
    python evaluate.py \\
        --ckpt   outputs/target_only_seed2026/best_model.pth \\
        --data_dir data/3dref_proper_split \\
        --split  test \\
        --seed   42 \\
        --out    results_seed2026_test.json
"""
import argparse, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import average_precision_score, confusion_matrix

from ghost_detector import GhostDataset, PointNet2GhostDetector


def seed_worker(worker_id):
    """Fix worker-level NumPy and Python random state for DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def set_reproducible(seed: int):
    """Set all known random sources to a fixed state."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[WARN] use_deterministic_algorithms not fully supported: {e}")
        print("       Results may vary slightly across runs on this hardware.")


def get_env():
    info = {"torch": torch.__version__,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def run_eval(ckpt_path, data_dirs, split="test", n_points=8192,
             batch_size=4, seed=42):
    set_reproducible(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = get_env()
    print(f"Environment: {env}")
    print(f"Eval seed: {seed} | split: {split} | num_workers: 0 (deterministic)")

    model = PointNet2GhostDetector(in_dim=8).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}, val F1={ckpt['metrics']['F1']:.4f}")

    ds = GhostDataset(data_dirs, split=split, n_points=n_points, augment=False)
    print(f"Dataset: {len(ds)} frames")

    # Use num_workers=0 for frozen evaluation — eliminates multiprocess RNG
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, generator=g)

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
    prec = float(tp / (tp + fp + 1e-8))
    rec  = float(tp / (tp + fn + 1e-8))
    f1   = float(2 * prec * rec / (prec + rec + 1e-8))
    iou_g = float(tp / (tp + fp + fn + 1e-8))
    iou_n = float(tn / (tn + fp + fn + 1e-8))
    miou  = (iou_g + iou_n) / 2
    ap    = float(average_precision_score(all_labels, all_probs))

    results = {
        "split":       split,
        "eval_seed":   seed,
        "n_frames":    len(ds),
        "Precision":   round(prec, 4),
        "Recall":      round(rec, 4),
        "F1":          round(f1, 4),
        "IoU_ghost":   round(iou_g, 4),
        "mIoU":        round(miou, 4),
        "AP":          round(ap, 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "environment": env,
        "reproducibility_note": (
            "Fixed-seed, same-environment re-run. num_workers=0 used for "
            "frozen test evaluation to eliminate multiprocess RNG. "
            "Bit-exact results require identical GPU model and driver version."
        ),
    }
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("environment", "reproducibility_note")}, indent=2))
    return results


def main():
    p = argparse.ArgumentParser(
        description="Reproducibility-aware frozen checkpoint evaluation")
    p.add_argument("--ckpt",       required=True,  help="Path to best_model.pth")
    p.add_argument("--data_dir",   required=True,  help="Dataset root (contains sequences/)")
    p.add_argument("--ref_dir",    default=None,   help="Additional data dir (optional)")
    p.add_argument("--split",      default="test", choices=["train", "val", "test"])
    p.add_argument("--n_points",   type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed",       type=int, default=42,
                   help="Fixed random seed for evaluation (default: 42)")
    p.add_argument("--out",        default=None,   help="Save JSON results to file")
    args = p.parse_args()

    data_dirs = [Path(args.data_dir) / "sequences"]
    if args.ref_dir:
        data_dirs.append(Path(args.ref_dir))

    results = run_eval(args.ckpt, data_dirs, args.split,
                       args.n_points, args.batch_size, args.seed)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
