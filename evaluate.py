#!/usr/bin/env python3
"""
evaluate.py — Run evaluation on a frozen checkpoint
Usage:
    python evaluate.py \
        --ckpt /path/to/best_model.pth \
        --data_dir /path/to/3dref_proper_split \
        --split test
"""
import argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import average_precision_score, confusion_matrix

from ghost_detector import GhostDataset, PointNet2GhostDetector

def run_eval(ckpt_path, data_dirs, split='test', n_points=8192, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = PointNet2GhostDetector(in_dim=8).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"val F1={ckpt['metrics']['F1']:.4f}")

    ds     = GhostDataset(data_dirs, split=split, n_points=n_points, augment=False)
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
    precision  = tp / (tp+fp+1e-8)
    recall     = tp / (tp+fn+1e-8)
    f1         = 2*precision*recall / (precision+recall+1e-8)
    iou_ghost  = tp / (tp+fp+fn+1e-8)
    iou_normal = tn / (tn+fp+fn+1e-8)
    miou       = (iou_ghost + iou_normal) / 2
    ap         = average_precision_score(all_labels, all_probs)

    results = dict(split=split, Precision=round(float(precision),4),
                   Recall=round(float(recall),4), F1=round(float(f1),4),
                   IoU_ghost=round(float(iou_ghost),4), mIoU=round(float(miou),4),
                   AP=round(float(ap),4),
                   TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))
    print(json.dumps(results, indent=2))
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt',      required=True,  help='Path to best_model.pth')
    p.add_argument('--data_dir',  required=True,  help='Dataset root (contains sequences/)')
    p.add_argument('--ref_dir',   default=None,   help='Additional data dir (optional)')
    p.add_argument('--split',     default='test', choices=['train','val','test'])
    p.add_argument('--n_points',  type=int, default=8192)
    p.add_argument('--batch_size',type=int, default=4)
    p.add_argument('--out',       default=None,   help='Save JSON results to file')
    args = p.parse_args()

    data_dirs = [Path(args.data_dir) / 'sequences']
    if args.ref_dir:
        data_dirs.append(Path(args.ref_dir))

    results = run_eval(args.ckpt, data_dirs, args.split, args.n_points, args.batch_size)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {args.out}")

if __name__ == '__main__':
    main()
