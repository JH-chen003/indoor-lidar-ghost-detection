#!/usr/bin/env python3
"""
ghost_detector.py — Indoor LiDAR Ghost Point Detection
=======================================================
Model:  PointNet2 (point cloud semantic segmentation, Qi et al. NeurIPS 2017)
Data:   Physics-inspired synthetic outdoor data + 3DRef real indoor data
Labels: 1=normal, 5=ghost (SemanticKITTI format)

References:
  - PointNet2:        Qi et al., NeurIPS 2017
  - Ghost definition: Yun & Sim, CVPR 2018; 3DRef, 3DV 2024
  - Intensity feats:  Gao et al., Remote Sensing 2021; 3DRef Table 4
  - Focal Loss:       Lin et al., ICCV 2017
"""
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score
import warnings
warnings.filterwarnings('ignore')

LABEL_NORMAL = 1
LABEL_GHOST  = 5
MAP_TO_TRAIN = {LABEL_NORMAL: 0, LABEL_GHOST: 1}

# ── Dataset ───────────────────────────────────────────────
class GhostDataset(Dataset):
    """
    Reads SemanticKITTI-format ghost detection data.
    Supports synth_C_v5 and 3DRef (label 1=normal, 5=ghost).
    """
    def __init__(self, data_dirs, split='train', n_points=8192,
                 augment=True, sampling_mode='priority'):
        self.n_points = n_points
        self.augment  = augment and (split == 'train')
        self.sampling_mode = sampling_mode
        self.files = []
        seq_map = {'train': '00', 'val': '08', 'test': '09'}
        seq_id  = seq_map[split]
        for seq_dir in data_dirs:
            seq_dir = Path(seq_dir)
            velo_dir  = seq_dir / seq_id / 'velodyne'
            label_dir = seq_dir / seq_id / 'labels'
            if not velo_dir.exists():
                print(f"[WARN] Not found: {velo_dir}, skipping")
                continue
            bins = sorted(velo_dir.glob('*.bin'))
            pairs = [(b, label_dir / (b.stem + '.label'))
                     for b in bins if (label_dir / (b.stem + '.label')).exists()]
            self.files.extend(pairs)
        print(f"[{split}] {len(self.files)} frames")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bin_path, label_path = self.files[idx]
        pts    = np.fromfile(bin_path,   dtype=np.float32).reshape(-1, 4)
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = (labels & 0xFFFF).astype(np.int32)
        valid  = (labels == LABEL_NORMAL) | (labels == LABEL_GHOST)
        pts, labels = pts[valid], labels[valid]
        train_labels = np.where(labels == LABEL_GHOST, 1, 0).astype(np.int64)
        n = len(pts)
        if n >= self.n_points:
            if self.sampling_mode == 'priority':
                ghost_idx  = np.where(train_labels == 1)[0]
                normal_idx = np.where(train_labels == 0)[0]
                n_ghost  = min(len(ghost_idx), self.n_points // 2)
                n_normal = self.n_points - n_ghost
                if len(ghost_idx) > 0 and n_ghost > 0:
                    gi = np.random.choice(ghost_idx,  n_ghost,
                                          replace=len(ghost_idx) < n_ghost)
                    ni = np.random.choice(normal_idx, n_normal,
                                          replace=len(normal_idx) < n_normal)
                    idx_sel = np.concatenate([gi, ni])
                else:
                    idx_sel = np.random.choice(n, self.n_points, replace=False)
            else:
                idx_sel = np.random.choice(n, self.n_points, replace=False)
        else:
            idx_sel = np.random.choice(n, self.n_points, replace=True)
        pts, train_labels = pts[idx_sel], train_labels[idx_sel]
        if self.augment:
            angle = np.random.uniform(0, 2 * np.pi)
            c, s  = np.cos(angle), np.sin(angle)
            R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
            pts[:, :3] = pts[:, :3] @ R.T
            pts[:, :3] *= np.random.uniform(0.95, 1.05)
            pts[:, :3] += np.random.randn(*pts[:, :3].shape).astype(np.float32) * 0.01
        xyz = pts[:, :3]
        i   = pts[:, 3:4]
        r   = np.linalg.norm(xyz, axis=1, keepdims=True)
        xyz_norm = xyz / (r + 1e-8)
        feat = np.concatenate([xyz, xyz_norm, i, r], axis=1)  # 8-dim
        return (torch.from_numpy(feat).float(),
                torch.from_numpy(xyz).float(),
                torch.from_numpy(train_labels).long())


# ── PointNet2 components ──────────────────────────────────
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def farthest_point_sample(xyz, n_samples):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    distance  = torch.ones(B, N, device=device) * 1e10
    farthest  = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = xyz[torch.arange(B, device=device), farthest].view(B, 1, 3)
        dist     = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def ball_query(radius, n_sample, xyz, new_xyz):
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    device  = xyz.device
    sqrdists = square_distance(new_xyz, xyz)
    idx = torch.arange(N, device=device).view(1, 1, N).repeat(B, S, 1)
    idx[sqrdists > radius ** 2] = N
    idx = idx.sort(dim=-1)[0][:, :, :n_sample]
    first = idx[:, :, 0:1].repeat(1, 1, n_sample)
    idx[idx == N] = first[idx == N]
    return idx

def index_points(points, idx):
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(
        points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_point, radius, n_sample, in_channel, mlp):
        super().__init__()
        self.n_point  = n_point
        self.radius   = radius
        self.n_sample = n_sample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        fps_idx  = farthest_point_sample(xyz, self.n_point)
        new_xyz  = index_points(xyz, fps_idx)
        idx      = ball_query(self.radius, self.n_sample, xyz, new_xyz)
        grouped_xyz  = index_points(xyz, idx) - new_xyz.unsqueeze(2)
        grouped_pts  = (torch.cat([grouped_xyz, index_points(points, idx)], dim=-1)
                        if points is not None else grouped_xyz)
        grouped_pts  = grouped_pts.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_pts = F.relu(bn(conv(grouped_pts)))
        return new_xyz, torch.max(grouped_pts, 2)[0].permute(0, 2, 1)

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2).sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            weight = (1.0 / (dists + 1e-8))
            weight = weight / torch.sum(weight, dim=2, keepdim=True)
            interpolated = torch.sum(
                index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
        new_points = (torch.cat([points1, interpolated], dim=-1)
                      if points1 is not None else interpolated)
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)

class PointNet2GhostDetector(nn.Module):
    """
    PointNet2 Ghost Point Detector.
    Input:  (B, N, 8) point features
    Output: (B, N, 2) per-point logits (normal vs ghost)
    """
    def __init__(self, in_dim=8):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.5, 32, in_dim-3, [32,32,64])
        self.sa2 = PointNetSetAbstraction(256,  1.0, 32, 64,       [64,64,128])
        self.sa3 = PointNetSetAbstraction(64,   2.0, 32, 128,      [128,128,256])
        self.sa4 = PointNetSetAbstraction(16,   4.0, 32, 256,      [256,256,512])
        self.fp4 = PointNetFeaturePropagation(256+512, [256,256])
        self.fp3 = PointNetFeaturePropagation(128+256, [256,128])
        self.fp2 = PointNetFeaturePropagation(64+128,  [128,128])
        self.fp1 = PointNetFeaturePropagation(in_dim-3+128, [128,128,128])
        self.cls = nn.Sequential(
            nn.Conv1d(128,128,1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.5), nn.Conv1d(128,2,1))

    def forward(self, feat, xyz):
        pts = feat[:, :, 3:]
        l0_xyz, l0_pts = xyz, pts
        l1_xyz, l1_pts = self.sa1(l0_xyz, l0_pts)
        l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
        l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
        l4_xyz, l4_pts = self.sa4(l3_xyz, l3_pts)
        l3_pts = self.fp4(l3_xyz, l4_xyz, l3_pts, l4_pts)
        l2_pts = self.fp3(l2_xyz, l3_xyz, l2_pts, l3_pts)
        l1_pts = self.fp2(l1_xyz, l2_xyz, l1_pts, l2_pts)
        l0_pts = self.fp1(l0_xyz, l1_xyz, l0_pts, l1_pts)
        return self.cls(l0_pts.permute(0,2,1)).permute(0,2,1)


# ── Focal Loss ────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. ICCV 2017) for extreme class imbalance."""
    def __init__(self, alpha=0.90, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.ignore_index = ignore_index

    def forward(self, logits, targets):
        B, N, C = logits.shape
        logits  = logits.reshape(-1, C)
        targets = targets.reshape(-1)
        valid   = targets != self.ignore_index
        logits, targets = logits[valid], targets[valid]
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt      = torch.exp(-ce_loss)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()


# ── Metrics ───────────────────────────────────────────────
def evaluate(preds, labels):
    tp = np.sum((preds==1)&(labels==1)); fp = np.sum((preds==1)&(labels==0))
    fn = np.sum((preds==0)&(labels==1)); tn = np.sum((preds==0)&(labels==0))
    precision = tp/(tp+fp+1e-8); recall = tp/(tp+fn+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    return {'F1': round(float(f1),4), 'Precision': round(float(precision),4),
            'Recall': round(float(recall),4),
            'IoU_ghost': round(float(tp/(tp+fp+fn+1e-8)),4),
            'mIoU': round(float(((tp/(tp+fp+fn+1e-8))+(tn/(tn+fp+fn+1e-8)))/2),4)}


# ── Training ──────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    synth_seq_dir = Path(args.synth_dir) / 'sequences'
    data_dirs = [synth_seq_dir]
    if args.ref_dir and Path(args.ref_dir).exists():
        data_dirs.append(Path(args.ref_dir))
        print("Using mixed data: synth + 3DRef")
    else:
        print("Using synthetic data only")

    train_ds = GhostDataset(data_dirs, 'train', args.n_points, True,  args.sampling_mode)
    val_ds   = GhostDataset(data_dirs, 'val',   args.n_points, False, args.sampling_mode)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = PointNet2GhostDetector(in_dim=8).to(device)
    if args.loss_type == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(device))
    else:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0; history = []; start_epoch = 1
    if hasattr(args,'resume') and args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1; best_f1 = ckpt['metrics']['F1']
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train(); total_loss = 0
        for feat, xyz, labels in train_loader:
            feat, xyz, labels = feat.to(device), xyz.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feat, xyz)
            loss = (criterion(logits.reshape(-1, 2), labels.reshape(-1))
                    if args.loss_type in ('ce','weighted_ce') else criterion(logits, labels))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); total_loss += loss.item()
        scheduler.step(); avg_loss = total_loss / len(train_loader)

        if epoch % args.eval_interval == 0:
            model.eval(); all_preds, all_labels = [], []
            with torch.no_grad():
                for feat, xyz, labels in val_loader:
                    logits = model(feat.to(device), xyz.to(device))
                    all_preds.append(logits.argmax(-1).cpu().numpy().ravel())
                    all_labels.append(labels.numpy().ravel())
            metrics = evaluate(np.concatenate(all_preds), np.concatenate(all_labels))
            metrics.update({'epoch': epoch, 'loss': round(avg_loss,4)})
            history.append(metrics)
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"F1: {metrics['F1']:.4f} | Recall: {metrics['Recall']:.4f} | "
                  f"Precision: {metrics['Precision']:.4f} | mIoU: {metrics['mIoU']:.4f}")
            ckpt_dict = {'epoch': epoch, 'model_state': model.state_dict(),
                         'optimizer_state': optimizer.state_dict(), 'metrics': metrics}
            torch.save(ckpt_dict, out_dir/'latest_checkpoint.pth')
            if metrics['F1'] > best_f1:
                best_f1 = metrics['F1']
                torch.save(ckpt_dict, out_dir/'best_model.pth')
                print(f"  ★ Saved best model (F1={best_f1:.4f})")
            del ckpt_dict
        else:
            print(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f}")

    with open(out_dir/'history.json','w') as f: json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    ckpt = torch.load(out_dir/'best_model.pth'); best_metrics = ckpt['metrics']
    print("="*60)
    print(f"{'Method':<25} {'F1':>8} {'Recall':>8} {'Precision':>10} {'mIoU':>8}")
    print("-"*60)
    print(f"{'PointNet2 (Ours)':<25} {best_metrics['F1']:>8.4f} "
          f"{best_metrics['Recall']:>8.4f} {best_metrics['Precision']:>10.4f} "
          f"{best_metrics['mIoU']:>8.4f}")
    print("="*60)
    with open(out_dir/'final_results.json','w') as f:
        json.dump({'pointnet2': best_metrics}, f, indent=2)
    print(f"Results saved to: {out_dir/'final_results.json'}")

def main():
    p = argparse.ArgumentParser(description='Indoor LiDAR Ghost Point Detection')
    p.add_argument('--synth_dir',      type=str,   required=True,
                   help='Path to synth_C_v5 dataset root (containing sequences/)')
    p.add_argument('--ref_dir',        type=str,   default=None,
                   help='Path to 3DRef sequences/ directory (optional, for mixed training)')
    p.add_argument('--out_dir',        type=str,   required=True,
                   help='Output directory for checkpoints and results')
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch_size',     type=int,   default=8)
    p.add_argument('--n_points',       type=int,   default=8192)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--eval_interval',  type=int,   default=5)
    p.add_argument('--seed',           type=int,   default=2026)
    p.add_argument('--sampling_mode',  type=str,   default='priority',
                   choices=['priority','random'])
    p.add_argument('--loss_type',      type=str,   default='focal',
                   choices=['ce','weighted_ce','focal'])
    p.add_argument('--focal_alpha',    type=float, default=0.90)
    p.add_argument('--focal_gamma',    type=float, default=2.0)
    p.add_argument('--resume',         type=str,   default=None)
    p.add_argument('--skip_baseline',  action='store_true')
    train(p.parse_args())

if __name__ == '__main__':
    main()
