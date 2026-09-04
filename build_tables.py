#!/usr/bin/env python3
"""
build_tables.py — Single authoritative table generator for the paper
=====================================================================
Design rules:
  1. Every group is filtered by experiment_id prefix + split + seed explicitly.
  2. Focal-priority val rows: ONLY 'e0_target_only_seed*' with split='val'.
     Never mixed with frozen target-test rows (split='test').
  3. All tables are written to --out as CSV files, not just printed.
  4. A build_tables.log is also written for audit.

Usage:
    python build_tables.py \
        --master results_master_formal.csv \
        --out    tables/

Output files:
    tables/table_e0_val.csv       Table 1:  E0 val (checkpoint selection)
    tables/table_e0_test.csv      Table 1b: E0 unified frozen target-test
    tables/table_e0_test_full.csv Table 1c: Focal-priority frozen metrics
    tables/table_e2.csv           Table 3b: E2 n=3 complete metrics
    tables/table_e3.csv           Table 4:  E3 FOV mIoU
    tables/table_e3_full.csv      Table 4b: E3 FOV complete metrics
    tables/table_e4_miou.csv      Table 5:  E4 mIoU
    tables/table_e4_full.csv      Table 5b: E4 complete metrics (mean only)
    tables/table_e6.csv           Table 6:  E6 intensity calibration mIoU
    tables/table_e6_full.csv      Table 6b: E6 complete metrics
    tables/build_tables.log       Full audit log
"""
import csv, argparse, sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


def load_formal(path):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    log(f"Loaded {len(rows)} rows from {path}")
    return rows


def get_group(rows, prefix, split, max_seeds=3):
    """
    Filter rows by experiment_id prefix AND split field AND presence of seed.
    Returns dict: metric -> list of values (one per seed, sorted by seed).
    """
    matched = [r for r in rows
               if r['experiment_id'].startswith(prefix)
               and r.get('seed', '').strip()
               and r.get('split', '').strip() == split]
    matched.sort(key=lambda r: r['seed'])
    if not matched:
        log(f"  WARNING: no rows for prefix='{prefix}' split='{split}'")
    elif len(matched) != max_seeds:
        log(f"  WARNING: found {len(matched)} rows for prefix='{prefix}' split='{split}' (expected {max_seeds})")

    out = defaultdict(list)
    for r in matched:
        for k in ['mIoU', 'precision', 'recall', 'F1', 'IoU_ghost', 'AP']:
            if r.get(k, '').strip():
                out[k].append(float(r[k]))
    return out, matched


def mean_sd(vals):
    if not vals:
        return None, None
    return round(float(np.mean(vals)), 4), round(float(np.std(vals, ddof=1)), 4)


def fmt_mean_sd(vals):
    m, s = mean_sd(vals)
    if m is None:
        return 'N/A'
    return f"{m}±{s}"


def write_csv(path, fieldnames, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    log(f"  Written: {path}")


def build_e0(rows, out_dir):
    log("\n" + "="*60)
    log("E0: THREE-STRATEGY BASELINE")

    strategies = [
        ('Target-only',  'e0_target_only_seed'),
        ('Source-only',  'e0_source_only_seed'),
        ('Direct-mixed', 'e0_direct_mixed_seed'),
    ]

    # Table 1: val mIoU (checkpoint selection only)
    t1_rows = []
    for label, prefix in strategies:
        g, _ = get_group(rows, prefix, 'val')
        m, s = mean_sd(g['mIoU'])
        t1_rows.append({'Strategy': label, 'mIoU_Mean': m, 'mIoU_SD': s,
                        'note': 'val; own domain; NOT for cross-strategy comparison'})
        log(f"  {label} val mIoU: {m}±{s}")
    write_csv(out_dir / 'table_e0_val.csv',
              ['Strategy', 'mIoU_Mean', 'mIoU_SD', 'note'], t1_rows)

    # Table 1b: frozen target-test mIoU (primary comparison)
    t1b_rows = []
    for label, prefix in strategies:
        g, _ = get_group(rows, prefix, 'test')
        m, s = mean_sd(g['mIoU'])
        t1b_rows.append({'Strategy': label, 'mIoU_Mean': m, 'mIoU_SD': s,
                         'note': 'frozen target-test seq09; each seed evaluated once'})
        log(f"  {label} test mIoU: {m}±{s}")
    write_csv(out_dir / 'table_e0_test.csv',
              ['Strategy', 'mIoU_Mean', 'mIoU_SD', 'note'], t1b_rows)

    # Table 1c: Focal-priority frozen complete metrics
    g, matched = get_group(rows, 'e0_target_only_seed', 'test')
    t1c_rows = []
    for metric in ['mIoU', 'precision', 'recall', 'F1', 'IoU_ghost', 'AP']:
        m, s = mean_sd(g[metric])
        t1c_rows.append({'Metric': metric, 'Mean': m, 'SD': s})
        log(f"  Focal-priority test {metric}: {m}±{s}")
    write_csv(out_dir / 'table_e0_test_full.csv',
              ['Metric', 'Mean', 'SD'], t1c_rows)


def build_e2(rows, out_dir):
    log("\n" + "="*60)
    log("E2: PROPORTION-ORIENTED REBALANCING (n=3)")

    seeds = ['2026', '2027', '2028']
    metrics = ['precision', 'recall', 'F1', 'IoU_ghost', 'AP']
    all_vals = defaultdict(list)
    per_seed_rows = []

    for seed in seeds:
        matched = [r for r in rows if r['experiment_id'] == f'e2_ratio_matched_seed{seed}']
        if not matched:
            log(f"  WARNING: e2_ratio_matched_seed{seed} not found")
            continue
        r = matched[0]
        row = {'Seed': seed}
        for k in metrics:
            v = r.get(k, '').strip()
            row[k] = v
            if v:
                all_vals[k].append(float(v))
        per_seed_rows.append(row)
        log(f"  seed{seed}: " + " ".join(f"{k}={r.get(k,'')}" for k in metrics))

    # Mean±SD row
    mean_row = {'Seed': 'Mean±SD'}
    for k in metrics:
        m, s = mean_sd(all_vals[k])
        mean_row[k] = f"{m}±{s}" if m is not None else 'N/A'
    per_seed_rows.append(mean_row)
    log(f"  Mean±SD: " + " ".join(f"{k}={mean_row[k]}" for k in metrics))

    write_csv(out_dir / 'table_e2.csv', ['Seed'] + metrics, per_seed_rows)


def build_e3(rows, out_dir):
    log("\n" + "="*60)
    log("E3: COMMON-FOV (val, n=3)")

    strategies = [('Target-only', 'e3_target_fovcommon_seed'),
                  ('Direct-mixed', 'e3_mixed_fovcommon_seed')]

    t4_rows = []
    for label, prefix in strategies:
        g, _ = get_group(rows, prefix, 'val')
        m, s = mean_sd(g['mIoU'])
        t4_rows.append({'Strategy': label, 'mIoU_Mean': m, 'mIoU_SD': s})
        log(f"  {label} mIoU: {m}±{s}")
    write_csv(out_dir / 'table_e3.csv', ['Strategy', 'mIoU_Mean', 'mIoU_SD'], t4_rows)

    t4b_rows = []
    for label, prefix in strategies:
        g, _ = get_group(rows, prefix, 'val')
        row = {'Strategy': label}
        for k in ['precision', 'recall', 'F1', 'IoU_ghost', 'AP']:
            m, _ = mean_sd(g[k])
            row[k] = m if m is not None else 'N/A'
        t4b_rows.append(row)
        log(f"  {label} full: " + " ".join(f"{k}={row[k]}" for k in ['precision','recall','F1','IoU_ghost','AP']))
    write_csv(out_dir / 'table_e3_full.csv',
              ['Strategy', 'precision', 'recall', 'F1', 'IoU_ghost', 'AP'], t4b_rows)


def build_e4(rows, out_dir):
    log("\n" + "="*60)
    log("E4: 3x2 ABLATION (val, n=3)")
    log("NOTE: Focal-priority uses e0_target_only_seed* with split=val ONLY.")

    groups = [
        ('CE-p',      'e4_ce_priority_seed',   'val'),
        ('WCE-p',     'e4_wce_priority_seed',  'val'),
        ('Focal-p*',  'e0_target_only_seed',   'val'),   # original setting; val only
        ('CE-r',      'e4_ce_random_seed',     'val'),
        ('WCE-r',     'e4_wce_random_seed',    'val'),
        ('Focal-r',   'e4_focal_random_seed',  'val'),
    ]

    t5_rows = []
    for label, prefix, split in groups:
        g, matched = get_group(rows, prefix, split)
        m, s = mean_sd(g['mIoU'])
        t5_rows.append({'Setting': label, 'mIoU_Mean': m, 'mIoU_SD': s, 'n_seeds': len(matched)})
        log(f"  {label} mIoU: {m}±{s} (n={len(matched)})")
    write_csv(out_dir / 'table_e4_miou.csv',
              ['Setting', 'mIoU_Mean', 'mIoU_SD', 'n_seeds'], t5_rows)

    t5b_rows = []
    for label, prefix, split in groups:
        g, _ = get_group(rows, prefix, split)
        row = {'Setting': label}
        for k in ['precision', 'recall', 'F1', 'IoU_ghost', 'AP']:
            m, _ = mean_sd(g[k])
            row[k] = m if m is not None else 'N/A'
        t5b_rows.append(row)
        log(f"  {label} full (mean): " + " ".join(f"{k}={row[k]}" for k in ['precision','recall','F1','IoU_ghost','AP']))
    write_csv(out_dir / 'table_e4_full.csv',
              ['Setting', 'precision', 'recall', 'F1', 'IoU_ghost', 'AP'], t5b_rows)


def build_e6(rows, out_dir):
    log("\n" + "="*60)
    log("E6: INTENSITY CALIBRATION (val, n=3)")

    methods = [
        ('uncalibrated', 'e0_direct_mixed_seed',          'val'),
        ('percentile',   'e6_intensity_percentile_seed',  'val'),
        ('z-score',      'e6_intensity_zscore_seed',      'val'),
        ('histogram',    'e6_intensity_histmatch_seed',   'val'),
    ]

    t6_rows = []
    for label, prefix, split in methods:
        g, matched = get_group(rows, prefix, split)
        m, s = mean_sd(g['mIoU'])
        t6_rows.append({'Method': label, 'mIoU_Mean': m, 'mIoU_SD': s, 'n_seeds': len(matched)})
        log(f"  {label} mIoU: {m}±{s} (n={len(matched)})")
    write_csv(out_dir / 'table_e6.csv',
              ['Method', 'mIoU_Mean', 'mIoU_SD', 'n_seeds'], t6_rows)

    t6b_rows = []
    for label, prefix, split in methods[1:]:   # exclude uncalibrated
        g, _ = get_group(rows, prefix, split)
        row = {'Method': label}
        for k in ['precision', 'recall', 'F1', 'IoU_ghost', 'AP']:
            m, _ = mean_sd(g[k])
            row[k] = m if m is not None else 'N/A'
        t6b_rows.append(row)
        log(f"  {label} full (mean): " + " ".join(f"{k}={row[k]}" for k in ['precision','recall','F1','IoU_ghost','AP']))
    write_csv(out_dir / 'table_e6_full.csv',
              ['Method', 'precision', 'recall', 'F1', 'IoU_ghost', 'AP'], t6b_rows)


def main():
    p = argparse.ArgumentParser(description='Generate all paper tables from formal master CSV')
    p.add_argument('--master', required=True, help='Path to results_master_formal.csv')
    p.add_argument('--out', default='tables/', help='Output directory for CSV tables and log')
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"build_tables.py run at {datetime.now().isoformat()}")
    log(f"Master: {args.master} | Output: {args.out}")

    rows = load_formal(args.master)
    build_e0(rows, out_dir)
    build_e2(rows, out_dir)
    build_e3(rows, out_dir)
    build_e4(rows, out_dir)
    build_e6(rows, out_dir)

    # Write log
    log_path = out_dir / 'build_tables.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG_LINES))
    log(f"\nLog written to {log_path}")
    log("Done. Verify every paper number against the CSV outputs before submission.")


if __name__ == '__main__':
    main()
