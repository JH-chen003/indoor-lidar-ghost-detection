#!/usr/bin/env python3
"""
build_tables.py — Generate all paper tables from results_master_formal.csv
===========================================================================
Key design principles:
  1. Only one authoritative source: results_master_formal.csv
  2. Each group is filtered by BOTH factor label AND split ('val' or 'test')
     to prevent mixing validation and frozen-target-test rows.
  3. Focal-priority val rows come from 'e0_target_only_seed*' (split=val),
     NOT from the FROZEN_TARGET_TEST rows (split=test).
  4. Outputs Markdown tables that can be copy-pasted directly into the paper.

Usage:
    python build_tables.py --master results_master_formal.csv --out tables/
"""
import csv, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_formal(path):
    return list(csv.DictReader(open(path)))


def mean_sd(vals):
    a = [float(v) for v in vals if v and str(v).strip()]
    if not a:
        return None, None
    return round(np.mean(a), 4), round(np.std(a, ddof=1), 4)


def group_by_prefix_and_split(rows, prefix, split):
    """Return metric->list for rows matching prefix AND split field."""
    matched = [r for r in rows
               if r['experiment_id'].startswith(prefix)
               and r.get('seed')
               and r.get('split', '').strip() == split]
    out = defaultdict(list)
    for r in matched:
        for k in ['mIoU', 'precision', 'recall', 'F1', 'IoU_ghost', 'AP']:
            if r.get(k) and str(r[k]).strip():
                out[k].append(r[k])
    return out


def fmt(m, s=None):
    if m is None:
        return 'N/A'
    if s is None:
        return str(m)
    return f"{m}±{s}"


def print_md_table(title, headers, rows_data):
    print(f"\n**{title}**\n")
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(['---'] * len(headers)) + '|')
    for row in rows_data:
        print('| ' + ' | '.join(str(c) for c in row) + ' |')


def build_e0(rows, out_dir):
    """E0: val split for checkpoint selection; test split for frozen target-test."""
    print("\n" + "="*60)
    print("E0 TABLES")

    # Table 1: val (for checkpoint selection only)
    t1_rows = []
    for label, prefix in [('Target-only','e0_target_only_seed'),
                           ('Source-only','e0_source_only_seed'),
                           ('Direct-mixed','e0_direct_mixed_seed')]:
        g = group_by_prefix_and_split(rows, prefix, 'val')
        m, s = mean_sd(g['mIoU'])
        t1_rows.append([label, fmt(m, s)])
    print_md_table("Table 1: Val mIoU (for checkpoint selection only; NOT for cross-strategy comparison)",
                   ['Strategy','mIoU (val, own domain)'], t1_rows)

    # Table 1b: unified frozen target-test
    t1b_rows = []
    for label, prefix in [('Target-only','e0_target_only_seed'),
                           ('Source-only','e0_source_only_seed'),
                           ('Direct-mixed','e0_direct_mixed_seed')]:
        g = group_by_prefix_and_split(rows, prefix, 'test')
        m, s = mean_sd(g['mIoU'])
        t1b_rows.append([label, fmt(m, s)])
    print_md_table("Table 1b: Unified frozen target-test mIoU (primary comparison)",
                   ['Strategy','mIoU (frozen target-test, n=3)'], t1b_rows)

    # Table 1c: Focal-priority frozen target-test complete metrics
    g = group_by_prefix_and_split(rows, 'e0_target_only_seed', 'test')
    metrics_row = ['Target-only']
    for k in ['precision','recall','F1','IoU_ghost','AP']:
        m, s = mean_sd(g[k])
        metrics_row.append(fmt(m, s))
    print_md_table("Table 1c: Focal-priority frozen target-test complete metrics (n=3)",
                   ['Strategy','Precision','Recall','F1','IoU_ghost','AP'], [metrics_row])


def build_e2(rows, out_dir):
    print("\n" + "="*60)
    print("E2 TABLES")
    seeds = ['2026','2027','2028']

    # Per-seed table
    per_seed = []
    all_vals = defaultdict(list)
    for seed in seeds:
        r_list = [r for r in rows if r['experiment_id'] == f'e2_ratio_matched_seed{seed}']
        if not r_list:
            print(f"  WARNING: e2_ratio_matched_seed{seed} not found")
            continue
        r = r_list[0]
        row = [seed]
        for k in ['precision','recall','F1','IoU_ghost','AP']:
            v = r.get(k,'')
            row.append(v)
            if v: all_vals[k].append(v)
        per_seed.append(row)

    mean_row = ['Mean±SD']
    for k in ['precision','recall','F1','IoU_ghost','AP']:
        m, s = mean_sd(all_vals[k])
        mean_row.append(fmt(m, s))
    per_seed.append(mean_row)

    print_md_table("Table 3b: E2 complete metrics per seed + Mean±SD (val, n=3)",
                   ['Seed','Precision','Recall','F1','IoU_ghost','AP'], per_seed)


def build_e3(rows, out_dir):
    print("\n" + "="*60)
    print("E3 TABLES")

    t4_rows = []
    for label, prefix in [('Target-only','e3_target_fovcommon_seed'),
                           ('Direct-mixed','e3_mixed_fovcommon_seed')]:
        g = group_by_prefix_and_split(rows, prefix, 'val')
        m, s = mean_sd(g['mIoU'])
        t4_rows.append([label, fmt(m, s)])
    print_md_table("Table 4: E3 FOV mIoU (val, n=3)",
                   ['Strategy','mIoU (Mean±SD)'], t4_rows)

    t4b_rows = []
    for label, prefix in [('Target-only','e3_target_fovcommon_seed'),
                           ('Direct-mixed','e3_mixed_fovcommon_seed')]:
        g = group_by_prefix_and_split(rows, prefix, 'val')
        row = [label]
        for k in ['precision','recall','F1','IoU_ghost','AP']:
            m, _ = mean_sd(g[k])
            row.append(str(m) if m else 'N/A')
        t4b_rows.append(row)
    print_md_table("Table 4b: E3 FOV complete metrics (val, Mean, n=3)",
                   ['Strategy','Precision','Recall','F1','IoU_ghost','AP'], t4b_rows)


def build_e4(rows, out_dir):
    print("\n" + "="*60)
    print("E4 TABLES")

    # Focal-priority comes from e0_target_only (val split), NOT target-test
    groups = [
        ('CE-p',    'e4_ce_priority_seed',  'val'),
        ('WCE-p',   'e4_wce_priority_seed', 'val'),
        ('Focal-p*','e0_target_only_seed',  'val'),   # val split only
        ('CE-r',    'e4_ce_random_seed',    'val'),
        ('WCE-r',   'e4_wce_random_seed',   'val'),
        ('Focal-r', 'e4_focal_random_seed', 'val'),
    ]

    t5_rows = []
    for label, prefix, split in groups:
        g = group_by_prefix_and_split(rows, prefix, split)
        m, s = mean_sd(g['mIoU'])
        t5_rows.append([label, fmt(m, s)])
    print_md_table("Table 5: E4 mIoU (val, n=3; *original setting = frozen model)",
                   ['Setting','mIoU (Mean±SD, n=3)'], t5_rows)

    t5b_rows = []
    for label, prefix, split in groups:
        g = group_by_prefix_and_split(rows, prefix, split)
        row = [label]
        for k in ['precision','recall','F1','IoU_ghost','AP']:
            m, _ = mean_sd(g[k])
            row.append(str(m) if m else 'N/A')
        t5b_rows.append(row)
    print_md_table("Table 5b: E4 complete metrics (val, Mean only, n=3)",
                   ['Setting','Precision','Recall','F1','IoU_ghost','AP'], t5b_rows)


def build_e6(rows, out_dir):
    print("\n" + "="*60)
    print("E6 TABLES")

    methods = [
        ('uncalibrated', 'e0_direct_mixed_seed'),
        ('percentile',   'e6_intensity_percentile_seed'),
        ('z-score',      'e6_intensity_zscore_seed'),
        ('histogram',    'e6_intensity_histmatch_seed'),
    ]

    t6_rows = []
    for label, prefix in methods:
        g = group_by_prefix_and_split(rows, prefix, 'val')
        m, s = mean_sd(g['mIoU'])
        t6_rows.append([label, fmt(m, s)])
    print_md_table("Table 6: E6 intensity calibration mIoU (val, n=3)",
                   ['Method','mIoU (Mean±SD)'], t6_rows)

    t6b_rows = []
    for label, prefix in methods[1:]:  # exclude uncalibrated
        g = group_by_prefix_and_split(rows, prefix, 'val')
        row = [label]
        for k in ['precision','recall','F1','IoU_ghost','AP']:
            m, _ = mean_sd(g[k])
            row.append(str(m) if m else 'N/A')
        t6b_rows.append(row)
    print_md_table("Table 6b: E6 complete metrics (val, Mean, n=3)",
                   ['Method','Precision','Recall','F1','IoU_ghost','AP'], t6b_rows)


def main():
    p = argparse.ArgumentParser(description='Generate paper tables from formal master CSV')
    p.add_argument('--master', required=True, help='Path to results_master_formal.csv')
    p.add_argument('--out', default='tables/', help='Output directory')
    args = p.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    rows = load_formal(args.master)
    print(f"Loaded {len(rows)} formal rows from {args.master}")

    build_e0(rows, args.out)
    build_e2(rows, args.out)
    build_e3(rows, args.out)
    build_e4(rows, args.out)
    build_e6(rows, args.out)

    print("\n" + "="*60)
    print("All tables generated. Verify each number matches the paper before submission.")


if __name__ == '__main__':
    main()
