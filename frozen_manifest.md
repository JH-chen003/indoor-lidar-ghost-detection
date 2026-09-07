# Frozen Manifest (v16)

**建立日期**: 2026-08-31（初版）／2026-09-02（v16更新）
**作者**: Anonymous Author（submission under review）

## 程式碼與重現材料

**匿名 Repository（審稿期間使用）**：
https://anonymous.4open.science/r/indoor-lidar-ghost-detection-F44B/

包含：`ghost_detector.py`、`evaluate.py`、`requirements.txt`、`README.md`（含逐步重現指令）

## 凍結模型：Focal-priority (Target-only, 原始方法設定, 正式模型)

| 項目 | Seed 2026 | Seed 2027 | Seed 2028 |
|---|---|---|---|
| 模型名稱 | e0_target_only_seed2026 | e0_target_only_seed2027 | e0_target_only_seed2028 |
| SHA256 | 1493cf874397e51b1fdb3294d8505205ddb9ccf3c828a5abf99aa10a57233818 | b443602040c00eae40e01d7bd708bdecc48bc8e3a969322a3d0f0233c0fc200b | 4ab30f01dad59d0ffc398df18439dacce80119e22c96204dd7071eb11b9672f4 |
| Loss / Sampling | Focal (α=0.90, γ=2.0) / priority | 同左 | 同左 |

### Target-only 於獨立3DRef Target Test（seq09, 746幀，僅評估一次）

| 指標 | Seed 2026 | Seed 2027 | Seed 2028 | Mean±SD |
|---|---|---|---|---|
| mIoU | 0.9615 | 0.9631 | 0.9624 | 0.9623±0.0008 |
| Precision | 0.9633 | 0.9643 | 0.9638 | 0.9638±0.0005 |
| Recall | 0.9974 | 0.9981 | 0.9978 | 0.9978±0.0004 |
| F1 | 0.9800 | 0.9809 | 0.9805 | 0.9805±0.0005 |
| IoU_ghost | 0.9609 | 0.9625 | 0.9618 | 0.9617±0.0008 |
| AP | 0.9920 | 0.9937 | 0.9924 | 0.9927±0.0009 |

---

## 對照模型：Source-only（同一frozen manifest下評估）

| 指標 | Seed 2026 | Seed 2027 | Seed 2028 | Mean±SD |
|---|---|---|---|---|
| mIoU | 0.2587 | 0.2583 | 0.2583 | 0.2584±0.0002 |
| Precision | 0.8171 | 0.25 | 0.25 | 0.439±0.327 |
| Recall | 0.0006 | 0.0 | 0.0 | 0.0002±0.0003 |
| F1 | 0.0013 | 0.0 | 0.0 | 0.0004±0.0008 |
| AP | 0.5614 | 0.5128 | 0.5053 | 0.5265±0.0305 |

---

## 對照模型：Direct-mixed（同一frozen manifest下評估）

| 指標 | Seed 2026 | Seed 2027 | Seed 2028 | Mean±SD |
|---|---|---|---|---|
| mIoU | 0.8458 | 0.9416 | 0.9417 | 0.9097±0.0553 |
| Precision | 0.8562 | 0.9457 | 0.9438 | 0.9152±0.0511 |
| Recall | 0.9945 | 0.9949 | 0.9972 | 0.9955±0.0015 |
| F1 | 0.9202 | 0.9697 | 0.9698 | 0.9532±0.0286 |
| AP | 0.9734 | 0.9898 | 0.9885 | 0.9839±0.0091 |

---

## 不可更改的規則

以上target-test結果僅供報告，不得回饋模型、超參數、epoch或threshold選擇。
