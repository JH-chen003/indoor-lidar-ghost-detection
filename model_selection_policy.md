# Model Selection Policy

**制定日期**: 2026-08-29（初版）／2026-09-01（v16複核，內容未變更）
**作者**: Anonymous Author
**狀態**: 事後制定（見下方誠實聲明）— 論文6.3節引用之補充材料S3

## 選模規則（三階段篩選）

### 第一階段：Recall門檻（安全性優先）
Ghost Point漏偵對下游應用（機器人避障、SLAM）風險遠高於誤報，故設定：
**Recall ≥ 0.97** 為入圍門檻，未達此門檻者直接淘汰，不進入後續排序。

### 第二階段：AP排序
通過第一階段門檻的候選者，依 **Average Precision (AP)** 由高到低排序，
AP同時反映模型在各門檻下的Precision-Recall權衡，較單一F1更全面。

### 第三階段：同分比較
若AP差距小於0.005（視為統計上不可區分），則依序比較：
1. F1（越高越好）
2. Precision（越高越好，避免過度誤報）

## 誠實聲明：時序限制

**本規則制定於E4六組完整分類指標已產出之後**，非嚴格意義下的「盲選」（pre-registration）。
本研究並未在觀察結果前預先凍結此規則，此為方法論上的限制。

為降低選規則時「配合已知答案」的疑慮，本規則刻意採用：
- Recall優先於mIoU/F1（因mIoU/F1與CE-priority的優勢並非直接掛鉤，Recall門檻是獨立於「哪組mIoU最高」的判準）
- AP而非mIoU作為主要排序依據（mIoU是CE-priority領先的指標，改用AP可避免直接照搬mIoU排名）

後續研究應在下一輪實驗前，於資料收集前預先凍結選模規則，以符合嚴格的pre-registration精神。

## 套用本規則於E4六組結果

| 組別 | Recall | 是否通過門檻(≥0.97) | AP |
|---|---|---|---|
| CE-priority (3 seeds平均) | 0.990 | 通過 | 0.9973 |
| Weighted CE-priority (3 seeds平均) | 0.997 | 通過 | 0.9950 |
| Focal-priority (原設定, 3 seeds平均) | 0.995 | 通過 | 0.9903 |
| CE-random (3 seeds平均) | 0.979 | 通過 | 0.9907 |
| Weighted CE-random (3 seeds平均) | 0.994 | 通過 | 0.9908 |
| Focal-random (3 seeds平均) | 0.992 | 通過 | 0.9868 |

**結果**：由於priority系列的Recall普遍已達0.97以上，本規則之Recall門檻未能有效篩選；
AP排序下CE-priority系列（AP約0.997）仍為最高，與單看mIoU的結論一致。

**誠實檢討**：本規則雖立意降低「配合答案」之疑慮，但套用結果顯示AP排序仍指向CE-priority，
與mIoU排序結論相同。這代表兩種可能：(1) CE-priority確實穩健地優於其他組別，跨多個指標一致；
或(2) 本規則設計仍不足以脫離mIoU導向的影響。本研究採取保守立場，暫不宣稱CE-priority為
最終確定之最佳模型，維持Focal-priority為主設定，待下一輪嚴格pre-registration實驗驗證。

## 最終決策（正式凍結）

**正式模型：Focal-priority**（原始方法設定）

- 已完成frozen target-test評估：F1=0.9805±0.0005（見frozen_manifest.md）
- 執行後不再回頭調整任何模型選擇決定
- CE-priority之驗證集優勢列為下一輪確認性研究（confirmatory study）之優先假說，非本文正式模型

## 適用範圍聲明

本policy僅適用於E4六組（Loss×Sampling）之選模決策，不適用於E0/E2/E3/E6之因子層級推論；
E0/E2/E3/E6之結論邊界另見論文6.1、6.2節之研究限制說明。
