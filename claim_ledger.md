# Claim Ledger (v16 — Final)

**更新日期**: 2026-09-03
**格式**: claim_id | 論文位置 | 使用數字 | 對應母表列 | 證據強度 | 限制語 | 動作

| claim_id | 論文位置 | 使用數字 | 對應experiment_id | 證據強度 | 限制語 | 動作 |
|---|---|---|---|---|---|---|
| C1-01 | 摘要/C1貢獻/表一之三 | mIoU=0.962±0.001 | e0_target_only_seed{2026,2027,2028}_FROZEN_TARGET_TEST | 強（n=3，frozen，獨立test） | 僅限既定3DRef target domain | 保留 |
| C1-02 | 摘要/C1貢獻 | F1=0.9805±0.0005, AP=0.9927±0.0009 | 同上 | 強 | 同上 | 保留 |
| C2-01 | 摘要/C2貢獻/表一之二 | Direct-mixed mIoU=0.909±0.056 | e0_direct_mixed_seed{2026,2027,2028}_TARGET_TEST | 強（n=3，target test） | 不解釋為必然負遷移 | 保留 |
| C2-02 | 表一之四 | 三策略完整分類指標（P/R/F1/IoU_g/AP） | e0_source_only/direct_mixed_seed{2026,2027,2028}_TARGET_TEST | 強（純推論無重訓） | Source-only高變異Precision已於note說明 | 保留 |
| C3-01 | 摘要/5.5節/表三 | E2配平後val mIoU=0.7820±0.0873 | e2_ratio_matched_seed{2026,2027,2028} | 中（n=3但變異大） | 僅為觀察性關聯，非單一因果 | 保留 |
| C3-01b | 表三之二 | Mean±SD: P=0.8442±0.1342, R=0.9023±0.0745, F1=0.8636±0.0452, IoU_g=0.7618±0.0685, AP=0.9586±0.0027 | e2_ratio_matched_seed{2026,2027,2028} | 中（AP穩定，其餘變異大） | 明確標示複合介入，非單一機制 | 保留 |
| C3-02 | 5.6節/表四 | E3-FOV落差=0.2002 | e3_target_fovcommon/e3_mixed_fovcommon各3seed | 強（n=3） | FOV非唯一候選因素 | 保留 |
| C3-03 | 5.7節/表五+表五之二 | E4六格mIoU與完整指標 | e4_*_seed{2026,2027,2028} 全18組 | 強（n=3全數完成） | 事後敏感性分析；完整指標為Mean，per-seed詳見results_master_formal.csv | 保留 |
| C3-04 | 5.8節/表六 | E6三方法mIoU（0.7865~0.7969） | e6_intensity_*_seed{2026,2027,2028} | 強（n=3） | 不宣稱單一校正法全面最佳 | 保留 |
| REF-01 | 4.1.3節 | Hahner積水引用 | N/A（文獻引用） | — | 軟化為類比推論，非直接驗證 | 保留 |
| REF-02 | 2.2節 | Ghost-FWL比較 | N/A（文獻引用） | — | 誠實揭露需特殊硬體，非同條件基線 | 保留 |
| REF-03 | 5.8節 | Intensity KS=1.000 | e6原始分布統計 | 強 | 「高度分離」非「完全不重疊」，不暗示唯一因果 | 保留 |
| EXCL-01 | 6.4節Coming Soon | E3 voxel size敏感度（0.03/0.05/0.10m） | 不列入本文正式主張 | — | 未完成多seed驗證，不構成正式主張，已移至Coming Soon | 不列入主文 |

## 可追溯性聲明
本表所有claim均可在 results_master_formal.csv 找到對應的 experiment_id、seed、split 與 metrics。
5.4節PR曲線AP數字（0.992/0.994/0.992）對應 e0_target_only_seed{2026,2027,2028} 的 val AP，
來源：e0_target_only_val_full_metrics.json，已補入母表。

## 程式碼可用性
匿名 Repository（審稿期間）：https://anonymous.4open.science/r/indoor-lidar-ghost-detection-F44B/
