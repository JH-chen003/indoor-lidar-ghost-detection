# Dataset Card

**建立日期**: 2026-09-01
**適用範圍**: 3DRef（target domain）與 synth_C_v5（source domain）

## 1. 資料概覽

| 項目 | 3DRef (Target Domain) | synth_C_v5 (Source Domain) |
|---|---|---|
| 場景類型 | 室內（走廊/大廳/玻璃隔間） | 戶外（KITTI底層） |
| 採集感測器 | Hesai QT64 | Velodyne HDL-64E（底層，物理合成疊加） |
| 垂直FOV | 104.2° | 26.9° |
| 標註方式 | 真人標註 | 物理模型合成（Snell反射/折射） |
| 授權 | 依3DRef原始授權取得，本卡不重新散布原始點雲 | 基於KITTI公開資料集衍生合成 |

## 2. 使用的序列與切分

| 用途 | 序列 | 幀數 |
|---|---|---|
| Train | 3DRef Seq1 (seq00, 時間區塊切分) | 2,240 |
| Val | 3DRef Seq1 (seq08) | 746 |
| Test（獨立，凍結後僅評估一次） | 3DRef Seq1 (seq09) | 746 |
| synth_C_v5 Train | — | 3,000 |
| synth_C_v5 Val | — | 600 |

**切分原則**：時間區塊切分（temporal-block split），避免相鄰幀時間相關性造成的資料洩漏。三個split互不重疊。

## 3. 標籤定義與二元化規則

3DRef原始標註含7類（Label 0~6）。本研究執行以下二元化規則：

- **Label 1（normal）→ 負類（0）**
- **Label 5（ghost/reflection）→ 正類（1）**
- **其餘類別（0/2/3/4/6，含glass/mirror/other_reflective/obstacle）於前處理階段直接排除，不參與訓練與評估**

## 4. Ghost Ratio 統計（依上述二元化規則，分母=有效點label1+5）

| 序列/資料集 | Ghost Ratio |
|---|---|
| 3DRef Seq1（訓練用） | 12.418% |
| 3DRef Seq2 | 4.300% |
| 3DRef Seq3 | 4.597% |
| synth_C_v5（train） | 0.895% |
| synth_C_v5（val） | 0.892% |

**Domain Gap（Ghost Ratio倍數）**: 12.418% / 0.895% ≈ 13.9倍

## 5. 感測器多樣性佐證（非訓練用，僅統計佐證）

| 感測器 | Seq1 | Seq2 | Seq3 | 平均 |
|---|---|---|---|---|
| Hesai QT64 | 12.418% | 4.300% | 4.597% | 6.017% |
| Ouster | 9.307% | 1.665% | 2.405% | 4.459% |
| Livox | 29.219% | 14.337% | 11.596% | 18.384% |

*註：此表分母為全部7類（含非二元化後保留的類別），與第4節數字算法不同，僅供跨感測器多樣性參考，未用於訓練。Livox數字異常偏高，推測與其非重複掃描固態LiDAR之掃描架構差異有關，不列入核心Domain Gap論述。*

## 6. 物理合成模型參數（synth_C_v5）

| 類型 | 參數 | 說明 |
|---|---|---|
| Metal Ghost（金屬鏡面反射） | 強度衰減係數 0.55~0.85 | physics-inspired工程近似，尚待實測校準 |
| Glass Ghost（玻璃折射） | n1=1.0, n2=1.5；偏移0.5~2.0m；強度衰減0.25~0.55 | 依Snell折射定律 |
| 生成機率 | 每個KITTI車輛bbox各70%機率獨立生成Metal/Glass Ghost | — |

## 7. 已知限制

1. 僅使用3DRef Seq1訓練，Seq2/Seq3/Ouster/Livox僅作統計佐證，未納入訓練
2. synth_C_v5物理參數為工程近似，非fully-physical校準
3. 積水路面Ghost未納入合成框架（見論文4.1.3節說明）
4. E2過取樣（13.44倍）後之synth_C_v5變體（ratio_matched）改變了原始資料分布，使用時須明確標註

## 8. 資料存取

本卡不重新散布3DRef原始點雲。具備3DRef原始授權之研究者可依frozen_manifest.md提供之frame manifest與前處理程式碼重現本研究之資料前處理流程。
