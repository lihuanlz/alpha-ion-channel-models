# 第三方复现指南 / Reproduction Guide

对应论文：*Gating time constants are the conserved coordinates of cardiac ion
channels: a unified reanalysis of seven voltage-clamp corpora*（latex_v2_unified/）。

## 一、从存缴表格复现（5 分钟，无需原始波形）

Zenodo 存缴包解压后，只需三个文件即可核对论文全部头条数字：

| 文件 | 内容 |
|---|---|
| `master_unified_v1.csv` | 统一总表，487,940 行 × 22 列（含 356 行文献锚） |
| `cipa2_cells.csv` | CiPA 细胞级 τ 比（ss 最高浓度块口径） |
| `robust_verdict_all_profile_2026-11-08.csv` | 四口径判决冻结表（212 行） |

```bash
python reproduce_check.py --root <解压目录>
```

输出 37 项 PASS/FAIL：主表规模与分源行数、CI 校准率（96.2%，n=255,411）、
五张判决表的组数/通过数/中位数（63/64 @ 2.47 等）、CiPA τ 比
（n=970，中位 0.948，分室 1.05/0.99/0.81/0.92/1.17，盲法 0.87）、
Beattie 锚点（−8.97 mV）。本次归档运行记录：**37/37 ALL PASS（2026-11-12）**。

论文六张主图由 `10_论文图与表/make_fig1_fig2.py`、`make_fig3_fig4.py`、
`make_fig5_fig6.py` 从同一总表重新生成；SI 判决长表由 `_make_si_table.py`
生成。这些脚本全部用 `Path(__file__)` 相对定位，无绝对路径。

## 二、从原始波形重提取（重活，可选）

各语料原始数据全部公开，重提取脚本在 `06_统一重算pipeline/batch*_*剖面升级.py`。
这些脚本顶部有一个 `DATA`/`LAW` 绝对路径变量（我们本机的数据根），
**第三方只需把这一行改成自己的数据目录**，其余逻辑不动。

| 语料 | 公开出处 | 重提取脚本 |
|---|---|---|
| Channelpedia (17,987 细胞, NWB) | channelpedia.epfl.ch | batch1（走 03 的 extract_features_v5 链） |
| CiPA 五实验室 hERG (62 GB) | OSF osf.io/a6k5t | batch2_cipa_剖面升级_v2.py |
| Lei 五温度 hERG | Biophys J 2019 配套 | batch3_lei_剖面升级.py |
| Beattie 九细胞 | figshare doi:10.6084/m9.figshare.4702546 | batch4_beattie9_剖面升级.py |
| Nav1.5 双温度 | figshare doi:10.6084/m9.figshare.27193878 | batch5_nav15_剖面升级.py |
| CaV1.2 药理 | OSF g3msb | batch5b_cav_剖面升级.py |
| IKs 激活系列 | Zenodo 8226585 | batch5c_iks_剖面升级.py |
| 文献锚三层 | CiPA 官方表 / 跨物种 / ICGenealogy | batch6_文献锚入表.py |

装表与校验：`build_master.py`（幂等）→ `qa_统一总表对拍.py`。

## 三、依赖

Python ≥ 3.10；pandas、numpy、scipy、matplotlib。
仅 Channelpedia 原始阶段需要 pynwb；仅编译论文需要 latexmk + pdflatex。
无 GPU、无网络服务、无付费组件依赖。

## 四、原始数据的格式/单位坑（已在代码中硬化处理）

全部登记在 `law_channelpedia/方法学存档_CiPA全药池pipeline冻结_2026-11-10.md`
（条目 1.1–1.22），代码侧对应处理：

1. **电流单位后缀不统一**：lab_2 是安培（`_current_A`），其余是 pA。
   batch2/stage 系列全部带单位映射 `{"pa":1, "na":1e3, "a":1e12, ...}`，
   未知单位直接报错拒绝（不静默猜）。
2. **列污染**：三个实验室存在电压串入电流列，由损坏列筛查剔除。
3. **模板电压 ≠ 实测电压**：一律用实测电压列（模板重建曾致 τ 偏差
   12.96×/0.14×，已登记为缺陷）。
4. **浓度单位三种写法并存**：逐目录归一化。
5. **官方汇总表两例数值灾难**（Hill 系数 299.86 / 101.55）：
   官方表级数值一律经独立重拟合复核后才使用。
6. **拟合撞界不删除**：`tau_edge` 旗标保留（Ba²⁺ 携带的 CaV1.2 失活等），
   统计时按旗标处理。
7. **重复记录**：同细胞多次记录先并优再拟合（IKs）。

## 五、已知限制（如实登记）

- QA 快查脚本里的 CI 覆盖率（95.6%）是全行口径；论文口径是
  τ 类且 qc ok 且有剖面区间（96.2%，n=255,411）。两者分母不同，都对。
- CiPA τ 比有两个口径：全 sweep 0.935 / ss 最高浓度块 0.948（论文用后者）。
- Lei 27°C 批次噪声使 τ_deact_slow 的表观 Q10 失真（6.13），
  论文温度律用 τ_rec（Q10 2.91）。
- `fig1` 脚本注释"与 QA 同口径"表述不准（分母不同），仅此注释问题。
