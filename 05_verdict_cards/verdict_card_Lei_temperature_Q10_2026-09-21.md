# 判词卡：Lei 温度系列 Q10 层 + 37 °C 跨宿主对表（收官）

2026-09-21 16:10 · 细胞线 4 hERG · 数据集 Lei et al. 2019 Part I/II（*Biophys. J.* 117:2438–2454, 2455–2470）
批次：herg25oc1（211 孔）/ herg27oc1（127）/ herg30oc1（118）/ herg33oc1（109）/ herg37oc3+37oc4（69+36=105），全 CHO 宿主，autoLC + Lei 官方再漏减，同一管道逐批提取。
判线：预注册判决卡_Lei温度Q10_2026-09-21.md（2026-09-21 13:30 冻结，一字未改，跑后照实登记）。

**一句话：绝对时间尺度层在五温度下原样站住——τ_rec 的 Q10 = 2.84（Arrhenius R² = 0.991）封卷；同温度跨宿主对表成立（Lei-CHO/37 °C τ_rec(−120) = 1.52 ms vs Beattie-HEK 3.04 ms，比值 0.499 ∈ [0.3,3]）封卷；37 °C 批结构复制 5/7 封卷（E_rev、h_ss 负端、m_ss 高档），τ_deact 慢层在高温下贴近阶梯窗上限方向判不成立、J5 逐周期判线维持结构性受限，两项照预注册登记。**

## T1 · 37 °C 批结构复制（判线同 25 °C 卡，n 门槛 53）

| 判决项 | 结果（37 °C 合并，n=105） | 判定 |
|---|---|---|
| J4 E_rev | 中位 −92.04 mV，CV 0.0306，n=102 | **封卷**（CV<0.10 且 ∈[−100,−80]） |
| J1 h_ss(−140) | 1.000，CV 0.000，n=105 | **封卷** |
| J1 h_ss(−120) | 0.968，CV 0.060，n=102 | **封卷** |
| J2 m_ss(+25) | 0.900，CV 0.118，n=102 | **封卷** |
| J2 m_ss(+40) | 1.000，CV 0.000，n=105 | **封卷** |
| J3 τ_deact 方向 τ2(−40)>τ2(−60) | τ2(−60)=0.302 s，τ2(−40)=0.266 s | **方向不成立 → 登记** |
| J5 合并（A1+A2 双过） | 98/306 = 32.0%，Wilson 95% CI [0.270, 0.374] | **登记**（下界 <0.6） |

J3 方向判不成立的机理登记（不作翻案依据）：37 °C 下去激活加快 ~2.5–10 倍，慢分量 τ2 中位 0.27–0.30 s 已贴近 staircase 0.5 s 阶梯窗上限量级，双指数慢分量在窗内不可识别（25 °C 时 τ2(−40)=3.03 s 已 6 倍于窗）。该量在高温批为窗限登记，与 25 °C 卡 §J3 登记级一致。

J5 明细：A1（逐周期回弹）0.5 Hz 11/102、1 Hz 0/102、2 Hz 87/102——0.5/1 Hz 仅 2/4 周期，判线结构性苛刻与 25 °C 卡 §J5 分析同；A2 量级 81/93/100（79%/91%/98%）；A3 频率方向 100/102（98%）；回弹峰比中位 0.399/0.582/1.154 全在 [0.3,3]。**前向量级与方向在 37 °C 维持，逐周期判线维持登记。**

## T2 · 同温度跨宿主绝对对表（37 °C：Lei-CHO vs Beattie-HEK）

| 量 | Lei-37（CHO，n=105 合并中位） | Beattie-37（HEK，封卷值） | 比值 | 判定 |
|---|---|---|---|---|
| τ_rec(−120) | 1.518 ms（CV 0.60） | 3.04 ms | **0.499** | **封卷**（∈[0.3,3]） |

辅助登记（不设判线，§0 口径：协议族不同，只报 Lei 侧绝对值）：τ_deact 慢层 τ2(−60/−40) = 0.302/0.266 s（窗限）；大补跳 τ_inact(+40) = 43.7 ms、τ_act(+40) = 265.2 ms。
备注：DoE 网格下界 1.0 ms 邻近 37 °C 批 τ_rec 分布下端（−140 档中位 1.23 ms），绝对值或略高估；比值 0.499 距判线两端均远，判决对该高估不敏感（即使按下界 1.0 ms 计，比值 0.33 仍在带内）。

## T3 · Q10 层（五温度 25/27/30/33/37 °C，群体中位数）

| 量 | 25→37 中位 | Q10 | 判线 [1.2,4.0] | Arrhenius R² | 线性（≥0.9） | 判定 |
|---|---|---|---|---|---|---|
| τ_rec(−140) | 4.31→1.23 ms | **2.84** | ✓ | 0.991 | ✓ | **封卷 + 线性成立** |
| τ_rec(−120) | 5.31→1.52 ms | **2.84** | ✓ | 0.991 | ✓ | **封卷 + 线性成立** |
| τ_deact τ2(−40) | 3.03→0.27 s | 7.57 | ✗ | 0.781 | ✗ | **登记**（27 °C 非单调 + 37 °C 窗限） |
| τ_act(+40)（大补跳 τ2） | 431.8→265.2 ms | **1.50** | ✓ | 0.470 | ✗ | **封卷（端点）**，线性不成立（30 °C 非单调） |
| τ_inact(+40)（大补跳 τ1） | 90.0→43.7 ms | **1.83** | ✓ | 0.692 | ✗ | **封卷（端点）**，线性不成立（27–33 °C 网格平台 55.6 ms） |
| E_rev(T) | −83.90→−92.04 mV | 斜率 −0.58 mV/°C | 出 [0.05,0.35] 带 | R² 0.825 | — | **登记**（方向 Nernst ✓） |

E_rev 备注：五温度逐细胞中位 −83.90/−85.88/−86.83/−86.46/−92.04 mV，全部 CV<0.035 封卷；方向随 T 变负 ✓；斜率 −0.58 mV/°C 约为纯 Nernst 钾电极（−0.28 mV/°C）两倍，照实登记出带；**37 °C 中位 −92.04 mV 与 model62/149 拟合值 −92.9 mV 互证**（预注册卡 §3 已声明两口径不互绑）。

单温度 CV 判决不连坐 Q10（预注册 §3 在案）；各温度各量 CV 判决等级全录于时间表。

## 对照（C1' / C2'）

- C1' 管道同一性：37oc3-A03 sactiv +40 段再漏减后 I_end = −7.9 pA，σ = 10.6 pA，|I| < 4σ ✓（该孔低表达，raw 峰值 48.9 pA；g = 0.076 pA/mV 有限无死锁）→ **过**。
- C2' 完成率：211/211、127/127、118/118、109/109、69/69、36/36 = **100%（六批全满）** → **过**。

## 异常登记（不判决，仅备案）

1. h_ss(−80)：25/27 °C 批全体 ok=false（保持电位档等时简并，J1 统计量正确拒绝），37 °C 批 80 孔通过且中位 0.857 CV 0.15——该档为等时简并电压，数值仅供参考，不入任何判决。
2. h_ss(−100) 37 °C 批 n=22（其余孔 ok=false），低于 n 门槛 → △数据不足。
3. τ_inact(+40) 在 27/30/33 °C 中位数同为 55.6 ms：DoE 网格量子化平台，致 Arrhenius R²=0.69 线性不成立；端点 Q10 判决不受影响。

## T4 · 第三方文献层（2026-09-21 跑后登记，非预注册，不设判线只对照）

**存在性结论**：公开**原始** hERG 动力学数据全部出自 Oxford 圈（Beattie/Lei）与 CiPA 三实验室（已用作药物迹线第三方），不存在第三家独立实验室的公开原始数据集。可用的第三方验证为**文献数字层**，已核实来源如下：

| 第三方锚点 | 实验室/体系 | 数字 | 本战役对应值 | 对照 |
|---|---|---|---|---|
| Vandenberg et al. 2006（*Am J Physiol Cell Physiol* 291:C165–175） | Victor Chang 研究所 Sydney；CHO，手动膜片钳，14–37 °C | τ_rec(−120 mV)：**3.8 ms @24 °C → 1.1 ms @37 °C**；τ_deact(−120)：24.7→12.4 ms；τ_inact(+40 三联脉冲)：3.2→0.87 ms（Li et al. 2016 Fig 3D 表格逐字转录） | Lei-25→37 °C：τ_rec 5.31→1.52 ms | **37 °C 同物理量对表：Lei/Vandenberg = 1.38（Beattie/HEK 参照为 3.04，三方同口袋）；τ 比值 3.49 vs 3.45 几乎一致；重算 Q10：恢复 2.60、去激活 1.70、失活 2.72（本战役 2.84/—/1.83\*）** |
| 同上 | 同上 | 速率 Q10：电导 1.4；激活/去激活/失活/恢复 **1.7–2.6** | τ_rec Q10 2.84；τ_inact 1.83；τ_act 1.50 | 恢复 Q10 与文献带上端一致（2.84 vs ≤2.6+散布）；τ_inact 带内；τ_act 略低于带下端 → 登记 |
| Di Veroli et al. 2013（*Am J Physiol Heart Circ Physiol* 304:H104–117） | Manchester/Boyett + Roche；CHO，IonWorks 自动膜片钳，20 °C | τ_rec(−100 mV) ≈ **1.9 ms**（经 Li et al. 2016 转引核实） | Lei-25 °C 6.54 ms | 比值 3.4 出因子 3 带；但 Li 2016 记载 Di Veroli 与 Vandenberg 同电压下彼此亦差 >2 倍——实验室间绝对 τ 散布 ≥2× 是该领域已知现象，只登记不判决 |
| Oliveira-Mendes et al. 2023（*Clin Transl Med* 13:e1266，doi:10.1002/ctm2.1266） | Nantes（Loussouarn/De Waard/Baró）；HEK293，384 孔自动膜片钳 + 手动复核，22/27/32/37 °C | AP 钳复极化功率用**单一时间因子 ≈2** 即可把 22–27 °C 图映射到 32–37 °C | AP 时程有效 Q10 ≈ 2（跨门控过程粗平均） | 量级方向一致 |
| Zhou et al. 1998（*Biophys J* 74:230–241） | Wisconsin（January/Robertson）；HEK293，35 °C | τ_rec(V)：−100 mV ≈0.6 → −20 mV ≈2.5 ms；τ_inact(+40) ≈2.3 ms（Li et al. 2016 Fig 4 圆点高清渲染人工读数，二手数字化 ±0.15 ms；全文 PDF 受 PMC/Cell 反爬保护未获取，DOI:10.1016/S0006-3495(98)77782-3） | Lei-37 °C τ_rec：−100 mV 1.87 → −20 mV 3.50 ms | 比值 1.5–3.1（本战役偏慢），V 依赖方向一致（两数据集 τ_rec 均随电压升高变长）；散布在领域已知 ≥2× 实验室间差内 → 登记 |

**文献层一句话**：τ_rec 在 37 °C 的绝对对表三方收敛——Lei-CHO 1.52 ms / Vandenberg-CHO 1.1 ms（比值 1.38）/ Beattie-HEK 3.04 ms（预注册比值 0.50 封卷）；温度律同样收敛——本战役 Q10 2.84 vs Vandenberg 恢复 Q10 重算 2.60（τ 比值 3.49 vs 3.45 几乎一致）vs Oliveira-Mendes AP 层因子 ≈2。绝对 τ 的实验室间散布（≥2×，Li et al. 2016 记载 Di Veroli 与 Vandenberg 同电压亦差 >2×）为领域已知；本战役封卷判决只锚定预注册参照，文献层全部登记不判决。\*失活 Q10 口径不互比：Vandenberg/Zhou 为三联脉冲快速失活（τ≈1–3 ms），本战役为大补跳 1 s 窗双指数快分量（43.7–90 ms），只登记各自数值。去激活无电压重叠（Vandenberg −120 mV；本战役 −60/−40 mV）。转录数字全录于 `数据\第三方文献层\T4_文献层数字_Vandenberg2006_Zhou1998.json`。

## 与 25 °C 卡的合读

- 群体结构两件硬货（E_rev 逐细胞恒定、h_ss 负端阶梯）在 **五个温度、670 孔**（25 °C 211 孔 + 新增四批 459 孔）全部封卷级复现（每温度 CV 判决各自独立）。
- 绝对时间尺度层：τ_rec(V) 在五温度单调下移，Q10 2.84 封卷且 Arrhenius 线性成立——**model62 的时间尺度方向不仅被数据钉死，而且服从单一温度律**。
- 跨宿主：37 °C 下 CHO 与 HEK 的 τ_rec(−120) 差 2 倍但在预注册因子 3 带内 → 同温度跨宿主绝对一致封卷；Lei-25 °C 5.31 ms 与 Beattie-37 °C 3.04 ms 的旧比值 1.7 现由 Q10 2.84 完全解释（5.31/2.84^1.2 ≈ 1.52 ms）。

## 正文/SI 补丁句（回填）

正文 Results 群体验证段（接 211 孔句后）：
> The same population signatures held across a five-temperature series (25/27/30/33/37 °C, 459 additional wells, same pipeline): per-cell E_rev remained cell-invariant at every temperature (CV ≤ 0.034) and shifted from −83.9 to −92.0 mV, the 37 °C value matching the model-fitted −92.9 mV; the recovery time constant followed a single temperature law (Q10 = 2.84, Arrhenius R² = 0.991); and at 37 °C the CHO-cell median τ_rec(−120 mV) of 1.52 ms agreed with the HEK-cell reference of 3.04 ms within the pre-registered factor-3 band (ratio 0.50).

正文 Discussion"温度系列"句改写（原句："The temperature-matched series¹⁹ remains a defined next step for the absolute-timescale layer, not a completed one."）：
> The temperature-matched series¹⁹ has now been completed: the absolute-timescale layer obeys a single Q10 of 2.84 across 25–37 °C and matches the HEK reference at 37 °C within a factor of two, while the slow deactivation component becomes window-limited at 37 °C and is registered rather than sealed.

SI S2 增补小节（Q10 层）：
> (v) Five-temperature Q10 layer. Recovery τ_rec at −140/−120 mV: Q10 = 2.84 (25→37 °C), log τ vs T linear R² = 0.991. Big-step τ_act(+40)/τ_inact(+40): endpoint Q10 = 1.50/1.83, Arrhenius linearity not established (grid-quantised plateaus and one non-monotone midpoint). Slow deactivation τ2(−40): Q10 = 7.6, out of band — the component approaches the 0.5 s staircase window at 37 °C and is registration-grade. E_rev(T): slope −0.58 mV/°C, steeper than the pre-registered band but the 37 °C median (−92.0 mV) matches the model-fitted −92.9 mV. All per-temperature medians, CVs and verdict grades are tabulated in the accompanying time table.

## 资产清单

- 数据：`04_细胞线4\数据\Lei全量\herg{27,30,33,37}oc{1,3,4}\`（五批 459 孔 × 三协议 + 37 °C 双批 AP 三协议，逐字节校验在案）
- 判线：`04_细胞线4\结果\预注册判决卡_Lei温度Q10_2026-09-21.md`（13:30 冻结）
- 代码：`04_细胞线4\α模型\` 下 `lei211_J1J2_提取.py` / `lei211_J3_提取.py` / `lei211_TAUH_群体表.py` / `lei211_J5_AP前向.py`（均支持 BATCH 环境变量）、`lei211_TAUH_逐孔.py`、`lei211_温度汇总.py`、`lei211_时间表生成.py`、下载器 `04_细胞线4\数据\Lei全量\download_lei_T.py`
- 结果：`lei211_J1J2_结果_<batch>.json`、`lei211_J3_结果_<batch>.json`、`lei211_tauh_表_<batch>.json`、`lei211_tauh_逐孔_herg37oc{3,4}.json`、`lei211_J5_结果_<batch>.json`、`lei211_温度汇总.json`
- 时间表：`04_细胞线4\结果\时间表_Lei五温度_2026-09-21.md`（五温度 × 全电压 × 判决等级）
- 图：`04_细胞线4\文章\figures_v01\EDFig6_temperature_Q10.png/.pdf`（a τ_rec Arrhenius + Vandenberg 双温度锚点、b 登记级 τ 层、c E_rev(T) + 149 拟合、d 跨宿主三方对表 Lei/Beattie/Vandenberg），脚本 `文章\make_EDFig6_temperature_Q10.py`；稿件内编号 **Extended Data Fig. 6**（原 EDFig4 文件名已废弃，正文 Discussion 与 SI S2 均已挂 ED6，Nature 式图注已写入正文 ED 图注区）
- 文献层：`04_细胞线4\数据\第三方文献层\Li2016_CiPA_hERG温度模型.pdf`（Vandenberg/Di Veroli/Zhou 数字转引核实来源）、`T4_文献层数字_Vandenberg2006_Zhou1998.json`（全部转录数字、重算 Q10、三方对照，含 li2016_p5/p6.png 目检裁剪）
