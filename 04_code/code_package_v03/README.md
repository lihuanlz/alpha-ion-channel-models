# code_package_v02 · 四通道数据计算代码归并包

**日期**：2026-09-21
**口径**：本包是从 `04_细胞线4\α模型\`、`04_细胞线4\w182\`、`04_细胞线4\文章\` **复制**（不移动）的全部数据计算相关脚本，共 84 个，按业务线分 10 个子文件夹。原始位置全部保留，本包仅为审计与复算提供单一入口。
**配套稿件**：`Nature_main_v02_2026-09-21.md` + `Nature_SI_v02_2026-09-21.md`。
**运行环境**：Windows + Anaconda Python（numpy / scipy / pandas / matplotlib / pyabf / neo）；数据根目录硬编码为 `D:\data` 与各工作区相对路径，复算时需按本机实际路径调整。

---

## 01_hERG_Beattie9_extraction（24 个）—— hERG 九细胞提取·封卷·前向

对应论文：Fig. 2a–d，Extended Data Fig. 1；SI S1.1、S2、S6。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| 2026-09-13_alpha_amplitude_table_extraction.py | 九细胞幅度/稳态表 | S1.1c h_ss 表 |
| 2026-09-13_alpha_inact_gate_inact_protocol_seal_verdict.py | h_ss(V) 12 档封卷 | S1.1c；Fig. 2b |
| 2026-09-13_alpha_tauobs_verdict_repol_window_fast_relax_seal.py | τ_rec/τ_d 复极窗四档 | S1.1b |
| 2026-09-13_alpha_tauobs_recon_testseg_early_phase.py | τ_obs 上界口径 | S1.1d |
| 2026-09-13_alpha_act_envelope_tauact_seal_verdict.py | τ_act 包络封卷/登记带 | S1.1 τ_act 段 |
| 2026-09-13_alpha_act_kinetics_envelope_const_verdict.py | 激活延迟 50–150 ms | 正文 hERG 段 |
| 2026-09-13_alpha_act_timecourse_ss_act_rise_verdict.py | m_ss 锚点 | S1.1f |
| 2026-09-13_alpha_pure_hssV_strip_verdict.py | B1（h=1 简化）否决 | 正文"0.42 at −70 mV" |
| 2026-09-13_alpha_hssmeasured_table_rerun_AP_forwardverdict.py | AP 前向 8/9 | Fig. 2d；S2 |
| 2026-09-13_alpha_hssmeasured_table_rerun_sineforward_verdict.py | sine 前向 7/9 | Fig. 2c；S2 |
| 2026-09-13_alpha_AP_forwardverdict.py | 同上早期版 | 存档 |
| 2026-09-13_alpha_assembly_smoke_sineforward_verdict.py | 组装冒烟 | 存档 |
| 2026-09-13_alpha_rebound_hook_inact_recovery_const_verdict.py | τ_rec hook 锚 | S1.1e G 表锚 |
| 2026-09-13_alpha_forward_validation_tail_and_sine.py | 尾段白化 28/33 | S2 |
| 2026-09-13_minus40_step_drift_assay.py | −40 mV 档数据缺陷鉴定 | S6 D 登记 |
| 2026-09-13_longtail_flatness_verdict_v5_nine_cells.py | 长尾判决 | S6 |
| 2026-09-14_alpha_tauobs_reanchor_AP_forwardverdict.py | 换锚敏感性 AP | S2 |
| 2026-09-14_alpha_tauobs_reanchor_sineforward_verdict.py | 换锚敏感性 sine | S2 |
| 2026-09-14_alpha_data_side_spike_baseline_recheck.py | 060/016 基线偏移 >3×MAD | S6 数据侧关闭 |
| 2026-09-14_alpha_B6_H1_highV_act_verdict.py | B6 候选 H1 判决 | S1.1 B6 段 |
| 2026-09-14_alpha_B6_H2_quantitative_prediction_card.py | B6 H2 定量预测 | S1.1 B6 段 |
| 2026-09-14_alpha_B6_H2_interbeat_accumulation_verdict.py | 跨拍累积主解释 | S1.1 B6 段 |
| 2026-09-14_alpha_B6_H3_envelope_convention_audit.py | B6 H3 口径 | S1.1 B6 段 |
| 2026-09-14_alpha_B6_act_foot_k3_cascade_dual_protocol_rerun.py | 激活足 k3 级联 | S1.1 B6 段 |

## 02_hERG_Lei_population_temperature（9 个）—— Lei 211 群体 + 五温度层

对应论文：正文 hERG 群体段（ref. 18/19），Extended Data Figs. 6–8；SI S2 群体/温度两节、S7。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| lei211_J1J2_extraction.py | E_rev / h_ss / m_ss 逐孔提取（211→670 孔） | S2 群体节；ED7/ED8 a,b |
| lei211_J3_extraction.py | 去激活阶梯慢分量提取 | 0.74/3.03 s 阶梯序 |
| lei211_TAUH_per_well.py | τ_rec 逐孔 DoE 拟合（4σ 噪声门） | S7 登记册 A 类来源 |
| lei211_TAUH_population_table.py | τ_rec 群体中位表 | ED6a；ED8c |
| lei211_J5_AP_forward.py | AP 钳前向（0.5/1/2 Hz，A1–A3） | 91–99% 幅度；A1 45.8% |
| lei211_C1_synthetic_recovery.py | 等时偏置合成回收对照 | S2 局限 (ii)；S7 C 类 |
| lei211_time_table_build.py | 五温度全档中位/CV/判级时间表 | ED6 数据源 |
| lei211_temperature_summary.py | Q10 = 2.84 / R² = 0.991 汇总 | 正文温度句；ED6 |
| lei211_population_plot.py | 群体图版 | ED7 图版 |

## 03_Nav1.5（6 个）

对应论文：Fig. 3a–b，ED3；SI S1.2、S2、S6。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| 2026-09-15_Nalpha1_Nav15inact_fast_component_constancyverdict.py | τ_h(V) 恒定性 6/7；−30 mV CV 0.029 | S1.2a |
| 2026-09-15_Nalpha2_Nav15whole_channel_oneshot_validation.py | 整迹前向 3/3（≤3.4% RMS） | 正文 Nav 段；S2 |
| 2026-09-16_Nalpha3_Nav15act_foot_portability_verdict.py | 脚部散布 CV 0.184 真实生物学 | 正文；S6 Nα-3 |
| 2026-09-16_Nalpha4_single_channel_minus40_anchor_verdict.py | τ_h(−40) 分布（n=69，CV 0.772）→ N 臂关闭 | Fig. 3b；S1.2f |
| 2026-09-15_Nalpha_drug_Nav15drug_shape_modulation_verdict.py | 双分量药物作用（|κ*|=0.20） | Fig. 4；S4 |
| 2026-09-15_Nalpha_drug_supplement_pack.py | 外部基准 7/7 在 ±50% 带 | S4 |

## 04_CaV1.2（6 个）

对应论文：Fig. 3c，ED4；SI S1.3、S4、S6。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| 2026-09-15_Calpha1_CaV12inact_temperature_law_carrier_verdict.py | f_inact 家族常数；sweep QC 门 | S1.3a；S7 A 类 |
| 2026-09-15_Calpha1b_CaV12inact_charge_carrier_verdict_17mV.py | +17 mV 锚段复核 | S1.3a |
| 2026-09-15_Calpha2_Ca_microdomain_verdict_current_inact_corr.py | CDI 非幅度分级（阴性） | S1.3 方程段 |
| 2026-09-15_Calpha3_CDI_structure_form_verdict.py | CDI 加性快分量 w_fast=0.289 | Fig. 3c |
| 2026-09-15_Calpha4_ramp_IV_temperature_law.py | Q10 非常数 → 作协变量 | S1.3c |
| 2026-09-15_Calpha5_CaV12drug_shape_modulation_verdict.py | 纯阻断否决 p=7.2e−16；双池反向 | Fig. 4；S4 |

## 05_IKs（6 个）

对应论文：Fig. 3d，ED5；SI S1.4、S4。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| 2026-09-15_Kalpha0_IKssmoke_GV_phenotype_cluster.py | G-V 表型摸底 | 存档 |
| 2026-09-15_Kalpha1_IKsact_deact_constancy_verdict.py | τ_app 否决、量级锚定、散布真实 | S1.4d |
| 2026-09-15_Kalpha2_Fedida2024_Fig3B_digitised.py | τ_act(V) 25 点表（钟形） | S1.4a；Fig. 3d |
| 2026-09-15_Kalpha2_Fedida2024_Fig3C_Deltatdigitised.py | 延迟表 d(V) | S1.4b |
| 2026-09-15_Kalpha3_IKsforward_validation.py | 激活前向 7/8 | S2 |
| 2026-09-15_Kalpha_drug_IKsdrug_table_decomposition_verdict.py | 多表调制；DIDS 池混杂登记 | Fig. 4；S4 |

## 06_identifiability_slowmemory（8 个）—— Fisher 审计 + 慢记忆谱

对应论文：正文 Fisher 段，Extended Data Fig. 2；SI S3。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| de182.py / judge182.py（w182） | model62 十三参 Fisher 谱：χ=7.4×10¹²、六零方向 | Fig. ED2；S3 |
| 2026-09-13_rate_coupling_identifiability_math_validation.py | 速率-电导联合标度精确简并 | S3 精确简并 (i) |
| 2026-09-13_slow_memory_layer_identifiability_math_validation.py | α_amp·ALPHA·XMAX 乘积简并 | S3 精确简并 (ii) |
| 2026-09-13_memory_kernel_identifiability_three_gate_validation.py | 模态坐标投影 | S3 |
| 2026-09-13_sameV_diff_history_coupling_verdictv4_nine_cells.py | 同压异史 17–112×（9/9） | S3 慢记忆审计 |
| 2026-09-13_slow_layer_discrete_const_verdictv5_noise_envelope_gate.py | 记忆谱 4.8 s–2000 s+ | S3 |
| 2026-09-13_slow_layer_master_curve_verdictP1.py | 慢层主曲线 | S3 |

## 07_drug_grammar_CiPA（8 个）—— 药物语法 + CiPA 65/68

对应论文：Fig. 2e、Fig. 4；SI S4。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| 2026-09-14_CiPA_full_drug_library_downloader.py | CiPA 三实验室 68 迹下载 | S4 数据源 |
| 2026-09-14_alpha_CiPA_crosslab_check_card.py | 跨实验室对拍（20/20, 16/16, 29/32） | Fig. 2d |
| 2026-09-14_alpha_drug_card1_block_kinetics_state_preference.py | k_on/k_off；捕获型状态偏好 | S4 |
| 2026-09-14_alpha_drug_card2_pmz_non_single_step_binding_attribution.py | pimozide 非单步 | S4 |
| 2026-09-14_alpha_drug_card3_full_library_panorama_parameter_table.py | 136 数据集参数全景 | S4 |
| 2026-09-14_alpha_drug_card4_AP_waveform_clinical_risk_crosscheck.py | AP 波形风险对拍 | S4 |
| 2026-09-14_alpha_drug_card5_shape_invariance_verdict.py | 纯阻断 30/30 否决 | 正文 hERG 药物段 |
| 2026-09-14_alpha_drug_card6_shape_deformation_decomposition.py | 幅度型归因（|κ*−1|=0.020） | Fig. 2e |

## 08_StageB_ORd_host（7 个）—— ORd 宿主整合 + 三臂药物打分

对应论文：Fig. 5；SI S5。

| 脚本 | 产出 | 论文落点 |
|---|---|---|
| 2026-09-14_alpha_formal_assembly_forward_engine.py | α 模型前向引擎（宿主共用） | S1/S5 |
| 2026-09-15_ORdthree_channel_alpha_assembly.py | ORd 内 IKr/IKs α 化替换 | Fig. 5a；J1–J6 |
| 2026-09-14_alpha_StageB_B0_ORd_port_foundation_recheck.py | 移植地基复核 | S5 |
| 2026-09-14_alpha_StageB_B1_official_static_arm_24drugs.py | B1 臂 ρ=−0.523 | Fig. 5b |
| 2026-09-14_alpha_StageB_B2_alpha_substitution_24drugs.py | B2/B2s 臂 ρ=−0.547/−0.506 | Fig. 5b |
| 2026-09-16_alpha_StageB_B3_dual_alpha_host_recheck_24drugs.py | 双 α 化复验；τ_act 敏感臂极差 0.016 | S5 稳健性 |
| 2026-09-16_B3_figure_redraw.py | Fig. 5 图版 | Fig. 5 |

## 09_figures（7 个）—— 稿件图版生成

| 脚本 | 产出 |
|---|---|
| make_figures_v01.py | Fig. 1–5 + EDFig2 + EDFig3（= 稿件 ED7）PNG/PDF |
| make_EDFig6_temperature_Q10.py | EDFig6 温度 Q10 图 |
| make_percell_violin_2026-09-21.py | ED8 v01 六面板逐细胞小提琴图（同时生成长表 CSV + 失败登记表） |
| make_percell_violin_v02_2026-09-21.py | ED8 v02：n<20 组去小提琴化只画点、全线性轴、IKs 加 Chan 锚星、CaV 加文献锚虚线 |
| make_percell_violin_v03_2026-09-21.py | ED8 v03：九面板首版（加药臂进图），版式被 v04 取代 |
| make_percell_violin_v04_2026-09-21.py | **ED8 v04（当前版）**：统一 box+strip 风格（中位/IQR/10–90% 须 + 全散点）、行内按组数配宽（f/i 加宽）、注记折行修掉右侧空白；同时幂等生成 `结果\per_cell_master_table_pharmacology_arms_2026-09-21.csv`（361 行） |
| make_EDFig345_channel_cards_2026-09-21.py | **ED3/ED4/ED5 通道卡（当前版）**：Nav1.5/CaV1.2/IKs 各 2×2，box+strip 同 ED8 v04 风格；数字全部直读封卷 JSON/主表（s_ref 池 CV 0.184、w_fast 0.289 vs Δf 0.293、前向 7/8 等）；ED5d 登记逐文件表重提取（n=8）与封卷 J2（n=5）的口径差异 |

## 10_verification_tau_rec（3 个）—— τ_rec 网格量化稳健性核验（2026-09-21，封后验证层）

| 脚本 | 产出 |
|---|---|
| verify_tauh_continuous_refit_2026-09-21.py | 连续精修 v1（幅度误播种留档版）：SSE 恶化 −3000% 的优化器假象记录，Q10≈0.9 为假象已丢弃 |
| verify_tauh_refit_v2_2026-09-21.py | **正确播种连续精修（当前有效版）**：1,340 拟合，SSE 仅改善 1.7–4.1%，Q10 精修 2.26(−140)/2.89(−120) |
| verify_tauh_modelfree_2026-09-21.py | 无模型时钟（t_half/t_ext/t10–90）：Q10 2.1–3.8（R²≥0.985）+ 网格边缘堆积审计（37 °C −140 mV 55% 删失） |

结论见 `结果\验证卡_taurec网格量化_连续精修复核_2026-09-21.md`：封卷 Q10=2.84 在精修/无模型带内，非网格 flattering。

---

## 不在本包内的相关资产（指针）

- 数据源下载/整理脚本：`04_细胞线4\数据\Lei全量\download_lei*.py`、`_releakcorrect.py`；`04_细胞线4\公开数据\Nav15_Tarasov2026_dryad\*.py`。
- 稿件版本生成脚本（v01→v02 锚点替换）：`文章\_make_v02_main.py`、`文章\_make_v02_si.py`。
- 早期探针/调试脚本（`_debug_*`、`_probe_*`、`peek_*`、`scan_*`）：留 `α模型\` 原处，属侦察层非产出层。
