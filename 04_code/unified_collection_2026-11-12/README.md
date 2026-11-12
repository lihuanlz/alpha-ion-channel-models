# 代码全集 · 离子通道统一重算工程（2026-11-12 归档）

本目录是细胞线4（hERG → 全通道）全部自产代码的**分类归档副本**。原位置脚本一律保留不动（冻结存档引用原路径），此处为按流水线阶段组织的完整副本，共 **126 个脚本 / 11 类**。

最终论文工程：`../latex_v2_unified/`（main.tex + si.tex，build/ 下 13+13 页 PDF）。
统一总表：`law_channelpedia/统一重算_2026-11-11/master_unified_v1.csv`（487,940 行，含 356 行文献锚）。

## 流水线总顺序

```
01 model62 构建与可识别性  →  02 数据下载与枚举  →  03 特征提取(时间常数)
→ 04 判决与统计(核心定律)  →  05 零记忆审计  →  06 统一重算 pipeline(2026-11-11)
→ 07 CiPA 全药池 stage 流水线  →  08 换表预言与宿主模型
→ 09 边界家族 Kir/K2P  →  10 论文图与表  →  11 冒烟与终验
```

v2 论文数字的**关键路径**：06（建统一总表）→ 04 的 robust_verdict_all_profile.py（判决）→ 10 的 make_fig1-6（图）→ 10 的 _make_si_table.py（SI 长表）。

---

## 01_model62构建与可识别性（19 个，源自 run182/）

model62（13 参 hERG 模型）从 model44 到可识别性谱的完整构建链。配套产物在 `run182/model62_identifiability/`（Fisher 矩阵等 JSON）。

| 脚本 | 用途 |
|---|---|
| runner_v148.py | 总驱动引擎 v1.29–1.49（拟合内核，含收敛标志） |
| run_all165.py | 代码165 一体驱动：K6a9Ji(model44) 双卡联合拟合+判决 |
| gate165.py | 代码165 门控侧指标单一实现源（防口径漂移） |
| gatepar.py / gatepar47.py | 门控包络拟合进程池后端（引擎 v2 / model47） |
| runner166_m46.py / runner166b_m46.py | 代码166/166b：慢可用度 u 态加边（model46，31参；166b τ_u 锚定版） |
| traj167_轨迹仪器化_2026-09-08.py | 167 判决点全内部态轨迹 dump（sim47_trace） |
| de169.py | 代码169：model50 结构重修卡（38 维拟合器） |
| de173.py | 代码173：model53 冻结版四臂重锚卡（30 维） |
| de174.py | 代码174：model53 减维 7 槽重拟合 @16713003 |
| de176.py / de177.py | 代码176/177：model60 全合成卡 / 特征锚定卡（官方 HH 环+X 慢件） |
| de178.py | 代码178：model61 机制补全卡（k43 双指数化，14 参） |
| de179.py | 代码179：model62 锚定收口卡（16 参→冻结 13 参） |
| judge179.py | 代码179 判官（§5 判线独立复算） |
| de182.py | **代码182：model62 可识别性谱卡·执行侧**（Fisher 走 expm 通道；硬墙：不重拟合/不改架构） |
| judge182.py | 代码182 判官（自含 expm 仿真，独立复算 Fisher） |
| judge_zero184.py | 代码184 零号测量：判官管线体检件 |

## 02_数据下载与枚举（7 个）

| 脚本 | 用途 |
|---|---|
| enumerate_full.py / enumerate_v2.py | Channelpedia 细胞全量枚举（19,282 细胞清单） |
| enumerate_rest.py | 枚举 silent/bad/excl 三类细胞 ID 并入总清单 |
| download_full.py | Channelpedia NWB+JSON 批量下载（72 GB） |
| 2026-11-09_CiPA全量下载器.py | CiPA 五实验室全量下载（62GB/14,973 文件，断点续传，429 限流退避） |
| 2026-11-10_数据归拢到数据总库.py | 全部数据分类归拢到 `数据总库/`（01_CiPA…06_官方锚） |
| _终验_2026-11-09.py | CiPA 终验：14,973 文件重算 sha256 对拍 OSF 官方清单 |

## 03_特征提取_时间常数（10 个）

| 脚本 | 用途 |
|---|---|
| extract_features.py | Channelpedia Kv 提取管线 v1（NWB → τ/V½ 特征） |
| extract_features_v2.py | v2 工程精度升级 |
| extract_features_v3.py | v3（协议段边界+窗宽修订） |
| extract_features_v4.py | v4（85,591 行版本，后被 v5 取代） |
| extract_features_v5_hcn.py | v5：Kv 全族 + HCN（反向门控极性） |
| extract_recovery_v5.py | 恢复协议 τ_rec 提取（v5 同代） |
| make_robust_corpus_v5r.py | v5r 稳健层：冻结提取器之上补旗标，重出判决表 |
| profile_everything.py | **轮廓似然·全数据完整重验（2026-11-08，profile_v1 起点）** |
| profile_likelihood_kv11.py | Tier-1 展示：Kv1.1@25 逐细胞不对称 CI vs 对称 SE |
| validate_acell_v5.py | v5 提取器的逐细胞验证 |

## 04_判决与统计_核心定律（15 个）

| 脚本 | 用途 |
|---|---|
| stats_family.py | 全族统计分拆 v1：τ 紧 vs V½ 松（判据跑前钉死） |
| stats_family_v2.py | v2：加方差分解（观测²=真实²+噪声²） |
| stats_v3.py | v3 统计（全族三温度扩展） |
| stats_qc.py | 统计层 QC |
| hier_bayes_kv11.py | 分层贝叶斯（经验贝叶斯·边际似然）Kv1.1 三温度展示 |
| robust_verdict_66.py | 稳健性重判：V½ 撞界过滤 + EB（66 组框架） |
| robust_verdict_profile.py | 判决第 D 口径：剖面似然全库重验 |
| robust_verdict_all_profile.py | **四张判决表四口径全量重判 v2（主判决产物 robust_verdict_all_profile_2026-11-08.csv，212 行）** |
| profile_full_verdict.py | 对判决实际使用的每一行做剖面重验 |
| recovery_verdict_rebuild.py | 恢复表判决重建 |
| icg_analysis.py | ICGenealogy 933 模型重拟合判决（K/Na/KCa/IH 层） |
| icg_ca_refine.py | ICG Ca 层精修 |
| analysis_BE.py | B+E 补充：全族温度定律 + 跨温度前向预言 |
| analysis_MNOK.py | M/N/O/K 收官：斜率 k 散布、窗电流假说（阴性）等 |
| analysis_pack_v5.py | v5 分析包（冻结口径来源） |

## 05_零记忆审计（5 个）

第二遍零记忆审计链，全部从冻结文件独立重写复算。

| 脚本 | 用途 |
|---|---|
| audit2_B_kv_verdict.py | B 项：Kv 判决链独立复算 |
| audit2_E_icg_verdict.py | E 项：ICG 判决链独立复算 |
| audit2_H_misc.py | H 项：跨物种/Q10/指纹三抽验 |
| audit2_I_dedup.py | I 项：v4/v5 两次独立提取对拍（正式口径） |
| run_all_audits.py | 审计总驱动 |

## 06_统一重算pipeline（14 个，2026-11-11）

**v2 论文的数据底座。** 全部语料统一升级到逐行剖面似然（profile_v1）并合表。

| 脚本 | 用途 |
|---|---|
| batch1_channelpedia_装入.py | Channelpedia 276,039 行装入统一 schema |
| batch2_cipa_剖面升级.py / _v2.py | CiPA 五室逐 sweep 剖面升级（v2 为 ss 最高浓度块口径） |
| batch2_装入统一表.py | cipa2_sweeps.csv（192,055 行）装入 |
| batch3_lei_剖面升级.py / batch3_装入统一表.py | Lei 五温度 10,203 行 |
| batch4_beattie9_剖面升级.py / batch4_装入统一表.py | Beattie 九细胞 207 行 |
| batch5_nav15_剖面升级.py | Nav1.5 双温度 371 行（tau_edge 旗标） |
| batch5b_cav_剖面升级.py | CaV1.2 562 行（连续浴温协变量） |
| batch5c_iks_剖面升级.py | IKs 8,147 行（重复记录先并优） |
| batch6_文献锚入表.py | 三层文献锚（CiPA 官方 250 + 跨物种 76 + ICG 30） |
| build_master.py | **幂等重建统一总表 master_unified_v1.csv** |
| qa_统一总表对拍.py | QA：关键判决从统一表重算，与冻结存档值对拍 |

## 07_CiPA全药池stage流水线（28 个，2026-11-10）

五实验室 28 药全药池 stage1–5 完整流水线（断点可续，ckpt 在原目录）。

- **stage1**：stage1_表级分析.py（剂量反应/Hill/IC50 重拟合）、stage1_出图.py（5 张全景图）
- **stage2 波形 τ 守恒**：stage2_波形级分析.py、stage2a_meta主表分析.py、stage2b_波形tau守恒.py + stage2b_汇总出图.py、stage2c_波形tau守恒v2.py + stage2c_汇总出图.py、stage2d_波形tau守恒_lab345列修复.py（列污染修复）、stage2e_最终汇总出图.py（fig6 干净版）
- **stage3 阻断时间进程**：stage3_阻断时间进程.py（未平台化/E-4031 时钟）、stage3_汇总出图.py
- **stage4 前向预言**：stage4a_前向预言_稳态阻断.py + stage4a_出图.py（fig10）、stage4b_波形级前向预言.py + stage4b_出图.py（fig11）
- **stage5 清剿**：stage5a_全段提取.py、stage5b_细胞属性_修正v2.py、stage5c_漏电流审计.py（⚠️第一版单位事故已作废重跑）、stage5d_QC失败普查.py、stage5e_波形预言_模板电压.py（模板电压系统性失败定案）、stage5f_lab2偏移漏电定案.py + stage5f_出图fig13.py、stage5g_三连分析.py + stage5g_出图fig14.py、stage5_出图fig12.py
- 其他：defect_probe_深挖.py（温度列/电压实现值/采样率/文件完整性）、make_fig8_正式稿.py

## 08_换表预言与宿主模型（14 个）

| 脚本 | 用途 |
|---|---|
| 2026-11-09_StageB_B0_路径修复.py | B0：宿主工程路径修复 |
| 2026-11-09_StageB_B2_换表.py / B3_换表.py / B3_换表重校G.py | B2/B3 换表臂执行（B3 重校 G 版） |
| make_b2_swap_2026-11-09.py / make_b3_swap / make_b3_recalg | 换表版文本手术生成器（每补丁带断言） |
| cp_swap_lib_2026_11_09.py | Channelpedia 群体表换入的公共库 |
| 2026-11-09_CiPA_crosslab_tauSwap_CP.py | **跨实验室预言 67/68：CP 群体 τ 表换入测 CiPA** |
| assemble_model_zoo.py | 逐通道模型动物园组装 v1 |
| 2026-11-09_MC_tau分布扰动_跨实验室卡.py / MC_tau扰动_汇总出图.py | 蒙特卡洛 τ 扰动跨实验室卡 + 汇总出图 |
| 2026-11-09_NavCaV_CiPA反向确认.py | 用 CiPA 数据反向确认 Nav/CaV 预测 |
| 2026-11-09_药物卡4_AP风险对拍_换表.py | 药物卡4：AP 风险对拍（换表版） |

## 09_边界家族Kir_K2P（2 个）

| 脚本 | 用途 |
|---|---|
| kir_recon.py | Kir 家族侦察：漏电样/整流主导、silent 细胞占比——不进判决的依据 |
| k2p_recon.py | K2P（TASK）侦察：纯漏电流、无 S 形门控——66→64 组框架修正的依据 |

## 10_论文图与表（11 个）

| 脚本 | 用途 |
|---|---|
| make_fig1_fig2.py | **v2 图1（语料+CI 校准）图2（判决全景）** |
| make_fig3_fig4.py | **v2 图3（温度）图4（药物五室）** |
| make_fig5_fig6.py | **v2 图5（通道卡）图6（ICG+边界登记）** |
| _make_si_table.py | **SI Table S5 判决长表生成（212 行；下划线转义+off-scale 处理）** |
| _wc.py | 正文词数统计（Nature 4000–5000 口径） |
| plot_fig4_law_2026-11-08.py | 旧版图4 定律六面板（v1 存档） |
| plot_fig5_boundary_2026-11-08.py | 旧版图5 独立印证与边界（v1 存档） |
| plot_figS_robustness.py | 旧版 SI 稳健性图（三口径对照） |
| make_fig2_identifiability_v2.py | v1 稿 Fig2 可识别性（去 0/451 字样版） |
| make_fig5_fivelab.py | v1 稿 fig5 五实验室六面板主图 |
| make_si_fig_defects.py | SI Figure S1 缺陷登记六面板（v2 沿用） |

## 11_冒烟与终验（1 个）

| 脚本 | 用途 |
|---|---|
| _bat_smoke.py | bat 冒烟：启动 25 秒抓输出后整树掐死（不留后台进程） |

---

## 运行环境与已知坑

- **第三方复现**：先读 `REPRODUCE.md`；一键核对入口 `06_统一重算pipeline/reproduce_check.py`（只读存缴表，37 项 PASS/FAIL，本次运行 37/37 全过，2026-11-12）。

- Python：Kimi 托管运行时（pandas/numpy/scipy/matplotlib/pynwb 等已装）。
- 画图先 `from daimon_runtime import setup_plot; setup_plot()` 再覆盖 rcParams（白底、无网格、Arial 7.2、300dpi）。
- Git Bash 下中文路径用正斜杠；`node run_latex.mjs` 的参数名是 `main` 不是 `entrypoint`。
- CiPA 数据坑全集见 `law_channelpedia/方法学存档_CiPA全药池pipeline冻结_2026-11-10.md`（1.1–1.22：单位后缀 lab_2 是安培、模板电压≠实测、lab_4 RS 名义值 80MΩ、官方 Hill 两例灾难 299.86/101.55 等）。
- LaTeX 源文件里通道名下划线必须转义（Kv10.1_003、Kv2.1_C1）。
