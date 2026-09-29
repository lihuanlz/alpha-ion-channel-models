# 侦察卡：Kir2.1 与 Kv4.3 公开数据集可行性

**日期**：2026-09-21
**任务**：评估 IK1（Kir2.1）与 Ito（Kv4.3）能否复刻 hERG 的"双数据集跨实验室互证"模式上 α 管线（extract–judge–assemble）。
**侦察标准**：① 同一细胞多协议原始电流轨迹（激活/失活/去激活）；② 多细胞群体（n≥10，最好上百）；③ 原始数据公开可下载。两通道各需两组独立来源才能互证。

---

## 一句话结论

**Kir2.1：公开原始数据 0 组，且生物物理上与 α 管线的"时间常数旅行"主线适配度低——暂缓。**
**Kv4.3：公开原始数据仅 1 组（Roeper 实验室 eLife 2025，Dryad 2.1 GB 原始电压钳轨迹），无第二独立来源，跨实验室互证目前凑不齐——可作单源可行性探针，不能作验证章节。**

---

## 一、Kir2.1 / IK1 侦察明细

### 候选清单

| 来源 | 内容 | 公开性判决 |
|---|---|---|
| Authier et al. 2019, J Pharmacol Toxicol Methods（Sanofi，QPatch，CHO-hKCNJ2，n=65 细胞完整 I–V，Ba²⁺/ML133/PA-6/氯喹药理）| CiPA 面板 Kir2.1 方法学正典 | **无数据可用性声明，原始轨迹未公开**。作者全部为 Sanofi 在职/前员工，数据为企业内部资产 |
| K Li 2025（Europe PMC, PMC12191713）SyncroPatch 384PE 高通量 APC | 384 孔原始轨迹截图 | 通道是 Kir6.1/Kir6.2（KATP），**不是 Kir2.1**；且仅截图 |
| Zenodo 检索 "Kir2.1"（10 条） | 命中：NaV1.5-KIR2.1 复合物**计算模型**（2024，无实验轨迹）、巨噬细胞 Kir2.1-DRG 神经元偶联（非心脏）、Kir2 质子化门控（单通道/结构方向） | 均不符标准①②③ |
| 其他（hypoxia-SUMO-PIP2 卵母细胞巨膜片；zacopride 大鼠心肌 IK1） | 体制不符 | 原始数据未公开 |

### 管线适配度分析（即使拿到数据也要面对的问题）

- Kir2.1 **几乎没有电压门控**：内向整流来自胞内 Mg²⁺/多胺的亚毫秒级瞬时阻塞，稳态 I–V 是主要可观测量；α 管线在这通道上能提取的是**稳态表 + 每细胞 G + E_rev([K⁺]out)**（零电流电位随 [K⁺]out 移动 54 mV/decade，Sanofi 实测，接近 Nernst 58）。
- 这意味着 Kir2.1 上**没有"时间常数旅行"故事可讲**——它天然是检验"midpoints per-cell / 散布真实"命题的极简对照，但对正文主线（恒等式 + 时间尺度常数化）贡献薄。
- **判决：暂缓。** 若要补强，唯一现实路径是向 Sanofi 请求 QPatch 原始数据或等待 CiPA phase II 数据发布；性价比低。

## 二、Kv4.3 / Ito 侦察明细

### 候选清单

| 来源 | 内容 | 公开性判决 |
|---|---|---|
| **Roeper/Kovacheva/Shin/Mankel 2025, eLife（Goethe University Frankfurt）**；Dryad doi:10.5061/dryad.76hdr7t6z，**2.10 GB** | 小鼠黑质多巴胺神经元急性脑片全细胞电压钳，**Kv4.3 介导 A 型电流原始轨迹（.mat）**：激活协议（−80 mV 预压 500 ms → −60…−20 mV 步阶 5 mV 递，1 s）+ 失活协议（−120…−20 mV 10 mV 递 1 s → 固定 −20 mV 1 s）+ HCN 协议；vehicle 组 17 细胞、6-OHDA 组 15 细胞；37 °C；20 kHz 采样 | ✅ **原始轨迹公开可下载**，满足标准①②③。**唯一命中** |
| Nanion SyncroPatch 384PE 应用笔记（CHO-Kv4.3，CiPA 协议，奎尼丁/美托洛尔/氟卡尼 IC50） | 网页截图 | ❌ 原始文件不可下载，仅厂商展示图 |
| Sophion QPatch 应用笔记（Kv4.3/Kir2.1 心脏钾通道面板，失活 V½ = −39.5 mV） | 网页摘要 | ❌ 同上 |
| Frontiers in Pharmacology 2025（Trovato 2022 流程，QPatch 测 hKv4.3 等四通道 × 37 化合物） | IC50 汇总表 | ❌ 只有 IC50 表，无原始轨迹 |
| PLOS ONE 2015（Kv4.2-/- 小鼠心肌 Ito 三指数拟合）；JMCC 2021（iPSC-CM 表达 Kv4.3/KChIP2.1，τ_fast 7.66 ms） | 方法学论文 | ❌ 原始数据均未公开 |
| Zenodo 检索 "Kv4.3"（10 条） | 仅 1 条 KCND3 遗传变异论文 | ❌ 无电生理原始数据 |
| Ma et al. 2022（Kv4.3±KChIP1/2/DPP6 冷冻电镜 + HEK 膜片钳） | 结构 + 门控曲线 | ❌ 只公开 EMDB/PDB 坐标，电生理轨迹未公开 |

### 管线适配度分析

- **生物物理适配度：A 级。** Ito 有激活 + 多指数失活 + 恢复（τ_rec 几十 ms 到数秒），动力学富矿，是 α 管线理想的下一个对象。
- **但唯一公开源是神经元原生体制**：黑质多巴胺神经元里的 A 型电流（含 KChIP/DPP 辅助亚基天然环境，AmmTx3 毒素定义 Kv4.3 贡献），电压窗偏阈下（−60…−20 mV 激活），温度 37 °C。可作"管线能否一天内吃下一个新通道原始轨迹"的可行性探针，但**不能直接当"心脏 Ito 跨实验室验证"**——物种、细胞类型、电压窗都不对齐，且没有第二组互证。
- **判决：登记为数据缺口。** 选项有二：
  - A. 下载 Dryad 2.1 GB 跑单源探针（extract τ_inact(V)、h∞(V)、每细胞散布），产出"管线可移植性"证据，明确标注单源不可互证；
  - B. 等待/索取第二来源（向 Nerbonne、Backx 或 Roeper 类实验室索取，或监测 CiPA phase II 心脏 Kv4.3 面板数据发布）。

## 三、给正文章节的含义

- **hERG/Nav1.5/Cav1.2/IKs 四通道主线不受影响**：本次侦察结论是"扩展暂时无弹药"，不是"现有结果有问题"。
- 若审稿人问"为何不覆盖 IK1/Ito（CiPA 全面板）"：可如实应答——CiPA 面板另两通道的**多协议原始轨迹目前无公开数据集**（本卡即证据链），这正是领域数据开放度的现状，也反向凸显 Beattie/Lei 系 hERG 数据与 IKs/Nav/Cav 各组数据的可贵。
- 若走 Kv4.3 选项 A 并跑出正面结果，可在 SI 加一节"管线对 Kv4.3 家族 A 型电流的可移植性演示（单源、神经元体制）"，措辞必须框死边界。

## 四、证据链接（第三方归属）

- Sanofi Kir2.1 QPatch 方法学：Authier et al., J Pharmacol Toxicol Methods 2019, https://pmc.ncbi.nlm.nih.gov/articles/PMC6479253/
- Kv4.3 唯一公开原始数据集：Roeper, Kovacheva, Shin, Mankel (2025), Dryad, https://doi.org/10.5061/dryad.76hdr7t6z （论文版：eLife, https://elifesciences.org/articles/104037）
- Nanion Kv4.3 CiPA 应用笔记（仅截图）：https://www.nanion.de/ （SyncroPatch 384PE application data）
- Sophion 心脏钾通道 QPatch 应用笔记（仅摘要）：https://sophion.com/publication/whole-cell-patch-clamp-recording-for-cardiac-potassium-ion-channels-kv1-5-kvlqt1-mink-kv4-3-and-kir2-1-on-qpatch-automated-patch-clamp-system/
- Frontiers in Pharmacology 2025（QPatch hKv4.3 × 37 化合物，仅 IC50 表）：https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2025.1671199/abstract
