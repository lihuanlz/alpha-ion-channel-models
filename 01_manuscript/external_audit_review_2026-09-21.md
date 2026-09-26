# 四通道 α 模型 Nature 稿件严格审计报告

**审计对象**：`nature_main_v02_2026-09-21.md`（168 行）／`nature_si_v02_2026-09-21.md`（326 行）／`external_audit_package_v03_2026-09-21.md`（140 行）
**审计日期**：2026-09-21
**方法**：三方逐条比对（main↔SI↔审计包）＋外部 ground truth 实抓（DOI content negotiation、figshare/Zenodo/Dryad/OSF/PMC 元数据、GitHub API、PubMed/eUtils、web 检索）＋数字复算（Wilson CI、CV、计数闭环、Q10 指数换算）。
**总结**：数字层质量极高（见 B 类通过清单）；**事实层暴露 6 个 A 类硬伤**，其中三个（A1/A2/A3）属于"自家文件互斥或与外部事实冲突"型——恰好是审计包纪律声明第 3 条（所有主张可审计）要防的类型。共同根源：**外部数字对表做得很足，但外部论文内容本身没有被核验**。

---

## A 类：投稿时刻必须改（6 项）

### A1. Beattie 2018 细胞系系统性错误：HEK293/hERG1a/1b → 实为 CHO/hERG1a 【影响面最大】

**证据（双源铁证）**：
- Beattie 2018 论文摘要（doi:10.1113/JP275733，实抓 CSL）："currents evoked by a novel 8 second sinusoidal voltage clamp in **CHO cells overexpressing hERG1a**."
- figshare 4702546 数据集摘要："Current recordings in **CHO cells over-expressing hERG1a**."

**错误位置（9 处连锁）**：

| # | 文件 | 位置 | 原文 |
|---|---|---|---|
| 1 | main | L93 (i) | "nine **HEK293** cells expressing **hERG1a/1b**" |
| 2 | main | L34 | "the **HEK construction-set** reference of 3.04 ms" |
| 3 | main | L85 | "matches the **HEK reference** at 37 °C" |
| 4 | SI | L197 | "the **HEK sealed reference** 3.04 ms" |
| 5 | main ED6d | L160 | "37 °C **cross-host** check"（图题）+ "the **HEK sealed reference**" |
| 6 | main ED8c | L164 | "star, Beattie **HEK reference** 3.04 ms" |
| 7 | 审计包 | C4 | "跨宿主对表 τ_rec(−120) CHO 1.52 ms vs **HEK** 3.04 ms" |
| 8 | 审计包 | L54 数据源表 | "hERG1a/1b **HEK293** 九细胞" |
| 9 | SI | L197 段 | "Cross-host absolute check" |

**结构性后果**：整个 "cross-host check"（跨宿主检查）解释框架失据——Beattie 2018 与 Lei 2019 都是 CHO。1.52 vs 3.04 ms 的真实对比是**跨实验室、跨平台（手工膜片钳 vs 自动膜片钳）、跨协议族**的同宿主对表。封卷判线数字（ratio 0.50 ∈ [0.3, 3]）无需改动，但判线的物理解释层需要重述。

**修法**：
- main (i) → "whole-cell recordings from **nine CHO cells overexpressing hERG1a** at 37 °C"
- 全文 "HEK construction-set / HEK sealed reference / Beattie HEK reference" → "construction-set reference"（或 "Beattie reference"），7 处
- "cross-host check" → "**cross-platform absolute check**"（ED6d 图题、SI S2、审计包 C4）——这仍是强 claim：不同实验室、不同记录平台、37 °C 绝对时间尺度对表
- 审计包数据源表 L54 同步修正

**反向确认**：main (ii) Nav1.5 的 "HEK cells over-expressing Nav1.5" **正确**（figshare 27193878 摘要铁证："NaV1.5-expressing HEK cells"）——细胞系错误仅限 hERG 侧。SI S8 的 Zhou 1998 "HEK293" 也正确。

### A2. main Discussion 与 SI S9 直接矛盾：Kv4.3 数据集存在性

- main L85："for the remaining two (IK1/Kir2.1 and Ito/Kv4.3), **no public dataset with multi-protocol raw traces currently exists**"
- SI S9："For **Kv4.3**, **exactly one public raw-trace dataset exists** (Roeper et al., eLife 2025; Dryad doi:10.5061/dryad.76hdr7t6z; 2.10 GB … 17 vehicle and 15 6-OHDA cells)"

审稿人对照 S9 一眼抓到；Dryad 数据集已实抓确认存活（200，Roeper/Kovacheva/Shin/Mankel，Goethe Frankfurt，2025）。

**修法**（main L85）→ "…no public dataset meets the two-source corroboration standard applied here (Kir2.1 has none; Kv4.3 has exactly one, precluding cross-laboratory corroboration), so their exclusion … is a documented data-availability boundary rather than an omission (SI S9)."

### A3. ref 17（TCS 预印本）标题张冠李戴

main ref 17 写 "Time-scale calibration and scale-free coordinates in biological measurement"。实抓 chemrxiv-2024-19rj6/**v10**（posted 2026-08-04）真实标题：

> **"Scale Degeneracy and Its Resolution: The Route to Calibration-Free Absolute Quantification"**

chemrxiv 主 DOI 的 v1 摘要确为早期"digital immunoassay techniques"版本——"Time-scale calibration and scale-free coordinates" 不匹配任何已知版本。审稿人点开 DOI 即穿，且像幻觉引用。

**修法**：标题替换为 v10 真题；若引用时 V11 已上传（新题 "Scale Degeneracy: The Physical Origin of Calibration in Molecular Binding"），改引 v11＋新题；[TCS preprint] 占位符一并规范为 Nature 格式。引用落点（标度简并→尺度自由坐标）概念成立，纯标题错误。

### A4. Nav1.5 外部数据源不可追溯 + 69/86 口径分裂

- main (ii) 称 τ_h(−40) 分布 "sources listed in SI S6"——**SI S6 无任何具体文献**，只有 Nα-4 判词卡指引
- 审计包数据源表给 "Tarasov 2026, Dryad（见 Nα4 判词卡来源清单）"——**无 DOI**；外部检索不可见（搜索阴性，无法正面确认该数据集公开存在）
- main refs 1–19 无 Tarasov 条目；Nα 药理 63 条件（abstract 四通道药物结论的支撑之一）同样只存在于审计包
- **口径分裂**：main/S1.2f "69 recordings/patches"（median 1.03、CV 0.772）vs ED8d 图注 "69 multi-channel + 14 single-channel + 3 whole-cell = 86 点"——分布统计基于哪个集合？图与正文各说各话

**修法**：①main/SI 落地 Tarasov 2026 完整引用＋DOI（从判词卡回抄）；②正文或图注声明 86 vs 69 口径（统计基于 69 multi-channel；14+3 为叠加显示的对照层）；③投稿前自查该 Dryad DOI 公开可及（外部索引目前搜不到）。

### A5. ED8 资产名 v01/v04 三方矛盾（含 main 文件内部自相矛盾）

- main ED8 图注尾（L164）："figures_v01/EDFig_percell_violin_**v04**" ← 描述九面板内容（a–i，含药理臂 e/f/h/i）＝v04
- main 尾注：EDFig_percell_violin_**v01** = ED8
- SI 尾注：EDFig_percell_violin_**v01**
- 审计包 v03：EDFig_percell_violin_**v04**（九面板，v01–v03 留档）

main 同一文件内两处互斥。外部审计者按尾注取 v01 会发现只有旧面板，与审计包承诺的九面板不符。

**修法**：main 尾注＋SI 尾注统一改 v04。

### A6. Abstract 全称声明被自家 SI 反驳

abstract："**Every parameter is a measured quantity**"（无拟合卖点）。SI 反例：
- IKs E_K = −85 mV（**registered assumption**，S1.4）
- Nav E_rev（I–V fit，**weakly identifiable**，S1.2e）
- CaV E_rev 文献锚 +46 mV（S1.3d）
- IKs β = 0.30（**pool constant, registered**）

全称量词一戳即破。

**修法**→ "every governing quantity is either measured directly or an explicitly registered assumption"——保住"零拟合门"卖点，堵住反例。

**已知 TODO 确认**（main 尾注 Open TODOs 自登记，不重复列 A）：refs 9/11 与 4/14 合并、首引重排、ref 15 Montnach 全引、Code DOI、致谢、作者贡献、作者列表、ED3–5 渲染、model62 外部名——清单与审计包形式层基本对齐，唯一低估见 B13。

---

## B 类：核验问题（14 项）

| # | 问题 | 位置 | 修法 |
|---|---|---|---|
| B1 | Authier 2019 期刊字段错：写 *J. Pharmacol. Toxicol. Methods*，PMC6479253 实际为 *Assay Drug Dev Technol*（2019 Apr 17；标题 "Electrophysiological and Pharmacological Characterization of Human Inwardly Rectifying K(ir)2.1 Channels on an Automated Patch-Clamp Platform"） | SI S9 | 期刊名替换 |
| B2 | Li et al. 2016 悬空引用：S8 三处使用（Fig. 3D、Fig. 4、"per Li et al. 2016"）但全库无完整条目，main refs 亦无 | SI S8 | 补全条目（Vandenberg/Zhou 数值的转录源，必须可溯源） |
| B3 | Vandenberg 值转录源归属分裂：ED6a 说 "transcribed from the tabulation in **ref. 19**"（Lei 2019 II），S8 说 "verbatim from **Li et al. 2016, Fig. 3D**" | main ED6a vs SI S8 | 统一为一级来源（Li 2016 或 Vandenberg 2006 原文） |
| B4 | Q10 2.84 归属撕裂：main "at −120 mV"；SI 封卷口径 "τ_rec(−140/−120 mV)"（合并）；验证层网格 −140 = 2.83 / −120 = 2.85。0.01 差＋单电压 vs 合并口径不一致 | main L34 | main 改 "Q10 = 2.84 (pooled −140/−120 mV)" 或对齐封卷卡原口径 |
| B5 | abstract "an independent 670-well automated-patch panel" 把 Lei 2019 I（211，25 °C）与 II（459，温度系列）两个数据集合并为单一 panel；Methods 是分开的 | abstract | 改 "an independent five-temperature automated-patch programme (670 wells)" |
| B6 | 30 drugs（hERG panel 分析）→ 24 drugs（Stage B 打分）→ 68 traces → 136 datasets → 517 cell-units 的数字关系在 main 无交代；"the CiPA panel" 一词两用（30 药库与 24 药验证集） | main §drug/Stage B | 一句话交代 24 为 CiPA 官方验证子集 |
| B7 | CiPA 档案数据可用性无具体入口："as distributed within the CiPA in silico validation materials" | main Data availability | 补 URL/DOI 或指明随 Li 2017/Dutta 2017 的 SI 分发 |
| B8 | −92.9 mV 措辞两处皆过强：main "matching"（−92.0 实测，差 0.9 mV）；SI "brackets"（−92.9 不在区间 [−92.0, −83.9] 内，数学上不成立） | main L34 / SI L197 | 统一 "within 1 mV of the model62-chain fitted value" |
| B9 | Host 检查 J1–J6 清单不完整：main Methods "include…"（非穷尽），SI S5 只展开 4 项（APD90/追溯迹/IKr 2.06/qNet）；main 独有数字（IKs peak ratio 1.52、"every other current agrees within 2%"）在 SI 无落点 | SI S5 | 补全六项定义与数值 |
| B10 | CV 0.772 双重出现：S1.2f 69-patch 分布 CV = 0.772 与同节 cell-attached "τ_decay CV 0.772"——两个不同样本同一数值到三位小数，疑似复制错误 | SI S1.2f | 回源卡核对 |
| B11 | Nav CV 0.029 复算不符：1.63/1.59/1.69 ms 的样本 CV = 0.031、总体 CV = 0.025，0.029 两者皆非 | main L23/SI S1.2a 表 | 回源卡核对（可能源卡有更多小数位） |
| B12 | 审计包自称"自洽"但 Tarasov 入口断链（"见 Nα4 判词卡"），包内未附 DOI | 审计包§2 | 并入 A4 修复 |
| B13 | 缺图范围低估：审计包形式层只登记 ED3–5 未绘制，main 标注 "Figure legends (**planned; figures to be drawn**)"——主图 Fig 1–5 与全部 ED 均未绘制 | 审计包§5 | 形式层登记改为"全部图未绘制"（外部审计方需要知道交付形态） |
| B14 | 配对检验口径：main "paired Wilcoxon" vs SI "Wilcoxon"（SI 的 93% cells above own-control drift 支持配对设计） | main/SI S4 | 统一写 paired |

---

## C 类：语言修缮（4 项）

1. SI S1.3：w_fast = 0.289 与 Δf = 0.293 "consistent **to three significant figures**"——三位有效数字下两者不同（0.289 ≠ 0.293），是两位（0.29）。改 "agree within 1.4%"。
2. E_rev(T)：端点差 −8.1 mV / 12 °C = −0.675 mV/°C vs SI 拟合斜率 −0.58 mV/°C。审稿人可手算出不一致。SI 注明斜率为五温度回归值、端点差另列。
3. main "within a factor of two"（ratio 0.50 = 恰好 2.0×，压线成立）。建议直接给数 "ratio 0.50 (a factor of exactly two)"。
4. Abstract 恰好 200 词整（实测）——达标但零余量；期刊字数算法（连字符、数字）可能溢出，建议留 2–3 词余量。

---

## B 类通过正清单（给用户吃定心丸）

**数字闭环（全部复算通过）**：
- Wilson CI 四组：7/9 → 45–94%；8/9 → 57–98%；65/68 → 88–99%；45.8% (287/627) → 42–50% ——全部数学正确
- 计数闭环：lab 20+16+29 = 65/68 ✓；211+127+118+109+105 = 670 ✓；69 组装 − 1 排除 = 68 CiPA ✓；E_rev 失败 2+1+0+1+3 = 7/670 与排除登记册闭环 ✓（ED8a n = 209/126/118/108/102 逐温度对上孔数−失败数）
- 主表分解：hERG 18,353 + Nav 323 + IKs 24 + CaV 10 = **18,710** ✓；臂表 94 + 204 + 63 = **361** ✓（CaV 204 = 10 对照 + 97×2，98−1 两值俱缺已登记）✓；登记册 73+9+5+2+1 = 90 ✓
- CV 复算：k_h 5.45/4.01/4.91 → CV 0.152≈0.15 且均值 = 封卷 4.79 ✓；τ_h(−40) 1.64/1.68/3.15 → CV 0.40 ✓
- Q10 链自洽：τ_rec(−120) 25→37 端点比 3.49 → Q10 = 3.49^(10/12) = 2.83 ≈ 封卷 2.84 ✓；Vandenberg 3.45 → 2.60 ✓；ED6a 五温度星标与 S8 外锚互证 ✓
- 三方数字矩阵：main↔SI↔审计包 C1–C11 逐项对齐（ρ 三臂 4–6 位一致、ε_x 0.088/0.223、|κ*−1| 0.020、|κ*| 0.20、Δf +0.12/+0.168/−0.088、APD90 0.36%、qNet 0.998、τ_act 0.016、χ 7.4×10¹²、0/451、17–112×、4.8–2000 s、Mef 15/15 中位 +0.296）✓
- 封卷零改动声明三方一致（v02/v03 变更链）✓；失败归因 main–SI 分层一致（047/060 data-side、014 model-side）✓

**外链实抓（8/8 存活）**：figshare 4702546（Beattie）／figshare 27193878（Lei 2025 数据）／Zenodo 8226585／GitHub CardiacModelling/hERGRapidCharacterisation（200；data-autoLC、temperature-dependency 目录印证 Methods 描述）／OSF g3msb／Dryad 76hdr7t6z（Roeper）／PMC10829594／chemrxiv v10

**外部文献核验（10 条通过）**：Lei 2025 *Adv Sci* 12(30):e00691, doi:10.1002/advs.202500691 ✓；Chan 2023 *eLife* 12:e87038（PMC10501768）✓——且 **Zenodo 8226585 归属确认**：九个 creator（Magnus/Harutyun/Jodene/Daniel/Yundi/Ying/Marc/Vitya/David）与 eLife 作者九人一一对应（=Chan M/Sahakyan H/Eldstrom J/Sastre D/Wang Y/Dou Y/Pourrier M/Vardanyan V/Fedida D），是 Zenodo 侧把名当姓存了元数据，稿件引用无误；Fedida 2024 *JGP* 156(3) 标题卷期全对 ✓；Beattie 2018 *J Physiol* 596(10):1813–1828 ✓；Vandenberg 2006 *AJP Cell Physiol* 291:C165–175 ✓；Oliveira-Mendes 2023 *Clin Transl Med* 13:e1266 ✓；Zhou 1998 *Biophys J* 74:230–241（HEK293 标注正确）✓；Lei 2019 I/II *Biophys J* 117:2438–2454/2455–2470 ✓（hERGRapidCharacterisation repo 互证）；Nav1.5 "HEK, rupture-patch" ✓（figshare 铁证）

**无法外部核验（需用户自查）**：Tarasov 2026（Dryad 入口在判词卡，外部索引未见）——见 A4

---

## 审计包自评六处攻击点 vs 本次发现

用户自评的六处攻击点（n=9 外推、Q10 封卷范围、CDI 对表独立性、时间扭曲 15%、IKs 小样本、ED8 一致性）全部是**数字/统计层**风险——该层本次审计确认质量过硬。本次抓到的 A1（Beattie 细胞系）、A2（Kv4.3 矛盾）、A3（TCS 标题）全部在**事实层**（外部论文内容、自家 SI 互斥），是自评盲区。

**修复优先级**：A1（9 处连锁＋解释框架重述，30 分钟）→ A3（1 分钟替换）→ A2（一句话）→ A5（两个尾注）→ A6（abstract 措辞）→ A4（回判词卡拿 DOI，依赖源文件）。

*凡审计指出的数字不一致，以源卡重算为准回写稿件——按审计包末节承诺执行。*
