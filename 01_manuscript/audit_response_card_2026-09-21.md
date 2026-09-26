# 审计应答卡 · 外部审计《四通道α模型_Nature投稿审计_2026-09-21》

**日期**：2026-09-21 晚
**应答方**：作者侧（本机管线）
**对应修复版本**：`Nature_main_v03_2026-09-21.md` / `Nature_SI_v03_2026-09-21.md` / `external_audit_package_v04_2026-09-21.md`
**总判**：审计 6 A + 14 B + 4 C 共 24 项，**全部可解答**。22 项按审计修法直接落地；2 项（B10、B11）源卡复核后证明原数字正确、系显示精度问题，仅澄清措辞未动数字。**另超额逮住 3 处审计未列的问题**（见末节）。封卷数字零改动；所有替换脚本化、逐条断言唯一命中（`_make_v03_main.py` 24 处 / `_make_v03_si.py` 15 处 / `_make_v04_auditpkg.py` 5 处，留档可复跑）。

---

## A 类：6/6 全部修复

| # | 审计发现 | 我们的复核 | 修复落点 |
|---|---|---|---|
| A1 | Beattie 2018 细胞系错（HEK293/hERG1a-1b → 实为 CHO/hERG1a），9 处连锁 | **确认属实**。独立实抓 Wiley 摘要（JP275733）："whole-cell patch-clamp voltage-clamp experiments using **CHO cells stably expressing hERG1a**"，双源铁证成立。Nav1.5 侧 HEK 标注复核**正确不动**（figshare 27193878 摘要 "NaV1.5-expressing HEK cells"；Zhou 1998、Oliveira-Mendes 2023 的 HEK293 亦正确保留） | main v03：Methods (i) 改 "nine CHO cells overexpressing hERG1a"；L34/L85 "HEK reference/construction-set" 改 "construction-set reference"；ED6 图题与 d 面板、ED8c 同步；"cross-host check" 全部改 "**cross-platform absolute check**"（同宿主、跨实验室/跨平台/跨协议族——claim 强度不降级）。SI v03 S2 同改。审计包 v04 C4 与数据源表同步。**封卷判线 1.52/3.04/0.50 一字未动** |
| A2 | main 与 SI S9 对 Kv4.3 数据集存在性互斥 | 确认属实 | main v03 Discussion 改按审计修法：Kir2.1 无公共数据集、Kv4.3 恰有一个（无法满足双源互证标准），排除系记录在案的数据可得性边界（SI S9） |
| A3 | ref 17 TCS 标题张冠李戴 | 确认属实（chemrxiv-2024-19rj6/**v10** 真题 "Scale Degeneracy and Its Resolution: The Route to Calibration-Free Absolute Quantification"） | main v03 ref 17 换真题 + 补 (2026) |
| A4 | Nav1.5 外部源不可追溯 + 69/86 口径分裂 | **可解答且超出审计预期**：Tarasov 数据集元数据已从 Dryad API v2 实抓——`doi:10.5061/dryad.0cfxpnwgh`（9.59 GB，curationStatus Published，2026-05-13），论文 *Nat. Commun.* doi:10.1038/s41467-026-72387-8（Tarasov M et al., Ohio State）。**审计"外部索引搜不到、需用户自查"一项由此关闭**。69/86 口径：分布统计基于 69 个多通道膜片；14 单通道 + 3 全细胞仅叠加显示 | main v03 新增 ref 21（Tarasov 全引 + 双 DOI）；Methods (ii) 写明统计/叠加口径；数据源计数 Six→Seven；Fig 3b 图注补 "multi-channel cell-attached patches (ref. 21)"。审计包 v04 数据源表同步补 DOI |
| A5 | ED8 资产名 v01/v04 三方矛盾 | 确认属实 | main v03 + SI v03 两处尾注统一 v04 |
| A6 | Abstract "Every parameter is a measured quantity" 被自家 SI 反例戳破 | 确认属实（E_K 注册假设、E_rev 弱可识别、β 注册池常数） | main v03 abstract 改审计修法原句；同时把摘要从 200 词整修到 **198 词**（见 C4） |

## B 类：14/14 全部解答

| # | 结论 | 落点 |
|---|---|---|
| B1 | 期刊错确认——且**作者也错**：PMC6479253 正文自引 "Authier et al. 2017" 为其参考文献第 29 条，该文本身是 Sanofi 团队作品；正确条目为 **Sanson et al. 2019, *Assay Drug Dev. Technol.* 17, 89–99**（uu.nl 学位论文参考书目 A 级源给出卷期页）。审计逮到期刊、我们补逮作者 | SI v03 S9 |
| B2 | Li et al. 2016 悬空引用确认 | SI v03 S8 补全条目（J. Pharmacol. Toxicol. Methods 81, 233–239, 2016）；main v03 同步列为 ref 20 |
| B3 | Vandenberg 转录源归属分裂确认 | 统一为 "Vandenberg et al. 2006 values as tabulated in Li et al. 2016 (ref. 20), Fig. 3D"（main ED6a 与 SI S8 一致） |
| B4 | Q10 口径确认 | main v03 改 "Q10 = 2.84, pooled −140/−120 mV"，对齐封卷卡原口径 |
| B5 | 670-well 合并表述确认 | abstract 改 "an independent five-temperature automated-patch programme (670 wells)" |
| B6 | 数字链已交代：**24 = 30 药库 ∩ CiPA 官方 28 药风险标签表（高 7/中 11/低 6）**（源卡：双协议前向收官判词 §21，newCiPA.csv sha256 在案） | main v03 Stage-B 段首句补此关系 |
| B7 | CiPA 入口具体化：随 refs 2/3 的 in silico validation 材料分发；风险标签/Cmax 为官方 newCiPA.csv，SHA-256 随判词卡落盘（不写无法核实的 URL） | main v03 Methods + Data availability |
| B8 | "matching/brackets" 两处过强确认（−92.9 不在 [−92.0, −83.9] 内） | main v03 + SI v03 统一 "within 1 mV of the model62-chain fitted value" |
| B9 | J1–J6 已从源卡（二审材料 2026-09-16）补齐六项定义+数值+判线全表（J1 0.005 ms；J2 APD90 251.206/0.36%、APA 128.51、RMP −87.99、qNet 0.056；J3a 1.1e-5；J3b 四项；J3c 预期失败签名；J4 240.7/236.6；J5 单调+夺获；J6 IKr 2.06、IKs 1.52、sanity 0.979–1.009） | SI v03 S5 表格化 |
| B10 | **源卡复核：不是复制错误**。Nα-4 结果 JSON：τ_decay CV = 0.7720552839346476（n=69 多通道系综）与 Po_peak CV = 0.6023794036828812（n=14 单通道）是两个不同量；两个 "0.772" 是**同一个 n=69 统计量在两处各引一次**（分布口径 + m 门关闭依据），不是两个样本撞数 | SI v03 S1.2f 措辞澄清，数字不动 |
| B11 | **源卡复核：0.029 正确**。Nα-2 结果 JSON 全精度值 1.633956381340646 / 1.5944765804158043 / 1.6881400606157897 → 样本 CV = 0.02869 ≈ 0.029；审计用显示值 1.63/1.59/1.69 复算才会得 0.031/0.025 | SI v03 S1.2 脚注补四位小数 + "sample CV from full-precision values"，数字不动 |
| B12 | 并入 A4，已随 ref 21 + DOI 关闭 | main v03 / 审计包 v04 |
| B13 | 缺图范围低估确认 | 审计包 v04 形式层改 "主图 Fig. 1–5 与全部 ED 均未绘制"（ED3–5/7/8 有落盘资产图，ED1/2/6 仅图注） |
| B14 | 确认（93% 超自身对照漂移 = 配对设计） | SI v03 S4 改 "paired Wilcoxon" |

## C 类：4/4 全部修复

1. **C1**：0.289 vs 0.293 三位有效数字确实不同——改 "agreeing with Δf = 0.293 within 1.4%"（SI v03 S1.3b）。
2. **C2**：SI v03 注明 −0.58 mV/°C 为五温度回归值，端点差 −0.68 mV/°C 另列。
3. **C3**：main v03 改 "ratio 0.50, a factor of exactly two"。
4. **C4**：abstract 200 → **198 词**（A6/B5 修复同时净 −2 词，零余量问题解决）。

## 超额自逮（审计未列，同源同纪律）

1. **Sanson 作者名**（见 B1）：审计修正期刊但未及作者。
2. **ED3d "61 多通道膜片" vs 全文 69**：ED3d 散点用的是同时具备 τ 与 late% 的 61/69 子集——图注已写明口径，避免下一个审计者当矛盾抓（main v03 ED3d）。
3. **main "every other current agrees within 2%"**：J6 sanity 里 ICaL = 0.979 实为 2.1% 偏差——改 "within 2.1%"（main v03 Results）。

## 仍未完成（与审计 TODO 对齐，投稿前处理）

refs 9/11 与 4/14 合并、首引重排（v03 新增 refs 20/21 后一并重排）、ref 15 Montnach 全引、Code DOI、致谢、作者贡献、作者列表、ED1/2/6 绘制、model62 外部名。

---

*执行纪律复核：全部 44 处替换经脚本断言唯一命中；摘要词数机器校验 198 ≤ 199；封卷数字零改动（逐替换项目检确认无数字串触碰）；外部事实仅引用本回合实抓来源（Wiley JP275733 摘要、Dryad API v2 元数据、uu.nl 书目、PMC6479253 正文）。*

---

## v04 增补（2026-09-21 晚，Tarasov 机制互证）

**触发**：用户核文献时确认 Tarasov 论文（*Nat. Commun.* 17, 6218 (2026)，doi:10.1038/s41467-026-72387-8，nature.com 实抓复核：开放获取、2026-05-08 发表）全文内容与本稿 Nav1.5 负结果直接互证。

**互证内容**：Tarasov 原文证明同一批记录里膜片内的 Nav1.5 并不独立门控——关闭态通道间相互作用以簇大小依赖方式稳定慢失活（多通道膜片 late 电流减小、用依赖性阻断利多卡因效力下降）。这为 S1.2f 登记的 τ_h(−40) 宽分布（median 1.03 ms，CV 0.772，0.31–6.64 ms）及其与 late 电流分数的相关（ρ = 0.44）提供了**独立物理机制**：膜片间簇组成差异使宽分布成为物理预期而非异常。

**执行**：
- main v03→v04：负结果段插入机制互证句（"That reading is independently corroborated by the source study's own analysis of the same recordings²¹…"），Nav1.5 负结果从"数据性质"升级为"机制锚定的物理性质"；ref 21 卷期补全（17, 6218）；
- SI v03→v04：S1.2f 补 "Mechanism corroboration (post-seal, registration-grade)" 段；
- 封卷判词、全部数字零改动；新增内容均为登记级解释层。

**严谨性说明**：互证仅 claim 到"consistent with / provides a physical basis"层级——Tarasov 原文报告单通道与多通道的**平均**衰减 τ 相近，并未报告 τ 的跨膜片散布本身；我们的散布量化（从其所存提取表独立算出）与其机制互补而不重复，无循环论证（我们的判词基于其数据表的自有统计，其机制基于其独立的 MINFLUX 结构证据与药理学实验）。
