# 判词卡：Lei 211 细胞 α 群体判决（收官）

2026-09-21 12:40 · 细胞线 4 hERG · 数据集 Lei et al. 2019 Part I（*Biophys. J.* 117:2438–2454）
herg25oc1（CHO，25 °C，autoLC + Lei 官方再漏减）· 判线：预注册判决卡_Lei211_α群体_2026-09-21.md（冻结 + 跑前修订 A1–A4）

**一句话：α 模型的两件硬货在独立实验室、独立宿主、独立温度、23 倍细胞数下原样站住——h_ss 负端阶梯逐细胞恒定（CV≤0.11，211 孔）与逐细胞 E_rev（CV 0.024，中位 −83.9 mV，与 149 拟合 −83.81 互证）；AP 前向量级与频率方向全对（A2 96%、A3 100%、回弹峰比中位 0.84），逐周期回弹判线在小周期数下结构性苛刻，合并双过 45.8% → 登记。**

## 逐条登记

| 判决项 | 内容 | 结果 | 判定 |
|---|---|---|---|
| J4 | E_rev 逐细胞（staircase 下坡过零） | n=209/211，中位 −83.90 mV，均值 −83.50，SD 2.02，CV 0.0242，范围 [−88.4, −72.6] | **封卷**（CV<0.10 且中位 ∈[−95,−70]） |
| J1 | h_ss(−140) | n=211，中位 1.000，CV 0.000 | **封卷** |
| J1 | h_ss(−120) | n=207，中位 0.955，CV 0.065 | **封卷** |
| J1 | h_ss(−100) | n=201，中位 0.887，CV 0.109；但 C1 回收误差 10.7%>10% → 统计量该档作废 | 不作废部分按规则不判决；数值仅供参考 |
| J1 | h_ss(−60…+40) | 中位 0.230→0.009→0.022，CV 0.38–2.34（小信号淹没，DF 校正放大噪声） | **登记**（逐档数值入总图 b） |
| J1-B1 类似 | 负端饱和 | −140/−120/−100 中位 1.000/0.955/0.887 全在 [0.7,1.3] | **负端饱和支持** |
| J2 | m_ss(+25/+40) | n=208/211，中位 0.623/1.000，CV 0.219/0.000 | **封卷** |
| J2 | m_ss(−50…+10) | 中位 0.104/0.104/0.106/0.131/0.271，CV 0.49–0.92；1 s 表观激活下界（25 °C 未达稳态，§1 已声明） | **登记** |
| J2-τ_act | sactiv 测试段上升沿 | 三孔冒烟全档 r²<0.9（激活+失活叠加非单调，修订 A4 已降级） | 判线作废，改源 staircase 上跳，全部登记 |
| J3 | τ_deact 快分量 τ1 | −60：n=210 中位 16.7 ms CV 1.19；−40：n=211 中位 16.7 ms CV 1.39 | **登记** |
| J3 | τ_deact 慢分量 τ2 | −60：n=210 中位 0.74 s；−40：n=211 中位 3.03 s，方向 τ(−40)>τ(−60) ✓ 阶梯方向成立 | **登记** |
| J3-τ_rec | sinactiv DoE 负档族 | −140:4.31 ms(CV 0.28,n=211)；−120:5.31(0.22,211)；−100:6.54(0.77,208)；−80:8.06(1.10,103)；−60:15.07(0.47,200)；−40:16.82(0.32,206)；−20:15.07(0.52,165) | **登记**（τ_h(V) 表入 J5） |
| J3-τ_act/inact | staircase 大补跳 −80→+40 | τ1=90.0 ms（失活，n=202），τ2=431.8 ms（激活，n=202） | **登记** |
| J5 | AP 前向 A1（复极回弹逐周期） | 0.5 Hz 77/209、1 Hz 53/209、2 Hz 157/209 | 见下 |
| J5 | A2（RMS 量级比∈[0.3,3]） | 191/209、202/209、206/209（≈96%） | 过 |
| J5 | A3（频率分级方向） | **209/209（100%）** | 过 |
| J5 | 合并判决 | 627 试验双过 287 = 45.8%，Wilson 95% CI [0.419, 0.497]，下界 <0.6 | **登记**（未达封卷判线） |
| J5 | 回弹峰比（中位） | 0.844，IQR [0.714, 1.007]，远在 [0.3,3] 内 | 物理量级正确 |
| C2 | 管道完成率 | 209/211（99.1%，C18/M19 两孔 E_rev 未过零，照实登记） | **过** |
| C1 | 合成回收（J1 统计量） | −140: 0.0% ✓、−120: 3.8% ✓、−100: 10.7% ✗、−60: 12.8% ✗、−40: 3.2% ✓、−20: 32.1% ✗ | **部分不过**（见后果） |

## C1 部分不过的后果（按预注册规则执行）

J1 统计量在 −100/−60/−20 档回收误差 >10%，该三档统计量作废不判决（−40 档 3.2% 通过但群体 CV 大 → 维持登记）。**J1 封卷肢体收缩为 {−140, −120}**。失效机理 = 等时近似在恢复/去激活竞速档产生 +10~30% 偏置（§5.3 已声明方向）。

## J5 登记的结构性分析（不作翻案依据，只作记录）

A1 判线"≥80% 周期出现回弹"继承自 Beattie（17 周期）；本数据集 0.5 Hz 仅 2 周期、1 Hz 仅 4 周期，一个边际周期低于 3σ 即整细胞判负：全周期命中率 0.5 Hz 77/209、1 Hz 53/209、2 Hz 0/209（需 6/7 故 2 Hz 反高）。回弹物理本身无问题：峰比中位 0.844、A2 96%、A3 100%。**结论：前向量级与方向封卷级，逐周期判线在小周期数下结构性苛刻 → 照预注册登记，不改判线。**

## 与 Beattie 线的分层对照（§0 声明口径）

| 结构性质 | Beattie（HEK/37 °C/9 细胞） | Lei（CHO/25 °C/211 细胞） | 复制？ |
|---|---|---|---|
| h_ss 负端饱和且逐细胞恒定 | B1 死、实测表 CV<0.3 | 1.000/0.955@−140/−120，CV≤0.065 | ✓ |
| E_rev 逐细胞、实验室属性 | −88.33（HEK/37） | −83.90（CHO/25，CV 0.024） | ✓（各为其值，不互比） |
| τ_deact 阶梯方向 | τ_late 随电压升高变快（e-fold≈12 mV） | τ2：−60 0.74 s → −40 3.03 s（同向） | ✓ |
| AP 复极回弹自发 | 8/9 细胞过 | 峰比 0.844、A3 100%、A1 结构性受限 | 方向 ✓ |
| Q10（描述性，不入判词） | τ_rec(−120)≈3 ms 级 | 5.31 ms | 比值 ≈1.7–1.8（25→37 °C 保守低估，合理） |

## 正文/SI 补丁句（回填预埋位）

正文（Results 末段 "124 cells nine protocols" 预埋句改写）：
> We further validated the two population-level signatures in an independent dataset of 211 QC-passed CHO cells at 25 °C (nine protocols; Lei et al., 2019a, b): per-cell E_rev was constant across cells (−83.9 mV median, CV 0.024, n = 209), and the steady-state inactivation limb was cell-invariant at −140/−120 mV (h_ss = 1.000/0.955, CV ≤ 0.065). Forward prediction of AP-clamp recordings at 0.5/1/2 Hz reproduced current magnitude (96% of cells within factor 3), repolarisation-rebound amplitude (median ratio 0.84) and frequency ordering (100%), with per-cycle rebound detection passing 45.8% of 627 trials (Wilson 95% CI [0.42, 0.50]).

SI Limitations 增补：
> (i) At 25 °C the 1.0 s activation pulses do not reach steady state below −20 mV; the measured m_ss(V) is an apparent 1 s curve (lower bound). (ii) The h_ss statistic carries a +10–30% isochronal bias where recovery and deactivation race (−100…−20 mV); those voltages are registration-grade only. (iii) Automatic leak correction was re-corrected following Lei et al.'s own pipeline (re-leak, pivot −80 mV); τ_act from activation-pulse rising edges is not identifiable under this correction and was sourced from staircase step transients instead (registration-grade). (iv) The per-cycle rebound criterion (≥80% of cycles) is structurally stringent at 2–4 cycles per sweep (0.5/1 Hz); magnitude- and direction-based criteria pass at 96–100%.

## 124 孔人工精选子集稳健性对照（2026-09-21 12:55 增补，跑后登记）

名单：Lei 论文人工精选 124 孔（repo room-temperature-only/manualv2selected-herg25oc1.txt）；
方法：同一管道结果按名单过滤重算，不重跑提取，判线同一。

| 项 | 211 自动 QC | 124 人工精选 | 方向 |
|---|---|---|---|
| E_rev CV / 范围 | 0.0242 / [−88.4,−72.6] | **0.0167 / [−87.1,−80.6]** | 收紧 ✓ |
| h_ss −140/−120 CV | 0.000 / 0.065 | 0.000 / **0.046** | 收紧 ✓ |
| h_ss −100/−60 CV | 0.109 / 0.379 | **0.074 / 0.251** | 收紧 ✓（C1 作废档判决维持，见上节） |
| m_ss +10/+25 CV | 0.488 / 0.219 | **0.300 / 0.147**（+10 边界） | 收紧 ✓ |
| J5 双过合并 | 45.8% [0.419,0.497] | **59.1% [0.541,0.640]** | 上升 ✓（下界仍<0.6，登记维持） |
| J5 A2（量级） | 96% | **124/124 = 100%（三频全满）** | 上升 ✓ |
| J5 A3（方向） | 100% | 100% | 持平 ✓ |
| 回弹峰比中位 | 0.844 | 0.814 | 持平 ✓ |

**结论：全部封卷项在人工精选子集上复现或增强，登记项维持登记——群体结论对 QC 口径稳健。**
方向效应与人工 QC 筛选的物理预期一致（更干净细胞 → 更小 CV），稳健性对照判【过】。

## 资产清单

- 数据：`04_细胞线4\数据\Lei全量\`（6 协议 × 211 孔 + 协议电压轴 + QC/人工精选名单，约 2 GB，逐字节校验）
- 判线：`04_细胞线4\结果\预注册判决卡_Lei211_α群体_2026-09-21.md`（冻结 + A1–A4 跑前修订）
- 代码：`04_细胞线4\α模型\lei211_J1J2_提取.py`（J1/J2/J4）、`lei211_J3_提取.py`、`lei211_TAUH_群体表.py`、`lei211_J5_AP前向.py`、`lei211_C1_合成回收.py`、`lei211_群体图.py`、下载器 `04_细胞线4\数据\Lei全量\download_lei.py`
- 结果：`lei211_J1J2_结果.json`、`lei211_J3_结果.json`、`lei211_tauh_表.json`、`lei211_J5_结果.json`
- 图：`lei211_群体判决总图.png/.pdf`
