# 失败细胞登记表 · 四通道（2026-09-21）

**口径**：逐细胞/逐电压档/逐 sweep 三级，全部来自封卷判词卡对应结果文件的重算或直接转录；类别定义——A 数据质量（记录本身异常或信噪不足）、B 协议限制（协议窗/等时简并/高温窗限）、C 方法边界（拟合贴界/等时近似偏置）、D 前向判决（模型预测不过，非数据问题）、E 真实物理（登记为非失败）。

**总计 90 条登记**（含批次聚合行；Lei 线逐电压档聚合行为每批每档一行，该档涉及细胞数见 detail 或 cell_or_file 列）。

## A 数据质量（73 条）

| 通道 | 数据集 | 细胞/文件 | 范围 | 失败项 | 说明 | 来源 |
|---|---|---|---|---|---|---|
| hERG | Lei-herg25oc1 | C18 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | M19 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 4孔:C22/L04/M19/P18 | 电压档 | h_ss(-120) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 10孔:C18/C22/G10/G22/H08/K03... | 电压档 | h_ss(-100) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 4孔:C18/C22/L04/P18 | 电压档 | h_ss(-60) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 2孔:C22/P18 | 电压档 | h_ss(-40) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 1孔:P18 | 电压档 | h_ss(-20) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 1孔:M19 | 电压档 | h_ss(20) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg25oc1 | 3孔 | 电压档 | tau_rec(-100) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg25oc1 | 108孔 | 电压档 | tau_rec(-80) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg25oc1 | 11孔 | 电压档 | tau_rec(-60) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg25oc1 | 5孔 | 电压档 | tau_rec(-40) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg25oc1 | 46孔 | 电压档 | tau_rec(-20) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg27oc1 | F13 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg27oc1 | 4孔:B08/E17/F13/M23 | 电压档 | h_ss(-120) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg27oc1 | 8孔:B04/B08/B23/E17/F13/I09... | 电压档 | h_ss(-100) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg27oc1 | 4孔:B23/C08/E17/I09 | 电压档 | h_ss(-60) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg27oc1 | 2孔:B23/E17 | 电压档 | h_ss(-40) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg27oc1 | 1孔 | 电压档 | tau_rec(-100) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg27oc1 | 27孔 | 电压档 | tau_rec(-80) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg27oc1 | 1孔 | 电压档 | tau_rec(-60) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg27oc1 | 7孔 | 电压档 | tau_rec(-20) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg30oc1 | 3孔:C15/J03/P01 | 电压档 | h_ss(-120) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 11孔:B08/B12/C15/F18/J03/J12... | 电压档 | h_ss(-100) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 3孔:B12/C15/P01 | 电压档 | h_ss(-60) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 2孔:C15/P01 | 电压档 | h_ss(-40) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 2孔:C15/P01 | 电压档 | h_ss(-20) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 1孔:P01 | 电压档 | h_ss(20) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 1孔:C15 | 电压档 | h_ss(40) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 1孔 | 电压档 | tau_rec(-100) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg30oc1 | 15孔 | 电压档 | tau_rec(-80) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg30oc1 | 3孔 | 电压档 | tau_rec(-60) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg33oc1 | P17 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg33oc1 | 2孔:K11/P17 | 电压档 | h_ss(-120) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg33oc1 | 5孔:B24/D08/K11/O12/P17 | 电压档 | h_ss(-100) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg33oc1 | 1孔:N05 | 电压档 | h_ss(40) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg33oc1 | 9孔 | 电压档 | tau_rec(-80) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg37oc3 | B05 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | P24 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 1孔:M06 | 电压档 | h_ss(-120) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 59孔:A03/B05/B06/B09/B11/B15... | 电压档 | h_ss(-100) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 4孔:B05/C03/M06/P24 | 电压档 | h_ss(-60) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 1孔:B05 | 电压档 | h_ss(-40) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 1孔:B05 | 电压档 | h_ss(-20) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 1孔:B05 | 电压档 | h_ss(0) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 1孔:M06 | 电压档 | h_ss(20) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 1孔 | 电压档 | tau_rec(-60) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| hERG | Lei-herg37oc4 | N05 | 细胞 | E_rev | staircase 下坡未过零，E_rev 不可定位（漏减/电导异常） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc4 | 2孔:A22/N05 | 电压档 | h_ss(-120) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc4 | 24孔:A19/A22/B12/B16/C05/C23... | 电压档 | h_ss(-100) | h_ss 该档 ok=false（等时近似失真/小信号） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc4 | 1孔 | 电压档 | tau_rec(-20) | DoE 拟合幅度 <4σ（保持段噪声门），该孔该档不取值 | 逐细胞重算（TAUH 逐孔 JSON） |
| Nav1.5 | Tarasov2026-Dryad | 22420004 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 22420011 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 23831026 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 23o09003 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 23o09026 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 23o11001 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 23o11008 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Tarasov2026-Dryad | 23o11021 | 膜片 | late_pct | 多通道膜片有 τ_decay 但无 late% 提取值（论文提取表缺项） | Nα4 结果 CSV 对账 |
| Nav1.5 | Nα3-Tarasov | 24503000 | 细胞 | 激活脚部 s_ref | 弦点低于 2σ_eff（6/45 pA, 2σ=11） | Nα3 逐细胞 CSV |
| Nav1.5 | Nα3-Tarasov | 24510033 | 细胞 | 激活脚部 s_ref | 弦点低于 2σ_eff（4/35 pA, 2σ=7） | Nα3 逐细胞 CSV |
| Nav1.5 | Nα3-Tarasov | 25407024 | 细胞 | 激活脚部 s_ref | 弦点低于 2σ_eff（56/530 pA, 2σ=99） | Nα3 逐细胞 CSV |
| IKs | Chan2023-Zenodo8226585 | 4个ABF(未具名) | 文件 | 原始读取 | 336 个 ABF 中 4 个 float 格式不可读，剔除；另 193→163 重复导出去重 | Kα1 判词卡 §三.6 |
| CaV1.2 | Ren2022-g3msb | 2021_05_20_0004.abf | sweep | QC门 G2G3/G4 | 206 sweeps 中 G2G3 失败 20、G4 失败 2、有效 184 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 2021_05_20_0008.abf | sweep | QC门 G2G3/G4 | 351 sweeps 中 G2G3 失败 278、G4 失败 1、有效 72 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 21May05_0001.abf | sweep | QC门 G2G3/G4 | 382 sweeps 中 G2G3 失败 97、G4 失败 194、有效 91 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 21May05_0005.abf | sweep | QC门 G2G3/G4 | 388 sweeps 中 G2G3 失败 337、G4 失败 51、有效 0 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 21may12_0007.abf | sweep | QC门 G2G3/G4 | 249 sweeps 中 G2G3 失败 9、G4 失败 240、有效 0 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 2021_06_30_0016.abf | sweep | QC门 G2G3/G4 | 300 sweeps 中 G2G3 失败 285、G4 失败 0、有效 0 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 2021_07_07_0008.abf | sweep | QC门 G2G3/G4 | 306 sweeps 中 G2G3 失败 296、G4 失败 8、有效 2 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 2021_07_07_0015.abf | sweep | QC门 G2G3/G4 | 188 sweeps 中 G2G3 失败 182、G4 失败 0、有效 0 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 2021_09_01_0005.abf | sweep | QC门 G2G3/G4 | 380 sweeps 中 G2G3 失败 210、G4 失败 170、有效 0 | Cα1 结果 JSON |
| CaV1.2 | Ren2022-g3msb | 2021_09_01_0007.abf | sweep | QC门 G2G3/G4 | 347 sweeps 中 G2G3 失败 292、G4 失败 49、有效 3 | Cα1 结果 JSON |

## B 协议限制（9 条）

| 通道 | 数据集 | 细胞/文件 | 范围 | 失败项 | 说明 | 来源 |
|---|---|---|---|---|---|---|
| hERG | Lei-herg25oc1 | 211孔:A01/A04/A06/A07/A09/A11... | 电压档 | h_ss(-80) | 保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg27oc1 | 127孔:A01/A05/A06/A07/A09/A11... | 电压档 | h_ss(-80) | 保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg30oc1 | 115孔:A01/A02/A05/A06/A10/A11... | 电压档 | h_ss(-80) | 保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg33oc1 | 107孔:A03/A04/A05/A06/A07/A09... | 电压档 | h_ss(-80) | 保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc3 | 12孔:B05/B16/C03/C08/C10/C21... | 电压档 | h_ss(-80) | 保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性） | 逐细胞重算（J1J2 JSON） |
| hERG | Lei-herg37oc4 | 13孔:B06/B18/D15/D22/E08/F08... | 电压档 | h_ss(-80) | 保持电位档等时简并，h_ss 统计量按规则拒绝（全批结构性） | 逐细胞重算（J1J2 JSON） |
| IKs | Chan2023-Zenodo8226585 | +30/+40/+50/+60档≥1/3细胞 | 电压档 | τ_act(V) | 4 s 激活协议对 IKs(τ 1.5–3 s) 过短，拟合贴 2.5 s 上界，按 caveat A 全部除名——判1 无票可投 | Kα1 判词卡 判1 |
| hERG | Lei-37°C批 | 全体80/105孔 | 电压档 | h_ss(−80) | 等时简并电压档；37 °C 批 80 孔通过但中位 0.857 CV 0.15——数值仅供参考不入判决 | Lei Q10 判词卡 异常登记1 |
| hERG | Lei-37°C批 | 83/105孔 | 电压档 | h_ss(−100) | 高温下该档 ok=false 过多，n=22 低于门槛 → 数据不足登记 | Lei Q10 判词卡 异常登记2 |

## C 方法边界（5 条）

| 通道 | 数据集 | 细胞/文件 | 范围 | 失败项 | 说明 | 来源 |
|---|---|---|---|---|---|---|
| Nav1.5 | Nα3-Tarasov | 24503033 | 细胞 | 激活脚部 s_ref | 2% 穿越不可定位 | Nα3 逐细胞 CSV |
| Nav1.5 | Nα3-Tarasov | 24510058 | 细胞 | 激活脚部 s_ref | 2% 穿越不可定位 | Nα3 逐细胞 CSV |
| Nav1.5 | Nα3-Lei | 005 | 细胞 | 激活脚部 s_ref | 2% 穿越不可定位 | Nα3 逐细胞 CSV |
| IKs | Chan2023-Zenodo8226585 | 2022_03_11_0005 | 细胞 | τ_app(−40) | 边界可疑细胞：inst=0.50 贴排除线、amp=201 pA 池内最小、τ_app=0.020 s 贴拟合下界；判0按条文纳入、判2登记为敏感点 | Kα1 判词卡 §三.2 |
| hERG | Lei-25°C批 | 统计档 | 方法 | C1 合成回收 −100/−60/−20 | 等时近似在恢复/去激活竞速档 +10~30% 偏置，三档统计量作废不判决 | Lei211 判词卡 C1 |

## D 前向判决（2 条）

| 通道 | 数据集 | 细胞/文件 | 范围 | 失败项 | 说明 | 来源 |
|---|---|---|---|---|---|---|
| hERG | Beattie2018-HEK9 | 16704007 | 细胞 | AP/sine 前向 | 九细胞前向验证：sine 7/9、AP 8/9——007 为已知问题细胞（死 sweep/坏节段史，见病灶审计卡） | 四线总览 v5 §1 |
| Nav1.5 | Lei-Nav-HEK35 | 3细胞中1细胞 | 细胞 | C2 整迹验证 | C2 1/3，机制清楚（见 Nα2 判词卡） | 四线总览 v5 §1 |

## E 真实物理（1 条）

| 通道 | 数据集 | 细胞/文件 | 范围 | 失败项 | 说明 | 来源 |
|---|---|---|---|---|---|---|
| Nav1.5 | Tarasov2026-Dryad | 池级 | 参数 | τ_decay(−40)/Po_peak | 池内散布 CV 0.77/0.60 失控但判4 ΔKPQ 方向 13× 正确——散布为 modal gating 真实涨落，非失败 | Nα4 判词卡 判1/2/4 |
