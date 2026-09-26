# v1.29.1：run_job 结果加收敛标志（converged/nit/nm_message）——136 批 M1 条款
# v1.36：新增 K6a2(model 29)/K6a1(model 30)=K6a 门指数臂 amix^2/amix^1（代码152门指数结构卡专用；K6a 原支不动，v1.34.5 行为逐位保持）；
# v1.37：新增 K6a2c(model 31，27参)=K6a2 + 态依赖失活 κ（r 前进速率 ×(1+κ·amix²)，κ=p[26] 线性坐标；κ=0 逐位退化为 K6a2；代码153 F窗残口结构卡专用）；
# v1.39：新增 K6a2f(model 32，28参)=K6a2 + a门前进速率电压不依赖地板（k1t+=p[26]、kb1t+=p[27]，线性坐标，/s；负值守卫 NaN；p[26]=p[27]=0 逐位退化为 K6a2；代码155 S1 地板结构卡专用，文献依据 Wang 1997 C1→C2 电压不依赖限速步）；
# v1.40（代码156）：K6a3=model 33——K6a2 的 a 侧双臂+wa混合 → Wang式三态串行链
# v1.41（代码157）：K6a5=model 34——a 侧整换 Wang 四态全链 C1⇄C2⇄C3⇄O（三关闭态，kf/kb 电压不依赖限速步，线性 o 输出）
# v1.42（代码158）：K6a6=model 35——a 侧整换柔性协同生灭链 k=0..4 + C4⇄O 协同门步（六态，24参）
# v1.43（代码159）：K6a5m=model 36——K6a5 + 逐协议速率调制 λ（定常性诊断：残口=机制 or 定常约束？）
# v1.45（代码161）：K6a7=model 38——十态二维梯形·文献锚定卡；VSD 轴锚死 Wang2013 门控电流
# v1.47（代码164）：K6a7Ji=model 41（24参=39块+失活层 KI/KB）/K6a7Jn=model 42（14参 rmix 卸下臂）
# v1.48（代码165）：恢复拓扑变体——K6a8Ji=model 43（24参，串行回流同层 I_k→C_k）/
#   K6a8Jd=model 45（24参，串行回流深复位 I_k→C_0）/K6a9Ji=model 44（26参，Lu式外加 C_4⇄I_c，
#   fi_c=p[24]·e^(ZI·v)、bi_c=p[25]·e^(−ZI·v)）；其余与 model 41 逐位同构。
#   退化自验证：43/45 在 p[22]=p[23]=0、44 在 p[24]=p[25]=0 时与 model 41 逐位一致。
# v1.46（代码163）：K6a7J=model 39——K6a7J=model 38 同构十态梯形，VSD 轴放开 3 参
#   （p[19]=A0 log、p[20]=ZA=ZB、p[21]=B0 log）受门控电流数据约束；
#   新增 sim_gate39 门控电流输出通道（相对单位通量）；H1 对称能垒保留。
#   （calib161.py：A0=27.492187/s，ZA=ZB=0.03558619/mV，B0=0.132122/s，Vh=-75mV 冻结），
#   自由参数仅 c/L/θ（顶替 KF 槽位语义）+ 继承 model34 的 r1/g/r2/zr/s 模块；判线见预注册_代码161。
# v1.44（代码160）：K6a5o=model 37——K6a5 全链 + 输出律 o²（门族遗产惯例；156 挂起正交轴；
#   159 判决定位残口在高压快端后按冻结预案启动；内核链动力学与 34 逐位相同，仅输出 g·o²·rmix·(v−EK)）
#   sse_of 内逐协议折参数走 34 内核，内核零改动；λ_sa=1 语义冻结参考窗
#   f_k=(4−k)·α(ve)·γ^k，b_k=k·β(ve)；γ=协同因子（电压不依赖，γ=1 独立 / ≫1 MWC / 中间 KNF）；
#   k4 -KF(p5,电压不依赖)- O -b2(ve)- k4；稳态=平均场 Ising p_k∝C(4,k)r^k γ^{k(k−1)/2}
#   机理依据：四聚体结构探针（结果/四聚体结构探针_2026-09-04/）+ γ 轴演示 gamma_axis.py + 用户直觉
#   C1-α/β-C2-f/b2-O（f 电压不依赖限速步；b2 电压依赖返回；后向 Euler 保正守恒）
#   机理依据：代码155机理探测（饱和型速率需求）+ Wang 1997/Tan 2012/Bett 2011
# v1.34.5：cipa 源+数据NaN剔除；v1.34.4：leiT 温度源；v1.34.3：sweep2 源；v1.34.2：lei2 多协议+dt dict；v1.34：Lei 数据源/dt 卡字段/零基线；v1.33.1：结果文件名含 tag（防跨批覆盖）；v1.32：fit_EK 卡级开关（E_K 第27自由参数）；v1.31：K8=model28（142a 快门解耦）；v1.30：协议权重 proto_w（141 加权臂）；v1.29：协议子集(protos)/跳变后mask(mask_pts)/Huber(huber_delta)——代码136剔AK重拟专用；默认行为与v1.28.2逐位一致
# -*- coding: utf-8 -*-
"""
herg 本地运行器（与沙箱代码82–95 同方程、同口径）
================================================

用途：在本机执行长时间拟合任务（Nelder-Mead 抛光），结果 JSON 落
results_local/，把该目录发回即可并录进主文档。

环境：Python ≥3.9；pip install numpy scipy pandas numba
（numba 必需：门动力学是时间序列递推，纯 Python 太慢；
 numba 第一次运行会花 10–30 秒编译，之后即 C 速度。）

数据：data/protocols/*.mat（电压协议）+ data/cells/<细胞号>/*.mat（电流轨迹）。
缺细胞数据时运行 scripts/download_data.py 或直接联系我补包。

任务卡：jobs/*.json，每张卡一次拟合：
  {"cell": "16704007", "model": "M2e", "start_extra": [1000.0],
   "maxiter": 5000, "note": "恢复天花板，起点 K=1000"}
执行：
  python runner.py jobs/job_demo.json          # 跑一张卡
  python runner.py jobs/                        # 跑整个目录（顺序）
结果：results_local/result_<cell>_<model>_s<start序号>.json

模型与参数约定（与沙箱一致，勿改）：
- M0（9 参）：I = g·a·r·(V−EK)；k1=p1·e^(p2·V)，k2=p3·e^(−p4·V)，
  k3=p5·e^(p6·V)，k4=p7·e^(−p8·V)；率在偶数位、斜率在奇数位。
- M1（13 参）：+ 串联第二失活门 r2（k5=p10·e^(p11·V)，k6=p12·e^(−p13·V)），
  I = g·a·r1·r2·(V−EK)。注意 p10=p[10], p11=p[9], p12=p[12], p13=p[11]。
- M2a（14 参）：a = w·a1+(1−w)·a2（并联快去活分量）。
- M2c（14 参）：r = w·r1+(1−w)·r2（并联慢恢复分量）。
- M2d（11 参）：k3、k4 加电压无关地板（+k3f, +k4f）。
- M2e（10 参）：k4 加天花板 K：k4_eff = k4·K/(k4+K)。
- M2ac（19 参）：a、r 同时并联第二分量。
- M2f（21 参，v1.8 新增）：M2ac + w(V)——w_a、w_r 放开为电压 logistic
  函数；结果 params 长度 21，注意其中 p[13]/p[18] 是 logit 截距
  （V=0 处的权重为 sigmoid(p[13])），p[19]/p[20] 是斜率（每 mV）。
- M2h（26 参，v1.9 新增）：M2f + a 侧第三慢去活分量（w3 恒定）。
  结果 params 长度 26：p[21..24]=kd1,sd1,kd2,sd2，p[25]=w3 的 logit。
- M0t2/M0t3/M0t4（均 9 参，v1.10 新增，代码104）：四聚体梯子入场考——
  I = g·a^n·r·(V−EK)，n=2/3/4（每个亚基一个独立激活门 a，电导要求
  n 位点同开；失活门 r 仍在整蛋白水平）。与 M0 同参数量，Δk=0，
  纯结构对拍。params 布局与 M0 完全相同（长度 9）。
- M2f2/M2f3（均 21 参，v1.11 新增，§52）：M2f 的 amix 升 n 次——
  I = g·amix^n·rmix·(V−EK)，n=2/3（aⁿ 结构与 w(V) 机制叠加考）。
  与 M2f 同参数量同布局（长度 21），Δk=12，对 M2f 为 Δk=0 纯结构对拍。
- M2fr2/M2fr3（均 21 参，v1.12 新增，§54 判词3）：M2f 的 rmix 升 n 次——
  I = g·amix·rmix^n·(V−EK)，n=2/3（r 侧协同放大，对称候选考；
  排队依据：§52 判词6 + 代码106 的 +60mV 读出尾回落小账）。
  与 M2f 同参数量同布局（长度 21），Δk=12，对 M2f 为 Δk=0 纯结构对拍。
- v1.12.1：sse_of 增加奇异点异常打回（exp 下溢致 1/0 等 → 返回 1e30 哨兵，
  与既有非有限守卫同口径）。内核与全部数值口径零改动，已完成批次结果零影响。
- v1.14：新增 K2（13 参，id=17）——C3-C2-C1-O-I 五态 Markov（O⇌I 耦合内生；
  激活前两步共享速率对；与乘性族同款欧拉/DT/罚守卫；起点=M2f3 谷映射翻译）。
- v1.13：新增 K1（23 参，id=16）——M2f3 + 失活速率耦合激活占有度
  （k3'=k3(1+κ·am³), k4'=k4(1+λ·am³)，整蛋白耦合入场考 K1，立项书 v1）；
  κ/λ 线性坐标，fac<1e-9 触发 NaN 走 1e30 罚；κ=λ=0 时应与 M2f3 逐位一致。
- v1.12.2：补登记 NPARAM/DK 的 M2fr2/M2fr3 条目（v1.12 打包时两处编辑被
  静默吞没，109 批次拟合完成后死于 DK 查键）。发卡核验电池自此增加：
  AST 注册表完备性检查 + 真 runner 端到端冒烟（各一张临时卡跑通全链）。
- v1.15：新增 K4u（25 参，id=18）/K4n（24 参，id=19）——M2f3 + 慢变量 s 的
  V½ 慢平移（整蛋白耦合入场考 K4，§64 可行性预演+§63 补录 HCN 双模态参照）：
  ds/dt=(s∞(v)−s)/τ_slow，s∞=σ((v−vh_s)/15)，a 侧整组速率对随 v+p[22]·s，
  r 侧随 v+ρ·p[22]·s（K4u ρ=σ(p[24]) 自由、K4n ρ≡0 窄版对拍，Δk=1）；
  w(V) 权重不移（预登记保守）；τ_slow<1ms 触发 NaN 走 1e30 罚；
  δ=0 时应与 M2f3 逐位一致（回归条款）。
- v1.16：新增 K4m（25 参，id=20）= K4n + ks 符号自由（§67.4 判词启动，
  沙箱结构审查后的存活写法）。ds/dt=(s∞(v)−s)/τ_slow，s∞=σ((v−vh_s)/ks)——
  ks 符号自由（K4n 固定 +15=单调增=预热快模态一族；ks<0=反转=静息沉降
  慢模态：(ii) 方向，ak1 重置长→s 高→慢、sa 阶梯去极化→s 低→快）。
  p[21]=τ_slow(ms,log10)，p[22]=vh_s(mV)，p[23]=δ(mV 符号自由)，p[24]=ks(mV
  符号自由)；τ_slow<1ms 或 |ks|<1mV（s∞ 奇异）触发 NaN 走 1e30 罚。
  【沙箱否决双录】①om 状态驱动版（om=am³·rm）：压制正比 ∫om dt，sa 的
  ∫om 比 ak1 大 10 倍→sa 结构性误压（0.19 vs 数据 0.957），τ_on 扫
  10–100ms 无解——与"静息史长→慢"的数据方向冲突，§63(b) 除名；
  ②双速率电压版（k_on=p[21]·exp(p[22]·v)）：z_on<0 静息沉降方向虽对，
  但 −80 推进/−60 回退要求 20mV 窗口内率变 >10 倍→|z_on|>0.115（z>3e0
  超物理）且端点欧拉爆炸，冒烟 C 起点 ak1/sa 同比例压（分化不足）——
  指数率做不出阈值，除名。存活者=ks 符号自由：sigmoid 才有真阈值。
  a 侧整组速率对随 ve_a=v+δ·s；r 侧不动（§67.2 对拍 ρ 无支持复用）；
  s 初值=s∞(V[0])；ks=+15 时应与 K4n 逐位一致、δ=0 时与 M2f3 逐位一致
  （双回归条款）。
- v1.28（2026-08-23，战役H135 P1）：新增 K7（30 参，id=23）=K6a + 慢失活门 u：
  du/dt=(u∞−u)/τu，u∞=σ((v−p[26])/p[28])，τu=p[27]（秒），
  电流乘 (1−p[29]·u)。守卫：τu<0.05s→NaN、|p[28]|<1mV→NaN、(1−λu)<0.05→NaN。
  回归条款：λ=p[29]=0 时与 K6a 逐位一致（u 积分但不进电流）。
- v1.17：新增 K6a（26 参，id=21）/K6b（26 参，id=22）——ak1 残口（am 差 1.25×，
  §70.3 解剖定位）双候选同批对拍（§70.4，用户拍板 116）。
  K6a=K4m+τ_slow 电压依赖：τ(v)=p[21]·exp(p[25]·v)，各段积累/回退速率不对称；
  K6b=K4m+wa 权重随 s：za=p[13]+p[19]·v+p[25]·s（解放"w 不移"保守项，
  结构对应物=hERG 激活双 VSD 分量权重随史变化；s 低偏慢分量 a2、s 高偏快 a1）。
  z_τ=0 / γ=0 时与 K4m 逐位一致（双回归条款）。
  【单位更正随版记录】K4 家族 τ_slow（p[21]）单位=秒（§70.1 勘误：dt=1e-4 秒
  与 p[21] 同单位）；v1.15/v1.16 版本头与卡注"ms"系标注错误，数值搜索不受影响
  （N-M 不认标签）；罚守卫 p[21]<1.0 实为 τ<1 秒（原意 1ms）——K4n/K4m 谷
  τ=7.8–172s 远离边界，已跑结果不受影响；K6a 逐点守卫 τ(v)<1s→NaN。
- v1.18：新增 audit 卡型（"mode":"audit"）——谷点数值 Hessian 辨识性审计
  （代码120，用户拍板）。不做拟合，只在 start_params 基点做参数扰动评估：
  dims/steps=逐维对角中心差分，cross=参数对 2×2 网格（s 模块内部相关）。
  输出 results_local/audit_<cell>_<model>_s<n>.json：
  base_sse + diag{维:{h,plus,minus}} + cross{"i,j":[++,+-,-+,--]}。
  NaN 不剔除原样记录（端点悬崖位置本身即信息，§66.3/§75.3 双录传统）。
  评估点数=1+2·len(dims)+4·len(cross)；全程只读数据，不改全局状态。
- v1.19：新增 Nav1.5 N1 内核与 nav 卡型（"mode":"nav"，代码121）——
  静默战场体系A（预注册稿 v1.0+附录 A1–A3，预期判词"静默"已先落笔）。
  N1（12 参，id=23）= HH 型 m³h 通道层 + Rs/Cm 钳制层（附录 A3 硬要求：
  钳制层建入内核，不得当噪声处理）：
    通道层：每门 kf=kf0·e^(+zf·Vm)、kb=kb0·e^(−zb·Vm)，x∞=kf/(kf+kb)、
      τ=1/(kf+kb)，指数精确积分（dt=0.04 ms=25 kHz，Nav1.5 快门 Euler 不稳）；
      I_Na = g·am³·ah·(Vm−ENa)；p=[kma,zma,kmb,zmb,kha,zha,khb,zhb,g,ENa,gl,Voff]，
      率单位 /ms（注意：与 hERG 段的秒不同，单位随版本头冻结，勿混）。
    钳制层：命令电位 Vc=Vcmd+Voff；门控状态冻结时电流对 Vm 线性 →
      解析解 Vm=(Vc+1e-3·Rs_eff·G·ENa)/(1+1e-3·Rs_eff·(G+gl))，G=g·am³·ah（nS），
      Rs_eff=meta 实测 Rs[MΩ]×(1−CP/100)（CP 由文件名解析）；量纲：MΩ·pA×1e-3=mV。
      漏流 gl·Vm 自拟合（附录 A2：不采纳作者经验校正值）；ENa 自由（初锚+55 mV）。
  nav 卡字段：cell（相对 nav_root 的细胞目录）、cps（CP 级别列表，跨补偿
    联合拟合=判线①跨协议自洽的硬执行：全套通道参数同时解释多 CP 数据）、
    nav_root（默认 HERE/data_nav，可写绝对路径）、start_params（12 物理参）、
    maxiter。数据加载复用 nav_adapter.py（v0.1 规格，附录 A2/A3）。
  输出 results_local/nav_<cell>_N1_s<n>.json：sse/params/nfev/分 CP sse 分解/
    IV 签名（每 CP 观测与模拟的峰电流·峰位）。ΔlogML 本批不算（无 M0 锚；
    扩展挑战=N2 vs N1 属后续批）。DK["N1"] 占位 0，无判决含义。
- v1.20：新增 N2（16 参，id=24）与 N1/N2 物理罚守卫（代码122 双卡对拍，
  §82.5 拍板）。背景：121 首谷=钳制层失控型病态（g→58 µS 入 Rs 饱和区、
  ENa→136 mV 代偿峰幅缺口与持续电流缺口，§82.3）——守卫为健康谷边界：
    【守卫（N1/N2 共用，出界 sse=1e30）】ENa∈[+40,+80] mV；g≤5000 nS；
    gl∈[1e-3,100] nS；Voff∈[−20,+20] mV。
  N2=N1+h2 慢失活分量（m³·h·h2，输出 ×ah2）：晚钠电流（+30~+60 mV 阶跃末
  −340~−382 pA 内向残余，观测峰幅 4.4%，§82.4）的结构承载——Nav1.5 失活
  快/慢双分量的文献身份。参数追加 [kf2,zf2,kb2,zb2]（同门参数化），
  p 长 16=[m 4, h 4, h2 4, g, ENa, gl, Voff]；Δk(N2−N1)=4，
  判线③ ΔlogML=−n/2·ln(SSE_N2/SSE_N1)−2·ln n（n=73,125，ln n=11.20）。
  【不启动登记】j 门（τ~秒级 IM）在 NaIV 20 ms 窗内不可激励、与 g 简并——
  预注册草案 m³h·j 缓行，待 NaInact 链（1,000 ms 条件段）入场。
  sim_nav_sweep 改 p 长度分支（len 12=N1／16=N2），N1 卡向后兼容。
- v1.21：钳制层 Cm 动态 + NaInact 链（代码123，§84.5 拍板 A+B 同批）。
  【Cm 动态】vm 升状态变量：Cm·dvm/dt=(1e3·(vc−vm)/Rs_eff)−G·(vm−ENa)−gl·vm
  （pA/pF=mV/ms）；G 冻结时线性 → 指数精确积分：vm∞=A/B、τv=Cm/B，
  A=1e3·vc/Rs+G·ENa，B=1e3/Rs+G+gl。τv=Rs·Cm=10–50 µs 与 dt=40 µs 同级，
  v1.19/1.20 的准静态解析解=τv→0 极限（回归测试条款：Cm 取极小值时
  新版须逐位复现旧版）。Cm 取 meta 实测 capacitance_pF（逐列）。
  P 补偿残余电容瞬态忽略（峰区窗在阶跃起 100 µs 后，双录在案）。
  【NaInact 链】失活协议入场：9 ms(−100)+1000 ms(V_cond)+40 ms(−20 测试)，
  26,225 点×13 列；加载器=nav_adapter v0.2 自写（官方 NotImplementedError，
  附录 A2 坑③；列结构自适应，电压以 meta 为准——坑②）。使命：h/h2 稳态
  辨识主场（h2 在 NaIV 20 ms 窗欠定，§84.3 简并污染嫌疑的清算）。
  卡新字段 protocols：["NaIV"] 或 ["NaIV","NaInact"]；sse 全点直加
  （hERG 五协议传统；条件段基线噪声占比入残差结构记录）。
  sim_nav_sweep 签名改 (p, Vcmd, rs_eff, cm, dt_ms)——v1.20 及前卡不兼容，
  须整体换包。守卫沿用 v1.20（ENa∈[+40,+80]／g≤5000 nS／gl∈[1e-3,100]／
  |Voff|≤20 mV）。
- v1.22/v1.23/v1.24：见 §87/§89/§90.6（放大器链／真值 rc／C_prs+基线口径）。
- v1.25（代码127）：EPC-10 输出滤波级联入场——card 新字段 filt:"epc10"
  触发，固定项 Δk=0（F1=6 极点 Bessel 10kHz + F2=4 极点 Bessel 5kHz，
  HEKA 手册结构 + 25kHz 标准对；§94 群延迟实证 0.176ms≈观测 0.16ms）。
  作用于 iin→iout 物理电流通道；filt 缺省=None 逐位兼容 v1.24。
  sse 与 125 锚（同数据同参直接可比，Δk=0）。
- v1.26（代码128）：filt:"epc10z" 精确化——模态展开 ZOH 精确离散化替换
  v1.25 双线性数字 Bessel（畸变双录：质心 0.1185ms vs 模拟 0.1762ms）。
  H(s)=k/Π(s−pi) 部分分式 → 每共轭对实二阶模态块 (c,d,er,ei,rR,rI)，
  矩阵指数解析解，无多项式病态；阶跃逐点 max|Δ|=3.4e-13，DC=1 精确，
  h[0]=0。质心 0.1962ms=模拟 0.1762+T/2（ZOH 保持内在含义，口径双录）。
  工程事故双录（§97.1）：①逐节 ZOH 错误——节间信号非分段常值，级联
  系统 ZOH 离散≠各节分别离散（dc 增益 0.827/质心 0.273 畸形当场捕获）；
  ②scipy 模拟 sos 把总增益 6e28 堆第一节致 tf/ss 路线全数病态，zpk 版
  h[0]≠0 提前一采样——均否，模态展开为唯一干净通路。
  v1.25 "epc10" 通路逐位保留（127 锚 5.7585812e10 已冻结）。
- EK = −84.15 mV（硬锚），dt = 0.1 ms，显式 Euler（与沙箱完全一致）。
173:- v1.38（代码154）：卡级 drop_windows={proto:[[t0,t1],...]} 时间段剔除（sa +60mV 伪影窗，数据管线审计定罪）；不带该字段=与 v1.37 逐位一致。

- v1.27（代码130）：N1r5=N1r+rs_scale 按 CP 五档（p 长 19=基12+rs×5+cm+垫位
  1.0，x 长 18；Δk=+4 判线①对齐 N1r）。动机=§100 结构残口：obs IV 峰值电压
  CP 依赖右移（011 −50→0mV），标量 rs_scale 修不动"低 CP 欠移+高 CP 过冲"
  双向误差。回归冻结：N1r5 五档同值时必须与 N1r 逐位一致（沙箱已核，见 §101）。

- v1.27a（132 批前修复，§104）：N1r5 的 rc 守卫补接（rs 五档+cm ∈[0.5,2.0]）；
  v1.27 守卫条件 len(p)∈(14,18) 漏掉 p=19，005/011 极端档由此漏出——131 s0 批
  判词挂起，132（s1）重跑裁决。morph 脚本 rs 双重补偿 bug 同批修复（morph129f/131f）。

判决口径：ΔlogML = −n/2·ln(SSE/SSE_M0) − Δk/2·ln(n)，Δk = 新增参数数
（M1:4, M2a:5, M2c:5, M2d:2, M2e:1, M2ac:10）；−120 窗 = 电压 ∈ [−121,−119)。
"""
import json
import os
import sys
import time

# v1.27b（代码134 诊断批）：rs/cm 守卫界环境变量化。
# 默认 0.5/2.0 与 v1.27a 逐位等价；仅 134 诊断卡（NAV_GUARD_LO=0.25
# NAV_GUARD_HI=4.0）放宽——判词冻结口径不变，放宽结果只作诊断登记。
_GUARD_LO = float(os.environ.get("NAV_GUARD_LO", "0.5"))
_GUARD_HI = float(os.environ.get("NAV_GUARD_HI", "2.0"))

import numpy as np
import scipy.io as sio
from numba import njit
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1e-4
EK = -84.15
PROTOS = ["activation_kinetics_1", "activation_kinetics_2",
          "steady_activation", "deactivation", "sine_wave"]
MODEL_ID = {"M0": 0, "M2a": 1, "M2c": 2, "M2d": 3, "M2e": 4, "M2ac": 5,
            "M1": 6, "M2f": 7, "M2h": 8, "M0t2": 9, "M0t3": 10, "M0t4": 11,
            "M2f2": 12, "M2f3": 13, "M2fr2": 14, "M2fr3": 15,
            "K1": 16,
            "K2": 17, "K4u": 18, "K4n": 19, "K4m": 20, "K6a": 21, "K6b": 22, "K7": 23, "K8": 28, "K6a2": 29, "K6a1": 30, "K6a2c": 31, "K6a2f": 32, "K6a3": 33, "K6a5": 34, "K6a6": 35, "K6a5m": 36, "K6a5o": 37, "K6a7": 38, "K6a7J": 39, "K6a7Ji": 41, "K6a7Jn": 42, "K6a8Ji": 43, "K6a9Ji": 44, "K6a8Jd": 45,
            "N1": 23, "N2": 24, "N1r": 25, "N2r": 26, "N1r5": 27}
NPARAM = {"M0": 9, "M2a": 14, "M2c": 14, "M2d": 11, "M2e": 10, "M2ac": 19,
          "M1": 13, "M2f": 21, "M2h": 26, "M0t2": 9, "M0t3": 9, "M0t4": 9,
          "M2f2": 21, "M2f3": 21, "M2fr2": 21, "M2fr3": 21,
          "K1": 23,
          "K2": 13, "K4u": 25, "K4n": 24, "K4m": 25, "K6a": 26, "K6b": 26, "K7": 30, "K8": 26, "K6a2": 26, "K6a1": 26, "K6a2c": 27, "K6a2f": 28, "K6a3": 23, "K6a5": 26, "K6a6": 24, "K6a5m": 30, "K6a5o": 26, "K6a7": 19, "K6a7J": 22, "K6a7Ji": 24, "K6a7Jn": 14, "K6a8Ji": 24, "K6a9Ji": 26, "K6a8Jd": 24,
          "N1": 12, "N2": 16, "N1r": 14, "N2r": 18, "N1r5": 19}
DK = {"M0": 0, "M1": 4, "M2a": 5, "M2c": 5, "M2d": 2, "M2e": 1, "M2ac": 10,
      "M2f": 12, "M2h": 17, "M0t2": 0, "M0t3": 0, "M0t4": 0,
      "M2f2": 12, "M2f3": 12, "M2fr2": 12, "M2fr3": 12,
      "K1": 14,
      "K2": 4, "K4u": 16, "K4n": 15, "K4m": 16, "K6a": 17, "K6b": 17, "K7": 21, "K8": 17, "K6a2": 17, "K6a1": 17, "K6a2c": 18, "K6a2f": 19,
      "N1": 0, "N2": 4,   # N1 占位无判决含义；N2=N1+h2，Δk(N2−N1)=4（判线③用）
      "N1r": 0, "N2r": 4}  # v1.23：r=真值 Rs/Cm 自由度（+2 参两侧对齐，Δk 不变）
DK["N1r5"] = 4
DK["K6a3"] = 14   # v1.40：23 参 vs M0 9 参（判线③对齐用）
DK["K6a5"] = 17   # v1.41：26 参 vs M0 9 参（判线③对齐用）
DK["K6a6"] = 15   # v1.42：24 参 vs M0 9 参（判线③对齐用）
DK["K6a5m"] = 21  # v1.43：30 参 vs M0 9 参（判线③对齐用）
DK["K6a5o"] = 17  # v1.44：26 参 vs M0 9 参（判线③对齐用）
DK["K6a7"] = 10   # v1.45：19 参 vs M0 9 参（判线③对齐用）
DK["K6a7J"] = 13  # v1.46：22 参 vs M0 9 参（判线③对齐用）   # v1.27：N1r5=N1r+rs_scale 按 CP 五档（+4 参，判线①对齐）
DK["K6a7Ji"] = 15  # v1.47：24 参 vs M0 9 参（判线③对齐用）
DK["K6a7Jn"] = 5   # v1.47：14 参 vs M0 9 参（判线③对齐用）
DK["K6a8Ji"] = 15  # v1.48：24 参 vs M0 9 参（判线③对齐用）
DK["K6a9Ji"] = 17  # v1.48：26 参 vs M0 9 参（判线③对齐用）
DK["K6a8Jd"] = 15  # v1.48：24 参 vs M0 9 参（判线③对齐用）


@njit(cache=True, fastmath=True, nogil=True)
def sim_kernel(p, V, model, dt, EK):
    """统一内核。p 为长度 19 的数组（未用分量填 0）。"""
    n = len(V)
    out = np.empty(n)
    p1, p2, p3, p4 = p[0], p[1], p[2], p[3]
    p5, p6, p7, p8, g = p[4], p[5], p[6], p[7], p[8]
    a1, a2, a3, r1, r2 = 0.0, 0.0, 0.0, 1.0, 1.0
    C3, C2, C1, O, Ic = 1.0, 0.0, 0.0, 0.0, 0.0
    s = 0.0
    if model == 18 or model == 19:
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[23]) / 15.0))
    if model in (20, 21, 22, 23, 28, 29, 30, 31, 32):
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[22]) / p[24])) if abs(p[24]) >= 1.0 else 0.0
    cc1, cc2, oo = 1.0, 0.0, 0.0   # v1.40：K6a3 链状态（避 K2 的 C1/C2 名）
    if model == 33:
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[19]) / p[21])) if abs(p[21]) >= 1.0 else 0.0
    cc3 = 0.0                       # v1.41：K6a5 第四链态（第三关闭态）
    if model == 34:
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[22]) / p[24])) if abs(p[24]) >= 1.0 else 0.0
    if model == 37:
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[22]) / p[24])) if abs(p[24]) >= 1.0 else 0.0
    x7 = np.zeros(10)               # v1.45：K6a7 梯形十态 [C0,O0,C1,O1,...,C4,O4]
    A00 = np.empty(5); A01 = np.empty(5); A10 = np.empty(5); A11 = np.empty(5)
    U00 = np.empty(5); U11 = np.empty(5)
    CP00 = np.zeros(5); CP01 = np.zeros(5); CP10 = np.zeros(5); CP11 = np.zeros(5)
    DP0 = np.zeros(5); DP1 = np.zeros(5)
    if model in (38, 39):
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0
        x7[0] = 1.0
    xi5 = np.zeros(5)               # v1.47（代码164）：失活层 I_k（41/42 用；初态 0，I 冻结 VSD 位置）
    xiC = 0.0                     # v1.48（代码165）：model 44 外加 C_4⇄I_c 的 I_c 人口
    if model in (41, 42, 43, 44, 45, 47):  # v1.48：43/44/45 同初态；v1.49b：47 补入（前哨验证逮住遗漏——sim_gate148 初态无条件设置故未连坐；41-45 路径逐位不变）
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0
        x7[0] = 1.0
    cc4, cc5 = 0.0, 0.0             # v1.42：K6a6 第五/六链态（k3/k4；cc1=k0,cc2=k1,cc3=k2,oo=O）
    if model == 35:
        s = 1.0 / (1.0 + np.exp(-(V[0] - p[20]) / p[22])) if abs(p[22]) >= 1.0 else 0.0
    u = 0.0
    if model == 23 and len(p) > 28:
        u = 1.0 / (1.0 + np.exp(-(V[0] - p[26]) / p[28])) if abs(p[28]) >= 1.0 else 0.0
    for i in range(n):
        v = V[i]
        k1 = p1 * np.exp(p2 * v)
        k2 = p3 * np.exp(-p4 * v)
        k3 = p5 * np.exp(p6 * v)
        k4 = p7 * np.exp(-p8 * v)
        if model == 3:            # M2d 失活地板
            k3 += p[9]
            k4 += p[10]
        elif model == 4:          # M2e 恢复天花板
            k4 = k4 * p[9] / (k4 + p[9])
        ta = 1.0 / (k1 + k2)
        tr = 1.0 / (k3 + k4)
        if model == 1:            # M2a: a 并联
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            out[i] = g * (p[13] * a1 + (1.0 - p[13]) * a2) * r1 * (v - EK)
        elif model == 2:          # M2c: r 并联
            kb3 = p[9] * np.exp(p[10] * v)
            kb4 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb3 + kb4)
            a1 += dt * ((k1 * ta) - a1) / ta
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kb4 * tb) - r2) / tb
            out[i] = g * a1 * (p[13] * r1 + (1.0 - p[13]) * r2) * (v - EK)
        elif model == 5:          # M2ac: 双侧并联
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            out[i] = (g * (p[13] * a1 + (1.0 - p[13]) * a2)
                      * (p[18] * r1 + (1.0 - p[18]) * r2) * (v - EK))
        elif model == 7:          # M2f: M2ac + w(V)（logistic 电压依赖权重）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            out[i] = (g * (wa * a1 + (1.0 - wa) * a2)
                      * (wr * r1 + (1.0 - wr) * r2) * (v - EK))
        elif model == 12:         # M2f2: M2f + amix^2（a 侧协同度 2）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            am = wa * a1 + (1.0 - wa) * a2
            out[i] = (g * (am * am)
                      * (wr * r1 + (1.0 - wr) * r2) * (v - EK))
        elif model == 13:         # M2f3: M2f + amix^3（a 侧协同度 3）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            am = wa * a1 + (1.0 - wa) * a2
            out[i] = (g * (am * am * am)
                      * (wr * r1 + (1.0 - wr) * r2) * (v - EK))
        elif model == 17:         # K2: C3-C2-C1-O-I 五态 Markov（整蛋白耦合入场考，耦合内生）
            a1f = p[0] * np.exp(p[1] * v)
            a1b = p[2] * np.exp(-p[3] * v)
            a2f = p[4] * np.exp(p[5] * v)
            a2b = p[6] * np.exp(-p[7] * v)
            fin = p[8] * np.exp(p[9] * v)
            fbk = p[10] * np.exp(-p[11] * v)
            dC3 = -a1f * C3 + a1b * C2
            dC2 = a1f * C3 - (a1b + a1f) * C2 + a1b * C1
            dC1 = a1f * C2 - (a1b + a2f) * C1 + a2b * O
            dO = a2f * C1 - (a2b + fin) * O + fbk * Ic
            dI = fin * O - fbk * Ic
            C3 += dt * dC3
            C2 += dt * dC2
            C1 += dt * dC1
            O += dt * dO
            Ic += dt * dI
            out[i] = p[12] * O * (v - EK)
        elif model == 18 or model == 19:
            # K4u(18,25参)/K4n(19,24参): M2f3 + 慢变量 s 的 V½ 慢平移
            # ds/dt=(s∞−s)/τ_slow, s∞=σ((v−p[23])/15), τ_slow=p[21] (ms)
            # a 侧整组速率对随 ve_a=v+p[22]·s；r 侧随 ve_r=v+ρ·p[22]·s（K4u ρ=σ(p[24])，K4n ρ≡0）
            if p[21] < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[23]) / 15.0))
            s += dt * (s_inf - s) / p[21]
            ve_a = v + p[22] * s
            if model == 18:
                ve_r = v + p[22] * s / (1.0 + np.exp(-p[24]))
            else:
                ve_r = v
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a)
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * ve_r)
            k4t = p[6] * np.exp(-p[7] * ve_r)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * ve_r)
            kc4t = p[16] * np.exp(-p[17] * ve_r)
            tct = 1.0 / (kc3t + kc4t)
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            out[i] = (g * am4 * am4 * am4
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 20:
            # K4m(20,25参): K4n 结构 + ks 符号自由（沙箱审查存活写法，§68）
            # ds/dt=(s∞−s)/τ_slow，s∞=σ((v−p[22])/p[24])，ve_a=v+p[23]·s，r 侧不动
            if p[21] < 1.0 or abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / p[21]
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a)
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * v)
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v)
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = 1.0 / (kc3t + kc4t)
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            out[i] = (g * am4 * am4 * am4
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 23:          # K7(23,30参): K6a + 慢失活门 u（v1.28，战役H135）
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0 or p[27] < 0.05 or abs(p[28]) < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a)
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * v)
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v)
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = 1.0 / (kc3t + kc4t)
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            u_inf = 1.0 / (1.0 + np.exp(-(v - p[26]) / p[28]))
            u += dt * (u_inf - u) / p[27]
            fac7 = 1.0 - p[29] * u
            if fac7 < 0.05:
                out[i] = np.nan
                continue
            out[i] = (g * am4 * am4 * am4
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK)) * fac7
        elif model == 21 or model == 22:
            # K6a(21): τ(v)=p[21]·exp(p[25]·v)；K6b(22): za 加 p[25]·s；余同 K4m
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            if model == 21:
                tau_v = p[21] * np.exp(p[25] * v)
            else:
                tau_v = p[21]
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a)
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * v)
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v)
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = 1.0 / (kc3t + kc4t)
            if model == 22:
                za4 = min(max(p[13] + p[19] * v + p[25] * s, -30.0), 30.0)
            else:
                za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            out[i] = (g * am4 * am4 * am4
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 29 or model == 30:
            # K6a2(29)/K6a1(30)（v1.36，代码152门指数结构卡）：K6a 的 amix 指数臂——
            # 29: amix^2；30: amix^1；余与 K6a(21) 逐行一致（τ(v) 电压依赖、za 无 s 项）
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a)
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * v)
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v)
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = (1.0 / (kc3t + kc4t))
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            if model == 29:
                out[i] = (g * am4 * am4
                          * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
            else:
                out[i] = (g * am4
                          * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 32:
            # K6a2f(32,28参)（v1.39，代码155 S1 地板结构卡）：K6a2 + a门前进速率电压不依赖地板——
            # k1t+=p[26]（a1）、kb1t+=p[27]（a2），单位 /s（与速率同）；负值守卫 NaN；
            # p[26]=p[27]=0 逐位=K6a2(29)；文献依据 Wang 1997 C1→C2 电压不依赖限速步
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            if p[26] < 0.0 or p[27] < 0.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a) + p[26]
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a) + p[27]
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * v)
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v)
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = (1.0 / (kc3t + kc4t))
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            out[i] = (g * am4 * am4
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 33:
            # K6a3(33,23参)（v1.40，代码156 Wang链结构卡）：
            # a 侧 = C1 -α/β- C2 -f/b2- O 三态串行链（替代 K6a2 的 a1/a2 双臂+wa 混合）
            #   α(ve)=p0·e^(p1·ve)   β(ve)=p2·e^(−p3·ve)   f=p4（电压不依赖限速步）
            #   b2(ve)=p5·e^(−p6·ve)（电压依赖返回；负电压爆炸=去激活快的结构来源）
            #   ve = v + p20·s（s 模块保留：τ_s=p18·e^(p22·v)，s∞=logistic(v−p19)/p21）
            #   p[7..10]=r1(k3_0,z3,k4_0,z4)  p[11]=g  p[12..15]=r2(kc3_0,zc3,kc4_0,zc4)
            #   p[16]=zr截距  p[17]=zr斜率
            # 链更新=后向 Euler（M 矩阵 3×3 Cramer，无条件稳定+保正+守恒；
            #   显式 Euler 在超极化段 β/b2 爆炸吹掉——概念验证 v1 工程双录 2026-09-04）
            if abs(p[21]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[18] * np.exp(p[22] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[19]) / p[21]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[20] * s
            al3 = p[0] * np.exp(p[1] * ve_a)
            be3 = p[2] * np.exp(-p[3] * ve_a)
            b23 = p[5] * np.exp(-p[6] * ve_a)
            f3 = p[4]
            m00 = 1.0 + dt * al3
            m01 = -dt * be3
            m10 = -dt * al3
            m11 = 1.0 + dt * (be3 + f3)
            m12 = -dt * b23
            m21 = -dt * f3
            m22 = 1.0 + dt * b23
            det3 = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22)
            bc0 = cc1
            bc1 = cc2
            bc2 = oo
            cc1 = (bc0 * (m11 * m22 - m12 * m21) - m01 * (bc1 * m22 - m12 * bc2)) / det3
            cc2 = (m00 * (bc1 * m22 - m12 * bc2) - bc0 * (m10 * m22)) / det3
            oo = (m00 * (m11 * bc2 - m21 * bc1) - m01 * (m10 * bc2) + bc0 * (m10 * m21)) / det3
            k3t = p[7] * np.exp(p[8] * v)
            k4t = p[9] * np.exp(-p[10] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[12] * np.exp(p[13] * v)
            kc4t = p[14] * np.exp(-p[15] * v)
            tct = (1.0 / (kc3t + kc4t))
            zr4 = min(max(p[16] + p[17] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[11] * oo * oo
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 34:
            # K6a5(34,26参)（v1.41，代码157 Wang全链换骨架卡）：
            # a 侧 = C1 -α1/β1- C2 -kf/kb- C3 -α2/β2- O 四态串行链（三个关闭态；
            #   kf/kb 电压不依赖限速步；输出线性 o——忠实 Wang 骨架，
            #   直接仿真证据（结果/Wang1997直接仿真_2026-09-04/）即基于线性 o）
            #   α1(ve)=p0·e^(p1·ve)  β1(ve)=p2·e^(−p3·ve)  kf=p4  kb=p5
            #   α2(ve)=p6·e^(p7·ve)  β2(ve)=p8·e^(−p9·ve)  ve = v + p23·s
            #   p[10..13]=r1  p[14]=g  p[15..18]=r2  p[19]/p[20]=zr 截距/斜率
            #   p[21..25]=s 模块(τ0,vh,δ,ks,z_τ)
            # 链更新=后向 Euler（Thomas 算法解三对角 M=I−dt·A，无条件稳定+保正+守恒；
            #   守恒性：A 列和=0 → M 列和=1 → 状态和不漂）
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            a15 = p[0] * np.exp(p[1] * ve_a)
            b15 = p[2] * np.exp(-p[3] * ve_a)
            a25 = p[6] * np.exp(p[7] * ve_a)
            b25 = p[8] * np.exp(-p[9] * ve_a)
            d0 = 1.0 + dt * a15
            d1 = 1.0 + dt * (b15 + p[4])
            d2 = 1.0 + dt * (p[5] + a25)
            d3 = 1.0 + dt * b25
            cp0 = (-dt * b15) / d0
            dp0 = cc1 / d0
            m1 = d1 - (-dt * a15) * cp0
            cp1 = (-dt * p[5]) / m1
            dp1 = (cc2 - (-dt * a15) * dp0) / m1
            m2 = d2 - (-dt * p[4]) * cp1
            cp2 = (-dt * b25) / m2
            dp2 = (cc3 - (-dt * p[4]) * dp1) / m2
            m3 = d3 - (-dt * a25) * cp2
            dp3 = (oo - (-dt * a25) * dp2) / m3
            oo = dp3
            cc3 = dp2 - cp2 * oo
            cc2 = dp1 - cp1 * cc3
            cc1 = dp0 - cp0 * cc2
            k3t = p[10] * np.exp(p[11] * v)
            k4t = p[12] * np.exp(-p[13] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[15] * np.exp(p[16] * v)
            kc4t = p[17] * np.exp(-p[18] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[19] + p[20] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[14] * oo
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 37:
            # K6a5o(37,26参)（v1.44，代码160 o²变体卡）：链动力学与 model 34 逐位相同，
            # 唯输出律 g·o²·rmix·(v−EK)（o²=门族遗产惯例，K6a/K6a2/K6a3 同款；
            # 参数布局同 K6a5：p[0..9]=链，p[10..13]=r1，p[14]=g，p[15..18]=r2，
            # p[19]/p[20]=zr，p[21..25]=s 模块）
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            a15 = p[0] * np.exp(p[1] * ve_a)
            b15 = p[2] * np.exp(-p[3] * ve_a)
            a25 = p[6] * np.exp(p[7] * ve_a)
            b25 = p[8] * np.exp(-p[9] * ve_a)
            d0 = 1.0 + dt * a15
            d1 = 1.0 + dt * (b15 + p[4])
            d2 = 1.0 + dt * (p[5] + a25)
            d3 = 1.0 + dt * b25
            cp0 = (-dt * b15) / d0
            dp0 = cc1 / d0
            m1 = d1 - (-dt * a15) * cp0
            cp1 = (-dt * p[5]) / m1
            dp1 = (cc2 - (-dt * a15) * dp0) / m1
            m2 = d2 - (-dt * p[4]) * cp1
            cp2 = (-dt * b25) / m2
            dp2 = (cc3 - (-dt * p[4]) * dp1) / m2
            m3 = d3 - (-dt * a25) * cp2
            dp3 = (oo - (-dt * a25) * dp2) / m3
            oo = dp3
            cc3 = dp2 - cp2 * oo
            cc2 = dp1 - cp1 * cc3
            cc1 = dp0 - cp0 * cc2
            k3t = p[10] * np.exp(p[11] * v)
            k4t = p[12] * np.exp(-p[13] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[15] * np.exp(p[16] * v)
            kc4t = p[17] * np.exp(-p[18] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[19] + p[20] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[14] * oo * oo
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 38:
            # K6a7(38,19参)（v1.45，代码161 二维梯形·文献锚定卡）：
            # 十态梯形 (k=0..4)×{C,O}；VSD 轴锚死 Wang2013 门控电流标定（calib161.py）：
            #   a(ve)=27.492187·e^(0.03558619·ve)  b(ve)=0.132122·e^(−0.03558619·ve)
            #   f_k=(4−k)a，b_k=k·b；O 行 VSD 前向 ×θ（环细致平衡）；
            #   垂直 C_k⇄O_k：开 c·L·θ^k / 关 c。
            # p[0]=c p[1]=L p[2]=θ p[3]=zk3 p[4]=k3_0 p[5]=zk4 p[6]=k4_0 p[7]=g
            # p[8..11]=r2 p[12]/p[13]=zr p[14..18]=s 模块(τ0,vh,δ,ks,z_τ)；输出=g·(ΣO_k)·rmix·(v−EK)
            # （r1 速率槽 4/6 与斜率槽 3/5 换位：共享头部 tr=1/(k3+k4) 用 p[4..7]，
            #   必须恒正 → 4/6 锁 log 槽；34 不崩同理——其 4/5/6 皆 log 槽）
            # 更新=后向 Euler 2×2 块 Thomas（列约定；守恒/保正/无条件稳定）
            if abs(p[17]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[14] * np.exp(p[18] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[16] * s
            aa7 = 27.492187 * np.exp(0.03558619 * ve_a)
            bb7 = 0.132122 * np.exp(-0.03558619 * ve_a)
            c7 = p[0]; L7 = p[1]; th7 = p[2]
            thp = 1.0
            for k in range(5):
                fk7 = (4.0 - k) * aa7
                if k == 4:
                    fk7 = 0.0
                bk7 = k * bb7
                ok7 = c7 * L7 * thp
                A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
                A01[k] = -dt * c7
                A10[k] = -dt * ok7
                A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7)
                U00[k] = -dt * fk7
                U11[k] = -dt * th7 * fk7
                thp *= th7
            # 前向块扫描（列约定 M=I−dtA）：
            #   次对角块 M[k][k−1]=−dt·G_{k−1}=diag(U00[k−1],U11[k−1])（前向速率）
            #   超对角块 M[k][k+1]=−dt·b_{k+1}·I（标量）；CP_k 携带超对角、W_k 消次对角
            det = A00[0] * A11[0] - A01[0] * A10[0]
            i00 = A11[0] / det; i01 = -A01[0] / det
            i10 = -A10[0] / det; i11 = A00[0] / det
            sup = -dt * bb7            # b_1 = 1·bb7
            CP00[0] = i00 * sup; CP01[0] = i01 * sup
            CP10[0] = i10 * sup; CP11[0] = i11 * sup
            DP0[0] = i00 * x7[0] + i01 * x7[1]
            DP1[0] = i10 * x7[0] + i11 * x7[1]
            for k in range(1, 5):
                s0u = U00[k - 1]; s1u = U11[k - 1]   # M[k][k−1] 的两个对角元
                m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
                m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
                det = m00 * m11 - m01 * m10
                i00 = m11 / det; i01 = -m01 / det
                i10 = -m10 / det; i11 = m00 / det
                sup = -dt * ((k + 1) * bb7)           # b_{k+1}
                CP00[k] = i00 * sup; CP01[k] = i01 * sup
                CP10[k] = i10 * sup; CP11[k] = i11 * sup
                rr0 = x7[2 * k] - s0u * DP0[k - 1]
                rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
                DP0[k] = i00 * rr0 + i01 * rr1
                DP1[k] = i10 * rr0 + i11 * rr1
            x7[8] = DP0[4]; x7[9] = DP1[4]
            for k in range(3, -1, -1):
                x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
                x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
            o7 = x7[1] + x7[3] + x7[5] + x7[7] + x7[9]
            k3t = p[4] * np.exp(p[3] * v)
            k4t = p[6] * np.exp(-p[5] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[8] * np.exp(p[9] * v)
            kc4t = p[10] * np.exp(-p[11] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[12] + p[13] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[7] * o7
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 39:
            # K6a7J(39,22参)（v1.46，代码163 门控电流联合拟合卡）：
            # 十态梯形 (k=0..4)×{C,O} 同 38；VSD 轴放开 3 参（p[19]/p[20]/p[21]，H1 对称保留）：
            #   a(ve)=p[19]·e^(p[20]·ve)  b(ve)=p[21]·e^(−p[20]·ve)；Vh 导出=−ln(p19/p21)/(2·p20)
            #   f_k=(4−k)a，b_k=k·b；O 行 VSD 前向 ×θ（环细致平衡）；
            #   垂直 C_k⇄O_k：开 c·L·θ^k / 关 c。
            # p[0]=c p[1]=L p[2]=θ p[3]=zk3 p[4]=k3_0 p[5]=zk4 p[6]=k4_0 p[7]=g
            # p[8..11]=r2 p[12]/p[13]=zr p[14..18]=s 模块(τ0,vh,δ,ks,z_τ)；输出=g·(ΣO_k)·rmix·(v−EK)
            # （r1 速率槽 4/6 与斜率槽 3/5 换位：共享头部 tr=1/(k3+k4) 用 p[4..7]，
            #   必须恒正 → 4/6 锁 log 槽；34 不崩同理——其 4/5/6 皆 log 槽）
            # 更新=后向 Euler 2×2 块 Thomas（列约定；守恒/保正/无条件稳定）
            if abs(p[17]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[14] * np.exp(p[18] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[16] * s
            aa7 = p[19] * np.exp(p[20] * ve_a)
            bb7 = p[21] * np.exp(-p[20] * ve_a)
            c7 = p[0]; L7 = p[1]; th7 = p[2]
            thp = 1.0
            for k in range(5):
                fk7 = (4.0 - k) * aa7
                if k == 4:
                    fk7 = 0.0
                bk7 = k * bb7
                ok7 = c7 * L7 * thp
                A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
                A01[k] = -dt * c7
                A10[k] = -dt * ok7
                A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7)
                U00[k] = -dt * fk7
                U11[k] = -dt * th7 * fk7
                thp *= th7
            # 前向块扫描（列约定 M=I−dtA）：
            #   次对角块 M[k][k−1]=−dt·G_{k−1}=diag(U00[k−1],U11[k−1])（前向速率）
            #   超对角块 M[k][k+1]=−dt·b_{k+1}·I（标量）；CP_k 携带超对角、W_k 消次对角
            det = A00[0] * A11[0] - A01[0] * A10[0]
            i00 = A11[0] / det; i01 = -A01[0] / det
            i10 = -A10[0] / det; i11 = A00[0] / det
            sup = -dt * bb7            # b_1 = 1·bb7
            CP00[0] = i00 * sup; CP01[0] = i01 * sup
            CP10[0] = i10 * sup; CP11[0] = i11 * sup
            DP0[0] = i00 * x7[0] + i01 * x7[1]
            DP1[0] = i10 * x7[0] + i11 * x7[1]
            for k in range(1, 5):
                s0u = U00[k - 1]; s1u = U11[k - 1]   # M[k][k−1] 的两个对角元
                m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
                m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
                det = m00 * m11 - m01 * m10
                i00 = m11 / det; i01 = -m01 / det
                i10 = -m10 / det; i11 = m00 / det
                sup = -dt * ((k + 1) * bb7)           # b_{k+1}
                CP00[k] = i00 * sup; CP01[k] = i01 * sup
                CP10[k] = i10 * sup; CP11[k] = i11 * sup
                rr0 = x7[2 * k] - s0u * DP0[k - 1]
                rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
                DP0[k] = i00 * rr0 + i01 * rr1
                DP1[k] = i10 * rr0 + i11 * rr1
            x7[8] = DP0[4]; x7[9] = DP1[4]
            for k in range(3, -1, -1):
                x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
                x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
            o7 = x7[1] + x7[3] + x7[5] + x7[7] + x7[9]
            k3t = p[4] * np.exp(p[3] * v)
            k4t = p[6] * np.exp(-p[5] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[8] * np.exp(p[9] * v)
            kc4t = p[10] * np.exp(-p[11] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[12] + p[13] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[7] * o7
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 41 or model == 42:
            # K6a7Ji(41,24参)/K6a7Jn(42,14参)（v1.47，代码164 失活门进骨架卡）：
            # 39 块同构十态梯形 + 失活层 O_k⇄I_k（k=0..4；I 冻结 VSD 位置、不导电）：
            #   fi(v)=p[22]·e^(ZI·v)、bi(v)=p[23]·e^(−ZI·v)，ZI=0.6/25.693（Wang1997 z=±0.6e0）；
            #   每步梯形 Thomas 后对每层 O_k⇄I_k 做精确弛豫子步（两态解析解，守恒/保正/任意 λdt 稳定）；
            #   p[22]=KI、p[23]=KB 锁 log 槽恒正；KI=0 时子步恒等 → 与 model 39 逐位一致（自验证口径）。
            # 41=rmix 自由臂：输出=g·(ΣO_k)·rmix·(v−EK)；42=rmix 卸下臂：输出=g·(ΣO_k)·(v−EK)。
            if abs(p[17]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[14] * np.exp(p[18] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[16] * s
            aa7 = p[19] * np.exp(p[20] * ve_a)
            bb7 = p[21] * np.exp(-p[20] * ve_a)
            c7 = p[0]; L7 = p[1]; th7 = p[2]
            thp = 1.0
            for k in range(5):
                fk7 = (4.0 - k) * aa7
                if k == 4:
                    fk7 = 0.0
                bk7 = k * bb7
                ok7 = c7 * L7 * thp
                A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
                A01[k] = -dt * c7
                A10[k] = -dt * ok7
                A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7)
                U00[k] = -dt * fk7
                U11[k] = -dt * th7 * fk7
                thp *= th7
            # 前向块扫描（列约定 M=I−dtA）：
            #   次对角块 M[k][k−1]=−dt·G_{k−1}=diag(U00[k−1],U11[k−1])（前向速率）
            #   超对角块 M[k][k+1]=−dt·b_{k+1}·I（标量）；CP_k 携带超对角、W_k 消次对角
            det = A00[0] * A11[0] - A01[0] * A10[0]
            i00 = A11[0] / det; i01 = -A01[0] / det
            i10 = -A10[0] / det; i11 = A00[0] / det
            sup = -dt * bb7            # b_1 = 1·bb7
            CP00[0] = i00 * sup; CP01[0] = i01 * sup
            CP10[0] = i10 * sup; CP11[0] = i11 * sup
            DP0[0] = i00 * x7[0] + i01 * x7[1]
            DP1[0] = i10 * x7[0] + i11 * x7[1]
            for k in range(1, 5):
                s0u = U00[k - 1]; s1u = U11[k - 1]   # M[k][k−1] 的两个对角元
                m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
                m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
                det = m00 * m11 - m01 * m10
                i00 = m11 / det; i01 = -m01 / det
                i10 = -m10 / det; i11 = m00 / det
                sup = -dt * ((k + 1) * bb7)           # b_{k+1}
                CP00[k] = i00 * sup; CP01[k] = i01 * sup
                CP10[k] = i10 * sup; CP11[k] = i11 * sup
                rr0 = x7[2 * k] - s0u * DP0[k - 1]
                rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
                DP0[k] = i00 * rr0 + i01 * rr1
                DP1[k] = i10 * rr0 + i11 * rr1
            x7[8] = DP0[4]; x7[9] = DP1[4]
            for k in range(3, -1, -1):
                x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
                x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
            # 失活子步（O_k⇄I_k 精确弛豫；λ=fi+bi，e^(−λdt)）
            fi7 = p[22] * np.exp((0.6 / 25.693) * v)
            bi7 = p[23] * np.exp(-(0.6 / 25.693) * v)
            lam7 = fi7 + bi7
            if lam7 > 0.0:
                ei7 = np.exp(-lam7 * dt)
                fro = bi7 / lam7
                for k in range(5):
                    Tk = x7[2 * k + 1] + xi5[k]
                    x7[2 * k + 1] = Tk * fro + (x7[2 * k + 1] - Tk * fro) * ei7
                    xi5[k] = Tk - x7[2 * k + 1]
            o7 = x7[1] + x7[3] + x7[5] + x7[7] + x7[9]
            if model == 42:
                out[i] = p[7] * o7 * (v - EK)
                continue
            k3t = p[4] * np.exp(p[3] * v)
            k4t = p[6] * np.exp(-p[5] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[8] * np.exp(p[9] * v)
            kc4t = p[10] * np.exp(-p[11] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[12] + p[13] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[7] * o7
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 43 or model == 44 or model == 45 or model == 47:
            # K6a8Ji(43,24参)/K6a9Ji(44,26参)/K6a8Jd(45,24参)（v1.48，代码165 恢复拓扑变体卡）：
            # 梯形/VSD/rmix 输出与 model 41 逐位同构；差异只在失活子步：
            #   43：串行回流同层——O_k→I_k(fi)、I_k→C_k(bi)（恢复出口落同层关闭库）；
            #   45：串行回流深复位——O_k→I_k(fi)、I_k→C_0(bi)（出口汇到最深关闭态，重爬全梯）；
            #       两者均为精确链式解析子步（O→I→C，C 在子步内吸收；守恒/保正）。
            #   44：Lu 式外加 C_4⇄I_c（fi_c=p[24]·e^(ZI·v)、bi_c=p[25]·e^(−ZI·v)，
            #       I_c 冻结 VSD 位置不导电）+ 原 O_k⇄I_k 不动。
            # 退化自验证：43/45 在 p[22]=p[23]=0、44 在 p[24]=p[25]=0 时与 model 41 逐位一致。
            #   47（v1.49，代码167 R1 行）：44+κ(s)——垂直开闭步 C_k⇄O_k 双向时标 κ=e^(−p[31]·s)；
            #       平衡比不变（KG 安全）、动力学变慢；β=0 时与 44 逐位一致（退化锚）。
            if abs(p[17]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[14] * np.exp(p[18] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[16] * s
            aa7 = p[19] * np.exp(p[20] * ve_a)
            bb7 = p[21] * np.exp(-p[20] * ve_a)
            c7 = p[0]; L7 = p[1]; th7 = p[2]
            kap = np.exp(-p[31] * s) if model == 47 else 1.0
            c7e = c7 * kap  # v1.49：IEEE x*1.0=x → 43/44/45 逐位不变
            thp = 1.0
            for k in range(5):
                fk7 = (4.0 - k) * aa7
                if k == 4:
                    fk7 = 0.0
                bk7 = k * bb7
                ok7 = c7e * L7 * thp
                A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
                A01[k] = -dt * c7e
                A10[k] = -dt * ok7
                A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7e)
                U00[k] = -dt * fk7
                U11[k] = -dt * th7 * fk7
                thp *= th7
            det = A00[0] * A11[0] - A01[0] * A10[0]
            i00 = A11[0] / det; i01 = -A01[0] / det
            i10 = -A10[0] / det; i11 = A00[0] / det
            sup = -dt * bb7
            CP00[0] = i00 * sup; CP01[0] = i01 * sup
            CP10[0] = i10 * sup; CP11[0] = i11 * sup
            DP0[0] = i00 * x7[0] + i01 * x7[1]
            DP1[0] = i10 * x7[0] + i11 * x7[1]
            for k in range(1, 5):
                s0u = U00[k - 1]; s1u = U11[k - 1]
                m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
                m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
                det = m00 * m11 - m01 * m10
                i00 = m11 / det; i01 = -m01 / det
                i10 = -m10 / det; i11 = m00 / det
                sup = -dt * ((k + 1) * bb7)
                CP00[k] = i00 * sup; CP01[k] = i01 * sup
                CP10[k] = i10 * sup; CP11[k] = i11 * sup
                rr0 = x7[2 * k] - s0u * DP0[k - 1]
                rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
                DP0[k] = i00 * rr0 + i01 * rr1
                DP1[k] = i10 * rr0 + i11 * rr1
            x7[8] = DP0[4]; x7[9] = DP1[4]
            for k in range(3, -1, -1):
                x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
                x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
            fi7 = p[22] * np.exp((0.6 / 25.693) * v)
            bi7 = p[23] * np.exp(-(0.6 / 25.693) * v)
            if model == 43 or model == 45:
                # 串行回流精确子步：O→I(fi)→C(bi)，C 子步内吸收
                efi = np.exp(-fi7 * dt); ebi = np.exp(-bi7 * dt)
                if fi7 > 0.0 or bi7 > 0.0:
                    if abs(fi7 - bi7) <= 1e-9 * max(max(fi7, bi7), 1.0):
                        for k in range(5):
                            O0 = x7[2 * k + 1]; I0 = xi5[k]
                            O1 = O0 * efi
                            I1 = (I0 + fi7 * O0 * dt) * ebi
                            dm = (O0 - O1) + (I0 - I1)
                            x7[2 * k + 1] = O1; xi5[k] = I1
                            if model == 45:
                                x7[0] += dm
                            else:
                                x7[2 * k] += dm
                    else:
                        cf = fi7 / (bi7 - fi7)
                        for k in range(5):
                            O0 = x7[2 * k + 1]; I0 = xi5[k]
                            O1 = O0 * efi
                            I1 = I0 * ebi + cf * O0 * (efi - ebi)
                            dm = (O0 - O1) + (I0 - I1)
                            x7[2 * k + 1] = O1; xi5[k] = I1
                            if model == 45:
                                x7[0] += dm
                            else:
                                x7[2 * k] += dm
            else:
                # model 44：O_k⇄I_k 原样 + C_4⇄I_c 精确对弛豫
                lam7 = fi7 + bi7
                if lam7 > 0.0:
                    ei7 = np.exp(-lam7 * dt)
                    fro = bi7 / lam7
                    for k in range(5):
                        Tk = x7[2 * k + 1] + xi5[k]
                        x7[2 * k + 1] = Tk * fro + (x7[2 * k + 1] - Tk * fro) * ei7
                        xi5[k] = Tk - x7[2 * k + 1]
                fic = p[24] * np.exp((0.6 / 25.693) * v)
                bic = p[25] * np.exp(-(0.6 / 25.693) * v)
                lamc = fic + bic
                if lamc > 0.0:
                    eic = np.exp(-lamc * dt)
                    frc = bic / lamc
                    Tc = x7[8] + xiC
                    x7[8] = Tc * frc + (x7[8] - Tc * frc) * eic
                    xiC = Tc - x7[8]
            o7 = x7[1] + x7[3] + x7[5] + x7[7] + x7[9]
            k3t = p[4] * np.exp(p[3] * v)
            k4t = p[6] * np.exp(-p[5] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[8] * np.exp(p[9] * v)
            kc4t = p[10] * np.exp(-p[11] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[12] + p[13] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[7] * o7
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 35:
            # K6a6(35,24参)（v1.42，代码158 柔性协同生灭链卡）：
            # a 侧 = 四亚基生灭链 k=0..4 + C4⇄O 协同门步（六态）：
            #   f_k(ve)=(4−k)·α(ve)·γ^k  b_k(ve)=k·β(ve)  k=0..3（生灭前段）
            #   k4 -KF- O（KF=p5 电压不依赖=天花板来源） O -b2(ve)- k4（去激活来源）
            #   α(ve)=p0·e^(p1·ve)  β(ve)=p2·e^(−p3·ve)  γ=p4（电压不依赖协同因子）
            #   b2(ve)=p6·e^(−p7·ve)  ve = v + p21·s
            #   稳态=平均场Ising：p_k∝C(4,k)·r^k·γ^{k(k−1)/2}（γ=1独立/≫1 MWC/中间 KNF）
            #   p[8..11]=r1  p[12]=g  p[13..16]=r2  p[17]/p[18]=zr 截距/斜率
            #   p[19..23]=s 模块(τ0,vh,δ,ks,z_τ)；输出=线性 o（同 v1.41 口径）
            # 链更新=后向 Euler Thomas 三对角（同 v1.41，六行；守恒/保正/无条件稳定）
            if abs(p[22]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[19] * np.exp(p[23] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[20]) / p[22]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[21] * s
            aa6 = p[0] * np.exp(p[1] * ve_a)
            bb6 = p[2] * np.exp(-p[3] * ve_a)
            g6 = p[4]
            f60 = 4.0 * aa6
            f61 = 3.0 * aa6 * g6
            f62 = 2.0 * aa6 * g6 * g6
            f63 = aa6 * g6 * g6 * g6
            gb6 = p[6] * np.exp(-p[7] * ve_a)
            d0 = 1.0 + dt * f60
            d1 = 1.0 + dt * (bb6 + f61)
            d2 = 1.0 + dt * (2.0 * bb6 + f62)
            d3 = 1.0 + dt * (3.0 * bb6 + f63)
            d4 = 1.0 + dt * (4.0 * bb6 + p[5])
            d5 = 1.0 + dt * gb6
            cp0 = (-dt * bb6) / d0
            dp0 = cc1 / d0
            m1 = d1 - (-dt * f60) * cp0
            cp1 = (-dt * 2.0 * bb6) / m1
            dp1 = (cc2 - (-dt * f60) * dp0) / m1
            m2 = d2 - (-dt * f61) * cp1
            cp2 = (-dt * 3.0 * bb6) / m2
            dp2 = (cc3 - (-dt * f61) * dp1) / m2
            m3 = d3 - (-dt * f62) * cp2
            cp3 = (-dt * 4.0 * bb6) / m3
            dp3 = (cc4 - (-dt * f62) * dp2) / m3
            m4 = d4 - (-dt * f63) * cp3
            cp4 = (-dt * gb6) / m4
            dp4 = (cc5 - (-dt * f63) * dp3) / m4
            m5 = d5 - (-dt * p[5]) * cp4
            dp5 = (oo - (-dt * p[5]) * dp4) / m5
            oo = dp5
            cc5 = dp4 - cp4 * oo
            cc4 = dp3 - cp3 * cc5
            cc3 = dp2 - cp2 * cc4
            cc2 = dp1 - cp1 * cc3
            cc1 = dp0 - cp0 * cc2
            k3t = p[8] * np.exp(p[9] * v)
            k4t = p[10] * np.exp(-p[11] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[13] * np.exp(p[14] * v)
            kc4t = p[15] * np.exp(-p[16] * v)
            tct = 1.0 / (kc3t + kc4t)
            zr4 = min(max(p[17] + p[18] * v, -30.0), 30.0)
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            out[i] = (p[12] * oo
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 31:
            # K6a2c(31)（v1.37，代码153态依赖失活卡）：K6a2 + r 前进速率 ×(1+κ·amix²)，κ=p[26]；κ=0 逐位=K6a2(29)
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * ve_a)
            kb2t = p[11] * np.exp(-p[12] * ve_a)
            tbt = 1.0 / (kb1t + kb2t)
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            am4c_pre = wa4 * a1 + (1.0 - wa4) * a2   # 耦合用上一步 amix（显式 Euler 口径）
            coup = 1.0 + p[26] * am4c_pre * am4c_pre
            if coup < 0.0:
                out[i] = np.nan
                continue
            k3t = p[4] * np.exp(p[5] * v) * coup
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v) * coup
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = (1.0 / (kc3t + kc4t))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4c = wa4 * a1 + (1.0 - wa4) * a2   # 输出用本步更新后 amix（与 29 支逐行同序）
            out[i] = (g * am4c * am4c
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 28:         # K8(28,26参)=142a: K6a骨架，快门a2回裸坐标(慢门a1留v+δ·s)
            if abs(p[24]) < 1.0:
                out[i] = np.nan
                continue
            tau_v = p[21] * np.exp(p[25] * v)
            if tau_v < 1.0:
                out[i] = np.nan
                continue
            s_inf = 1.0 / (1.0 + np.exp(-(v - p[22]) / p[24]))
            s += dt * (s_inf - s) / tau_v
            ve_a = v + p[23] * s
            k1t = p[0] * np.exp(p[1] * ve_a)
            k2t = p[2] * np.exp(-p[3] * ve_a)
            tat = 1.0 / (k1t + k2t)
            kb1t = p[9] * np.exp(p[10] * v)
            kb2t = p[11] * np.exp(-p[12] * v)
            tbt = 1.0 / (kb1t + kb2t)
            k3t = p[4] * np.exp(p[5] * v)
            k4t = p[6] * np.exp(-p[7] * v)
            trt = 1.0 / (k3t + k4t)
            kc3t = p[14] * np.exp(p[15] * v)
            kc4t = p[16] * np.exp(-p[17] * v)
            tct = 1.0 / (kc3t + kc4t)
            za4 = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr4 = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa4 = 1.0 / (1.0 + np.exp(-za4))
            wr4 = 1.0 / (1.0 + np.exp(-zr4))
            a1 += dt * ((k1t * tat) - a1) / tat
            a2 += dt * ((kb1t * tbt) - a2) / tbt
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4t * tct) - r2) / tct
            am4 = wa4 * a1 + (1.0 - wa4) * a2
            out[i] = (g * am4 * am4 * am4
                      * (wr4 * r1 + (1.0 - wr4) * r2) * (v - EK))
        elif model == 16:         # K1         # K1: M2f3 + k3'=k3(1+κ·am³), k4'=k4(1+λ·am³)（整蛋白耦合入场考）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            am = wa * a1 + (1.0 - wa) * a2
            f3 = 1.0 + p[21] * am * am * am
            f4 = 1.0 + p[22] * am * am * am
            if f3 < 1e-9 or f4 < 1e-9:
                out[i] = np.nan
                continue
            k3t = k3 * f3
            k4t = k4 * f4
            trt = 1.0 / (k3t + k4t)
            r1 += dt * ((k4t * trt) - r1) / trt
            r2 += dt * ((kc4 * tc) - r2) / tc
            out[i] = (g * (am * am * am)
                      * (wr * r1 + (1.0 - wr) * r2) * (v - EK))
        elif model == 14:         # M2fr2: M2f + rmix^2（r 侧协同放大 2）         # M2fr2: M2f + rmix^2（r 侧协同放大 2）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            rm = wr * r1 + (1.0 - wr) * r2
            out[i] = (g * (wa * a1 + (1.0 - wa) * a2)
                      * (rm * rm) * (v - EK))
        elif model == 15:         # M2fr3: M2f + rmix^3（r 侧协同放大 3）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            rm = wr * r1 + (1.0 - wr) * r2
            out[i] = (g * (wa * a1 + (1.0 - wa) * a2)
                      * (rm * rm * rm) * (v - EK))
        elif model == 8:          # M2h: M2f + a 侧第三慢分量（w3 恒定）
            kb1 = p[9] * np.exp(p[10] * v)
            kb2 = p[11] * np.exp(-p[12] * v)
            tb = 1.0 / (kb1 + kb2)
            kc3 = p[14] * np.exp(p[15] * v)
            kc4 = p[16] * np.exp(-p[17] * v)
            tc = 1.0 / (kc3 + kc4)
            kd1 = p[21] * np.exp(p[22] * v)
            kd2 = p[23] * np.exp(-p[24] * v)
            td = 1.0 / (kd1 + kd2)
            za = min(max(p[13] + p[19] * v, -30.0), 30.0)
            zr = min(max(p[18] + p[20] * v, -30.0), 30.0)
            z3 = min(max(p[25], -30.0), 30.0)
            wa = 1.0 / (1.0 + np.exp(-za))
            wr = 1.0 / (1.0 + np.exp(-zr))
            w3 = 1.0 / (1.0 + np.exp(-z3))
            a1 += dt * ((k1 * ta) - a1) / ta
            a2 += dt * ((kb1 * tb) - a2) / tb
            a3 += dt * ((kd1 * td) - a3) / td
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((kc4 * tc) - r2) / tc
            amix = (1.0 - w3) * (wa * a1 + (1.0 - wa) * a2) + w3 * a3
            out[i] = (g * amix * (wr * r1 + (1.0 - wr) * r2) * (v - EK))
        elif model == 6:          # M1: 串联 r2
            k5 = p[10] * np.exp(p[9] * v)
            k6 = p[12] * np.exp(-p[11] * v)
            t2 = 1.0 / (k5 + k6)
            a1 += dt * ((k1 * ta) - a1) / ta
            r1 += dt * ((k4 * tr) - r1) / tr
            r2 += dt * ((k6 * t2) - r2) / t2
            out[i] = g * a1 * r1 * r2 * (v - EK)
        elif model == 9:          # M0t2: I = g·a^2·r·(V−EK)
            a1 += dt * ((k1 * ta) - a1) / ta
            r1 += dt * ((k4 * tr) - r1) / tr
            out[i] = g * (a1 * a1) * r1 * (v - EK)
        elif model == 10:         # M0t3: I = g·a^3·r·(V−EK)
            a1 += dt * ((k1 * ta) - a1) / ta
            r1 += dt * ((k4 * tr) - r1) / tr
            out[i] = g * (a1 * a1 * a1) * r1 * (v - EK)
        elif model == 11:         # M0t4: I = g·(a*a)^2·r·(V−EK)
            a1 += dt * ((k1 * ta) - a1) / ta
            r1 += dt * ((k4 * tr) - r1) / tr
            aa = a1 * a1
            out[i] = g * (aa * aa) * r1 * (v - EK)
        else:                     # M0 / M2d / M2e
            a1 += dt * ((k1 * ta) - a1) / ta
            r1 += dt * ((k4 * tr) - r1) / tr
            out[i] = g * a1 * r1 * (v - EK)
    return out


@njit(cache=True, fastmath=True)
def sim_nav_sweep(p, Vcmd, rs_eff, cm, dt_ms):
    """N1/N2 单扫点：HH m³h(+h2) + Rs/Cm 钳制层动态（v1.21，代码123）。
    p=N1(12)=[kma,zma,kmb,zmb,kha,zha,khb,zhb,g,ENa,gl,Voff]；
    N2(16)=[m 4, h 4, h2 4, g, ENa, gl, Voff]（率 /ms，g/gl nS，电压 mV）；
    Vcmd=mV 命令电位数组；rs_eff=meta Rs×(1−CP/100)，MΩ；dt_ms=0.04。
    返回总电流预测 I_sim（pA）。守门：斜率·电压 clip ±30；分母下溢保护。"""
    n = len(Vcmd)
    out = np.empty(n)
    kma, zma, kmb, zmb = p[0], p[1], p[2], p[3]
    kha, zha, khb, zhb = p[4], p[5], p[6], p[7]
    g, ENa, gl, Voff = p[8], p[9], p[10], p[11]
    is_n2 = len(p) == 16
    if is_n2:                    # N2：h2 慢失活分量（v1.20）
        kf2, zf2, kb2, zb2 = p[8], p[9], p[10], p[11]
        g, ENa, gl, Voff = p[12], p[13], p[14], p[15]
    vm = Vcmd[0] + Voff          # 段首：holding 电位平衡
    # 段首稳态锁定（hERG 内核同传统）
    rfa = kma * np.exp(min(max(zma * vm, -30.0), 30.0))
    rba = kmb * np.exp(min(max(-zmb * vm, -30.0), 30.0))
    rfh = kha * np.exp(min(max(zha * vm, -30.0), 30.0))
    rbh = khb * np.exp(min(max(-zhb * vm, -30.0), 30.0))
    am = rfa / max(rfa + rba, 1e-300)
    ah = rfh / max(rfh + rbh, 1e-300)
    ah2 = 1.0
    if is_n2:
        rf2 = kf2 * np.exp(min(max(zf2 * vm, -30.0), 30.0))
        rb2 = kb2 * np.exp(min(max(-zb2 * vm, -30.0), 30.0))
        ah2 = rf2 / max(rf2 + rb2, 1e-300)
    for i in range(n):
        # ① 门控指数精确积分一步（用当前 vm）
        rfa = kma * np.exp(min(max(zma * vm, -30.0), 30.0))
        rba = kmb * np.exp(min(max(-zmb * vm, -30.0), 30.0))
        rfh = kha * np.exp(min(max(zha * vm, -30.0), 30.0))
        rbh = khb * np.exp(min(max(-zhb * vm, -30.0), 30.0))
        tm = 1.0 / max(rfa + rba, 1e-300)
        th = 1.0 / max(rfh + rbh, 1e-300)
        am += (rfa * tm - am) * (1.0 - np.exp(-dt_ms / tm))
        ah += (rfh * th - ah) * (1.0 - np.exp(-dt_ms / th))
        if is_n2:
            rf2 = kf2 * np.exp(min(max(zf2 * vm, -30.0), 30.0))
            rb2 = kb2 * np.exp(min(max(-zb2 * vm, -30.0), 30.0))
            t2 = 1.0 / max(rf2 + rb2, 1e-300)
            ah2 += (rf2 * t2 - ah2) * (1.0 - np.exp(-dt_ms / t2))
        # ② 钳制层 Cm 动态（v1.21）：G 冻结时 vm 线性 ODE → 指数精确积分
        G = g * am * am * am * ah * ah2
        vc = Vcmd[i] + Voff
        bb = 1e3 / rs_eff + G + gl                 # nS
        vm_inf = (1e3 / rs_eff * vc + G * ENa) / bb
        tv = cm / bb                               # ms
        vm += (vm_inf - vm) * (1.0 - np.exp(-dt_ms / tv))
        # ③ 总电流（钠 + 漏；电容电流设 P 补偿理想中和，v1.21 双录）
        out[i] = G * (vm - ENa) + gl * vm
    return out


# ── v1.22（代码124）：放大器链钳制层（官方 nav-artefact-model voltage-clamp.mmt 复刻）──
# 结构来源：Lei/Clark/Clercx 等 Eq.3–9（GitHub CardiacModelling/nav-artefact-model，公开直链）。
# 延迟常数固定（Δk=0 不动判线）：τ_stim=20µs τ_sum=1µs τ_clamp=0.8µs τ_out=0.2µs；
# C_prs=0（作者亦不拟合）；α_R=α_P=CP/100（官方 get_naiv_alphas 同口径）；
# Rs*/Cm*=meta 实测值且真值=估计值（本批冻结，自由度留作 125 候选）；
# 观测=I_post=I_out−g_leak*·(V_c−E_leak)，g_leak*=1e3/rseal，E_leak=−80mV 固定。
TAU_STIM, TAU_SUM, TAU_CLAMP, TAU_OUT, E_LEAK_POST = 0.02, 0.001, 0.0008, 0.0002, -80.0

# ── v1.25（代码127）：EPC-10 输出滤波级联（固定项，Δk=0 不动判线）──
# 结构来源：Patchliner=HEKA EPC-10 放大器（多源文献实证）；HEKA 手册——
# Filter 1 = 6 极点 Bessel 10kHz（电流通道固定前级），Filter 2 = 4 极点 Bessel
# 5kHz（25kHz 采样的 Patchmaster 标准对）；τ_stim=20µs 即 EPC-10 StimFilter 20µs
# 档（v1.22 已在链内）。数字 bessel(norm='phase', fs=25kHz) 级联逼近模拟级联，
# 双录：10kHz=0.8 Nyquist，双线性逼近有畸变，作为一阶口径。
# 依据（§94）：实测数据 ~0.16ms 固定延迟 vs 级联群延迟 0.176ms 吻合；
# 滤波作用于物理电流通道（iin 之后、iout 之前）。filt=None 时逐位同 v1.24。
EPC10_SOS = None   # run_nav 按 card["filt"]=="epc10" 以 scipy 生成 (5,5) [b0,b1,b2,a1,a2]
EPC10_MB = None    # v1.26：filt=="epc10z" 模态块 (5,6) [c,d,er,ei,rR,rI]（ZOH 精确离散）


@njit(cache=True, fastmath=True)
def _ex1(x):
    return 1.0 - np.exp(-x)


@njit(cache=True, fastmath=True)
def sim_nav_sweep_amp(p, Vcmd, alpha, rs, rseal, cm, dt_ms, c_prs=0.0,
                      sos=None, modal=None):
    """N1/N2 单扫点：HH m³h(+h2) + 全放大器链钳制层（v1.22，代码124）。
    alpha=CP/100（α_R=α_P）；rs/rseal MΩ（真值=放大器估计，冻结）；cm pF。
    v1.24（代码126）：c_prs=寄生电容固定值（pF），官方 mmt 双侧原式
    +C_prs·dV_p/dt − C_prs_est·dV_clamp/dt、估计=真值（作者不拟合 C_prs）；
    固定项不进拟合，Δk=0。稳定域冒烟：≤0.5pF 稳、≥1pF 发散（1e30 守卫接住）。
    返回 I_post 预测（pA，漏流后处理口径，与 CSV 对齐）。"""
    n = len(Vcmd)
    out = np.empty(n)
    kma, zma, kmb, zmb = p[0], p[1], p[2], p[3]
    kha, zha, khb, zhb = p[4], p[5], p[6], p[7]
    g, ENa, gl, Voff = p[8], p[9], p[10], p[11]
    is_n2 = len(p) in (16, 18)
    if is_n2:
        kf2, zf2, kb2, zb2 = p[8], p[9], p[10], p[11]
        g, ENa, gl, Voff = p[12], p[13], p[14], p[15]
    # v1.23：N1r/N2r（len 14/18）真值 Rs/Cm 自由度——放大器估计侧仍用 meta 值
    rs_t, cm_t = rs, cm
    if len(p) in (14, 18):
        rs_t = rs * p[-2]
        cm_t = cm * p[-1]
    if len(p) == 19:
        # v1.27 N1r5：rs_scale 按 CP 分档（cp_idx=round(alpha*5)∈[0,4]）
        cp_idx = int(round(alpha * 5.0))
        assert 0.0 <= cp_idx <= 4, f"N1r5: alpha={alpha} 越出五档"
        rs_t = rs * p[12 + cp_idx]
        cm_t = cm * p[17]
    rs_g = rs_t * 1e-3                     # 真值 MΩ → GΩ（膜方程/串阻电流用）
    rs_est_g = rs * 1e-3                   # 估计值（补偿环路与超充用）
    gl_est = 1e3 / rseal                   # nS = pA/mV
    tau_est = max((1.0 - alpha) * cm * rs_est_g, 1e-6)   # ms（α_P<1 已断言）
    # v1.25：滤波状态（每扫点清零）
    if sos is not None:
        fs1 = np.zeros(sos.shape[0])
        fs2 = np.zeros(sos.shape[0])
    else:
        fs1 = np.zeros(1)
        fs2 = np.zeros(1)
    # v1.26：模态滤波状态（每扫点清零；与 sos 互斥，run_nav 保证）
    if modal is not None:
        mxR = np.zeros(modal.shape[0])
        mxI = np.zeros(modal.shape[0])
    else:
        mxR = np.zeros(1)
        mxI = np.zeros(1)
    v0 = Vcmd[0]
    vc, vest, vclamp, vp, iout = v0, v0, v0, v0, 0.0
    vm = v0 + Voff
    # 段首稳态锁定（门控）
    rfa = kma * np.exp(min(max(zma * vm, -30.0), 30.0))
    rba = kmb * np.exp(min(max(-zmb * vm, -30.0), 30.0))
    rfh = kha * np.exp(min(max(zha * vm, -30.0), 30.0))
    rbh = khb * np.exp(min(max(-zhb * vm, -30.0), 30.0))
    am = rfa / max(rfa + rba, 1e-300)
    ah = rfh / max(rfh + rbh, 1e-300)
    ah2 = 1.0
    if is_n2:
        rf2 = kf2 * np.exp(min(max(zf2 * vm, -30.0), 30.0))
        rb2 = kb2 * np.exp(min(max(-zb2 * vm, -30.0), 30.0))
        ah2 = rf2 / max(rf2 + rb2, 1e-300)
    for i in range(n):
        # ① 放大器链（全精确指数小步）
        vc += (Vcmd[i] - vc) * _ex1(dt_ms / TAU_STIM)
        vest += (vc - vest) * _ex1(dt_ms / tau_est)
        dvest = (vc - vest) / tau_est                  # mV/ms
        vtgt = vc + (iout * alpha + cm * dvest * alpha) * rs_est_g  # 估计侧（v1.23）
        dvclamp = (vtgt - vclamp) / TAU_SUM  # mV/ms（v1.24：官方 C_prs 对消项用，更新前差）
        vclamp += (vtgt - vclamp) * _ex1(dt_ms / TAU_SUM)
        dvp = (vclamp - vp) / TAU_CLAMP      # mV/ms（v1.24：取更新前差，与 dvest 同规；
                                             # 取更新后差会被 e^-50 清零，冒烟曾捕获，双录）
        vp += (vclamp - vp) * _ex1(dt_ms / TAU_CLAMP)
        # ② 膜电位：G 冻结线性 ODE 精确积分（v1.21 同法，vp 驱动；真值 rs_t/cm_t）
        G = g * am * am * am * ah * ah2
        bb = 1.0 / (cm_t * rs_g) + (G + gl) / cm_t     # /ms
        vm_inf = ((vp + Voff) / (cm_t * rs_g) + G * ENa / cm_t) / bb
        vm += (vm_inf - vm) * _ex1(dt_ms * bb)
        # ③ 门控指数精确积分（用新 vm）
        rfa = kma * np.exp(min(max(zma * vm, -30.0), 30.0))
        rba = kmb * np.exp(min(max(-zmb * vm, -30.0), 30.0))
        rfh = kha * np.exp(min(max(zha * vm, -30.0), 30.0))
        rbh = khb * np.exp(min(max(-zhb * vm, -30.0), 30.0))
        tm = 1.0 / max(rfa + rba, 1e-300)
        th = 1.0 / max(rfh + rbh, 1e-300)
        am += (rfa * tm - am) * _ex1(dt_ms / tm)
        ah += (rfh * th - ah) * _ex1(dt_ms / th)
        if is_n2:
            rf2 = kf2 * np.exp(min(max(zf2 * vm, -30.0), 30.0))
            rb2 = kb2 * np.exp(min(max(-zb2 * vm, -30.0), 30.0))
            t2 = 1.0 / max(rf2 + rb2, 1e-300)
            ah2 += (rf2 * t2 - ah2) * _ex1(dt_ms / t2)
        # ④ 观测电流：I_in=串阻电流−Cm* 补偿电流(+C_prs 双侧对消项)；I_out 低通；I_post 漏流后处理
        # v1.24：官方 mmt 原式 +C_prs·dV_p − C_prs_est·dV_clamp，取估计=真值（作者不拟合 C_prs）；
        # 单边去掉对消项在 dt=0.04ms 显式反馈下环增益>1 必发散（c_prs≥0.5pF 冒烟发散，双录）。
        iin = (vp + Voff - vm) / rs_g - cm * dvest + c_prs * (dvp - dvclamp)
        iout += (iin - iout) * _ex1(dt_ms / TAU_OUT)
        # v1.25：EPC-10 Bessel 级联只作用监视/记录通道（反馈抽头在滤波器前——
        # 否则会往补偿环注入 0.18ms 环延迟必振，§95.1 冒烟 8.9e10 发散实证双录）；
        # DF2T，a0=1 已归一，每扫点状态清零。
        if sos is not None:
            y = iout
            for k in range(sos.shape[0]):
                xk = y
                y = sos[k, 0] * xk + fs1[k]
                fs1[k] = sos[k, 1] * xk - sos[k, 3] * y + fs2[k]
                fs2[k] = sos[k, 2] * xk - sos[k, 4] * y
            out[i] = y - gl_est * (vc - E_LEAK_POST)
        elif modal is not None:
            # v1.26：模态块（ZOH 精确离散）——输出取更新前状态（h[0]=0 严格真）
            y = 0.0
            for k in range(modal.shape[0]):
                y += 2.0 * (modal[k, 4] * mxR[k] - modal[k, 5] * mxI[k])
                nR = modal[k, 0] * mxR[k] - modal[k, 1] * mxI[k] + modal[k, 2] * iout
                mxI[k] = modal[k, 1] * mxR[k] + modal[k, 0] * mxI[k] + modal[k, 3] * iout
                mxR[k] = nR
            out[i] = y - gl_est * (vc - E_LEAK_POST)
        else:
            out[i] = iout - gl_est * (vc - E_LEAK_POST)
    return out


def unpack(x, model):
    """优化坐标 x → 物理参数 p（率取 10^x，w 取 sigmoid）。"""
    p = np.zeros(19)
    p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
    p[[1, 3, 5, 7]] = x[5:9]
    if model == "M1":
        # x 布局：[7 个 log 率][6 个斜率]，斜率须从 x[7:11] 读
        # （v1.3 修复：此前误用 x[5:9]，与 embed_x0 布局不一致；
        #   该分支在 v1.0–v1.2 中从未被任何任务卡执行，双录备查）
        p[[1, 3, 5, 7]] = x[7:11]
        p[[10, 12]] = 10.0 ** x[[5, 6]]
        p[[9, 11]] = x[[11, 12]]
    elif model in ("M2a", "M2c"):
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = 1.0 / (1.0 + np.exp(-x[13]))
    elif model == "M2d":
        p[9], p[10] = 10.0 ** x[9], 10.0 ** x[10]
    elif model == "M2e":
        p[9] = 10.0 ** x[9]
    elif model == "M2ac":
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = 1.0 / (1.0 + np.exp(-x[13]))
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = 1.0 / (1.0 + np.exp(-x[18]))
    elif model == "K2":
        # C3-C2-C1-O-I 五态 Markov；p=[k1_0,z1,k2_0,z2,k3_0,z3,k4_0,z4,k5_0,z5,k6_0,z6,g]
        p = np.zeros(23)
        p[[0, 2, 4, 6, 8, 10, 12]] = 10.0 ** x[:7]
        p[[1, 3, 5, 7, 9, 11]] = x[7:13]
    elif model == "K1":
        # M2f3 + 失活速率耦合激活占有度：p[21]=κ, p[22]=λ（线性坐标原样）
        p = np.zeros(23)
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
        p[21], p[22] = x[21], x[22]
    elif model in ("K4u", "K4n"):
        # p 长 25(K4u)/24(K4n) = M2f3 的 21 + τ_slow(ms, log10坐标) + δ(mV,线性)
        # + vh_s(mV,线性) + ρ(logit, 仅 K4u)
        p = np.zeros(25 if model == "K4u" else 24)
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
        p[21] = 10.0 ** x[21]
        p[22] = x[22]
        p[23] = x[23]
        if model == "K4u":
            p[24] = x[24]
    elif model == "K4m":
        # p 长 25 = M2f3 的 21 + τ_slow(ms,log10) + vh_s(mV) + δ(mV) + ks(mV,符号自由)
        p = np.zeros(25)
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
        p[21] = 10.0 ** x[21]
        p[22] = x[22]
        p[23] = x[23]
        p[24] = x[24]
    elif model == "K7":
        # p 长 30 = K6a 的 26 + vh_u(p26,mV) + τu(p27,秒,log10) + ku(p28,mV,符号自由) + λ(p29,线性)
        p = np.zeros(30)
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
        p[21] = 10.0 ** x[21]
        p[22] = x[22]
        p[23] = x[23]
        p[24] = x[24]
        p[25] = x[25]
        p[26] = x[26]
        p[27] = 10.0 ** x[27]
        p[28] = x[28]
        p[29] = x[29]
    elif model in ("K6a", "K6b", "K8", "K6a2", "K6a1", "K6a2c", "K6a2f"):
        # p 长 26 = K4m 的 25 + z_τ(K6a,线性) / γ(K6b,线性)；K6a2c 长 27 追加 p[26]=κ(线性)；K6a2f 长 28 追加 p[26]=kf_a1、p[27]=kf_a2（线性，/s）
        p = np.zeros(27 if model == "K6a2c" else (28 if model == "K6a2f" else 26))
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
        p[21] = 10.0 ** x[21]
        p[22] = x[22]
        p[23] = x[23]
        p[24] = x[24]
        p[25] = x[25]
        if model == "K6a2c":
            p[26] = x[26]
        if model == "K6a2f":
            p[26] = x[26]
            p[27] = x[27]
    elif model == "K6a3":
        # v1.40（代码156）：K6a3 23 参——p[0..6]=链(α0,zα,β0,zβ,f,b2_0,zb2)，
        # p[7..10]=r1，p[11]=g，p[12..15]=r2，p[16]/p[17]=zr 截距/斜率，
        # p[18..22]=s 模块(τ0,vh,δ,ks,z_τ)
        # x 坐标：log10 作用于 [0,2,4,5,7,9,11,12,14,18]，余者线性
        p = np.zeros(23)
        lg = [0, 2, 4, 5, 7, 9, 11, 12, 14, 18]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(23) if i not in lg]
        p[ln] = x[ln]
    elif model in ("K6a5", "K6a5o"):
        # v1.44：K6a5o(37) 与 K6a5 同布局同坐标（唯输出律 o² 差异在内核）
        # v1.41（代码157）：K6a5 26 参——p[0..9]=Wang链(α1_0,zα1,β1_0,zβ1,kf,kb,α2_0,zα2,β2_0,zβ2)，
        # p[10..13]=r1，p[14]=g，p[15..18]=r2，p[19]/p[20]=zr 截距/斜率，
        # p[21..25]=s 模块(τ0,vh,δ,ks,z_τ)
        # x 坐标：log10 作用于 [0,2,4,5,6,8,10,12,14,15,17,21]，余者线性
        p = np.zeros(26)
        lg = [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 17, 21]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(26) if i not in lg]
        p[ln] = x[ln]
    elif model == "K6a7":
        # v1.45（代码161）：K6a7 19 参——p[0..2]=c,L,θ；p[3..6]=r1；p[7]=g；p[8..11]=r2；
        # p[12]/p[13]=zr 截距/斜率；p[14..18]=s 模块(τ0,vh,δ,ks,z_τ)
        # VSD 轴无参数（文献锚死常量在内核）；log10 槽=[0,1,2,4,6,7,8,10,14]
        # （p[4]/p[6] 锁 log 槽恒正：共享头部 tr=1/(k3+k4) 防 0/0）
        p = np.zeros(19)
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(19) if i not in lg]
        p[ln] = x[ln]
    elif model == "K6a7J":
        # v1.46（代码163）：K6a7J 22 参——K6a7 的 19 参 + p[19]=A0(log)、p[20]=ZA(=ZB)、p[21]=B0(log)
        # VSD 轴放开受门控数据约束（H1 对称能垒保留；Vh 导出=−ln(A0/B0)/(ZA+ZB)）
        # log10 槽=[0,1,2,4,6,7,8,10,14,19,21]（19/21 锁 log 槽恒正；20 线性）
        p = np.zeros(22)
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(22) if i not in lg]
        p[ln] = x[ln]
    elif model == "K6a7Ji":
        # v1.47（代码164）：K6a7Ji 24 参 = K6a7J 22 参 + p[22]=KI(log)、p[23]=KB(log)
        # log10 槽=[0,1,2,4,6,7,8,10,14,19,21,22,23]
        p = np.zeros(24)
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21, 22, 23]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(24) if i not in lg]
        p[ln] = x[ln]
    elif model == "K6a9Ji":
        # v1.48（代码165）：K6a9Ji 26 参 = K6a7Ji 24 参 + p[24]=fi_c0(log)、p[25]=bi_c0(log)
        # log10 槽=Ji 槽 + 24, 25
        p = np.zeros(26)
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21, 22, 23, 24, 25]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(26) if i not in lg]
        p[ln] = x[ln]
    elif model == "K6a7Jn":
        # v1.47（代码164）：K6a7Jn 14 优化坐标 → 24 物理槽（rmix 卸下臂；惰性 rmix 槽置 0，内核 42 块不读）
        # x=[c,L,θ,g,τ0,vh,δ,ks,z_τ,A0,ZA,B0,KI,KB] → p 槽 [0,1,2,7,14,15,16,17,18,19,20,21,22,23]
        # log10 槽（x 下标）=[0,1,2,3,4,9,11,12,13]（c/L/θ/g/τ0/A0/B0/KI/KB 恒正）
        p = np.zeros(24)
        mp = [0, 1, 2, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        lg = [0, 1, 2, 3, 4, 9, 11, 12, 13]
        for i in range(14):
            p[mp[i]] = 10.0 ** x[i] if i in lg else x[i]
        # 惰性 rmix 槽置中性值（内核 42 块与 sim_gate41 均不读；但共享头部 tr=1/(k3+k4) 须有限）
        p[4] = 1.0; p[6] = 1.0; p[8] = 1.0; p[10] = 1.0
    elif model == "K6a5m":
        # v1.43（代码159）：K6a5m 物理 30 参——[0..25]=K6a5（坐标同 K6a5），
        # p[26]=λ_sa=1.0 语义冻结（参考窗不进优化），p[27..29]=λ_deact/λ_sine/λ_ap（log10 坐标）
        # x 长度 29 = K6a5 的 26 坐标 + 3 个 log10 λ
        p = np.zeros(30)
        lg = [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 17, 21]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(26) if i not in lg]
        p[ln] = x[ln]
        p[26] = 1.0
        p[27:30] = 10.0 ** x[26:29]
    elif model == "K6a6":
        # v1.42（代码158）：K6a6 24 参——p[0..7]=生灭链+门(α_0,zα,β_0,zβ,γ,KF,b2_0,zb2)，
        # p[8..11]=r1，p[12]=g，p[13..16]=r2，p[17]/p[18]=zr 截距/斜率，
        # p[19..23]=s 模块(τ0,vh,δ,ks,z_τ)
        # x 坐标：log10 作用于 [0,2,4,5,6,8,10,12,13,15,19]，余者线性
        p = np.zeros(24)
        lg = [0, 2, 4, 5, 6, 8, 10, 12, 13, 15, 19]
        p[lg] = 10.0 ** x[lg]
        ln = [i for i in range(24) if i not in lg]
        p[ln] = x[ln]
    elif model in ("N1", "N2", "N1r", "N2r", "N1r5"):
        # Nav1.5 N1（12 参）/N2（16 参，m/h 后插 h2 门 4 参）：
        # v1.27：N1r5（19 参物理布局）= N1 基 12 + rs_scale×5（CP 0/20/40/60/80）
        #        + cm_scale + p[18]=1.0 对齐垫位（不进优化，语义冻结）
        # x=[log10kma,zma,log10kmb,zmb, log10kha,zha,log10khb,zhb,
        #    (N2: log10kf2,zf2,log10kb2,zb2,) log10g, ENa, log10gl, Voff]
        # v1.23：N1r/N2r 末尾追加 [rs_scale, cm_scale]（线性坐标，真值=meta×scale）
        npar = {"N1": 12, "N2": 16, "N1r": 14, "N2r": 18, "N1r5": 19}[model]
        p = np.zeros(npar)
        p[0], p[2], p[4], p[6] = 10.0 ** x[0], 10.0 ** x[2], 10.0 ** x[4], 10.0 ** x[6]
        p[1], p[3], p[5], p[7] = x[1], x[3], x[5], x[7]
        off = 4 if model in ("N2", "N2r") else 0
        if off:
            p[8], p[10] = 10.0 ** x[8], 10.0 ** x[10]
            p[9], p[11] = x[9], x[11]
        p[8 + off] = 10.0 ** x[8 + off]
        p[9 + off] = x[9 + off]
        p[10 + off] = 10.0 ** x[10 + off]
        p[11 + off] = x[11 + off]
        if model in ("N1r", "N2r"):
            p[12 + off] = x[12 + off]          # rs_scale（线性）
            p[13 + off] = x[13 + off]          # cm_scale（线性）
        if model == "N1r5":
            p[12:17] = x[12:17]                # rs_scale ×5（按 CP 分档，线性）
            p[17] = x[17]                      # cm_scale（线性）
            p[18] = 1.0                        # 对齐垫位（不进优化）
    elif model in ("M2f", "M2f2", "M2f3", "M2fr2", "M2fr3"):
        # p 长度 21；p[13]/p[18] 存 logit 截距（非权重！），p[19]/p[20] 为斜率
        p = np.zeros(21)
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
    elif model == "M2h":
        # p 长度 26；p[13]/p[18]/p[25] 存 logit（p[25]=w3 恒定权重的 logit）
        p = np.zeros(26)
        p[[0, 2, 4, 6, 8]] = 10.0 ** x[:5]
        p[[1, 3, 5, 7]] = x[5:9]
        p[9], p[11] = 10.0 ** x[9], 10.0 ** x[11]
        p[10], p[12] = x[10], x[12]
        p[13] = x[13]
        p[14], p[16] = 10.0 ** x[14], 10.0 ** x[16]
        p[15], p[17] = x[15], x[17]
        p[18] = x[18]
        p[19], p[20] = x[19], x[20]
        p[21], p[23] = 10.0 ** x[21], 10.0 ** x[23]
        p[22], p[24] = x[22], x[24]
        p[25] = x[25]
    return p


def embed_x0(model, p0, x9, extra):
    """从收敛 M0 锚点嵌入新门起点（extra 为物理量，率/概率在此转坐标）。"""
    if model == "M2d":
        return np.r_[x9, np.log10(extra[0]), np.log10(extra[1])]
    if model == "M2e":
        return np.r_[x9, np.log10(extra[0])]
    if model in ("M2a", "M2c"):
        kb1, sb1, kb2, sb2, w = extra
        return np.r_[x9, np.log10(kb1), sb1, np.log10(kb2), sb2,
                     np.log(w / (1 - w))]
    if model == "M2ac":
        kb1, sb1, kb2, sb2, wa, kc3, sc3, kc4, sc4, wr = extra
        return np.r_[x9, np.log10(kb1), sb1, np.log10(kb2), sb2,
                     np.log(wa / (1 - wa)),
                     np.log10(kc3), sc3, np.log10(kc4), sc4,
                     np.log(wr / (1 - wr))]
    if model == "M1":
        return np.r_[x9[:5], np.log10(extra[0]), np.log10(extra[1]),
                     x9[5:], extra[2], extra[3]]
    raise ValueError(model)


def x_from_params(p, model):
    """完整物理参数向量 → 优化坐标（用于从已有拟合结果继续抛光/移植起点）。"""
    p = np.asarray(p, float)
    x9 = np.r_[np.log10(p[[0, 2, 4, 6, 8]]), p[[1, 3, 5, 7]]]
    logit = lambda w: np.log(w / (1.0 - w))
    if model in ("M0", "M0t2", "M0t3", "M0t4"):
        return x9
    if model == "M2d":
        return np.r_[x9, np.log10(p[9]), np.log10(p[10])]
    if model == "M2e":
        return np.r_[x9, np.log10(p[9])]
    if model in ("M2a", "M2c"):
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     logit(p[13])]
    if model == "M2ac":
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     logit(p[13]), np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], logit(p[18])]
    if model == "K2":
        # 物理布局同 unpack 注释；x=[log10(k1_0,k2_0,k3_0,k4_0,k5_0,k6_0,g), z1..z6]
        return np.r_[np.log10(p[[0, 2, 4, 6, 8, 10, 12]]), p[[1, 3, 5, 7, 9, 11]]]
    if model == "K1":
        # 物理布局：p[0..20] 同 M2f 族（p[13]/p[18] 为 0–1 权重），p[21]/p[22]=κ/λ
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     logit(p[13]), np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], logit(p[18]), p[19], p[20], p[21], p[22]]
    if model == "K4u":
        # 物理布局：p[0..20] 同 M2f 族；p[21]=τ_slow(ms), p[22]=δ(mV),
        # p[23]=vh_s(mV), p[24]=ρ(0–1)
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     logit(p[13]), np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], logit(p[18]), p[19], p[20],
                     np.log10(p[21]), p[22], p[23], logit(p[24])]
    if model == "K4n":
        # 同 K4u 但无 ρ；v1.28 修复：p[13]/p[18] 线性截距
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     p[13], np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], p[18], p[19], p[20],
                     np.log10(p[21]), p[22], p[23]]
    if model == "K4m":
        # 物理布局：p[0..20] 同 M2f 族；p[21]=τ_slow(s), p[22]=vh_s, p[23]=δ, p[24]=ks(符号自由)
        # v1.28 修复：p[13]/p[18] 线性截距（同 K6a 注记）
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     p[13], np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], p[18], p[19], p[20],
                     np.log10(p[21]), p[22], p[23], p[24]]
    if model in ("K6a", "K6b", "K8", "K6a2", "K6a1", "K6a2c", "K6a2f"):
        # 物理布局：p[0..24] 同 K4m；p[25]=z_τ(K6a)/γ(K6b)；K6a2c 追加 p[26]=κ；K6a2f 追加 p[26]/p[27]=kf_a1/kf_a2
        # v1.28 修复：p[13]/p[18] 为 za/zr 线性截距（内核 za4=p[13]+p[19]*v 线性使用），
        # 此前 logit 系从 M2f 族（该处为 0–1 权重、unpack 用 sigmoid）误继承——K 族线性
        xr = np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                   p[13], np.log10(p[14]), p[15], np.log10(p[16]),
                   p[17], p[18], p[19], p[20],
                   np.log10(p[21]), p[22], p[23], p[24], p[25]]
        if model == "K6a2c":
            xr = np.r_[xr, p[26]]
        if model == "K6a2f":
            xr = np.r_[xr, p[26], p[27]]
        return xr
    if model == "K7":
        # 物理布局：p[0..25] 同 K6a（v1.28 起 p[13]/p[18] 线性）；p[26]=vh_u, p[27]=τu(秒), p[28]=ku, p[29]=λ
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     p[13], np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], p[18], p[19], p[20],
                     np.log10(p[21]), p[22], p[23], p[24], p[25],
                     p[26], np.log10(p[27]), p[28], p[29]]
    if model in ("M2f", "M2f2", "M2f3", "M2fr2", "M2fr3"):
        # 物理布局：p[0..18] 同 M2ac（p[13]/p[18] 为 0–1 权重），p[19]/p[20] 斜率
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     logit(p[13]), np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], logit(p[18]), p[19], p[20]]
    if model == "M2h":
        # 物理布局：p[0..20] 同 M2f，p[21..24] = 第三分量 kd1,sd1,kd2,sd2，
        # p[25] = w3（0–1 权重）
        return np.r_[x9, np.log10(p[9]), p[10], np.log10(p[11]), p[12],
                     logit(p[13]), np.log10(p[14]), p[15], np.log10(p[16]),
                     p[17], logit(p[18]), p[19], p[20],
                     np.log10(p[21]), p[22], np.log10(p[23]), p[24],
                     logit(p[25])]
    if model == "K6a3":
        # v1.40：K6a3 23 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 2, 4, 5, 7, 9, 11, 12, 14, 18]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model in ("K6a5", "K6a5o"):
        # v1.44：K6a5o 同 K6a5 坐标；v1.41：K6a5 26 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 17, 21]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model == "K6a6":
        # v1.42：K6a6 24 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 2, 4, 5, 6, 8, 10, 12, 13, 15, 19]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model == "K6a7":
        # v1.45：K6a7 19 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model == "K6a7J":
        # v1.46：K6a7J 22 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model == "K6a7Ji":
        # v1.47：K6a7Ji 24 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21, 22, 23]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model == "K6a9Ji":
        # v1.48：K6a9Ji 26 参（布局见 unpack 注释；unpack(x_from_params(p)) 须逐位回读）
        xr = np.asarray(p, float).copy()
        lg = [0, 1, 2, 4, 6, 7, 8, 10, 14, 19, 21, 22, 23, 24, 25]
        xr[lg] = np.log10(xr[lg])
        return xr
    if model == "K6a7Jn":
        # v1.47：K6a7Jn 24 物理槽 → 14 优化坐标（与 unpack 的 mp/lg 逐位互逆）
        mp = [0, 1, 2, 7, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        lg = [0, 1, 2, 3, 4, 9, 11, 12, 13]
        xr = np.asarray(p, float)[mp].copy()
        for i in lg:
            xr[i] = np.log10(xr[i])
        return xr
    if model == "K6a5m":
        # v1.43：K6a5m 30 参物理布局 → 29 优化坐标（λ_sa=p[26] 语义冻结不进 x）
        xr = np.asarray(p[:26], float).copy()
        lg = [0, 2, 4, 5, 6, 8, 10, 12, 14, 15, 17, 21]
        xr[lg] = np.log10(xr[lg])
        return np.r_[xr, np.log10(np.asarray(p[27:30], float))]
    if model in ("N1", "N2", "N1r", "N2r", "N1r5"):
        # 物理布局：N1 p=[kma,zma,kmb,zmb,kha,zha,khb,zhb,g,ENa,gl,Voff]
        #           N1r5 p=基12+[rs×5]+[cm]+[垫位1.0]（v1.27）
        #           N2 p=[m 4, h 4, kf2,zf2,kb2,zb2, g,ENa,gl,Voff]
        #           N*r 末尾追加 [rs_scale, cm_scale]（v1.23）
        base = [np.log10(p[0]), p[1], np.log10(p[2]), p[3],
                np.log10(p[4]), p[5], np.log10(p[6]), p[7]]
        if model in ("N2", "N2r"):
            base += [np.log10(p[8]), p[9], np.log10(p[10]), p[11]]
            off = 4
        else:
            off = 0
        base += [np.log10(p[8 + off]), p[9 + off],
                 np.log10(p[10 + off]), p[11 + off]]
        if model in ("N1r", "N2r"):
            base += [p[12 + off], p[13 + off]]
        if model == "N1r5":
            base += [p[12], p[13], p[14], p[15], p[16], p[17]]
        return np.array(base)
    if model == "M1":
        return np.r_[np.log10(p[0]), np.log10(p[2]), np.log10(p[4]),
                     np.log10(p[6]), np.log10(p[8]), np.log10(p[10]),
                     np.log10(p[12]), p[1], p[3], p[5], p[7], p[9], p[11]]
    raise ValueError(model)


def load_cell(cell, protos=None):
    if str(cell).startswith("cipa:"):  # v1.34.5：CiPA TED 源
        z = np.load(os.path.join(HERE, "data", "cipa", f"{str(cell)[5:]}.npz"))
        return {"stepramp": (z["v"], z["i"])}
    if str(cell).startswith("leiT:"):  # v1.34.4：温度系列源（batch_well.npz）
        z = np.load(os.path.join(HERE, "data", "lei_temp", f"{str(cell)[5:]}.npz"))
        return {"staircase": (z["v"], z["i"])}
    if str(cell).startswith("lei2sw:"):  # v1.34.3：sweep2 复测源
        well = str(cell)[7:]
        z = np.load(os.path.join(HERE, "data", "lei_sw2", f"{well}.npz"))
        return {"staircase": (z["v"], z["i"])}
    if str(cell).startswith("lei:"):  # v1.34：Lei-I 外部数据源（npz 适配格式）
        well = str(cell)[4:]
        fp2 = os.path.join(HERE, "data", "lei2", f"{well}.npz")
        if os.path.exists(fp2):  # v1.34.2：多协议版
            z = np.load(fp2)
            out = {}
            for pr in (protos if protos is not None else ["staircase"]):
                out[pr] = (z[f"{pr}_v"], z[f"{pr}_i"])
            return out
        z = np.load(os.path.join(HERE, "data", "lei", f"{well}.npz"))
        return {"staircase": (z["v"], z["i"])}
    # v1.29.1：run_job 结果加收敛标志（converged/nit/nm_message）——136 批 M1 条款
# v1.29：protos 子集（剔坏协议用；None=全部，行为与旧版逐位一致）
    data = {}
    for proto in (protos if protos is not None else PROTOS):
        vpath = os.path.join(HERE, "data", "protocols", f"{proto}_protocol.mat")
        cpath = os.path.join(HERE, "data", "cells", cell,
                             f"{proto}_{cell}_dofetilide_subtracted"
                             f"_leak_subtracted.mat")
        voltage = sio.loadmat(vpath)["T"].ravel()
        current = sio.loadmat(cpath)["T"].ravel()
        n = min(len(voltage), len(current))
        data[proto] = (voltage[:n], current[:n])
    return data


def make_keep_mask(v, mask_pts, dV=10.0):
    """v1.29：跳变后 mask_pts 点剔除（容性伪影窗）。keep=True 为保留点。"""
    keep = np.ones(len(v), dtype=bool)
    idx = np.where(np.abs(np.diff(v)) > dV)[0] + 1
    for si in idx:
        keep[si:si + mask_pts] = False
    return keep


_TPE = None
def _tpe():
    global _TPE
    if _TPE is None:
        from concurrent.futures import ThreadPoolExecutor
        _TPE = ThreadPoolExecutor(4)
    return _TPE

def _sse_proto(args):
    p, v, c, mid, ek, dt = args
    try:
        m = sim_kernel(p, v, mid, DT if dt is None else dt, EK if ek is None else ek)
    except (ZeroDivisionError, OverflowError, FloatingPointError):
        return None
    if not np.all(np.isfinite(m)):
        return None
    return m - c  # v1.34.5：NaN 保留全长，在 sse_of 内 keep/window 之后再剔

def sse_of(p, data, model, window=None, keep=None, huber=0.0, w=None, ek=None, decim=None, dt=None):
    # v1.29.1：run_job 结果加收敛标志（converged/nit/nm_message）——136 批 M1 条款
# v1.29：keep={proto: bool数组} 剔点；huber>0 时 ρ=r²(|r|≤δ) / 2δ|r|−δ²(>|δ|)
    tot = 0.0
    mid = MODEL_ID[model]
    prots = list(data.items())
    def _dt_of(pr):
        return dt.get(pr, None) if isinstance(dt, dict) else dt  # v1.34.2：dt 可逐协议
    # v1.43（代码159）：K6a5m(36) 逐协议速率调制——λ 折进参数向量后走 34 内核：
    #   速率槽 [0,2,4,5,6,8,10,12,15,17]×λ、s τ0 槽 [21]÷λ（≡内核内全速率×λ，严格等价）；
    #   λ=p[26..29] 按协议取槽；λ≤0 → 该协议残差 None（=NaN 守卫）
    _RSLOT36 = {"steady_activation": 26, "deactivation": 27, "sine_wave": 28, "ap": 29}
    def _peff36(pr):
        if mid != 36:
            return p, mid
        lam = float(p[_RSLOT36[pr]])
        if not (lam > 0.0):
            return None, 34
        p2 = np.array(p[:26], float)
        p2[[0, 2, 4, 5, 6, 8, 10, 12, 15, 17]] *= lam
        p2[21] /= lam
        return p2, 34
    rs = list(_tpe().map(_sse_proto, [(_peff36(pr)[0], v, c, _peff36(pr)[1], ek, _dt_of(pr)) for pr, (v, c) in prots])) \
        if len(prots) > 1 else [_sse_proto((_peff36(prots[0][0])[0], prots[0][1][0], prots[0][1][1], _peff36(prots[0][0])[1], ek, _dt_of(prots[0][0])))]
    for (proto, (v, c)), r in zip(prots, rs):
        if r is None:
            return 1e30
        wmul = 1.0 if w is None else float(w.get(proto, 1.0))  # v1.30：协议权重（141 加权臂）
        if window is not None:
            r = r[(v >= window[0]) & (v < window[1])]
        if keep is not None and proto in keep:
            r = r[keep[proto]] if window is None else r  # window 与 keep 不叠加（window 仅诊断用）
        r = r[np.isfinite(r)]  # v1.34.5：NaN 剔除（在 keep/window 之后、decim 之前）
        if decim is not None and proto in decim:  # v1.33：降采样（探索期；r·√ds 使 r@r≈×ds 保无偏；置于 keep 后防错位）
            ds = int(decim[proto])
            r = r[::ds] * np.sqrt(ds)
        if huber > 0.0:
            a = np.abs(r)
            quad = r * r
            lin = 2.0 * huber * a - huber * huber
            tot += wmul * float(np.where(a <= huber, quad, lin).sum())
        else:
            tot += wmul * float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def run_audit(card):
    """v1.18 辨识性审计：谷点数值 Hessian（对角中心差分 + cross 2×2 网格）。
    不拟合，逐点 sse_of 评估；NaN 原样记录。"""
    cell, model = card["cell"], card["model"]
    data = load_cell(cell, card.get("protos", None))   # v1.29.1：run_job 结果加收敛标志（converged/nit/nm_message）——136 批 M1 条款
# v1.29：审计亦可剔协议
    au = card["audit"]
    dims = [int(d) for d in au["dims"]]
    steps = [float(h) for h in au["steps"]]
    cross = [(int(i), int(j)) for i, j in au.get("cross", [])]
    x0 = np.array(x_from_params(card["start_params"], model), dtype=float)
    # v1.29.2：审计与拟合同口径——mask/Huber 从卡传入（138 批臂B审计需要）
    keep = None
    if int(card.get("mask_pts", 0)) > 0:
        keep = {k: make_keep_mask(v, int(card["mask_pts"])) for k, (v, c) in data.items()}
    hub = float(card.get("huber_delta", 0.0))
    t0 = time.time()
    base = sse_of(unpack(x0, model), data, model, keep=keep, huber=hub)
    print(f"  [audit] base sse={base:.1f} ({time.time()-t0:.0f}s)", flush=True)
    diag = {}
    for d, h in zip(dims, steps):
        xp = x0.copy(); xp[d] += h
        xm = x0.copy(); xm[d] -= h
        sp = sse_of(unpack(xp, model), data, model, keep=keep, huber=hub)
        sm = sse_of(unpack(xm, model), data, model, keep=keep, huber=hub)
        diag[str(d)] = {"h": h, "plus": float(sp), "minus": float(sm)}
        print(f"  [audit] dim {d} ±{h:g}: {sp:.1f} / {sm:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    cross_out = {}
    hmap = dict(zip(dims, steps))
    for i, j in cross:
        hi, hj = hmap[i], hmap[j]
        vals = []
        for si in (+1.0, -1.0):
            for sj in (+1.0, -1.0):
                xp = x0.copy(); xp[i] += si * hi; xp[j] += sj * hj
                vals.append(float(sse_of(unpack(xp, model), data, model, keep=keep, huber=hub)))
        cross_out[f"{i},{j}"] = vals   # 固定顺序 [++, +-, -+, --]
        print(f"  [audit] cross {i},{j}: "
              f"{['%.1f' % v for v in vals]} ({time.time()-t0:.0f}s)", flush=True)
    res = {"cell": cell, "model": model, "start": int(card.get("start", 0)),
           "mode": "audit", "note": card.get("note", ""),
           "base_sse": float(base), "diag": diag, "cross": cross_out,
           "seconds": time.time() - t0, "runner": "local"}
    os.makedirs(os.path.join(HERE, "results_local"), exist_ok=True)
    out = os.path.join(HERE, "results_local",
                       f"audit_{cell}_{model}_s{int(card.get('start', 0))}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[done] audit {cell} {model} s{int(card.get('start', 0))}: "
          f"base={base:.1f} ({res['seconds']:.0f}s) -> {out}", flush=True)
    return res


DT_NAV = 0.04   # ms（25 kHz，预注册附录 A2 冻结）


def _find_one(nav_root, cell_rel, prefix, cp):
    """按协议前缀+CP 递归定位唯一数据文件（多匹配取 35C）。"""
    import glob as _glob
    pat = os.path.join(nav_root, cell_rel, "**", f"{prefix}_*_{int(cp)}CP.csv")
    hits = [h for h in _glob.glob(pat, recursive=True)
            if not h.endswith("_meta.csv")]
    if len(hits) == 0:
        raise FileNotFoundError(f"{prefix} CP={cp}: 无匹配 {pat}")
    if len(hits) > 1:
        c35 = [h for h in hits if "_35C_" in os.path.basename(h)]
        if len(c35) == 1:
            return c35[0]
        raise ValueError(f"{prefix} CP={cp}: 匹配不唯一 {hits}")
    return hits[0]


def load_nav(cell_rel, cps, nav_root, protocols=("NaIV",), baseline_ms=0.0):
    """Nav1.5 装载（v1.21：双协议+Cm 字段；规格=预注册附录 A2/A3，nav_adapter v0.2）。
    返回 sweeps：list of dict(Vcmd[mV], Iobs[pA], rs_eff[MΩ], cm[pF], v_step, cp, proto)。
    每 CP×协议 须恰好 1 个数据文件。
    v1.24（代码126）：baseline_ms>0 时按官方口径做每扫点前 baseline_ms 毫秒
    均值平移（官方 fit_artefacts 前 5ms 基线扣除；§87 双录的我们此前不做、
    由 gl 吸收——本批试算该口径差异，数据变了 sse 与旧批不可比）。"""
    import nav_adapter as na
    sweeps = []
    for cp in cps:
        for proto in protocols:
            csv_path = _find_one(nav_root, cell_rel, proto, cp)
            meta_path = csv_path[:-4] + "_meta.csv"
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"缺 meta：{meta_path}")
            meta = na.load_meta(meta_path)
            if proto == "NaIV":
                t, volts, cur = na.load_naiv_wide(csv_path)
                Vax = na.naiv_voltage_axis(volts, len(t))   # v0.3：batch1 1000 点窗自适应（129-补5）
            elif proto == "NaInact":
                volts, cur = na.load_nainact_wide(csv_path, meta)
                Vax = na.nainact_voltage_axis(volts)
            else:
                raise ValueError(f"未知协议 {proto}")
            for v in volts:
                row = meta[np.isclose(meta["voltage_mV"], v)]
                if len(row) == 0:
                    raise ValueError(f"{os.path.basename(csv_path)}: meta 缺电压行 {v}")
                rs_raw = float(row["rseries_Mohm"].iloc[0])
                rs = rs_raw * (1.0 - cp / 100.0)
                cm = float(row["capacitance_pF"].iloc[0])
                rseal = float(row["rseal_Mohm"].iloc[0]) if "rseal_Mohm" in row else 1e9
                iobs = np.array(cur[v], dtype=float, copy=True)
                if baseline_ms > 0.0:        # v1.24：每扫点前段均值平移（官方口径试算）
                    kb = max(1, int(round(baseline_ms / na.DT)))
                    iobs = iobs - float(iobs[:kb].mean())
                sweeps.append({"Vcmd": Vax[v], "Iobs": iobs, "rs_eff": rs,
                               "cm": cm, "v_step": float(v), "cp": int(cp),
                               "proto": proto,
                               "alpha": cp / 100.0, "rs": rs_raw,
                               "rseal": rseal})  # v1.22 放大器链字段
    return sweeps


def make_epc10_sos():
    """v1.25：EPC-10 输出滤波级联 sos→(5,5) [b0,b1,b2,a1,a2]，fs=25kHz 固定。
    F1=6 极点 Bessel 10kHz + F2=4 极点 Bessel 5kHz（HEKA 手册结构 +
    25kHz 采样标准对；§94 群延迟实证 0.176ms≈观测 0.16ms）。"""
    from scipy import signal as _sig
    s1 = _sig.bessel(6, 10000.0 / 12500.0, norm="phase", output="sos")
    s2 = _sig.bessel(4, 5000.0 / 12500.0, norm="phase", output="sos")
    sos = np.vstack([s1, s2])              # (5,6)
    return np.ascontiguousarray(sos[:, [0, 1, 2, 4, 5]])   # 去 a0（=1）


def make_epc10_modal():
    """v1.26：EPC-10 级联的模态展开 ZOH 精确离散 → (5,6) [c,d,er,ei,rR,rI]。
    每行一个共轭极点对的实二阶块：x'=c·xR−d·xI+er·u（虚部同规），
    输出贡献 2(rR·xR−rI·xI)。阶跃逐点 max|Δ|=3.4e-13（§97.1 核验）。"""
    from scipy import signal as _sig
    zs, ps, ks = [], [], []
    for N, fc in ((6, 10000.0), (4, 5000.0)):
        z, p, k = _sig.bessel(N, 2 * np.pi * fc, norm="phase",
                              analog=True, output="zpk")
        zs.append(z); ps.append(p); ks.append(k)
    pa = np.concatenate(ps)                # 全极点（zs 皆空）
    ka = float(np.prod(ks))
    T = DT_NAV * 1e-3                      # ms → s
    blocks = []
    for i, pi in enumerate(pa):
        if pi.imag <= 0:
            continue                       # 每对只取 Im>0 代表
        d = 1.0 + 0.0j
        for j, pj in enumerate(pa):
            if i != j:
                d *= (pi - pj)
        r = ka / d                         # 部分分式留数
        e = np.exp(pi * T)                 # c+jd
        g = (e - 1.0) / pi                 # er+j·ei
        blocks.append([e.real, e.imag, g.real, g.imag, r.real, r.imag])
    return np.ascontiguousarray(np.array(blocks))


def sse_nav(p, sweeps, clamp="cm", c_prs=0.0, sos=None, modal=None):
    """N1/N2 全 sweeps 残差平方和；NaN/非有限→1e30（hERG 同口径）。
    v1.20 物理罚守卫（出界 1e30）：ENa∈[+40,+80] mV；g≤5000 nS；
    gl∈[1e-3,100] nS；Voff∈[−20,+20] mV（121 病态谷教训，§82.3）。
    v1.22：clamp="amp" 走全放大器链引擎（代码124），"cm" 走 v1.21 引擎。
    v1.24：c_prs 透传 amp 引擎（固定值，Δk=0）。"""
    gg, ee, ll, oo = (p[12], p[13], p[14], p[15]) if len(p) in (16, 18) else (p[8], p[9], p[10], p[11])
    if not (40.0 <= ee <= 80.0) or gg > 5000.0 or not (1e-3 <= ll <= 100.0) or abs(oo) > 20.0:
        return 1e30
    if len(p) in (14, 18):               # v1.23 rc 守卫：真值缩放 ∈[_GUARD_LO, _GUARD_HI]（v1.27b）
        if not (_GUARD_LO <= p[-2] <= _GUARD_HI) or not (_GUARD_LO <= p[-1] <= _GUARD_HI):
            return 1e30
    if len(p) == 19:                     # v1.27a N1r5 守卫（129-补6/§104：v1.27 漏接，
        if not np.all((_GUARD_LO <= p[12:17]) & (p[12:17] <= _GUARD_HI)):  # rs 五档（v1.27b 界可配）
            return 1e30                  # 005 rs0=3.63/cm=0.4526、011 rs80=0.4712 即由此漏出）
        if not (_GUARD_LO <= p[17] <= _GUARD_HI):    # cm
            return 1e30
    tot = 0.0
    for sw in sweeps:
        if clamp == "amp":
            m = sim_nav_sweep_amp(p, sw["Vcmd"], sw["alpha"], sw["rs"],
                                  sw["rseal"], sw["cm"], DT_NAV, c_prs, sos,
                                  modal)
        else:
            m = sim_nav_sweep(p, sw["Vcmd"], sw["rs_eff"], sw["cm"], DT_NAV)
        if not np.all(np.isfinite(m)):
            return 1e30
        r = m - sw["Iobs"]
        tot += float(r @ r)
        if not np.isfinite(tot):
            return 1e30
    return tot


def run_nav(card):
    """v1.20 Nav1.5 拟合卡（"mode":"nav"）：N1/N2，跨 CP 联合（通道参数全共享，
    仅 Rs_eff 随 CP 变化且无自由度——判线①跨补偿自洽的硬执行）。"""
    cell, cps = card["cell"], [int(c) for c in card["cps"]]
    model = card["model"]
    if model not in ("N1", "N2", "N1r", "N2r", "N1r5"):
        raise ValueError(f"v1.27 支持 N1/N2/N1r/N2r/N1r5，收到 {model}")
    nav_root = card.get("nav_root", os.path.join(HERE, "data_nav"))
    protocols = tuple(card.get("protocols", ["NaIV"]))
    clamp = card.get("clamp", "cm")          # v1.22：amp=放大器链（代码124）
    c_prs = float(card.get("c_prs", 0.0))          # v1.24：C_prs 固定值敏感性（pF）
    baseline_ms = float(card.get("baseline_ms", 0.0))  # v1.24：官方基线平移口径（ms）
    if (c_prs or baseline_ms) and clamp != "amp":
        raise ValueError("c_prs/baseline_ms 试算仅在 amp 引擎发卡（v1.24）")
    if model in ("N1r", "N2r", "N1r5") and clamp != "amp":
        raise ValueError("N1r/N2r/N1r5 真值 rc 自由度仅在 amp 引擎实现（v1.23/v1.27）")
    filt = card.get("filt")                # v1.25：EPC-10 输出滤波级联（固定项）
    sos = None
    modal = None                           # v1.26：epc10z 模态块
    if filt:
        if filt not in ("epc10", "epc10z"):
            raise ValueError(f"未知 filt={filt}（v1.26 实现 'epc10'/'epc10z'）")
        if clamp != "amp":
            raise ValueError("filt 仅在 amp 引擎发卡（v1.25）")
        if filt == "epc10":
            sos = make_epc10_sos()
        else:
            modal = make_epc10_modal()     # v1.26 ZOH 精确离散
    # v1.26c（129-补3）：实树缺档双录后的分协议档位表。cbp 存在时逐协议
    # 各取各的 cps；sse 合并、数值路径与 v1.26 逐位一致（仅装载层分支）。
    cbp = card.get("cps_by_proto")
    if cbp:
        sweeps = []
        for pr in protocols:
            sweeps += load_nav(cell, [int(c) for c in cbp[pr]],
                               nav_root, (pr,), baseline_ms)
    else:
        sweeps = load_nav(cell, cps, nav_root, protocols, baseline_ms)
    n = sum(len(sw["Iobs"]) for sw in sweeps)
    x0 = np.array(x_from_params(card["start_params"], model), dtype=float)
    maxiter = int(card.get("maxiter", 20000))
    si = int(card.get("start", 0))
    t0 = time.time()
    state = {"it": 0}

    def cb(xk):
        state["it"] += 1
        if state["it"] % 250 == 0:
            print(f"  ... iter {state['it']} ({time.time()-t0:.0f}s)", flush=True)

    sse0 = sse_nav(unpack(x0, model), sweeps, clamp, c_prs, sos, modal)
    print(f"  [nav] 起跑 sse={sse0:.3e}（{len(sweeps)} 扫点，n={n}，clamp={clamp}，"
          f"c_prs={c_prs}，baseline_ms={baseline_ms}，filt={filt}）", flush=True)
    r = minimize(lambda x: sse_nav(unpack(x, model), sweeps, clamp, c_prs, sos,
                                   modal),
                 x0, method="Nelder-Mead", callback=cb,
                 options={"maxiter": maxiter, "maxfev": maxiter + 200,
                          "xatol": 1e-6, "fatol": 1e-3})
    p_fit = unpack(r.x, model)
    # 分 CP sse 分解 + IV 签名（每 CP 观测/模拟的峰电流与峰位电压）
    per_cp, iv = {}, {}
    for sw in sweeps:
        if clamp == "amp":
            m = sim_nav_sweep_amp(p_fit, sw["Vcmd"], sw["alpha"], sw["rs"],
                                  sw["rseal"], sw["cm"], DT_NAV, 0.0, sos,
                                  modal)
        else:
            m = sim_nav_sweep(p_fit, sw["Vcmd"], sw["rs_eff"], sw["cm"], DT_NAV)
        rr = m - sw["Iobs"]
        key = str(sw["cp"])
        per_cp[key] = per_cp.get(key, 0.0) + float(rr @ rr)
        if sw.get("proto") == "NaInact":   # 失活协议不进 IV 峰签名
            continue
        d = iv.setdefault(key, {"i_peak_obs": 0.0, "v_peak_obs": None,
                                "i_peak_sim": 0.0, "v_peak_sim": None})
        if sw["Iobs"].min() < d["i_peak_obs"]:
            d["i_peak_obs"] = float(sw["Iobs"].min())
            d["v_peak_obs"] = sw["v_step"]
        if m.min() < d["i_peak_sim"]:
            d["i_peak_sim"] = float(m.min())
            d["v_peak_sim"] = sw["v_step"]
    cell_tag = cell.replace("/", "_").replace("\\", "_").replace("*", "")
    proto_tag = "".join("Inact" if pr == "NaInact" else pr.replace("NaIV", "IV")
                        for pr in protocols)
    if clamp != "cm":                        # v1.22：钳制层标签防跨批撞名（§85.1 同规）
        proto_tag += "_" + clamp
    if c_prs:                                # v1.24 防撞名（§90.6 试算批）
        proto_tag += f"_cprs{c_prs:g}"
    if baseline_ms:
        proto_tag += f"_bl{baseline_ms:g}"
    if filt:                               # v1.25 防撞名（代码127）
        proto_tag += "_" + filt
    res = {"cell": cell, "model": model, "start": si, "mode": "nav",
           "cps": cps, "protocols": list(protocols), "clamp": clamp,
           "c_prs": c_prs, "baseline_ms": baseline_ms, "filt": filt,
           "cps_by_proto": cbp,             # v1.26c：无则为 None
           "note": card.get("note", ""),
           "sse": float(r.fun), "sse_start": float(sse0), "n": n,
           "sse_per_cp": per_cp, "iv": iv,
           "params": p_fit.tolist(), "nfev": int(r.nfev),
           "converged": bool(r.success), "nit": int(r.nit),
           "fit_EK": fit_ek, "EK_used": ek_fit,
           "decimate": decim, "early_stopped": state.get("early_stopped", False),
           "nm_message": str(r.message),
           "seconds": time.time() - t0, "runner": "local"}
    os.makedirs(os.path.join(HERE, "results_local"), exist_ok=True)
    out = os.path.join(HERE, "results_local",
                       f"nav_{cell_tag}_{model}_{proto_tag}_s{si}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[done] nav {cell_tag} {model} {proto_tag} s{si}: sse={r.fun:.3e} "
          f"(起跑 {sse0:.3e}) nfev={r.nfev} ({res['seconds']:.0f}s) -> {out}",
          flush=True)
    return res


def run_job(card_path):
    with open(card_path, encoding="utf-8") as f:   # v1.10.1：显式 UTF-8，防 Windows GBK 终端炸中文 note
        card = json.load(f)
    if card.get("mode") == "audit":
        return run_audit(card)
    if card.get("mode") == "nav":
        return run_nav(card)
    cell, model = card["cell"], card["model"]
    anchor_fp = os.path.join(HERE, "anchors", f"code89_percell_{cell}.json")
    lei_mode = str(cell).startswith(("lei:", "lei2sw:", "leiT:", "cipa:"))  # v1.34：无锚数据源 → 零电流基线
    if lei_mode:
        p0 = None
        x9 = None
    else:
        with open(anchor_fp, encoding="utf-8") as f:
            d89 = json.load(f)
        p0 = np.array(d89["params_polished"])
        x9 = np.r_[np.log10(p0[[0, 2, 4, 6, 8]]), p0[[1, 3, 5, 7]]]
    dt_card = card.get("dt", None)  # v1.34：卡级采样间隔（Lei 5kHz=2e-4）
    # v1.29.1：run_job 结果加收敛标志（converged/nit/nm_message）——136 批 M1 条款
# v1.29：协议子集 + mask + Huber（默认全协议/无mask/纯SSE=旧行为）
    protos = card.get("protos", None)
    mask_pts = int(card.get("mask_pts", 0))
    huber_delta = float(card.get("huber_delta", 0.0))
    proto_w = card.get("proto_w", None)  # v1.30
    data = load_cell(cell, protos)
    keep = None
    if mask_pts > 0:
        keep = {k: make_keep_mask(v, mask_pts) for k, (v, c) in data.items()}
    drop = card.get("drop_windows", None)  # v1.38：时间段剔除（与 mask_pts 可叠加）
    if drop:
        if keep is None:
            keep = {k: np.ones(len(v), dtype=bool) for k, (v, c) in data.items()}
        for k, wins in drop.items():
            assert k in data, f"drop_windows 协议不存在：{k}"
            tt = np.arange(len(data[k][0])) * (dt_card if dt_card is not None else DT)
            for t0w, t1w in wins:
                keep[k][(tt >= t0w) & (tt < t1w)] = False
    n = sum(int(keep[k].sum()) if keep is not None else len(v[0])
            for k, v in ((k, data[k]) for k in data))
    if lei_mode:
        sse_m0 = float(sum(float(np.nansum(c * c)) for (v, c) in data.values()))  # v1.34.5：NaN 安全
    else:
        p0m = p0_full19(p0)
        sse_m0 = sse_of(p0m, data, "M0", keep=keep, huber=huber_delta, w=proto_w)

    si = int(card.get("start", 0))
    if "start_params" in card:
        x0 = x_from_params(card["start_params"], model)
    else:
        x0 = embed_x0(model, p0, x9, card["start_extra"])
    fit_ek = bool(card.get("fit_EK", False))  # v1.32
    if fit_ek:
        x0 = np.r_[x0, float(card.get("EK0", EK))]
    maxiter = int(card.get("maxiter", 5000))
    t0 = time.time()
    state = {"it": 0}

    def cb(xk):
        state["it"] += 1
        if state["it"] % 250 == 0:
            print(f"  ... iter {state['it']} ({time.time()-t0:.0f}s)",
                  flush=True)

    decim = card.get("decimate", None)  # v1.33：{proto: ds} 阶段一降采样
    early_stop = bool(card.get("early_stop", False))  # v1.33：每1000轮收益<0.1%连续两次收兵
    state["fchk"] = [None, 0]  # [上次检查点最优, 连续低收益次数]

    def _mk_obj(use_decim):
        def f(x):
            if fit_ek:
                return sse_of(unpack(x[:-1], model), data, model,
                              keep=keep, huber=huber_delta, w=proto_w, ek=x[-1],
                              decim=decim if use_decim else None, dt=dt_card)
            return sse_of(unpack(x, model), data, model,
                          keep=keep, huber=huber_delta, w=proto_w,
                          decim=decim if use_decim else None, dt=dt_card)
        return f

    class _EarlyStop(Exception):
        pass

    def cb2(xk):
        state["it"] += 1
        if state["it"] % 250 == 0:
            print(f"  ... iter {state['it']} ({time.time()-t0:.0f}s)",
                  flush=True)
        if early_stop and state["it"] % 1000 == 0:
            f_now = state.get("f_best", None)
            if state["fchk"][0] is not None and f_now is not None:
                gain = (state["fchk"][0] - f_now) / max(abs(state["fchk"][0]), 1e-12)
                state["fchk"][1] = state["fchk"][1] + 1 if gain < 1e-3 else 0
                if state["fchk"][1] >= 2:
                    raise _EarlyStop
            if f_now is not None:
                state["fchk"][0] = f_now

    import scipy.optimize as _so
    f1 = _mk_obj(decim is not None)
    f2 = _mk_obj(False)
    budget1 = int(maxiter * 0.6) if decim is not None else maxiter
    converged_flag, nit_tot, msg = False, 0, ""
    stopped_early = False

    def _track(xk, *a):
        # 记录 N-M 当前最优（callback 只给 xk；用闭包算 f 太费，这里用 xk 对应 f 由 objf 包装侧记录）
        pass

    # objf 包装侧记录最优 f（供 early_stop 判定）
    best = {"f": None}
    state["xk_best"] = x0.copy()
    state["nfev_true"] = 0
    def wrap(f):
        def g(x):
            state["nfev_true"] += 1
            v = f(x)
            if best["f"] is None or v < best["f"]:
                best["f"] = v
                state["xk_best"] = np.array(x).copy()
            state["f_best"] = best["f"]
            return v
        return g
    try:
        r = minimize(wrap(f1), x0, method="Nelder-Mead", callback=cb2,
                     options={"maxiter": budget1, "maxfev": budget1 + 200,
                              "xatol": 1e-6, "fatol": 1e-3})
        x_stage, converged_flag, nit_tot, msg = r.x, bool(r.success), int(r.nit), str(r.message)
    except _EarlyStop:
        x_stage, msg, stopped_early = state["xk_best"], "early_stop(stage1)", True
    if decim is not None:
        state["it"] = 0
        best["f"] = None
        try:
            r2 = minimize(wrap(f2), x_stage, method="Nelder-Mead", callback=cb2,
                          options={"maxiter": maxiter - budget1, "maxfev": maxiter - budget1 + 200,
                                   "xatol": 1e-6, "fatol": 1e-3})
            x_stage, converged_flag = r2.x, bool(r2.success)
            nit_tot += int(r2.nit); msg = msg + "|stage2:" + str(r2.message)
        except _EarlyStop:
            x_stage, msg, stopped_early = state["xk_best"], msg + "|early_stop(stage2)", True

    class _R:  # 兼容后续 r.x/r.fun 读取
        pass
    r = _R()
    r.x = x_stage
    r.fun = f2(x_stage)
    r.success = converged_flag
    r.nit = nit_tot
    r.message = msg
    r.nfev = state.get("nfev_true", 0)
    state["early_stopped"] = stopped_early
    p_fit = unpack(r.x[:-1], model) if fit_ek else unpack(r.x, model)
    ek_fit = float(r.x[-1]) if fit_ek else EK
    dlogml = float(-n / 2 * np.log(r.fun / sse_m0)
                   - (DK[model] / 2) * np.log(n))
    sse120 = sse_of(p_fit, data, model, window=(-121.0, -119.0))
    if lei_mode:
        sse120_m0 = float("nan")  # v1.34：Lei 卡无 M0 锚基线
    else:
        sse120_m0 = sse_of(p0m, data, "M0", window=(-121.0, -119.0))
    res = {"cell": cell, "model": model, "start": si,
           "note": card.get("note", ""),
           "protos": protos, "mask_pts": mask_pts, "huber_delta": huber_delta,
           "proto_w": proto_w,
           "n_eff": n,
           "sse": float(r.fun), "sse_m0": sse_m0, "dlogml": dlogml,
           "dk": DK[model], "sse120": sse120, "sse120_m0": sse120_m0,
           "lei_mode": lei_mode,
           "gain120_pct": float((1 - sse120 / sse120_m0) * 100),
           "params": p_fit.tolist(), "nfev": int(r.nfev),
           "converged": bool(r.success), "nit": int(r.nit),
           "fit_EK": fit_ek, "EK_used": ek_fit,
           "decimate": decim, "early_stopped": state.get("early_stopped", False),
           "nm_message": str(r.message),
           "seconds": time.time() - t0, "runner": "local"}
    os.makedirs(os.path.join(HERE, "results_local"), exist_ok=True)
    out = os.path.join(HERE, "results_local",
                       f"result_{str(cell).replace(':','_')}_{model}_{card.get('tag', 's'+str(si))}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"[done] {cell} {model} s{si}: sse={r.fun:.1f} "
          f"ΔlogML={dlogml:+.2e} g120={res['gain120_pct']:+.1f}% "
          f"({res['seconds']:.0f}s) -> {out}", flush=True)
    return res


@njit(cache=True, fastmath=True, nogil=True)
def sim_gate39(p, V, dt):
    """v1.46（代码163）：model 39 门控电流通道（相对单位，每步等电荷归一）。
    Ig_au(t) = Σ_k [ f_k·(C_k + θ·O_k) − b_{k+1}·(C_{k+1} + O_{k+1}) ]
    （VSD 跃迁净通量；垂直跃迁 C_k⇄O_k 不移动电荷不进 Ig；不含离子成分）。
    推进器与 sim_kernel 的 model 39 块同构（后向 Euler 2×2 块 Thomas）。
    输出符号约定：去极化时 Ig>0（外向门控电流）。"""
    n = len(V)
    out = np.empty(n)
    x7 = np.zeros(10)
    A00 = np.empty(5); A01 = np.empty(5); A10 = np.empty(5); A11 = np.empty(5)
    U00 = np.empty(5); U11 = np.empty(5)
    CP00 = np.zeros(5); CP01 = np.zeros(5); CP10 = np.zeros(5); CP11 = np.zeros(5)
    DP0 = np.zeros(5); DP1 = np.zeros(5)
    s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0
    x7[0] = 1.0
    for i in range(n):
        v = V[i]
        if abs(p[17]) < 1.0:
            out[i] = np.nan
            continue
        tau_v = p[14] * np.exp(p[18] * v)
        if tau_v < 1.0:
            out[i] = np.nan
            continue
        s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
        s += dt * (s_inf - s) / tau_v
        ve_a = v + p[16] * s
        aa7 = p[19] * np.exp(p[20] * ve_a)
        bb7 = p[21] * np.exp(-p[20] * ve_a)
        c7 = p[0]; L7 = p[1]; th7 = p[2]
        thp = 1.0
        for k in range(5):
            fk7 = (4.0 - k) * aa7
            if k == 4:
                fk7 = 0.0
            bk7 = k * bb7
            ok7 = c7 * L7 * thp
            A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
            A01[k] = -dt * c7
            A10[k] = -dt * ok7
            A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7)
            U00[k] = -dt * fk7
            U11[k] = -dt * th7 * fk7
            thp *= th7
        det = A00[0] * A11[0] - A01[0] * A10[0]
        i00 = A11[0] / det; i01 = -A01[0] / det
        i10 = -A10[0] / det; i11 = A00[0] / det
        sup = -dt * bb7
        CP00[0] = i00 * sup; CP01[0] = i01 * sup
        CP10[0] = i10 * sup; CP11[0] = i11 * sup
        DP0[0] = i00 * x7[0] + i01 * x7[1]
        DP1[0] = i10 * x7[0] + i11 * x7[1]
        for k in range(1, 5):
            s0u = U00[k - 1]; s1u = U11[k - 1]
            m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
            m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
            det = m00 * m11 - m01 * m10
            i00 = m11 / det; i01 = -m01 / det
            i10 = -m10 / det; i11 = m00 / det
            sup = -dt * ((k + 1) * bb7)
            CP00[k] = i00 * sup; CP01[k] = i01 * sup
            CP10[k] = i10 * sup; CP11[k] = i11 * sup
            rr0 = x7[2 * k] - s0u * DP0[k - 1]
            rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
            DP0[k] = i00 * rr0 + i01 * rr1
            DP1[k] = i10 * rr0 + i11 * rr1
        x7[8] = DP0[4]; x7[9] = DP1[4]
        for k in range(3, -1, -1):
            x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
            x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
        ig = 0.0
        thp = 1.0
        for k in range(4):
            fk7 = (4.0 - k) * aa7
            bk1 = (k + 1.0) * bb7
            ig += fk7 * (x7[2 * k] + th7 * x7[2 * k + 1]) - bk1 * (x7[2 * k + 2] + x7[2 * k + 3])
            thp *= th7
        out[i] = ig
    return out


@njit(cache=True, fastmath=True, nogil=True)
def sim_gate41(p, V, dt):
    """v1.47（代码164）：model 41/42 门控电流通道（sim_gate39 同构 + 失活层）。
    梯形推进与失活子步同 sim_kernel 41 块；Ig=仅 VSD 跃迁净通量（O⇄I 与垂直 C⇄O 不移动电荷，
    但失活经 O_k 人口间接影响 Ig 时间过程——已含在仿真内）。（相对单位，每步等电荷归一）。
    Ig_au(t) = Σ_k [ f_k·(C_k + θ·O_k) − b_{k+1}·(C_{k+1} + O_{k+1}) ]
    （VSD 跃迁净通量；垂直跃迁 C_k⇄O_k 不移动电荷不进 Ig；不含离子成分）。
    推进器与 sim_kernel 的 41 块同构（后向 Euler 2×2 块 Thomas + 失活弛豫子步）。
    输出符号约定：去极化时 Ig>0（外向门控电流）。"""
    n = len(V)
    out = np.empty(n)
    x7 = np.zeros(10)
    A00 = np.empty(5); A01 = np.empty(5); A10 = np.empty(5); A11 = np.empty(5)
    U00 = np.empty(5); U11 = np.empty(5)
    CP00 = np.zeros(5); CP01 = np.zeros(5); CP10 = np.zeros(5); CP11 = np.zeros(5)
    DP0 = np.zeros(5); DP1 = np.zeros(5)
    s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0
    x7[0] = 1.0
    xi5 = np.zeros(5)
    for i in range(n):
        v = V[i]
        if abs(p[17]) < 1.0:
            out[i] = np.nan
            continue
        tau_v = p[14] * np.exp(p[18] * v)
        if tau_v < 1.0:
            out[i] = np.nan
            continue
        s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
        s += dt * (s_inf - s) / tau_v
        ve_a = v + p[16] * s
        aa7 = p[19] * np.exp(p[20] * ve_a)
        bb7 = p[21] * np.exp(-p[20] * ve_a)
        c7 = p[0]; L7 = p[1]; th7 = p[2]
        thp = 1.0
        for k in range(5):
            fk7 = (4.0 - k) * aa7
            if k == 4:
                fk7 = 0.0
            bk7 = k * bb7
            ok7 = c7 * L7 * thp
            A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
            A01[k] = -dt * c7
            A10[k] = -dt * ok7
            A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7)
            U00[k] = -dt * fk7
            U11[k] = -dt * th7 * fk7
            thp *= th7
        det = A00[0] * A11[0] - A01[0] * A10[0]
        i00 = A11[0] / det; i01 = -A01[0] / det
        i10 = -A10[0] / det; i11 = A00[0] / det
        sup = -dt * bb7
        CP00[0] = i00 * sup; CP01[0] = i01 * sup
        CP10[0] = i10 * sup; CP11[0] = i11 * sup
        DP0[0] = i00 * x7[0] + i01 * x7[1]
        DP1[0] = i10 * x7[0] + i11 * x7[1]
        for k in range(1, 5):
            s0u = U00[k - 1]; s1u = U11[k - 1]
            m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
            m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
            det = m00 * m11 - m01 * m10
            i00 = m11 / det; i01 = -m01 / det
            i10 = -m10 / det; i11 = m00 / det
            sup = -dt * ((k + 1) * bb7)
            CP00[k] = i00 * sup; CP01[k] = i01 * sup
            CP10[k] = i10 * sup; CP11[k] = i11 * sup
            rr0 = x7[2 * k] - s0u * DP0[k - 1]
            rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
            DP0[k] = i00 * rr0 + i01 * rr1
            DP1[k] = i10 * rr0 + i11 * rr1
        x7[8] = DP0[4]; x7[9] = DP1[4]
        for k in range(3, -1, -1):
            x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
            x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
        fi7 = p[22] * np.exp((0.6 / 25.693) * v)
        bi7 = p[23] * np.exp(-(0.6 / 25.693) * v)
        lam7 = fi7 + bi7
        if lam7 > 0.0:
            ei7 = np.exp(-lam7 * dt)
            fro = bi7 / lam7
            for k in range(5):
                Tk = x7[2 * k + 1] + xi5[k]
                x7[2 * k + 1] = Tk * fro + (x7[2 * k + 1] - Tk * fro) * ei7
                xi5[k] = Tk - x7[2 * k + 1]
        ig = 0.0
        thp = 1.0
        for k in range(4):
            fk7 = (4.0 - k) * aa7
            bk1 = (k + 1.0) * bb7
            ig += fk7 * (x7[2 * k] + th7 * x7[2 * k + 1]) - bk1 * (x7[2 * k + 2] + x7[2 * k + 3])
            thp *= th7
        out[i] = ig
    return out




@njit(cache=True, fastmath=True, nogil=True)
def sim_gate148(p, V, dt, model):
    """v1.48（代码165）：model 43/44/45 门控电流通道（sim_gate41 同构 + 拓扑变体子步）。
    Ig 口径不变：仅 VSD 跃迁净通量（失活子步各向均不跨层、不移动电荷）。
    Ig_au(t) = Σ_k [ f_k·(C_k + θ·O_k) − b_{k+1}·(C_{k+1} + O_{k+1}) ]。去极化 Ig>0。
    v1.49（代码167）：model 47=44+κ(s) 走本通道；κ 作用于垂直步（不跨层、不移动电荷），
    Ig 公式不变；门控协议 s≈0 → κ≈1 → 门控读数对 κ 一阶免疫（补记61 §3-2 时标分离保护）。"""
    n = len(V)
    out = np.empty(n)
    x7 = np.zeros(10)
    A00 = np.empty(5); A01 = np.empty(5); A10 = np.empty(5); A11 = np.empty(5)
    U00 = np.empty(5); U11 = np.empty(5)
    CP00 = np.zeros(5); CP01 = np.empty(5); CP10 = np.empty(5); CP11 = np.empty(5)
    DP0 = np.zeros(5); DP1 = np.empty(5)
    s = 1.0 / (1.0 + np.exp(-(V[0] - p[15]) / p[17])) if abs(p[17]) >= 1.0 else 0.0
    x7[0] = 1.0
    xi5 = np.zeros(5)
    xiC = 0.0
    for i in range(n):
        v = V[i]
        if abs(p[17]) < 1.0:
            out[i] = np.nan
            continue
        tau_v = p[14] * np.exp(p[18] * v)
        if tau_v < 1.0:
            out[i] = np.nan
            continue
        s_inf = 1.0 / (1.0 + np.exp(-(v - p[15]) / p[17]))
        s += dt * (s_inf - s) / tau_v
        ve_a = v + p[16] * s
        aa7 = p[19] * np.exp(p[20] * ve_a)
        bb7 = p[21] * np.exp(-p[20] * ve_a)
        c7 = p[0]; L7 = p[1]; th7 = p[2]
        kap = np.exp(-p[31] * s) if model == 47 else 1.0  # v1.49
        c7e = c7 * kap
        thp = 1.0
        for k in range(5):
            fk7 = (4.0 - k) * aa7
            if k == 4:
                fk7 = 0.0
            bk7 = k * bb7
            ok7 = c7e * L7 * thp
            A00[k] = 1.0 + dt * (fk7 + bk7 + ok7)
            A01[k] = -dt * c7e
            A10[k] = -dt * ok7
            A11[k] = 1.0 + dt * (th7 * fk7 + bk7 + c7e)
            U00[k] = -dt * fk7
            U11[k] = -dt * th7 * fk7
            thp *= th7
        det = A00[0] * A11[0] - A01[0] * A10[0]
        i00 = A11[0] / det; i01 = -A01[0] / det
        i10 = -A10[0] / det; i11 = A00[0] / det
        sup = -dt * bb7
        CP00[0] = i00 * sup; CP01[0] = i01 * sup
        CP10[0] = i10 * sup; CP11[0] = i11 * sup
        DP0[0] = i00 * x7[0] + i01 * x7[1]
        DP1[0] = i10 * x7[0] + i11 * x7[1]
        for k in range(1, 5):
            s0u = U00[k - 1]; s1u = U11[k - 1]
            m00 = A00[k] - s0u * CP00[k - 1]; m01 = A01[k] - s0u * CP01[k - 1]
            m10 = A10[k] - s1u * CP10[k - 1]; m11 = A11[k] - s1u * CP11[k - 1]
            det = m00 * m11 - m01 * m10
            i00 = m11 / det; i01 = -m01 / det
            i10 = -m10 / det; i11 = m00 / det
            sup = -dt * ((k + 1) * bb7)
            CP00[k] = i00 * sup; CP01[k] = i01 * sup
            CP10[k] = i10 * sup; CP11[k] = i11 * sup
            rr0 = x7[2 * k] - s0u * DP0[k - 1]
            rr1 = x7[2 * k + 1] - s1u * DP1[k - 1]
            DP0[k] = i00 * rr0 + i01 * rr1
            DP1[k] = i10 * rr0 + i11 * rr1
        x7[8] = DP0[4]; x7[9] = DP1[4]
        for k in range(3, -1, -1):
            x7[2 * k] = DP0[k] - (CP00[k] * x7[2 * k + 2] + CP01[k] * x7[2 * k + 3])
            x7[2 * k + 1] = DP1[k] - (CP10[k] * x7[2 * k + 2] + CP11[k] * x7[2 * k + 3])
        fi7 = p[22] * np.exp((0.6 / 25.693) * v)
        bi7 = p[23] * np.exp(-(0.6 / 25.693) * v)
        if model == 43 or model == 45:
            efi = np.exp(-fi7 * dt); ebi = np.exp(-bi7 * dt)
            if fi7 > 0.0 or bi7 > 0.0:
                if abs(fi7 - bi7) <= 1e-9 * max(max(fi7, bi7), 1.0):
                    for k in range(5):
                        O0 = x7[2 * k + 1]; I0 = xi5[k]
                        O1 = O0 * efi
                        I1 = (I0 + fi7 * O0 * dt) * ebi
                        dm = (O0 - O1) + (I0 - I1)
                        x7[2 * k + 1] = O1; xi5[k] = I1
                        if model == 45:
                            x7[0] += dm
                        else:
                            x7[2 * k] += dm
                else:
                    cf = fi7 / (bi7 - fi7)
                    for k in range(5):
                        O0 = x7[2 * k + 1]; I0 = xi5[k]
                        O1 = O0 * efi
                        I1 = I0 * ebi + cf * O0 * (efi - ebi)
                        dm = (O0 - O1) + (I0 - I1)
                        x7[2 * k + 1] = O1; xi5[k] = I1
                        if model == 45:
                            x7[0] += dm
                        else:
                            x7[2 * k] += dm
        else:
            lam7 = fi7 + bi7
            if lam7 > 0.0:
                ei7 = np.exp(-lam7 * dt)
                fro = bi7 / lam7
                for k in range(5):
                    Tk = x7[2 * k + 1] + xi5[k]
                    x7[2 * k + 1] = Tk * fro + (x7[2 * k + 1] - Tk * fro) * ei7
                    xi5[k] = Tk - x7[2 * k + 1]
            fic = p[24] * np.exp((0.6 / 25.693) * v)
            bic = p[25] * np.exp(-(0.6 / 25.693) * v)
            lamc = fic + bic
            if lamc > 0.0:
                eic = np.exp(-lamc * dt)
                frc = bic / lamc
                Tc = x7[8] + xiC
                x7[8] = Tc * frc + (x7[8] - Tc * frc) * eic
                xiC = Tc - x7[8]
        ig = 0.0
        for k in range(4):
            fk7 = (4.0 - k) * aa7
            bk1 = (k + 1.0) * bb7
            ig += fk7 * (x7[2 * k] + th7 * x7[2 * k + 1]) - bk1 * (x7[2 * k + 2] + x7[2 * k + 3])
        out[i] = ig
    return out


def p0_full19(p0):
    p = np.zeros(19)
    p[:9] = p0
    return p


def main():
    target = sys.argv[1]
    if os.path.isdir(target):
        cards = sorted(os.path.join(target, f)
                       for f in os.listdir(target) if f.endswith(".json"))
        for c in cards:
            run_job(c)
    else:
        run_job(target)


if __name__ == "__main__":
    main()
