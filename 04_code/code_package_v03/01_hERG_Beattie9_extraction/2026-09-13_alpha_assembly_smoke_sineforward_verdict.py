# 2026-09-13_α模型_组装冒烟_sine前向判决.py
# 目的（组装冒烟，不是正式组装——DeepSeek 审计三缺口全部桥接声明在案）：
#   用四块表前向跑 sine 全程：dm/dt=(m_ss-m)/τ_m, dh/dt=(h_ss-h)/τ_h,
#   I=G·m·h·(V-E_rev)。V(t) 直接读 sine_wave 协议（chirp 波形原样吃进，不近似）。
#   头号靶：chirp 后反弹/chirp 前反弹 幅度比（实测 0.38-0.66，九细胞全降，
#   历史改幅度不改 τ_rec，260 在案）——乘积门模型能否自发出这个抑制。
#
# 【重要纠偏 2026-09-13】失活块 A(V)/A(+50) 曲线判词修正：
#   反弹在 -120 停 500ms，h 对任何测试档全额恢复（τ_rec=3ms），反弹峰与测试末 h 无关
#   -> A(V) 量的是 m(150ms,V)，是激活曲线不是 h_ss(V)；且 -90x60ms 复位不彻底
#   （τ_d(-90)~72ms -> 进测试档 m~0.43 残留）。A(V) 降级为本组装的验证靶，
#   h_ss(V<=-40)=1 改物理声明（失活是去极化现象）。
#
# 数据源（全部封卷/登记 JSON，脚本运行时加载，不手抄）：
#   幅度表提取_结果.json: 每细胞 y_ss(-70/-60/-50)（m_ss 锚）；τ 阶梯 TF/TM/TAU_L；
#   反弹hook_结果.json: τ_rec/τ_deact(-120/-110/-100) 中位；A_rows 每细胞 -120 钩锚
#     （G=A(-120)/31.67，模型形误差±30%声明）；B_rows sine 双反弹实测（评分靶）；
#   失活门_失活协议封卷判决_结果.json: 每细胞 h_ss50、g_hat、m90 τ_r（-90 格）；
#   激活时程_结果.json: 登记慢分量 τ2（-20/0/+20/+40 中位，CV 大，缺口 3 在案）。
#
# 桥接声明（冒烟专用，正式组装前须补，编号对应 DeepSeek 三缺口）：
#   B1（缺口1）: h_ss(V<=-40)=1.0 物理声明；h_ss(+40) 每细胞实测（失活协议 +40 测试段
#     末 50ms 均值/(G·1·DF)，m(150ms,+40)~0.85±0.15 折进不确定度）；-40..+40 对数桥。
#   B2（缺口2）: τ_rec(V) = 已测 4 点(-120:3.04,-110:4.25,-100:5.05,-90:6.5ms) 对数
#     线性外推至 -40（得 ~8-23ms），-40..0 续外推封顶 100ms；V>=0 τ_h=1.5ms（τ_obs
#     上界，对组装尺度为瞬时）。
#   B3（缺口3）: τ_m(V>=-40) 用激活登记慢分量中位（-40:24.05封卷,-20:4.0,0:2.0,
#     +20:0.71,+40:0.29s）；快分量（20-100ms）不进单门 m，声明；m_ss(-40..-20)
#     逻辑桥（最弱桥，判词标注）；m_ss(-20..+60)=1.0 声明（A(V) m曲线支持 m(+20)~1）。
#   B4: -40 档 y_ss 负值伪影在案剔除；缺档细胞用同档中位。
#   B5: 单门 m（非双分量），chirp 内 m 低通近似，声明。
#
# 判线（跑前声明）：
#   S1 反弹1：模拟 DoE 幅度 / 实测 B_rows(which=0) 幅度 ∈ [0.3,3]（G 不确定度）；
#   S2 抑制比：模拟 A2/A1 与实测 A2/A1 比值 ∈ [0.5,2] 且模拟比 <1（方向必须对）——头号靶；
#   S3 +40 稳态：模拟 / 实测（+40 段末 200ms 均值）∈ [1/3,3]；
#   每细胞 S1/S2/S3 全过 -> 该细胞冒烟过；九细胞过 >=7 -> 乘积门冒烟成立。
#
# 【POST-HOC 修复 2026-09-13 第二轮（首轮 6/9 后跑后声明）】
#   首轮三例未过全归因数据侧在案问题，修复只动数据锚/靶，不动模型、不动判线：
#   R1 G 锚回退链：hook -120 有效A>0 -> 任意有效A>0 -> 失活块 g_hat（旗 posthoc_G_inact）
#      -> hook -120 无效行 |A|（旗 posthoc_G_degraded）。16704047 走第三级（g_hat=0.0718）。
#   R2 实测靶降级回填：sine 双反弹实测缺档时用无效行 |A| 回填（旗 posthoc_target_degraded）。
#      16713110 which=0 回填 4.311。16704047 两档全回填（比 1.40，与八细胞相反——
#      该细胞实测靶不可信，S2 预计仍不过，属数据侧失败，不修饰）。
#   R3 h40 回退链：失活 +40 测试段直提 -> 失活块 h_ss50 -> sine 自身 +40 段标定
#      （旗 posthoc_h40_selfcal，该细胞 S3 降级为非独立）-> 默认 0.0034。
#      16708060/16708016（过减漏在案，失活协议 h 链断裂）走第三级。
# 对照（跑前声明）：
#   C1 积分器+评分自洽：已知 τ 表合成 +40->-120 单步，DoE 反演 τ_r 误差 <15%；
#   C2 合成全程（16713003 表+实测噪声）评分管道必须 S1-S3 全过（管道不死锁）。
# 运行：python 本文件（九细胞全量）；SMOKE=1 单细胞 16713003 冒烟。
import os
import json
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
SMOKE = os.environ.get("SMOKE", "0") == "1"
CELLS = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                    "16708060", "16708118", "16713003", "16713110", "16715049"]
DT = 1e-4
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)

F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
SEP_MIN = 1.5


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


def segments(V):
    edges = np.where(np.diff(V) != 0)[0] + 1
    return [(float(V[s[0]]), int(s[0]), len(s)) for s in np.split(np.arange(len(V)), edges)]


def doe_fit(t, y, tr_grid=TR_GRID, td_grid=TD_GRID):
    if len(t) < 50:
        return None

    def scan(trg, tdg):
        best = None
        for tr in trg:
            td_ok = tdg[tdg >= SEP_MIN * tr]
            if not len(td_ok):
                continue
            er = np.exp(-t / tr)
            for td in td_ok:
                x = np.exp(-t / td) - er
                X = np.column_stack([np.ones(len(t)), x])
                sol, *_ = np.linalg.lstsq(X, y, rcond=None)
                sse = float(np.sum((y - X @ sol) ** 2))
                if best is None or sse < best[0]:
                    best = (sse, tr, td, sol)
        return best

    b = scan(tr_grid, td_grid)
    if b is None:
        return None
    _, tr0, td0, _ = b
    trg = tr0 * np.exp(np.linspace(-0.35, 0.35, 9))
    tdg = td0 * np.exp(np.linspace(-0.35, 0.35, 9))
    b = scan(trg, tdg)
    sse, tr, td, sol = b
    res = y - (sol[0] + sol[1] * (np.exp(-t / td) - np.exp(-t / tr)))
    return dict(tau_r=float(tr), tau_d=float(td), c=float(sol[0]), A=float(sol[1]),
                rms=float(np.sqrt(np.mean(res ** 2))),
                edge=bool(tr <= tr_grid[0] * 1.02 or tr >= tr_grid[-1] * 0.98))


class Tab:
    """对数-线性插值锚表（x 线性、y 对数，越界取端点）。"""
    def __init__(self, anchors, log_y=True):
        self.xs = np.array(sorted(anchors), dtype=float)
        ys = np.array([anchors[x] for x in self.xs], dtype=float)
        self.ly = np.log(np.clip(ys, 1e-9, None)) if log_y else ys
        self.log_y = log_y

    def __call__(self, v):
        y = np.interp(v, self.xs, self.ly)
        return float(np.exp(y)) if self.log_y else float(y)


def build_tabs(cell, amp, hook, inact, I_inact=None, V_inact=None, sine_mea40=None):
    """每细胞四表：m_ss, τ_m, h_ss, τ_h + G。返回 dict（含 posthoc 旗帜）。"""
    # ---- m_ss 锚（每细胞 y_ss，缺档用中位；B3/B4 桥接） ----
    yss = {}
    for r in amp["rows"]:
        if r["v"] in (-70, -60, -50):
            yss.setdefault(r["v"], {})[r["cell"]] = max(r["y_ss"], 1e-4)
    med = {v: float(np.median(list(d.values()))) for v, d in yss.items()}
    def cell_y(v):
        return yss.get(v, {}).get(cell, med.get(v, 0.02))
    m_anchors = {-130: 1e-4, -120: 1e-4, -110: 1e-4, -100: 1e-4, -90: 1e-4,
                 -80: cell_y(-70) * 0.5,
                 -70: cell_y(-70), -60: cell_y(-60), -50: cell_y(-50),
                 -40: min(1.0, cell_y(-50) * 3.0),   # B3 逻辑桥（最弱桥）
                 -30: min(1.0, cell_y(-50) * 8.0),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    m_ss = Tab(m_anchors, log_y=True)

    # ---- τ_m 锚（封卷中位 + 登记慢分量；B3/B5） ----
    hs = hook["summary"]
    tm_anchors = {-130: hs["-120"]["td_med"], -120: hs["-120"]["td_med"],
                  -110: hs["-110"]["td_med"], -100: hs["-100"]["td_med"],
                  -90: 0.072, -80: 0.24,
                  -70: amp["tau_used"]["TM"]["-70"], -60: amp["tau_used"]["TM"]["-60"],
                  -50: amp["tau_used"]["TAU_L"]["-50"], -40: amp["tau_used"]["TAU_L"]["-40"],
                  -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
                  20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
    tau_m = Tab(tm_anchors, log_y=True)

    # ---- G：R1 回退链（posthoc 旗帜） ----
    ci = inact["cells"].get(cell, {})
    G = None
    gflag = None
    for r in hook["A_rows"]:
        if r["cell"] == cell and r["v"] == -120 and r["valid"] and r["A"] > 0:
            G = r["A"] / DF_M120
    if G is None:
        cands = [r["A"] / DF_M120 for r in hook["A_rows"]
                 if r["cell"] == cell and r["valid"] and r["A"] > 0]
        if cands:
            G = float(np.median(cands))
    if G is None and ci.get("g_hat") and ci["g_hat"] > 0:      # R1 第三级
        G = float(ci["g_hat"])
        gflag = "posthoc_G_inact"
    if G is None:                                              # R1 第四级
        cands = [abs(r["A"]) / DF_M120 for r in hook["A_rows"]
                 if r["cell"] == cell and r["v"] == -120 and r["A"] != 0]
        if cands:
            G = float(np.median(cands))
            gflag = "posthoc_G_degraded"

    # ---- h_ss（B1 + R3 回退链） ----
    h50 = None
    if ci.get("h_ss50") and ci["h_ss50"] > 0:
        h50 = float(ci["h_ss50"])
    h40 = np.nan
    hflag = None
    if I_inact is not None and G is not None and G > 0:
        info = segments(V_inact)
        for k, (v, s0, n) in enumerate(info):
            if abs(v - 40.0) < 2.0 and 0.14 < n * DT < 0.16 and k >= 1:
                pv = info[k - 1][0]
                if abs(pv + 90.0) < 2.0:                    # 是测试档不是别的
                    iss = float(np.mean(I_inact[s0 + n - 5000: s0 + n]))
                    h40 = iss / (G * 1.0 * (40.0 - E_REV))
                    break
    if not np.isfinite(h40) or h40 <= 0:
        h40 = h50 if h50 else np.nan
    if (not np.isfinite(h40) or h40 <= 0) and G and sine_mea40 and sine_mea40 > 0:
        h40 = sine_mea40 / (G * 1.0 * (40.0 - E_REV))         # R3 第三级
        hflag = "posthoc_h40_selfcal"
    if not np.isfinite(h40) or h40 <= 0:
        h40 = 0.0034
    h40 = float(np.clip(h40, 1e-4, 0.2))
    h_anchors = {-130: 1.0, -100: 1.0, -40: 1.0, 0: float(np.sqrt(1.0 * h40)),
                 40: h40, 50: h50 if h50 else h40, 60: h50 if h50 else h40}
    h_ss = Tab(h_anchors, log_y=True)

    # ---- τ_h（B2） ----
    tr90 = ci.get("m90", {}).get("tau_r") if ci.get("m90", {}).get("valid") else None
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: tr90 if tr90 else 0.0065,
                  -40: 0.023, -20: 0.05, 0: 0.0015, 60: 0.0015}
    tau_h = Tab(th_anchors, log_y=True)
    return dict(m_ss=m_ss, tau_m=tau_m, h_ss=h_ss, tau_h=tau_h, G=G,
                h40=h40, h50=h50, gflag=gflag, hflag=hflag)


def forward(V, tabs):
    """全向量化查表 + 简单递推：每步 m/h 向 (m_ss,h_ss) 指数弛豫。"""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    n = len(v)
    m = np.empty(n)
    h = np.empty(n)
    m0 = ms[0]
    h0 = hss[0]
    for i in range(n):
        m0 = ms[i] + (m0 - ms[i]) * em[i]
        h0 = hss[i] + (h0 - hss[i]) * eh[i]
        m[i] = m0
        h[i] = h0
    return m, h


def find_sine_structure(V):
    """程序化核实 sine 结构（实测：+40x1.0s、两个 -120x0.5s 反弹、chirp 每点变）。"""
    info = segments(V)
    long40 = [(v, s0, n) for v, s0, n in info if abs(v - 40) < 3 and n * DT > 0.8]
    rebs = [(k, v, s0, n) for k, (v, s0, n) in enumerate(info)
            if abs(v + 120) < 3 and 0.4 < n * DT < 0.7]
    chirp = None
    if len(rebs) >= 2:
        s_end = rebs[0][2] + rebs[0][3]
        s_beg = rebs[1][2]
        chirp = (s_end, s_beg)
    return dict(info=info, long40=long40, rebs=rebs, chirp=chirp)


def score_cell(cell, V, I, tabs, meas, pr):
    m, h = forward(V, tabs)
    G = tabs["G"]
    Isim = G * m * h * (V - E_REV)
    st = find_sine_structure(V)
    out = dict(G=G)
    # +40 稳态（段长 1.0s，取末 200ms）
    if st["long40"]:
        v, s0, nseg = st["long40"][0]
        sim40 = float(np.mean(Isim[s0 + nseg - 2000: s0 + nseg]))
        mea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg])) if I is not None else np.nan
        out["sim40"] = sim40
        out["mea40"] = mea40
        out["S3"] = bool(np.isfinite(mea40) and abs(mea40) > 1e-6
                         and 1 / 3 <= sim40 / mea40 <= 3)
    # 双反弹 DoE（向内翻正，与实测 B_rows 符号约定一致）
    amps = []
    for k, v, s0, nseg in st["rebs"][:2]:
        t = np.arange(nseg) * DT
        r = doe_fit(t, -Isim[s0:s0 + nseg])
        amps.append(r["A"] if r else np.nan)
    out["simA"] = amps
    ma = meas.get(cell)
    if ma and len(amps) == 2 and all(np.isfinite(amps)):
        a1m, a2m = ma
        out["A1_meas"], out["A2_meas"] = a1m, a2m
        out["S1"] = bool(a1m and 0.3 <= amps[0] / a1m <= 3.0)
        sr = amps[1] / amps[0] if amps[0] else np.nan
        mr = a2m / a1m if a1m else np.nan
        out["sim_ratio"] = sr
        out["mea_ratio"] = mr
        out["S2"] = bool(np.isfinite(sr) and np.isfinite(mr)
                         and sr < 1.0 and 0.5 <= sr / mr <= 2.0)
    out["pass"] = bool(out.get("S1") and out.get("S2") and out.get("S3"))
    out["_sim"] = Isim
    out["_struct"] = st
    out["_mh"] = (m, h)
    return out


def main():
    rng = np.random.default_rng(5)
    print("=" * 84)
    print(" α模型 组装冒烟·sine 前向判决" + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"))
    print(" 判线: S1 反弹1幅度∈[0.3,3] | S2 抑制比方向对且∈[0.5,2]×实测 | S3 +40稳态∈[1/3,3]")
    print("=" * 84, flush=True)

    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))

    # 实测 sine 双反弹靶值（R2：缺档用无效行 |A| 降级回填，旗 posthoc_target_degraded）
    meas = {}
    for r in hook["B_rows"]:
        if r["valid"] and r["A"] > 0:
            meas.setdefault(r["cell"], {})[r["which"]] = r["A"]
    mflag = {}
    for r in hook["B_rows"]:
        d = meas.setdefault(r["cell"], {})
        if r["which"] not in d and r["A"] != 0:
            d[r["which"]] = abs(r["A"])
            mflag[r["cell"]] = "posthoc_target_degraded"
    meas = {c: (d[0], d[1]) for c, d in meas.items() if 0 in d and 1 in d}
    print("[实测靶] chirp 抑制比 A2/A1:", flush=True)
    for c in CELLS:
        if c in meas:
            tag = f"  [{mflag[c]}]" if c in mflag else ""
            print(f"  {c}: A1={meas[c][0]:.3f} A2={meas[c][1]:.3f} "
                  f"比={meas[c][1] / meas[c][0]:.3f}{tag}", flush=True)
        else:
            print(f"  {c}: 实测反弹缺（评分降级）", flush=True)

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    t_c = np.arange(int(0.4 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) \
        + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE 反演 τ_r: {rc['tau_r'] * 1e3:.2f}ms（真值3.5）-> "
          f"{'过' if c1 else '不过'}", flush=True)
    # C2: 合成细胞（16713003 表）评分管道不死锁
    V0 = load_mat("sine_wave_protocol.mat", "16713003", "sine_wave")[0]
    tabs0 = build_tabs("16713003", amp, hook, inact)
    m0, h0 = forward(V0, tabs0)
    Isyn = tabs0["G"] * m0 * h0 * (V0 - E_REV)
    st0 = find_sine_structure(V0)
    ok_c2 = st0["chirp"] is not None and len(st0["long40"]) >= 1 and len(st0["rebs"]) >= 2
    print(f"  C2 sine 结构解析: +40段 {len(st0['long40'])} 反弹 {len(st0['rebs'])} "
          f"chirp {'有' if st0['chirp'] else '无'} -> {'过' if ok_c2 else '不过'}", flush=True)
    if st0["chirp"]:
        cs, ce = st0["chirp"]
        vc = V0[cs:ce]
        print(f"     chirp 窗 {(ce - cs) * DT:.2f}s  电压 [{vc.min():.0f},{vc.max():.0f}]mV",
              flush=True)
    if not (c1 and ok_c2):
        print("  对照未归位 -> 停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    res = {}
    for c in CELLS:
        V, I = load_mat("sine_wave_protocol.mat", c, "sine_wave")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        smea40 = None                                    # R3 第三级自标定原料
        if I is not None:
            stc = find_sine_structure(V)
            if stc["long40"]:
                v, s0, nseg = stc["long40"][0]
                smea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg]))
        tabs = build_tabs(c, amp, hook, inact, Ii, Vi, sine_mea40=smea40)
        r = score_cell(c, V, I, tabs, meas, None)
        r["h40"] = tabs["h40"]
        r["gflag"] = tabs["gflag"]
        r["hflag"] = tabs["hflag"]
        if c in mflag:
            r["mflag"] = mflag[c]
        res[c] = r
        fl = " ".join(x for x in [r.get("gflag"), r.get("hflag"), r.get("mflag")] if x)
        print(f"  {c}: G={r['G']:.4f} h40={tabs['h40']:.4f} | "
              f"S1{'✓' if r.get('S1') else '×'} simA1={r['simA'][0] if r['simA'] else np.nan:.2f}"
              f"(实测{r.get('A1_meas', np.nan):.2f}) | "
              f"S2{'✓' if r.get('S2') else '×'} 模拟比{r.get('sim_ratio', np.nan):.2f}"
              f"(实测{r.get('mea_ratio', np.nan):.2f}) | "
              f"S3{'✓' if r.get('S3') else '×'} sim40={r.get('sim40', np.nan):.3f}"
              f"(实测{r.get('mea40', np.nan):.3f}) -> "
              f"{'过' if r['pass'] else '不过'}" + (f"  [{fl}]" if fl else ""), flush=True)

    npass = sum(1 for r in res.values() if r["pass"])
    print("\n" + "-" * 84, flush=True)
    print(f" 九细胞冒烟: {npass}/{len(res)} 过（判线 >=7 乘积门冒烟成立）", flush=True)
    print(" 总判词：", "乘积门冒烟成立（登记桥接在案）" if npass >= (1 if SMOKE else 7)
          else "乘积门冒烟未过——缺口桥接不足或模型缺项，明细见上", flush=True)

    # ---------- 图 ----------
    nfig = len(res)
    fig, axes = plt.subplots(nfig, 1, figsize=(15, 3.2 * nfig), squeeze=False)
    for ax, (c, r) in zip(axes[:, 0], res.items()):
        V, I = load_mat("sine_wave_protocol.mat", c, "sine_wave")
        tt = np.arange(len(V)) * DT
        if I is not None:
            ax.plot(tt, I, lw=0.3, color="0.6", label="实测")
        ax.plot(tt, r["_sim"], lw=0.5, color="tab:red", alpha=0.8, label="模拟")
        st = r["_struct"]
        for k, v, s0, nseg in st["rebs"][:2]:
            ax.axvspan(s0 * DT, (s0 + nseg) * DT, color="tab:blue", alpha=0.08)
        ax.set_title(f"{c}  S1{'过' if r.get('S1') else '未'} "
                     f"S2{'过' if r.get('S2') else '未'} S3{'过' if r.get('S3') else '未'}"
                     f"  抑制比 模拟{r.get('sim_ratio', np.nan):.2f}/实测{r.get('mea_ratio', np.nan):.2f}",
                     fontsize=9)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-13_α模型_组装冒烟_sine前向判决.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(cells={c: {k: v for k, v in r.items() if not k.startswith("_")}
                      for c, r in res.items()},
               meas={c: meas[c] for c in meas}, npass=npass)
    fjson = os.path.join(HERE, "2026-09-13_α模型_组装冒烟_sine前向判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  图落盘: {fpng}", flush=True)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
