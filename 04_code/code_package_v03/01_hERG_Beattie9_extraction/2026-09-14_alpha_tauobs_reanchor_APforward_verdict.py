# 2026-09-14_α模型_τobs换锚_AP前向判决.py
# 【τ_obs 换锚·B2 退役 2026-09-14】本文件 = hss实测表重跑_AP前向判决.py 的唯一改动版：
#   τ_h 复极段 B2 桥锚退役，换 τ_obs 封卷实测上界中位表（τobs判决_复极窗快弛豫封卷_结果.json，
#   封卷 5/6 档，§13）：新增显式锚 -70: 7.0ms、-60: 7.0ms、-50: 8.9ms、-40: 14.4ms、
#   -30: 14.4ms；原桥锚 -40: 23ms 让位；-20: 50ms 无实测覆盖，桥锚保留（登记）；0/+60: 1.5ms 不动。
#   其余（m_ss/τ_m/h_ss/G 表、P1-P5、R1-R3、对照 C1/C2、评分口径）逐字不动。
# 判线（跑前钉死，与 hss 实测表重跑版逐字相同，一字未动）：
#   A1 复极回弹 >=80% 周期且峰比中位∈[0.3,3]；A2 hook DoE∈[0.3,3]；A3 间期累积方向；
#   每细胞 A1/A2/A3 全过 -> 过；九细胞 >=7 -> AP 前向对 B2 退役稳健。
# 意义：>=7 仍过 -> B2 桥无伤退役，τ_h 复极段实测化收口；掉 -> 照实登记，桥值回滚备案。
# 运行：python 本文件（九细胞全量）；SMOKE=1 单细胞 16713003 冒烟。
# ---------- 以下为 hss 实测表重跑版文件头（历史声明，改动以本条为准） ----------
# 2026-09-13_α模型_hss实测表重跑_AP前向判决.py
# 【B1 退役·换表重跑 2026-09-13 深夜】本文件 = AP前向判决.py 的唯一改动版：
#   h_ss B1 段（-80..+30，12 档）锚换实测剥离表（纯hssV剥离判决_结果.json，运行时
#   加载），m_ss=y_ss/h_ss 同步重分裂；政策 P1-P5 与 sine 换表版逐字相同（见该文件头）。
# 判线与 B1 版逐字相同（A1/A2/A3、九细胞>=7，一字未动）；对照 C1/C2 相同。
# 意义：>=7 仍过 -> AP 前向对 B1 之死稳健；掉 -> 原 7/9 依赖错误假设，照实登记。
# ---------- 以下为原 B1 版文件头（除 B1 条已作废外，其余声明仍然有效） ----------
# 目的（AP clamp 决战，不是调参——与 sine 冒烟同一套四表、同一 G、同一 E_rev=-88.33，
#   零调参前向打 AP 协议）：dm/dt=(m_ss-m)/τ_m, dh/dt=(h_ss-h)/τ_h, I=G·m·h·(V-E_rev)。
#   V(t) 直接读 ap_protocol.mat（17 个去极化峰的数字化 AP 钳制波形，原样吃进）。
#
# 【协议结构（2026-09-13 程序化解析在案）】
#   前置 -80x0.25s -> -120x0.05s 预脉冲 -> -80；0.57s 起 17 个去极化峰
#   （峰 +8.8~+71.8mV，间期回 -80）；7.325s 尾部 -120x0.5s 反弹 hook；尾 -80x1.0s。
#   全长 8.82s，88245 点 @DT=1e-4。
#
# 【实测形态依据（16713003 抽样，判线设计依据）】
#   复极窗峰 0.09~2.05nA >> 平台窗 -0.001~0.335（hERG 复极回弹 signature）；
#   间期 -80 窗电流 0.020 -> 0.189nA 跨周期累积（~8倍）；
#   -120 hook 翻正幅度 ~1.2nA；噪声 std 0.0069nA（前置 -80 段）。
#
# 判线（跑前声明）：
#   A1 复极回弹自发（头号靶）：>=80% 周期（17个中>=14）模拟复极窗
#      （峰后至 V<-40）出现电流峰（max > 平台窗均值+0.021=3x噪声），
#      且这些周期 模拟复极峰/实测复极峰 的中位比 ∈ [0.3,3]（G 不确定度）；
#   A2 -120 hook 反弹幅度：DoE 反演 模拟/实测 ∈ [0.3,3]（同 sine S1 口径）；
#   A3 间期累积方向：模拟间期 -80 窗电流 末1/3均值 > 首1/3均值（方向判，
#      实测 0.020->0.189 约8倍）；
#   每细胞 A1/A2/A3 全过 -> 该细胞过；九细胞过 >=7 -> α模型 AP 前向成立。
#
# 桥接声明（与 sine 冒烟同一套，一字未动）：
#   B1【已作废·本版换实测表】h_ss(V<=-40)=1.0 物理声明；h40 回退链（失活协议直提 ->
#      h_ss50 -> 默认0.0034；+40 端 R3 链本版保留 P4，16708060 停在 h50=0.0022 在案）。
#   B2 τ_rec(V) 4点对数外推；B3 τ_m 锚/逻辑桥；B4 -40档y_ss伪影剔除；
#   B5 单门 m 声明。R1 G 回退链同 sine（16704047 走 posthoc_G_inact=0.0718）。
# 对照（跑前声明）：
#   C1 DoE 反演 τ_r 误差 <15%；
#   C2 合成全程（16713003 表）结构解析 17峰+hook 且 A1/A2/A3 可计算（管道不死锁）。
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
NOISE3 = 3 * 0.0069          # A1 峰门槛（16713003 前置段噪声，九细胞同口径）

F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")

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
    """对数-线性插值锚表（x 线性、y 对数，越界取端点）。与 sine 冒烟同一份。"""
    def __init__(self, anchors, log_y=True):
        self.xs = np.array(sorted(anchors), dtype=float)
        ys = np.array([anchors[x] for x in self.xs], dtype=float)
        self.ly = np.log(np.clip(ys, 1e-9, None)) if log_y else ys
        self.log_y = log_y

    def __call__(self, v):
        y = np.interp(v, self.xs, self.ly)
        return float(np.exp(y)) if self.log_y else float(y)


def build_tabs(cell, amp, hook, inact, hss, I_inact=None, V_inact=None):
    """与 sine 换表版同一份四表（sine_mea40 级无原料，其余逐字相同）。返回 dict。
    换表版：h_ss B1 段（-80..+30）用实测剥离表；m_ss=y_ss/h_ss 同步重分裂。"""
    # ---- h_ss 实测剥离表（P3：每细胞 qc 过 -> 每细胞值；否 -> 群体中位） ----
    hc = hss["cells"].get(cell, {}).get("curve", {})
    hB1, h_pop = {}, []
    for v in (-80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30):
        e = hc.get(str(v))
        if e and e.get("qc") and e["h"] > 0:
            hB1[v] = float(e["h"])
        else:
            hB1[v] = float(hss["gears"][str(v)]["med"])
            h_pop.append(v)
    # ---- m_ss 锚（每细胞 y_ss，缺档用中位；换表重分裂 m_ss=y_ss/h_ss） ----
    yss = {}
    for r in amp["rows"]:
        if r["v"] in (-70, -60, -50):
            yss.setdefault(r["v"], {})[r["cell"]] = max(r["y_ss"], 1e-4)
    med = {v: float(np.median(list(d.values()))) for v, d in yss.items()}
    def cell_y(v):
        return yss.get(v, {}).get(cell, med.get(v, 0.02))
    m_anchors = {-130: 1e-4, -120: 1e-4, -110: 1e-4, -100: 1e-4, -90: 1e-4,
                 -80: min(1.0, cell_y(-70) * 0.5 / hB1[-80]),
                 -70: min(1.0, cell_y(-70) / hB1[-70]),
                 -60: min(1.0, cell_y(-60) / hB1[-60]),
                 -50: min(1.0, cell_y(-50) / hB1[-50]),
                 -40: min(1.0, cell_y(-50) * 3.0 / hB1[-40]),
                 -30: min(1.0, cell_y(-50) * 8.0 / hB1[-30]),
                 -20: 1.0, 0: 1.0, 20: 1.0, 40: 1.0, 60: 1.0}
    m_ss = Tab(m_anchors, log_y=True)

    hs = hook["summary"]
    tm_anchors = {-130: hs["-120"]["td_med"], -120: hs["-120"]["td_med"],
                  -110: hs["-110"]["td_med"], -100: hs["-100"]["td_med"],
                  -90: 0.072, -80: 0.24,
                  -70: amp["tau_used"]["TM"]["-70"], -60: amp["tau_used"]["TM"]["-60"],
                  -50: amp["tau_used"]["TAU_L"]["-50"], -40: amp["tau_used"]["TAU_L"]["-40"],
                  -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
                  20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
    tau_m = Tab(tm_anchors, log_y=True)

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
    if G is None and ci.get("g_hat") and ci["g_hat"] > 0:
        G = float(ci["g_hat"])
        gflag = "posthoc_G_inact"
    if G is None:
        cands = [abs(r["A"]) / DF_M120 for r in hook["A_rows"]
                 if r["cell"] == cell and r["v"] == -120 and r["A"] != 0]
        if cands:
            G = float(np.median(cands))
            gflag = "posthoc_G_degraded"

    # ---- h_ss（B1 退役：实测剥离表；+40 端 R3 回退链保留，P4） ----
    h50 = None
    if ci.get("h_ss50") and ci["h_ss50"] > 0:
        h50 = float(ci["h_ss50"])
    h40 = np.nan
    if I_inact is not None and G is not None and G > 0:
        edges = np.where(np.diff(V_inact) != 0)[0] + 1
        info = [(float(V_inact[s[0]]), int(s[0]), len(s))
                for s in np.split(np.arange(len(V_inact)), edges)]
        for k, (v, s0, n) in enumerate(info):
            if abs(v - 40.0) < 2.0 and 0.14 < n * DT < 0.16 and k >= 1:
                pv = info[k - 1][0]
                if abs(pv + 90.0) < 2.0:
                    iss = float(np.mean(I_inact[s0 + n - 5000: s0 + n]))
                    h40 = iss / (G * 1.0 * (40.0 - E_REV))
                    break
    if not np.isfinite(h40) or h40 <= 0:
        h40 = h50 if h50 else np.nan
    if not np.isfinite(h40) or h40 <= 0:
        h40 = 0.0034
    h40 = float(np.clip(h40, 1e-4, 0.2))
    h_anchors = {-130: 1.0, -100: 1.0}                       # P1
    h_anchors.update(hB1)                                    # B1 段实测（-80..+30）
    h_anchors[40] = h40                                      # P4：+40 端 R3 链保留
    h_anchors[50] = h50 if h50 else h40
    h_anchors[60] = h50 if h50 else h40
    h_ss = Tab(h_anchors, log_y=True)

    # ---- τ_h（B2 退役：τ_obs 封卷实测上界中位表；-20 桥锚保留登记） ----
    tr90 = ci.get("m90", {}).get("tau_r") if ci.get("m90", {}).get("valid") else None
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: tr90 if tr90 else 0.0065,
                  -70: 0.0070, -60: 0.0070, -50: 0.0089, -40: 0.0144, -30: 0.0144,
                  -20: 0.05, 0: 0.0015, 60: 0.0015}
    tau_h = Tab(th_anchors, log_y=True)
    return dict(m_ss=m_ss, tau_m=tau_m, h_ss=h_ss, tau_h=tau_h, G=G,
                h40=h40, h50=h50, gflag=gflag, h_pop=h_pop)


def forward(V, tabs):
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    n = len(V)
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


def find_ap_structure(V):
    """程序化解析 AP 结构：>0mV 去极化组、复极窗、平台窗、间期窗、-120 hook。"""
    n = len(V)
    idx = np.where(V > 0)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 100)[0] + 1) if len(idx) else []
    cycles = []
    for g in grp:
        pk = g[np.argmax(V[g])]
        after = np.where(V[pk:] < -40)[0]
        w_end = pk + int(after[0]) if len(after) else min(pk + 2000, n - 1)
        plat = (max(0, pk - 100), min(pk + 100, n))
        cycles.append(dict(pk=int(pk), repol=(int(pk), int(w_end)),
                           plat=plat, vmax=float(V[pk])))
    inters = []
    for k in range(len(grp) - 1):
        a, b = int(grp[k][-1]), int(grp[k + 1][0])
        if b - a > 500:
            inters.append((a, b))
    h120 = np.where(V < -119.5)[0]
    gh = np.split(h120, np.where(np.diff(h120) > 100)[0] + 1) if len(h120) else []
    hook = None
    for g in gh:
        if len(g) * DT > 0.3:                      # 0.5s 尾 hook（不要 0.05s 预脉冲）
            hook = (int(g[0]), int(g[-1]) + 1)
    return dict(cycles=cycles, inters=inters, hook=hook)


def score_cell(cell, V, I, tabs):
    m, h = forward(V, tabs)
    G = tabs["G"]
    Isim = G * m * h * (V - E_REV)
    st = find_ap_structure(V)
    out = dict(G=G, h40=tabs["h40"], gflag=tabs["gflag"])
    # ---- A1 复极回弹 ----
    ratios, npeak = [], 0
    for cyc in st["cycles"]:
        a, b = cyc["repol"]
        if b - a < 20:
            continue
        p0, p1 = cyc["plat"]
        s_pk = float(np.max(Isim[a:b]))
        s_pl = float(np.mean(Isim[p0:p1]))
        has = s_pk > s_pl + NOISE3
        npeak += int(has)
        if I is not None:
            m_pk = float(np.max(I[a:b]))
            if m_pk > NOISE3:
                ratios.append(s_pk / m_pk)
    ncy = len(st["cycles"])
    out["repol_npeak"] = npeak
    out["repol_ncy"] = ncy
    out["repol_ratio_med"] = float(np.median(ratios)) if ratios else np.nan
    out["A1"] = bool(ncy and npeak >= 0.8 * ncy
                     and np.isfinite(out["repol_ratio_med"])
                     and 0.3 <= out["repol_ratio_med"] <= 3.0)
    # ---- A2 -120 hook DoE ----
    if st["hook"]:
        a, b = st["hook"]
        t = np.arange(b - a) * DT
        rs = doe_fit(t, -Isim[a:b])
        out["hook_simA"] = rs["A"] if rs else np.nan
        if I is not None:
            rm = doe_fit(t, -I[a:b])
            out["hook_meaA"] = rm["A"] if rm else np.nan
            out["A2"] = bool(rs and rm and rm["A"] > 0
                             and 0.3 <= rs["A"] / rm["A"] <= 3.0)
    # ---- A3 间期累积方向 ----
    if len(st["inters"]) >= 3:
        k = max(1, len(st["inters"]) // 3)
        first = np.concatenate([Isim[a:b] for a, b in st["inters"][:k]])
        last = np.concatenate([Isim[a:b] for a, b in st["inters"][-k:]])
        out["inter_first"] = float(np.mean(first))
        out["inter_last"] = float(np.mean(last))
        out["A3"] = bool(out["inter_last"] > out["inter_first"])
    out["pass"] = bool(out.get("A1") and out.get("A2") and out.get("A3"))
    out["_sim"] = Isim
    out["_struct"] = st
    return out


def main():
    rng = np.random.default_rng(5)
    print("=" * 84)
    print(" α模型 AP 前向判决·τ_obs 换锚重跑（B2 退役）"
          + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"))
    print(" 判线: A1 复极回弹>=80%周期且峰比中位∈[0.3,3] | A2 hook DoE∈[0.3,3] | A3 累积方向"
          "（与 hss 实测表重跑版逐字相同）")
    print(" 改动: 仅 τ_h 复极段 -70..-30 换 τ_obs 实测上界中位 7.0/7.0/8.9/14.4/14.4ms")
    print(" 纪律: 与 sine 换表版同一套四表同一 G 同一 E_rev，零调参")
    print("=" * 84, flush=True)

    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))

    # ---------- 对照 ----------
    print("\n[对照]", flush=True)
    t_c = np.arange(int(0.4 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) \
        + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE 反演 τ_r: {rc['tau_r'] * 1e3:.2f}ms（真值3.5）-> "
          f"{'过' if c1 else '不过'}", flush=True)
    V0 = load_mat("ap_protocol.mat", "16713003", "ap")[0]
    st0 = find_ap_structure(V0)
    ok_c2 = len(st0["cycles"]) >= 14 and st0["hook"] is not None
    print(f"  C2 AP 结构解析: 去极化峰 {len(st0['cycles'])} 个, 间期窗 {len(st0['inters'])}, "
          f"hook {'有' if st0['hook'] else '无'} -> {'过' if ok_c2 else '不过'}", flush=True)
    if not (c1 and ok_c2):
        print("  对照未归位 -> 停。", flush=True)
        return
    print("  对照归位。", flush=True)

    # ---------- 真实数据 ----------
    print("\n[真实数据]", flush=True)
    res = {}
    for c in CELLS:
        V, I = load_mat("ap_protocol.mat", c, "ap")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        tabs = build_tabs(c, amp, hook, inact, hss, Ii, Vi)
        r = score_cell(c, V, I, tabs)
        r["h_pop"] = tabs["h_pop"]
        res[c] = r
        fl = (f"  [{r['gflag']}]" if r.get("gflag") else "") + \
             (f"  [hss_pop{r['h_pop']}]" if r["h_pop"] else "")
        print(f"  {c}: G={r['G']:.4f} h40={r['h40']:.4f} | "
              f"A1{'✓' if r.get('A1') else '×'} 峰{r.get('repol_npeak')}/{r.get('repol_ncy')}"
              f" 比中位{r.get('repol_ratio_med', np.nan):.2f} | "
              f"A2{'✓' if r.get('A2') else '×'} hook 模拟{r.get('hook_simA', np.nan):.2f}"
              f"/实测{r.get('hook_meaA', np.nan):.2f} | "
              f"A3{'✓' if r.get('A3') else '×'} 间期{r.get('inter_first', np.nan):.4f}"
              f"->{r.get('inter_last', np.nan):.4f} -> "
              f"{'过' if r['pass'] else '不过'}{fl}", flush=True)

    npass = sum(1 for r in res.values() if r["pass"])
    print("\n" + "-" * 84, flush=True)
    print(f" 九细胞 AP τ_obs 换锚重跑: {npass}/{len(res)} 过（判线 >=7 对 B2 退役稳健）", flush=True)
    print(" 总判词：", "AP 前向对 B2 退役稳健（τ_obs 实测锚，判线未动）——B2 桥退役收口"
          if npass >= (1 if SMOKE else 7)
          else "换锚后未过判线——B2 桥值有载，照实登记并回滚备案，明细见上", flush=True)

    # ---------- 图 ----------
    nfig = len(res)
    fig, axes = plt.subplots(nfig, 1, figsize=(15, 3.0 * nfig), squeeze=False)
    for ax, (c, r) in zip(axes[:, 0], res.items()):
        V, I = load_mat("ap_protocol.mat", c, "ap")
        tt = np.arange(len(V)) * DT
        if I is not None:
            ax.plot(tt, I, lw=0.3, color="0.6", label="实测")
        ax.plot(tt, r["_sim"], lw=0.5, color="tab:red", alpha=0.8, label="模拟")
        if r["_struct"]["hook"]:
            a, b = r["_struct"]["hook"]
            ax.axvspan(a * DT, b * DT, color="tab:blue", alpha=0.08)
        ax.set_title(f"{c}  A1{'过' if r.get('A1') else '未'} A2{'过' if r.get('A2') else '未'} "
                     f"A3{'过' if r.get('A3') else '未'}  复极峰比中位 "
                     f"{r.get('repol_ratio_med', np.nan):.2f}", fontsize=9)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("I (nA)")
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fpng = os.path.join(HERE, "2026-09-14_α模型_τobs换锚_AP前向判决.png")
    fig.savefig(fpng, dpi=120, bbox_inches="tight")

    out = dict(note="B2 退役 τ_obs 换锚重跑：τ_h(-70/-60/-50/-40/-30)=实测上界中位 "
                    "7.0/7.0/8.9/14.4/14.4ms；-20 桥锚 50ms 保留登记；"
                    "判线与 hss 实测表重跑版逐字相同",
               cells={c: {k: v for k, v in r.items() if not k.startswith("_")}
                      for c, r in res.items()}, npass=npass)
    fjson = os.path.join(HERE, "2026-09-14_α模型_τobs换锚_AP前向判决_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n  图落盘: {fpng}", flush=True)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
