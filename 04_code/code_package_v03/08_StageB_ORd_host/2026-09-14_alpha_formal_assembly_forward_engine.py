# 2026-09-14_α模型_正式组装_前向引擎.py
# ============================================================================
# α 模型 · 正式组装 · 唯一正式前向引擎（2026-09-14 定稿）
#
# 模型方程（单门 m，乘积门）：
#   dm/dt = (m_ss(V) - m) / tau_m(V)
#   dh/dt = (h_ss(V) - h) / tau_h(V)
#   I(t)  = G * m * h * (V - E_rev)          E_rev = -88.33 mV
#
# 本文件 = 2026-09-13_α模型_hss实测表重跑_AP前向判决.py 的模型部分继承
#   （Tab / forward 逐字；build_tabs 仅 th_anchors 一处换锚——2026-09-14 τ_obs 换锚，
#    与 τobs换锚版判决脚本一致；评分件 doe_fit/find_ap_structure/score_cell 属判决管道
#    不入引擎）。四张测量表运行时从封卷 JSON 加载，不手抄：
#     幅度表提取_结果.json        y_ss(-70/-60/-50) 每细胞、τ 阶梯 TF/TM/TAU_L
#     反弹hook_结果.json          τ_rec/τ_deact(-120/-110/-100) 中位、G 锚 A(-120)
#     失活门_失活协议封卷判决_结果.json  h_ss50、g_hat、m90 τ_r(-90)
#     纯hssV剥离判决_结果.json    h_ss(-80..+30) 12 档每细胞值+群体中位
#
# 封卷战绩（本引擎原样产生，判线跑前钉死）：
#   AP 前向 8/9（稳健峰口径；唯一败例 16707014，模型侧登记：h 表全群体中位+D4）
#   sine 前向 7/9（败例 047 数据侧 posthoc_target_degraded、060 数据侧 D1）
#   去激活阶梯白化 28/33；h_ss 12 档实测剥离（4 档封卷/8 档登记）；
#   τ_obs 复极窗快弛豫封卷 5/6 档（h_ss=h_150 身份封卷）；
#   τ_obs 换锚重跑（B2 退役 2026-09-14）：sine 7/9、AP 8/9 同集合同归因，判线未动。
#
# 登记限制（随附，逐条见模型卡 §7-§9）：
#   B2 已退役（2026-09-14 换锚重跑过线：sine 7/9、AP 8/9 同集合同归因，判线未动）：
#     τ_h(-70/-60/-50/-40/-30)=τ_obs 实测上界中位 7.0/7.0/8.9/14.4/14.4ms；
#     -20: 50ms 无实测覆盖，桥锚保留登记（τ_h 链唯一残留桥锚）；
#   B3 τ_m(V>=-40) 登记带（激活慢分量中位，CV 0.36 未封卷）；
#   B5 单门 m 声明——B6 判决（2026-09-14）：激活足（延迟 50-150ms）在 +40 阶跃
#     封卷成立，但整流级联全局施加被双协议拒绝（sine 1/9、AP 2/9）-> 单门保留为
#     组装形；代价登记：去极化头 100-200ms m 高估 ~3x（AP A1 定量散布机制之一）；
#     3x 矛盾（包络 m≈0.08·m∞ @100ms vs AP 要求 m≈0.29·m∞）调和候选 H1/H2/H3
#     全部登记未决。
#
# 【已拒·登记】k3 整流级联（B6b 换形，被拒，不入模型，留存备查）：
#   激活方向（m_ss>=m3）三级等τ串联（tau_c(+40) 逐细胞 48-108ms 中位 86ms），
#   去激活方向单门弛豫（梯子不动）；对照 C3 6.7e-4 / C4 4.6e-14 归位。
#   拒绝原因：chirp 期 m 过抑制 2x（sine S2 塌）、AP 短峰期 m 压低 ~3x（A1 塌）。
#   对称级联更早被否（胖尾伪影：003 AP 间期累积 0.10->0.54 出窗）。
#
# 运行：本文件为模块；python 本文件 = 16713003 锚表打印自检（非判决，不产生判词）。
#   正式判决管道见同目录 *hss实测表重跑_*前向判决.py（判线跑前钉死版）。
# ============================================================================
import os
import json
import numpy as np
import scipy.io as sio

DATA = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/04_细胞线4/wA1/代码A1_16704007_流程固定重标卡_2026-09-11"
HERE = os.path.dirname(os.path.abspath(__file__))
DT = 1e-4
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)

F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")


def load_mat(proto, cell, tag):
    V = sio.loadmat(f"{DATA}/data/protocols/{proto}")['T'].flatten().astype(float)
    fp = f"{DATA}/data/cells/{cell}/{tag}_{cell}_dofetilide_subtracted_leak_subtracted.mat"
    if not os.path.exists(fp):
        return V, None
    I = sio.loadmat(fp)['T'].flatten().astype(float)
    n = min(len(I), len(V))
    return V[:n], I[:n]


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

    # ---- τ_h（B2 已退役 2026-09-14：τ_obs 封卷实测上界中位表；-20 桥锚保留登记） ----
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


# ----------------------------------------------------------------------------
# 【已拒·登记】k3 整流级联换形（B6b，2026-09-14 判决：sine 1/9、AP 2/9 被拒）
# 留存备查，不入模型。参考实现（来自 B6 判决脚本，注释态）：
#
#   def forward_k3_rectified(V, tabs, tauc_tab):
#       # 激活方向（m_ss >= m3）：三级等τ串联（足）
#       #   dm1/dt=(m_ss-m1)/τc, dm2/dt=(m1-m2)/τc, dm3/dt=(m2-m3)/τc
#       # 去激活方向（m_ss < m3）：m3 以封卷 tau_m 单门弛豫（梯子逐字不动）
#       ...（见 2026-09-14_α模型_B6激活足_k3级联判决_双协议换形重跑.py）
#   τc(+40) 逐细胞锚 48-108ms（中位 86ms，D11 两细胞 007/047 取中位）。
#   拒绝机制：chirp 期 m 过抑制 ~2x（S2 塌）；AP 短峰期 m 压低 ~3x（A1 塌）。
# ----------------------------------------------------------------------------


def _selfcheck():
    """锚表打印自检（非判决，不产生判词）：16713003 四表锚值 + G。"""
    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))
    Vi, Ii = load_mat("inactivation_protocol.mat", "16713003", "inactivation")
    tabs = build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)
    print("α模型正式引擎自检（16713003，非判决）")
    print(f"  G = {tabs['G']:.4f} nA/mV  h40 = {tabs['h40']:.4f}  gflag = {tabs['gflag']}")
    print(f"  h_pop（群体中位回退档）= {tabs['h_pop']}")
    for v in (-120, -90, -70, -50, -40, -30, 0, 20, 40):
        print(f"  V={v:+5.0f}mV: m_ss={tabs['m_ss'](v):.4f}  tau_m={tabs['tau_m'](v)*1e3:9.2f}ms"
              f"  h_ss={tabs['h_ss'](v):.4f}  tau_h={tabs['tau_h'](v)*1e3:7.2f}ms")


if __name__ == "__main__":
    _selfcheck()
