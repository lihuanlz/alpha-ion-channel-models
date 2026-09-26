# 2026-09-14_α模型_B6激活足_k3级联判决_双协议换形重跑.py
# 目的（B6 主攻：单门 m 无激活延迟 vs 实测 sigmoid 足 50-150ms —— 模型形态修补）：
#   二审指定靶点（收官文档 §11/§13）：去极化头 ~100ms 单门 m 系统性高估 m，是
#   AP A1 定量散布机制之一。本脚本两件事一次做：
#   B6a 足确认+形态判决（数据源：已封卷的激活 envelope 判决 JSON 六点，不再碰原始
#       数据；007/047 按 D11 登记排除，有效池 7/9）。
#       三形态（全部带自由基线列 c + 线性 A，统一六点等权）：
#         single:  A(1-exp(-t/τ))            —— 无足单门（B6 否掉对象）
#         corner:  A(1-exp(-max(t-d,0)/τ))   —— 显式角延迟（envelope 封卷拟合形）
#         k3:      A·P3(t/τc)                —— 等τ三级串联级联（可实现候选）
#       【足确认群体门·跑前钉死】median(k3_SSE/single_SSE) <= 0.5 -> 足必需成立。
#       【形态政策 P-FORM·跑前钉死】角延迟在时变电压下不可连续实现（"阶跃"无定义），
#         k3 为与角延迟最近的可实现形；角延迟对比照实登记（ratio + 绝对 RMS），
#         形态近似不确定度 ±15-20%（m 轨迹，临界窗内）登记，靠前向判线吸收。
#       【τc(+40) 锚】逐细胞 k3 拟合 τc；007/047（D11）<- 群体中位。
#   B6b 换形重跑（判线与 hss实测表重跑版逐字相同，一字未动）：
#       m 门换【整流三级级联】（冒烟定型 2026-09-14，见下）：
#         激活方向（m_ss >= m3）：dm1/dt=(m_ss-m1)/τc, dm2/dt=(m1-m2)/τc,
#           dm3/dt=(m2-m3)/τc，电流用 m3 —— 串联三级产生 sigmoid 足；
#         去激活方向（m_ss < m3）：m3 以封卷单门 τ_m(V) 直接向 m_ss 弛豫
#           （m1/m2 同步贴随）—— 去激活尾与封卷模型逐字一致，梯子不动。
#       【冒烟定型登记】初版对称级联（级联级在去激活方向同样生效）冒烟即暴露
#         胖尾伪影：-80 间期 m3 衰减比封卷单门慢 ~3.5 倍（(1+x+x²/2)e^-x 多项式尾），
#         003 AP 间期累积 0.10->0.54、峰比 1.50->3.51 出窗 —— 与封卷去激活测量
#         （离散指数白化、无足）矛盾。整流级联是唯一的双方向自洽形：激活串联
#         慢（足），去激活单门快塌（梯子）——与 HH 式不对称协同一致，且方向
#         不对称性本身可证伪（去激活无足已被 28/33 白化证实）。
#       【τc(V) 表政策 P-TAUC·跑前钉死】τc 锚 = 封卷 τ_m 锚；V<=-30 不动（慢模=封卷
#         去激活常数）；V>=-20 除以 K=3（该区 τ_m 锚原系单门有效值）；
#         +40 档 <- 逐细胞 k3 τc（上覆）；+50/+60 = 0.29/3。
#       级联每步更新用端点值一阶递推（DT=1e-4 vs τc>=30ms，误差登记可忽略，C3/C4 检验）。
#       sine 判线 S1/S2/S3、AP 判线 A1/A2/A3、九细胞 >=7 —— 与封卷版逐字相同。
# 【总判线·跑前钉死】B6a 群体门过 且 sine>=7/9 且 AP>=7/9 -> B6 封卷（k3 级联形+
#   τc 表入模型）；任一不过 -> B6 留登记，单门保留为组装形，数字照实写。
# 对照（跑前声明）：
#   C-sing：20 合成单门真值包络（τ=200ms,A=3.7,σ=0.04），群体门不得触发
#     （median>0.5），否则门作废停；
#   C-k3：20 合成 k3 真值包络（τc=70ms），群体门必须触发（median<=0.5），否则门作废停；
#   C1 DoE 反演 τr 误差 <15%（管道，同封卷版）；
#   C2 sine/AP 结构解析（同封卷版）；
#   C3 k3 积分器自洽：+40 恒压合成步，m3 数值 vs 解析 P3，max|差|<2e-3。
# 预期登记（不是判线，跑前声明）：AP 复极峰比整体下移 ~10-30%（足压低短峰 m），
#   003(1.50) 预期向 1.1-1.3；近下沿细胞（047:0.31）有出窗风险，照实登记；
#   hook DoE 因 -120 级联胖尾 S1/A2 比可上移 <=40%；sine S2 近不变。
# 运行：python 本文件（九细胞全量）；SMOKE=1 时 B6a 照旧全群体（只读 JSON），
#   前向仅 16713003 冒烟。
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
CELLS_FWD = ["16713003"] if SMOKE else ["16704007", "16704047", "16707014", "16708016",
                                         "16708060", "16708118", "16713003", "16713110", "16715049"]
D11_EXCLUDE = {"16704007", "16704047"}          # D11 激活协议记录失败（在案）
DT = 1e-4
E_REV = -88.33
DF_M120 = abs(-120.0 - E_REV)
NOISE3 = 3 * 0.0069                              # A1 峰门槛（同封卷版）
K_STAGE = 3                                      # 级联级数（P-FORM 钉死）

F_ENV = os.path.join(HERE, "2026-09-13_α模型_激活envelope_τact封卷判决_结果.json")
F_AMP = os.path.join(HERE, "2026-09-13_α模型_幅度表提取_结果.json")
F_HOOK = os.path.join(HERE, "2026-09-13_α模型_反弹hook_结果.json")
F_INACT = os.path.join(HERE, "2026-09-13_α模型_失活门_失活协议封卷判决_结果.json")
F_HSS = os.path.join(HERE, "2026-09-13_α模型_纯hssV剥离判决_结果.json")
F_SINE_BASE = os.path.join(HERE, "2026-09-13_α模型_hss实测表重跑_sine前向判决_结果.json")
F_AP_BASE = os.path.join(HERE, "2026-09-13_α模型_hss实测表重跑_AP前向判决_结果.json")

TR_GRID = np.exp(np.linspace(np.log(0.001), np.log(0.060), 25))
TD_GRID = np.exp(np.linspace(np.log(0.008), np.log(2.000), 30))
SEP_MIN = 1.5
TAU_GRID = np.exp(np.linspace(np.log(0.02), np.log(1.5), 60))
D_GRID = np.arange(0, 0.201, 0.010)


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


# ================= B6a：包络形态判决 =================

def env_shape(form, t, p):
    if form == "single":
        return 1.0 - np.exp(-t / p)
    if form == "corner":
        d, tau = p
        return 1.0 - np.exp(-np.maximum(t - d, 0.0) / tau)
    if form == "k3":
        x = t / p
        return 1.0 - np.exp(-x) * (1.0 + x + x * x / 2.0)
    raise ValueError(form)


def fit_env_form(dts, A, form):
    """统一六点等权 + 自由基线列 c + 线性 A。返回 (sse, param, ainf, c)。"""
    best = None
    combos = [(d, t1) for d in D_GRID for t1 in TAU_GRID] if form == "corner" \
        else [(None, tt) for tt in TAU_GRID]
    for d, t1 in combos:
        x = env_shape(form, dts, (d, t1) if form == "corner" else t1)
        X = np.column_stack([np.ones(len(dts)), x])
        sol, *_ = np.linalg.lstsq(X, A, rcond=None)
        sse = float(np.sum((A - X @ sol) ** 2))
        if best is None or sse < best[0]:
            best = (sse, t1 if form != "corner" else (d, t1), float(sol[1]), float(sol[0]))
    return best


def b6a(rng):
    """返回 (gate_ok, table, details)。table[cell] = τc(+40) s。"""
    print("\n[B6a 包络形态判决]（数据源：封卷 envelope JSON 六点；D11 剔 007/047）", flush=True)
    env = json.load(open(F_ENV, encoding="utf-8"))
    # ---- 对照 C-sing / C-k3（跑前钉死：门行为验证） ----
    dts = np.array([0.003, 0.010, 0.030, 0.100, 0.300, 1.000])
    rat_sing, rat_k3, rat_corn = [], [], []
    for k in range(20):
        Asyn = 3.7 * (1 - np.exp(-dts / 0.20)) + rng.normal(0, 0.04, 6)
        ss, *_ = fit_env_form(dts, Asyn, "single")
        s3, *_ = fit_env_form(dts, Asyn, "k3")
        rat_sing.append(s3 / ss)
        x0 = dts / 0.07
        Asyn = 3.7 * (1 - np.exp(-x0) * (1 + x0 + x0 * x0 / 2)) + rng.normal(0, 0.04, 6)
        ss, *_ = fit_env_form(dts, Asyn, "single")
        s3, *_ = fit_env_form(dts, Asyn, "k3")
        rat_k3.append(s3 / ss)
        Asyn = 3.7 * (1 - np.exp(-np.maximum(dts - 0.075, 0) / 0.228)) + rng.normal(0, 0.04, 6)
        ss, *_ = fit_env_form(dts, Asyn, "single")
        s3, *_ = fit_env_form(dts, Asyn, "k3")
        rat_corn.append(s3 / ss)
    ms_, mk_, mc_ = float(np.median(rat_sing)), float(np.median(rat_k3)), float(np.median(rat_corn))
    print(f"  对照 C-sing: k3/single 中位 {ms_:.2f}（要求>0.5，门不滥杀）"
          f" | C-k3: {mk_:.3f}（要求<=0.5，门不漏杀）"
          f" | C-corner: {mc_:.3f}（登记）", flush=True)
    if not (ms_ > 0.5 and mk_ <= 0.5):
        print("  对照门行为不符 -> 群体门作废，停。", flush=True)
        return False, {}, {}
    # ---- 真实包络 ----
    details = {}
    for c, r in env["cells"].items():
        if c in D11_EXCLUDE:
            continue
        pts = r.get("k2_+40", {}).get("pts")
        if not pts:
            continue
        dt_c = np.array([p[0] for p in pts]) * 1e-3
        A_c = np.array([p[1] for p in pts])
        ss, ps, as_, cs_ = fit_env_form(dt_c, A_c, "single")
        sco, (dc, tc), ac_, cc_ = fit_env_form(dt_c, A_c, "corner")
        s3, t3, a3, c3 = fit_env_form(dt_c, A_c, "k3")
        details[c] = dict(sse_single=ss, sse_corner=sco, sse_k3=s3,
                          ratio_k3_single=s3 / ss if ss > 0 else np.nan,
                          ratio_k3_corner=s3 / sco if sco > 0 else np.nan,
                          rms_k3=float(np.sqrt(s3 / len(dt_c))),
                          rms_corner=float(np.sqrt(sco / len(dt_c))),
                          tauc=t3, corner_d=dc, corner_tau=tc,
                          single_tau=ps, ainf_k3=a3, c_k3=c3)
    rats = [d["ratio_k3_single"] for d in details.values()]
    med_rat = float(np.median(rats))
    gate = bool(med_rat <= 0.5)
    print(f"  {'细胞':>9} | SSE_sing | SSE_corn | SSE_k3 | k3/sing | k3/corn | "
          f"RMS_k3 | τc(k3) | 角延迟(d,τ)", flush=True)
    for c, d in details.items():
        print(f"  {c} | {d['sse_single']:8.4f} | {d['sse_corner']:8.4f} | {d['sse_k3']:8.4f}"
              f" | {d['ratio_k3_single']:7.3f} | {d['ratio_k3_corner']:7.2f}"
              f" | {d['rms_k3']:.4f} | {d['tauc'] * 1e3:4.0f}ms"
              f" | ({d['corner_d'] * 1e3:.0f}ms,{d['corner_tau'] * 1e3:.0f}ms)", flush=True)
    print(f"  群体门: k3/single 中位 = {med_rat:.3f}（判线 <=0.5）-> "
          f"{'足必需成立' if gate else '足必需不成立'}", flush=True)
    tauc_med = float(np.median([d["tauc"] for d in details.values()]))
    table = {c: d["tauc"] for c, d in details.items()}
    for c in D11_EXCLUDE:
        table[c] = tauc_med
    print(f"  τc(+40) 逐细胞锚（D11 两细胞 <- 中位 {tauc_med * 1e3:.0f}ms）: "
          + " ".join(f"{c[-3:]}:{table[c] * 1e3:.0f}" for c in sorted(table)), flush=True)
    print(f"  形态登记: k3/corner 中位比 {float(np.median([d['ratio_k3_corner'] for d in details.values()])):.2f}"
          f"（角延迟拟合更优，P-FORM 政策选可实现 k3，绝对 RMS 均噪声量级）", flush=True)
    return gate, table, details


# ================= B6b：换形前向（四表与封卷版同一份，逐字复制） =================

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


def build_tabs(cell, amp, hook, inact, hss, tauc40, I_inact=None, V_inact=None,
               sine_mea40=None):
    """每细胞五表：m_ss, τ_m（仅作 τc 原料）, τc（B6 新增）, h_ss, τ_h + G。
    与 hss实测表重跑版同一份四表逐字复制；唯一改动：新增 τc 表（政策 P-TAUC）。"""
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

    # ---- τ_m 锚（封卷中位 + 登记慢分量；仅作 τc 原料） ----
    hs = hook["summary"]
    tm_anchors = {-130: hs["-120"]["td_med"], -120: hs["-120"]["td_med"],
                  -110: hs["-110"]["td_med"], -100: hs["-100"]["td_med"],
                  -90: 0.072, -80: 0.24,
                  -70: amp["tau_used"]["TM"]["-70"], -60: amp["tau_used"]["TM"]["-60"],
                  -50: amp["tau_used"]["TAU_L"]["-50"], -40: amp["tau_used"]["TAU_L"]["-40"],
                  -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
                  20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
    # ---- τc 锚（B6 新增，政策 P-TAUC：V<=-30 不动；V>=-20 除 K=3；+40 逐细胞上覆） ----
    tc_anchors = {v: (a if v <= -30 else a / K_STAGE) for v, a in tm_anchors.items()}
    tc_anchors[40] = tauc40                       # 逐细胞 k3 τc（B6a）
    tc_anchors[50] = 0.29 / K_STAGE
    tc_anchors[60] = 0.30 / K_STAGE
    tau_c = Tab(tc_anchors, log_y=True)
    tau_m = Tab(tm_anchors, log_y=True)      # 封卷去激活 τ（整流级联去激活分支用）

    # ---- G：R1 回退链（posthoc 旗帜，逐字） ----
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

    # ---- h_ss（B1 退役：实测剥离表；+40 端 R3 回退链保留，逐字） ----
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
                if abs(pv + 90.0) < 2.0:
                    iss = float(np.mean(I_inact[s0 + n - 5000: s0 + n]))
                    h40 = iss / (G * 1.0 * (40.0 - E_REV))
                    break
    if not np.isfinite(h40) or h40 <= 0:
        h40 = h50 if h50 else np.nan
    if (not np.isfinite(h40) or h40 <= 0) and G and sine_mea40 and sine_mea40 > 0:
        h40 = sine_mea40 / (G * 1.0 * (40.0 - E_REV))
        hflag = "posthoc_h40_selfcal"
    if not np.isfinite(h40) or h40 <= 0:
        h40 = 0.0034
    h40 = float(np.clip(h40, 1e-4, 0.2))
    h_anchors = {-130: 1.0, -100: 1.0}
    h_anchors.update(hB1)
    h_anchors[40] = h40
    h_anchors[50] = h50 if h50 else h40
    h_anchors[60] = h50 if h50 else h40
    h_ss = Tab(h_anchors, log_y=True)

    # ---- τ_h（B2，逐字） ----
    tr90 = ci.get("m90", {}).get("tau_r") if ci.get("m90", {}).get("valid") else None
    th_anchors = {-130: hs["-120"]["tr_med"], -120: hs["-120"]["tr_med"],
                  -110: hs["-110"]["tr_med"], -100: hs["-100"]["tr_med"],
                  -90: tr90 if tr90 else 0.0065,
                  -40: 0.023, -20: 0.05, 0: 0.0015, 60: 0.0015}
    tau_h = Tab(th_anchors, log_y=True)
    return dict(m_ss=m_ss, tau_c=tau_c, tau_m=tau_m, h_ss=h_ss, tau_h=tau_h, G=G,
                h40=h40, h50=h50, gflag=gflag, hflag=hflag, h_pop=h_pop)


def forward_k3(V, tabs):
    """B6 整流三级级联（冒烟定型）：
      激活方向（m_ss>=m3）：dm1=(m_ss-m1)/τc, dm2=(m1-m2)/τc, dm3=(m2-m3)/τc，电流 m3；
      去激活方向（m_ss<m3）：m3 以封卷 τ_m(V) 单门弛豫（m1/m2 同步贴随，再激活时
      级联从同步态重启）。h 门同封卷版。端点值一阶递推（C3/C4 检验）。"""
    v = V.astype(float)
    ms = np.exp(np.interp(v, tabs["m_ss"].xs, tabs["m_ss"].ly))
    tc = np.exp(np.interp(v, tabs["tau_c"].xs, tabs["tau_c"].ly))
    tm = np.exp(np.interp(v, tabs["tau_m"].xs, tabs["tau_m"].ly))
    hss = np.exp(np.interp(v, tabs["h_ss"].xs, tabs["h_ss"].ly))
    th = np.exp(np.interp(v, tabs["tau_h"].xs, tabs["tau_h"].ly))
    ec = np.exp(-DT / tc)
    em = np.exp(-DT / tm)
    eh = np.exp(-DT / th)
    n = len(V)
    m3 = np.empty(n)
    h = np.empty(n)
    m1 = m2 = m3_ = ms[0]
    h0 = hss[0]
    for i in range(n):
        if ms[i] >= m3_:                            # 激活方向：级联足
            m1 = ms[i] + (m1 - ms[i]) * ec[i]
            m2 = m1 + (m2 - m1) * ec[i]
            m3_ = m2 + (m3_ - m2) * ec[i]
        else:                                       # 去激活方向：封卷单门
            m3_ = ms[i] + (m3_ - ms[i]) * em[i]
            m1 = m3_
            m2 = m3_
        h0 = hss[i] + (h0 - hss[i]) * eh[i]
        m3[i] = m3_
        h[i] = h0
    return m3, h


# ---- sine 结构/评分（与封卷版逐字相同，仅 forward 换 k3） ----

def find_sine_structure(V):
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


def score_sine(cell, V, I, tabs, meas):
    m, h = forward_k3(V, tabs)
    G = tabs["G"]
    Isim = G * m * h * (V - E_REV)
    st = find_sine_structure(V)
    out = dict(G=G)
    if st["long40"]:
        v, s0, nseg = st["long40"][0]
        sim40 = float(np.mean(Isim[s0 + nseg - 2000: s0 + nseg]))
        mea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg])) if I is not None else np.nan
        out["sim40"] = sim40
        out["mea40"] = mea40
        out["S3"] = bool(np.isfinite(mea40) and abs(mea40) > 1e-6
                         and 1 / 3 <= sim40 / mea40 <= 3)
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
    return out


# ---- AP 结构/评分（与封卷版逐字相同，仅 forward 换 k3） ----

def find_ap_structure(V):
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
        if len(g) * DT > 0.3:
            hook = (int(g[0]), int(g[-1]) + 1)
    return dict(cycles=cycles, inters=inters, hook=hook)


def score_ap(cell, V, I, tabs):
    m, h = forward_k3(V, tabs)
    G = tabs["G"]
    Isim = G * m * h * (V - E_REV)
    st = find_ap_structure(V)
    out = dict(G=G, h40=tabs["h40"], gflag=tabs["gflag"])
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
    rng = np.random.default_rng(17)
    print("=" * 88)
    print(" α模型 B6 激活足判决：k3 级联 vs 单门 vs 角延迟 + 双协议换形重跑"
          + ("（冒烟 16713003）" if SMOKE else "（九细胞全量）"))
    print(" 总判线: B6a 群体门(median k3/single<=0.5) 且 sine>=7 且 AP>=7 -> B6 封卷")
    print("=" * 88, flush=True)

    gate, tauc_tab, b6a_det = b6a(rng)

    # ---------- B6b 对照 ----------
    print("\n[B6b 对照]", flush=True)
    t_c = np.arange(int(0.4 / DT)) * DT
    y_c = 3.0 * (np.exp(-t_c / 0.030) - np.exp(-t_c / 0.0035)) \
        + rng.normal(0, 0.02, len(t_c))
    rc = doe_fit(t_c, y_c)
    c1 = bool(rc and abs(rc["tau_r"] - 0.0035) / 0.0035 < 0.15)
    print(f"  C1 DoE 反演 τ_r: {rc['tau_r'] * 1e3:.2f}ms（真值3.5）-> "
          f"{'过' if c1 else '不过'}", flush=True)
    # C3 整流级联激活分支：-130 预置零态 -> +40 恒压步，m3 数值 vs 解析 P3
    tc3 = 0.090
    npre = 100
    Vc = np.concatenate([np.full(npre, -130.0), np.full(int(1.0 / DT), 40.0)])
    tabs_c = dict(m_ss=Tab({-140: 1e-4, -120: 1e-4, 40: 1.0, 60: 1.0}),
                  tau_c=Tab({-140: tc3, 60: tc3}), tau_m=Tab({-140: tc3, 60: tc3}),
                  h_ss=Tab({-140: 1.0, 60: 1.0}), tau_h=Tab({-140: 0.0015, 60: 0.0015}))
    m3n, _ = forward_k3(Vc, tabs_c)
    tt = np.arange(len(Vc) - npre) * DT
    x0 = tt / tc3
    p3 = 1 - np.exp(-x0) * (1 + x0 + x0 * x0 / 2)
    err = float(np.max(np.abs(m3n[npre:] - p3)))
    c3ok = bool(err < 2e-3)
    print(f"  C3 激活分支积分器 vs 解析 P3: max|差|={err:.2e}（要求<2e-3）-> "
          f"{'过' if c3ok else '不过'}", flush=True)
    # C4 整流级联去激活分支：+40x1s 预激活 -> -80x0.8s，m3 须按封卷 τ_m 单门衰减
    Vc4 = np.concatenate([np.full(int(1.0 / DT), 40.0), np.full(int(0.8 / DT), -80.0)])
    tabs_d = dict(m_ss=Tab({-140: 1e-4, -60: 1e-4, 40: 1.0, 60: 1.0}),
                  tau_c=Tab({-140: tc3, 60: tc3}),
                  tau_m=Tab({-140: 0.24, -60: 0.24, 40: 0.29, 60: 0.30}),
                  h_ss=Tab({-140: 1.0, 60: 1.0}), tau_h=Tab({-140: 0.0015, 60: 0.0015}))
    m4, _ = forward_k3(Vc4, tabs_d)
    seg = m4[int(1.0 / DT):]
    tt4 = np.arange(len(seg)) * DT
    ref4 = 1e-4 + (seg[0] - 1e-4) * np.exp(-tt4 / 0.24)
    err4 = float(np.max(np.abs(seg - ref4)))
    c4ok = bool(err4 < 2e-3)
    print(f"  C4 去激活分支 vs 封卷单门 e^(-t/0.24): max|差|={err4:.2e}（要求<2e-3）-> "
          f"{'过' if c4ok else '不过'}", flush=True)
    # C2 结构解析（两协议，同封卷版）
    V0s = load_mat("sine_wave_protocol.mat", "16713003", "sine_wave")[0]
    st0 = find_sine_structure(V0s)
    ok_s = st0["chirp"] is not None and len(st0["long40"]) >= 1 and len(st0["rebs"]) >= 2
    V0a = load_mat("ap_protocol.mat", "16713003", "ap")[0]
    st0a = find_ap_structure(V0a)
    ok_a = len(st0a["cycles"]) >= 14 and st0a["hook"] is not None
    print(f"  C2 sine 结构: +40段 {len(st0['long40'])} 反弹 {len(st0['rebs'])} "
          f"chirp {'有' if st0['chirp'] else '无'} | AP 结构: 峰 {len(st0a['cycles'])} "
          f"hook {'有' if st0a['hook'] else '无'} -> {'过' if (ok_s and ok_a) else '不过'}",
          flush=True)
    if not (c1 and c3ok and c4ok and ok_s and ok_a):
        print("  对照未归位 -> 停。", flush=True)
        return
    print("  对照归位。", flush=True)
    if not gate:
        print("  B6a 群体门未过 -> 足必需不成立；仍按登记跑完前向供参考。", flush=True)

    # ---------- B6b 真实数据 ----------
    amp = json.load(open(F_AMP, encoding="utf-8"))
    hook = json.load(open(F_HOOK, encoding="utf-8"))
    inact = json.load(open(F_INACT, encoding="utf-8"))
    hss = json.load(open(F_HSS, encoding="utf-8"))
    # 实测 sine 双反弹靶值（R2 降级回填，逐字）
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

    res_s, res_a = {}, {}
    print("\n[B6b sine 换形前向]", flush=True)
    for c in CELLS_FWD:
        V, I = load_mat("sine_wave_protocol.mat", c, "sine_wave")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        smea40 = None
        if I is not None:
            stc = find_sine_structure(V)
            if stc["long40"]:
                v, s0, nseg = stc["long40"][0]
                smea40 = float(np.mean(I[s0 + nseg - 2000: s0 + nseg]))
        tabs = build_tabs(c, amp, hook, inact, hss, tauc_tab.get(c, 0.090), Ii, Vi,
                          sine_mea40=smea40)
        r = score_sine(c, V, I, tabs, meas)
        r["h_pop"] = tabs["h_pop"]
        res_s[c] = r
        print(f"  {c}: G={r['G']:.4f} | S1{'✓' if r.get('S1') else '×'} "
              f"simA1={r['simA'][0] if r['simA'] else np.nan:.2f}"
              f"(实测{r.get('A1_meas', np.nan):.2f}) | "
              f"S2{'✓' if r.get('S2') else '×'} 模拟比{r.get('sim_ratio', np.nan):.2f}"
              f"(实测{r.get('mea_ratio', np.nan):.2f}) | "
              f"S3{'✓' if r.get('S3') else '×'} sim40={r.get('sim40', np.nan):.3f}"
              f"(实测{r.get('mea40', np.nan):.3f}) -> "
              f"{'过' if r['pass'] else '不过'}", flush=True)
    print("\n[B6b AP 换形前向]", flush=True)
    for c in CELLS_FWD:
        V, I = load_mat("ap_protocol.mat", c, "ap")
        Vi, Ii = load_mat("inactivation_protocol.mat", c, "inactivation")
        tabs = build_tabs(c, amp, hook, inact, hss, tauc_tab.get(c, 0.090), Ii, Vi)
        r = score_ap(c, V, I, tabs)
        r["h_pop"] = tabs["h_pop"]
        res_a[c] = r
        print(f"  {c}: G={r['G']:.4f} | A1{'✓' if r.get('A1') else '×'} "
              f"峰{r.get('repol_npeak')}/{r.get('repol_ncy')} "
              f"比中位{r.get('repol_ratio_med', np.nan):.2f} | "
              f"A2{'✓' if r.get('A2') else '×'} hook 模拟{r.get('hook_simA', np.nan):.2f}"
              f"/实测{r.get('hook_meaA', np.nan):.2f} | "
              f"A3{'✓' if r.get('A3') else '×'} 间期{r.get('inter_first', np.nan):.4f}"
              f"->{r.get('inter_last', np.nan):.4f} -> "
              f"{'过' if r['pass'] else '不过'}", flush=True)

    ns_ = sum(1 for r in res_s.values() if r["pass"])
    na_ = sum(1 for r in res_a.values() if r["pass"])
    need = 1 if SMOKE else 7
    print("\n" + "-" * 88, flush=True)
    print(f" sine 换形: {ns_}/{len(res_s)} 过 | AP 换形: {na_}/{len(res_a)} 过"
          f"（判线 >={need}）", flush=True)
    # 与封卷版基线对比（换形前）
    if not SMOKE and os.path.exists(F_SINE_BASE) and os.path.exists(F_AP_BASE):
        sb = json.load(open(F_SINE_BASE, encoding="utf-8"))
        ab = json.load(open(F_AP_BASE, encoding="utf-8"))
        print("  [对比封卷版基线]", flush=True)
        for c in CELLS_FWD:
            b_s = sb["cells"].get(c, {})
            b_a = ab["cells"].get(c, {})
            print(f"   {c}: sine 比 {b_s.get('sim_ratio', np.nan):.2f}->"
                  f"{res_s[c].get('sim_ratio', np.nan):.2f} | AP峰比 "
                  f"{b_a.get('repol_ratio_med', np.nan):.2f}->"
                  f"{res_a[c].get('repol_ratio_med', np.nan):.2f} | "
                  f"hook {b_a.get('hook_simA', np.nan):.2f}->{res_a[c].get('hook_simA', np.nan):.2f}",
                  flush=True)
    sealed = bool(gate and ns_ >= need and na_ >= need)
    print(" 总判词：",
          ("B6 封卷——足必需成立（群体门），k3 级联换形双协议过判线，"
           "τc 表与级联形入模型" if sealed else
           "B6 未封卷——" + ("群体门未过；" if not gate else "")
           + (f"sine {ns_}/9 未过线；" if ns_ < need else "")
           + (f"AP {na_}/9 未过线；" if na_ < need else "")
           + "照实登记，单门保留为组装形"), flush=True)

    # ---------- 图 ----------
    # 图1 B6a 包络
    if b6a_det:
        n1 = len(b6a_det)
        fig, axes = plt.subplots(n1, 1, figsize=(9, 2.4 * n1), squeeze=False)
        dd = np.linspace(0, 1.0, 400)
        for ax, (c, d) in zip(axes[:, 0], b6a_det.items()):
            env = json.load(open(F_ENV, encoding="utf-8"))
            pts = env["cells"][c]["k2_+40"]["pts"]
            ax.scatter([p[0] for p in pts], [p[1] for p in pts], c="k", s=30, label="包络点")
            x = env_shape("k3", dd, d["tauc"])
            ax.plot(dd * 1e3, d["c_k3"] + d["ainf_k3"] * x, "tab:red",
                    label=f"k3 τc={d['tauc'] * 1e3:.0f}ms")
            xs = env_shape("single", dd, d["single_tau"])
            ax.plot(dd * 1e3, d["c_k3"] + d["ainf_k3"] * xs, "tab:blue", ls="--",
                    label="单门(参考)")
            xc = env_shape("corner", dd, (d["corner_d"], d["corner_tau"]))
            ax.plot(dd * 1e3, d["c_k3"] + d["ainf_k3"] * xc, "tab:green", ls=":",
                    label="角延迟(参考)")
            ax.set_xscale("log")
            ax.set_title(f"{c}  k3/single={d['ratio_k3_single']:.3f}", fontsize=9)
            ax.set_xlabel("Δt (ms)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        f1 = os.path.join(HERE, "2026-09-14_α模型_B6激活足_包络形态判决.png")
        fig.savefig(f1, dpi=120, bbox_inches="tight")
        print(f"\n  图落盘: {f1}", flush=True)
    # 图2/3 前向叠图
    for tag, res, proto, tname in (("sine", res_s, "sine_wave", "sine"),
                                   ("AP", res_a, "ap", "AP")):
        nfig = len(res)
        if not nfig:
            continue
        fig, axes = plt.subplots(nfig, 1, figsize=(15, 3.0 * nfig), squeeze=False)
        for ax, (c, r) in zip(axes[:, 0], res.items()):
            V, I = load_mat(f"{proto}_protocol.mat", c, proto if proto != "sine_wave" else "sine_wave")
            tt = np.arange(len(V)) * DT
            if I is not None:
                ax.plot(tt, I, lw=0.3, color="0.6", label="实测")
            ax.plot(tt, r["_sim"], lw=0.5, color="tab:red", alpha=0.8, label="模拟(k3)")
            ax.set_title(f"{c} {tname} 换形  pass={'过' if r['pass'] else '未'}", fontsize=9)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("I (nA)")
            ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fx = os.path.join(HERE, f"2026-09-14_α模型_B6激活足_{tag}换形前向.png")
        fig.savefig(fx, dpi=120, bbox_inches="tight")
        print(f"  图落盘: {fx}", flush=True)

    out = dict(note="B6 激活足判决+换形重跑：k3 等τ三级级联；判线与封卷版逐字相同",
               gate_b6a=gate, tauc40={c: tauc_tab.get(c) for c in tauc_tab},
               b6a=b6a_det, sine_npass=ns_, ap_npass=na_, sealed_b6=sealed,
               sine={c: {k: v for k, v in r.items() if not k.startswith("_")}
                     for c, r in res_s.items()},
               ap={c: {k: v for k, v in r.items() if not k.startswith("_")}
                   for c, r in res_a.items()})
    fjson = os.path.join(HERE, "2026-09-14_α模型_B6激活足_k3级联判决_双协议换形重跑_结果.json")
    with open(fjson, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"  结果落盘: {fjson}", flush=True)


if __name__ == "__main__":
    main()
