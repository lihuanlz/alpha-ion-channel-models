# lei211_J3_提取.py
# 判线来源：预注册判决卡 §2-J3-Amended（修订A1/A4）
# 内容：τ_deact(-40,-60)（staircase 下跳尾，封卷判线 CV<0.3 且 n>=105）
#       τ_rec 负档族（sinactiv 测试段 DoE，登记）
#       τ_act/τ_inact（staircase 上跳双相 + -80→+40 大补跳，登记）
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "数据", "Lei全量")
SMOKE = os.environ.get("SMOKE", "0") == "1"

BATCH = os.environ.get("BATCH", "herg25oc1")
WELLS = [l.strip() for l in open(os.path.join(DATA, "qc", f"selected-{BATCH}.txt"), encoding="utf-8")
         if l.strip() and not l.startswith("#")]
if SMOKE:
    WELLS = ["A01", "B03", "C01"]

DT_I = 2e-4
DT_V = 1e-4

# staircase 阶梯段序列（程序化解析在案，0.5s 档）
# 序列: -80x1.0, -40,-60,-20,-40,0,-20,20,0,40,20,40,0,20,-20,0,-40,-20,-60,-40, -80x1.0
# 下跳: ->-40 (来自-20, 第4段), ->-60 (来自-40, 第2段), ->-40 (来自0?否), ->-60 (来自-20), ->-40 (来自-20末)
STAIR_SEQ = [-40, -60, -20, -40, 0, -20, 20, 0, 40, 20, 40, 0, 20, -20, 0, -40, -20, -60, -40]


def load_protocol(name):
    d = np.genfromtxt(os.path.join(DATA, "protocol", f"protocol-{name}.csv"),
                      delimiter=",", skip_header=1)
    return d[:, 0], d[:, 1:]


def load_current(proto, well):
    d = np.genfromtxt(os.path.join(DATA, proto if BATCH=="herg25oc1" else os.path.join(BATCH, proto), f"{BATCH}-{proto}-{well}.csv"),
                      delimiter=",", skip_header=1)
    if d.ndim == 1:
        d = d[:, None]
    return d


def releak_sweep(i5, v10):
    v = v10[::2][:len(i5)]
    v0 = v[0]
    wa, wb = int(0.175 / DT_I), int(0.2 / DT_I)
    mv = np.mean(v[wa:wb] - v0)
    g = 0.0 if abs(mv) < 1e-9 else -np.mean(i5[wa:wb]) / mv
    return i5 + g * (v - v0), g


def segments(v10):
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]


def fit_biexp_tail(tt, y, sig):
    """带锚双指数网格：y=y0+a1*e(-t/t1)+a2*e(-t/t2)，t1<t2 对数网格"""
    best = None
    for t1 in np.exp(np.linspace(np.log(0.005), np.log(0.3), 18)):
        for t2 in np.exp(np.linspace(np.log(max(0.02, t1 * 1.8)), np.log(6.0), 22)):
            X = np.column_stack([np.ones(len(tt)), np.exp(-tt / t1), np.exp(-tt / t2)])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, t1, t2, sol)
    if best is None:
        return None
    sse, t1, t2, sol = best
    a0, a1, a2 = sol
    amp = abs(a1) + abs(a2)
    if amp < 4 * sig:
        return None
    return {"tau1": float(t1), "tau2": float(t2), "a1": float(a1), "a2": float(a2), "y0": float(a0)}


def fit_doe(tt, y, sig):
    """DoE: y = y0 + A*(e(-t/td) - e(-t/tr))，tr<td"""
    best = None
    for tr in np.exp(np.linspace(np.log(0.001), np.log(0.08), 22)):
        for td in np.exp(np.linspace(np.log(max(0.004, tr * 1.5)), np.log(2.0), 26)):
            x = np.exp(-tt / td) - np.exp(-tt / tr)
            X = np.column_stack([np.ones(len(tt)), x])
            sol, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(np.sum((y - X @ sol) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol)
    sse, tr, td, sol = best
    if abs(sol[1]) < 4 * sig:
        return None
    return {"tau_r": float(tr), "tau_d": float(td), "A": float(sol[1]), "y0": float(sol[0])}


def extract_J3(well):
    out = {"tdeact": {}, "trec": {}, "tact": {}, "tinact": {}}
    # ---------- staircase 下跳尾 ----------
    _, Vs = load_protocol("staircaseramp")
    v10 = Vs[:, 0]
    Is = load_current("staircaseramp", well)[:, 0]
    segs = segments(v10)
    sig_s = float(np.std(Is[: int(0.2 / DT_I)]))  # 首段 -80x0.25
    # 找阶梯段（0.5s 档序列），对照 STAIR_SEQ
    step_segs = [(vv, a, b) for vv, a, b in segs if 0.45 <= (b - a) * DT_V <= 0.55]
    # 下跳目标：进入 -40 / -60 的档
    for k in range(1, len(step_segs)):
        vv, a, b = step_segs[k]
        vp = step_segs[k - 1][0]
        if vv in (-40, -60) and vp > vv:
            seg = Is[a // 2: b // 2]
            tt = np.arange(len(seg)) * DT_I
            r = fit_biexp_tail(tt, seg, sig_s)
            if r is not None:
                out["tdeact"].setdefault(str(vv), []).append(r)
    # -80→+40 大补跳（staircase 首个 +40x1.0 段）→ τ_act(+40)/τ_inact(+40) 登记
    for vv, a, b in segs:
        if vv == 40 and 0.9 <= (b - a) * DT_V <= 1.1:
            seg = Is[a // 2: b // 2]
            tt = np.arange(len(seg)) * DT_I
            r = fit_biexp_tail(tt, seg, sig_s)
            if r is not None:
                out["tact"]["40_bigstep"] = r
            break
    # 阶梯段上跳（升相 τobs 登记）：-60→-20, -40→0, -20→20, 0→40, 20→40, 0→20, -20→0
    for k in range(1, len(step_segs)):
        vv, a, b = step_segs[k]
        vp = step_segs[k - 1][0]
        if vv > vp and (vv - vp) == 20:
            seg = Is[a // 2: b // 2]
            tt = np.arange(len(seg)) * DT_I
            r = fit_biexp_tail(tt, seg, sig_s)
            if r is not None:
                out["tact"].setdefault(str(vv), []).append(r)
    # ---------- sinactiv τ_rec DoE ----------
    _, Vn = load_protocol("sinactiv")
    In = load_current("sinactiv", well)
    SIN_V = [-140, -120, -100, -80, -60, -40, -20, 0, 20, 40]
    for sw in [0, 1, 2]:  # -140/-120/-100
        ic, _ = releak_sweep(In[:, sw], Vn[:, sw])
        segs_n = segments(Vn[:, sw])
        test = None
        for vv, a, b in segs_n:
            if vv == SIN_V[sw] and (b - a) * DT_V >= 0.4:
                test = (a, b)
        if test is None:
            continue
        a, b = test[0] // 2, test[1] // 2
        seg = ic[a: a + int(0.15 / DT_I)]  # 前 150ms
        tt = np.arange(len(seg)) * DT_I
        r = fit_doe(tt, seg, sig_s)
        if r is not None:
            out["trec"][str(SIN_V[sw])] = r
    return out, sig_s


def main():
    res = {}
    for w in WELLS:
        r, sig = extract_J3(w)
        res[w] = {"sigma": sig, "J3": r}
        td = {k: [round(x["tau1"], 3) for x in v] for k, v in r["tdeact"].items()}
        tr = {k: round(v["tau_r"], 4) for k, v in r["trec"].items()}
        print(w, "sig=", round(sig, 1), "τdeact(τ1)", td, "τrec", tr, flush=True)
    fp = os.path.join(ROOT, f"lei211_J3_结果_{BATCH}.json" if not SMOKE else f"lei211_J3_冒烟_{BATCH}.json")
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved", fp)


if __name__ == "__main__":
    main()
