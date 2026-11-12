# judge_zero184.py — 代码184 零号测量·判官管线体检件（2026-09-11）
# 口径：DeepSeek 零号测量稿——判官独立 expm 通道（judge182.sim_judge）+ 独立拟合器
#      （scipy least_squares，与执行侧 curve_fit 不同例程不同初值）重提 τ_rec@−40；
#       扰动阵测提取散布。判据：判官-执行 τ_rec 相对差 <1e-4 过；>5% → 管线不稳，锚点暂停。
import json, math, os, sys, time
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import judge182
from de174 import load174
from de173 import DT

W184 = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\w184\代码184_tau_rec锚点最小验证卡_16713003_2026-09-11"
os.makedirs(W184, exist_ok=True)
ONSET40 = 85.2404  # −40mV 档 onset（零号测量在案）
TAU_DATA = 0.0134202  # 183 注册件数据侧 τ_rec@−40（16713003）


def fit_tail_ls(cur, t_onset, w0=0.005, w1=5.5, decim=1):
    """判官独立提取：least_squares，初值/例程均与执行侧 curve_fit 不同。"""
    i0 = int((t_onset + w0) / DT)
    i1 = min(int((t_onset + w1) / DT), len(cur))
    y = np.asarray(cur[i0:i1], float)[::decim]
    t = (np.arange(len(y)) * decim + 0.5 * decim) * DT + w0

    def resid(p):
        af, tf, as_, ts, c = p
        return af * np.exp(-t / tf) + as_ * np.exp(-t / ts) + c - y

    c0 = float(np.mean(y[-200:]))
    i00 = float(y[0])
    best = None
    for tf0, ts0 in [(0.01, 0.3), (0.1, 2.0), (0.003, 5.0)]:
        try:
            r = least_squares(resid, [(i00 - c0) * 0.7, tf0, (i00 - c0) * 0.3, ts0, c0],
                              bounds=([-np.inf, 1e-4, -np.inf, 1e-4, -np.inf],
                                      [np.inf, 20.0, np.inf, 100.0, np.inf]),
                              max_nfev=20000)
            if best is None or r.cost < best.cost:
                best = r
        except Exception:
            pass
    if best is None:
        return None
    af, tf, as_, ts, c = best.x
    if tf > ts:
        af, tf, as_, ts = as_, ts, af, tf
    m = resid(best.x) + y
    r2 = 1.0 - float(((m - y) ** 2).sum()) / float(y @ y)
    return {"tau_f": float(tf), "tau_s": float(ts), "r2": r2,
            "a_f": float(af), "a_s": float(as_), "c": float(c)}


def main():
    t0 = time.time()
    x16v = json.load(open(os.path.join(HERE, "counter_181_model62_判官复核件.json"), encoding='utf-8'))
    x16 = np.asarray(x16v["x16"], float)
    data2, keep = load174()
    vd, cd = data2["deactivation"]
    m = judge182.sim_judge(vd, x16)   # 判官独立 expm 通道
    # —— ① 判官重提（基线窗 5ms/5.5s 不抽稀）——
    rm = fit_tail_ls(m, ONSET40)
    rd = fit_tail_ls(np.asarray(cd, float), ONSET40)
    # —— ② 扰动阵（窗起点 × 窗长 × 抽稀）——
    spread = {"model": [], "data": []}
    for w0 in (0.003, 0.005, 0.008, 0.012):
        for w1 in (3.0, 4.0, 5.5):
            for dc in (1, 25):
                a = fit_tail_ls(m, ONSET40, w0, w1, dc)
                b = fit_tail_ls(np.asarray(cd, float), ONSET40, w0, w1, dc)
                if a and a["r2"] >= 0.9:
                    spread["model"].append(a["tau_f"])
                if b and b["r2"] >= 0.9:
                    spread["data"].append(b["tau_f"])
    sm = np.array(spread["model"]); sd = np.array(spread["data"])
    out = {
        "卡": "代码184 零号测量 τ_rec@−40mV", "日期": "2026-09-11",
        "判官通道": "judge182.sim_judge（自写 expm）", "判官提取": "least_squares 多初值",
        "模型侧": rm, "数据侧": rd,
        "τ_rec_模型_ms": rm["tau_f"] * 1e3, "τ_rec_数据_ms": rd["tau_f"] * 1e3,
        "相对差": abs(rm["tau_f"] - rd["tau_f"]) / rd["tau_f"],
        "扰动阵": {"model_tauf_ms_均值": float(sm.mean() * 1e3),
                  "model_tauf_ms_std": float(sm.std(ddof=1) * 1e3) if len(sm) > 1 else None,
                  "model_n": len(sm),
                  "data_tauf_ms_均值": float(sd.mean() * 1e3),
                  "data_tauf_ms_std": float(sd.std(ddof=1) * 1e3) if len(sd) > 1 else None,
                  "data_n": len(sd)},
        "执行侧对照": {"τ_rec_模型_ms": 6.70836, "τ_rec_数据_ms": 13.4202},
        "判词判线": "相对差<5% 不开卡；5–20% 开 λ 扫描；>20% 先查管线（本件）+ 判官-执行<1e-4 过",
    }
    out["判官vs执行_模型侧相对差"] = abs(rm["tau_f"] * 1e3 - 6.70836) / 6.70836
    out["判官vs执行_数据侧相对差"] = abs(rd["tau_f"] * 1e3 - 13.4202) / 13.4202
    json.dump(out, open(os.path.join(W184, "zero_measure_trec_40_181endpoint.json"), "w",
                        encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f"[zero184] 判官侧：模型 τ_rec={rm['tau_f']*1e3:.3f}ms 数据 {rd['tau_f']*1e3:.3f}ms "
          f"相对差={out['相对差']:.3f}", flush=True)
    print(f"[zero184] 判官vs执行：模型侧 {out['判官vs执行_模型侧相对差']:.2e} "
          f"数据侧 {out['判官vs执行_数据侧相对差']:.2e}（判据1e-4）", flush=True)
    print(f"[zero184] 扰动阵：模型 {out['扰动阵']['model_tauf_ms_均值']:.2f}±"
          f"{out['扰动阵']['model_tauf_ms_std']:.2f}ms (n={len(sm)})；数据 "
          f"{out['扰动阵']['data_tauf_ms_均值']:.2f}±{out['扰动阵']['data_tauf_ms_std']:.2f}ms "
          f"(n={len(sd)})", flush=True)
    print(f"[zero184] 用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
