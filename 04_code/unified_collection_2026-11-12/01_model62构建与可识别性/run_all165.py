# -*- coding: utf-8 -*-
"""run_all165.py：代码165 一体驱动（runner v148）：K6a9Ji(model 44) 双卡联合拟合 + 判决。
用法：
  python run_all165.py            # 预检 → 双卡 → 判决
  python run_all165.py fit        # 只跑双卡
  python run_all165.py fit job_165A_16704007_K6a9Ji.json   # 单卡（并行用）
  python run_all165.py judge      # 只判决
双目标：J = SSE_ion（口径同161/163/164：n=1,625,321 剔窗） + λ·SSE_gate（λ=1e4 冻结，gate165.py 单一实现）。
结构硬界（J=1e12，预注册_代码165 §2 冻结）：c∈[0.01,1e6]、KI≥0.5、KB≥0.5、KIC≥0.5、KBC≥0.5
  ——防 161 式 c→0、163 式 c→∞/KI→0 逃逸；x 下标 K6a9Ji=[0]/[22]/[23]/[24]/[25]。
两阶段 NM + early_stop 逻辑照 runner run_job 祖传结构；结果落盘 results_local/ 并镜像主目录。
"""
import json, os, sys, time, hashlib
import numpy as np
import runner_v148 as R
import gate165

MD5 = "d472f53f8d6814bd7f1ba7d2c7055623"   # v1.49b（双录链：f0e2cf72=v1.48 → 067e8c01=v1.49a 链块 → d472f53f=v1.49b 初态补47；2026-09-08 代码167 R1 行，旧路径逐位不变）
HERE = os.path.dirname(os.path.abspath(__file__))
JOBDIR = os.path.join(HERE, "jobs165")
DT = 1e-4
XB = {"K6a9Ji": (0, 22, 23, 24, 25)}   # (c, KI, KB, KIC, KBC) 的 x 下标
XC_LO, XC_HI, XK_LO = -2.0, 6.0, -0.3010299956639812  # c∈[0.01,1e6]、KI/KB/KIC/KBC≥0.5（log10）

def load_keep(protos):
    data = R.load_cell("16704007", protos)
    keep = {k: np.ones(len(v), dtype=bool) for k, (v, c) in data.items()}
    tt = np.arange(len(data["steady_activation"][0])) * DT
    keep["steady_activation"][(tt >= 50.18) & (tt < 55.18)] = False
    return data, keep

def obj165(x, data, keep, lam, decim, model):
    xi = x[:-1]
    xc, xki, xkb, xkic, xkbc = (xi[i] for i in XB[model])
    if not (XC_LO <= xc <= XC_HI) or xki < XK_LO or xkb < XK_LO \
            or xkic < XK_LO or xkbc < XK_LO:
        return 1e12                      # 结构硬界：逃逸方向一律墙
    p = R.unpack(xi, model)
    si = R.sse_of(p, data, model, keep=keep, ek=x[-1], decim=decim)
    if not np.isfinite(si):
        return 1e12
    return si + lam * gate165.gate_pen(R, p)

def fit165(card_path):
    card = json.load(open(card_path, encoding="utf-8"))
    model = card["model"]
    assert model == "K6a9Ji" and len(card["start_params"]) == 26
    lam = float(card.get("lambda", 1e4))
    assert lam == 1e4, "λ 冻结 1e4，卡面不符拒跑"
    data, keep = load_keep(card["protos"])
    decim = card.get("decimate", None)
    x0 = np.r_[R.x_from_params(np.array(card["start_params"], float), model),
               float(card["EK0"])]
    xc, xki, xkb, xkic, xkbc = (x0[i] for i in XB[model])
    assert XC_LO <= xc <= XC_HI and xki >= XK_LO and xkb >= XK_LO \
        and xkic >= XK_LO and xkbc >= XK_LO, "起跑点出结构界，拒跑"
    maxiter = int(card.get("maxiter", 60000))
    early_stop = bool(card.get("early_stop", False))
    t0 = time.time()
    state = {"it": 0, "fchk": [None, 0]}
    best = {"f": None}
    state["xk_best"] = x0.copy()

    class _EarlyStop(Exception):
        pass

    def cb(xk):
        state["it"] += 1
        if state["it"] % 250 == 0:
            print(f"  ... iter {state['it']} ({time.time()-t0:.0f}s)", flush=True)
        if early_stop and state["it"] % 1000 == 0:
            f_now = best["f"]
            if state["fchk"][0] is not None and f_now is not None:
                gain = (state["fchk"][0] - f_now) / max(abs(state["fchk"][0]), 1e-12)
                state["fchk"][1] = state["fchk"][1] + 1 if gain < 1e-3 else 0
                if state["fchk"][1] >= 2:
                    raise _EarlyStop
            if f_now is not None:
                state["fchk"][0] = f_now

    def wrap(f):
        def g(x):
            v = f(x)
            if best["f"] is None or v < best["f"]:
                best["f"] = v
                state["xk_best"] = np.array(x).copy()
            return v
        return g

    from scipy.optimize import minimize
    nfev0 = [0]
    def counted(f):
        def g(x):
            nfev0[0] += 1
            return f(x)
        return g

    f1 = counted(wrap(lambda x: obj165(x, data, keep, lam, decim, model)))
    f2 = counted(wrap(lambda x: obj165(x, data, keep, lam, None, model)))
    budget1 = int(maxiter * 0.6)
    converged, nit_tot, msg, stopped = False, 0, "", False
    try:
        r = minimize(f1, x0, method="Nelder-Mead", callback=cb,
                     options={"maxiter": budget1, "maxfev": budget1 + 200,
                              "xatol": 1e-6, "fatol": 1e-3})
        x_stage, converged, nit_tot, msg = r.x, bool(r.success), int(r.nit), str(r.message)
    except _EarlyStop:
        x_stage, msg, stopped = state["xk_best"], "early_stop(stage1)", True
    state["it"] = 0
    best["f"] = None
    try:
        r2 = minimize(f2, x_stage, method="Nelder-Mead", callback=cb,
                      options={"maxiter": maxiter - budget1, "maxfev": maxiter - budget1 + 200,
                               "xatol": 1e-6, "fatol": 1e-3})
        x_stage, converged = r2.x, bool(r2.success)
        nit_tot += int(r2.nit); msg = msg + "|stage2:" + str(r2.message)
    except _EarlyStop:
        x_stage, msg, stopped = state["xk_best"], msg + "|early_stop(stage2)", True

    p_fit = R.unpack(x_stage[:-1], model)
    ek_fit = float(x_stage[-1])
    sse_ion = R.sse_of(p_fit, data, model, keep=keep, ek=ek_fit)
    gm = gate165.gate_metrics(R, p_fit)
    gp = 0.0 if gm is None else gate165.gate_pen(R, p_fit)
    n = sum(int(keep[k].sum()) for k in keep)
    res = {"cell": card["cell"], "model": model, "start": int(card.get("start", 0)),
           "tag": card.get("tag"), "note": card.get("note", ""),
           "protos": card["protos"], "n_eff": n,
           "sse": float(sse_ion), "sse_gate": float(gp), "J": float(sse_ion + lam * gp),
           "lambda": lam,
           "gate_metrics": None if gm is None else [float(v) for v in gm],
           "params": p_fit.tolist(), "nfev": int(nfev0[0]),
           "converged": bool(converged), "nit": int(nit_tot),
           "fit_EK": True, "EK_used": ek_fit,
           "decimate": decim, "early_stopped": stopped, "nm_message": str(msg),
           "seconds": time.time() - t0, "runner": "local-v148"}
    os.makedirs(os.path.join(HERE, "results_local"), exist_ok=True)
    out = os.path.join(HERE, "results_local",
                       f"result_{card['cell']}_{model}_{card.get('tag','s'+str(card.get('start',0)))}.json")
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[done] 16704007 {model} {card.get('tag')}: sse_ion={sse_ion:.1f} gate={gp:.2f} "
          f"({res['seconds']:.0f}s) -> {out}", flush=True)
    return res

def preflight(jobs):
    m = hashlib.md5(open(os.path.abspath(R.__file__), 'rb').read()).hexdigest()
    assert m == MD5, f"runner_v148 md5 不符：{m}≠{MD5}（拒跑，版本守卫尽责）"
    assert R.MODEL_ID.get("K6a9Ji") == 44 and R.NPARAM.get("K6a9Ji") == 26, "注册表缺 K6a9Ji"
    cal = json.load(open(os.path.join(HERE, "calib165.json"), encoding="utf-8"))
    assert abs(cal["A0"]-23.088575018644143) < 1e-6 and abs(cal["ZA"]-0.03486909592566613) < 1e-10, \
        "calib165 常量漂移"
    protos = ["steady_activation", "deactivation", "sine_wave", "ap"]
    data, keep = load_keep(protos)
    for j in jobs:
        card = json.load(open(os.path.join(JOBDIR, j), encoding="utf-8"))
        model = card["model"]
        assert model == "K6a9Ji" and len(card["start_params"]) == 26, "卡字段不符"
        p = np.array(card["start_params"], float)
        sse_ion = R.sse_of(p, data, model, keep=keep, ek=card["EK0"])
        gp = gate165.gate_pen(R, p)
        J0 = sse_ion + float(card.get("lambda", 1e4)) * gp
        rel = abs(J0/card["sse0_anchor"] - 1)
        # 容差 1e-4（补记33 修订沿用）：守卫防的是拷错文件/错数据
        # （相对差 O(1e-2..1) 量级）；Windows↔Linux 跨平台浮点漂移实测 2.6e-9（λ·gate 项
        # 含 curve_fit，平台敏感），1e-4 留足千倍余量仍抓得住真错。
        assert rel < 1e-4, \
            f"{j} 起跑锚复算不符：{J0:.5f} ≠ {card['sse0_anchor']:.5f}（相对差 {rel:.2e}）"
        note = f"（相对差 {rel:.2e}，跨平台浮点双录）" if rel > 1e-9 else ""
        print(f"  预检 {j}: 起跑锚 ✓ ({J0:.5f}){note}", flush=True)
    print("预检全过。")

def run_fits(jobs):
    import shutil
    for j in jobs:
        print(f"\n===== 跑 {j} =====", flush=True)
        t0 = time.time()
        res = fit165(os.path.join(JOBDIR, j))
        stem = f"result_16704007_{res['model']}_{res['tag']}.json"
        src = os.path.join(HERE, "results_local", stem)
        shutil.copyfile(src, os.path.join(HERE, stem))
        print(f"===== {j} 完：sse_ion={res['sse']:.4f} 用时 {(time.time()-t0)/3600:.2f}h =====",
              flush=True)

def main():
    args = sys.argv[1:]
    mode = args[0] if args else "all"
    jobs = sorted(os.listdir(JOBDIR))
    if mode == "fit" and len(args) > 1:
        jobs = [args[1]]
    if mode in ("all", "fit"):
        preflight(jobs)
        run_fits(jobs)
    if mode in ("all", "judge"):
        import judge165

if __name__ == "__main__":
    main()
