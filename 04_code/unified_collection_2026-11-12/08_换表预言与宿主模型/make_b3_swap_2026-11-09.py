# -*- coding: utf-8 -*-
"""
2026-11-09 · 手术生成 B3 换表版
================================
以 code_package_v03/08_StageB_ORd_host/2026-09-16_alpha_StageB_B3_dual_alpha_host_recheck_24drugs.py
为底（逐字节读取），按锚点替换/插入，输出 2026-11-09_StageB_B3_换表.py。

换表范围（单变量）：仅 hERG α 门 τ_m 锚表换 Channelpedia QC 中位（21.5°C 基准，秒）；
IKs 侧 Fedida 25 点表不动（KCNQ1 不在本次 Channelpedia Kv11.1 提取范围，存档已登记此限制）。
m_ss/h_ss/τ_h/E_rev/Q10/G_Kr/G_Ks 冻结值全部不动（v7 item 1.1 不重新校准）。

R 线对照口径：orig 配置对照封卷 B2；full/act 配置对照同配置换表 B2（保持单变量原则）。

每个替换断言旧串唯一出现，否则报错中止。
"""
import os

SRC = (r"D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
       r"alpha-ion-channel-models/04_code/code_package_v03/08_StageB_ORd_host/"
       "2026-09-16_alpha_StageB_B3_dual_alpha_host_recheck_24drugs.py")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "2026-11-09_StageB_B3_换表.py")

src = open(SRC, encoding="utf-8").read()
n0 = len(src)


def rep(old, new, count=1):
    """锚点替换，断言出现次数。"""
    global src
    n = src.count(old)
    assert n == count, f"锚点出现 {n} 次（预期 {count}）: {old[:60]!r}..."
    src = src.replace(old, new)


# ---------------------------------------------------------------- A1 路径块整体替换
OLD_PATHS = '''ANCHOR = os.path.join(BASE, "CiPA官方锚")
CIPA_CSV = os.path.join(ANCHOR, "newCiPA.csv")
KA3_JSON = os.path.join(BASE, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")
B0_SS = os.path.join(BASE, "StageB_B0稳态_CL1000.json")
B1_JSON = os.path.join(BASE, "2026-09-14_α模型_StageB_B1_官方静态臂_24药_结果.json")
B2_JSON = os.path.join(BASE, "2026-09-14_α模型_StageB_B2_α替换_24药_tauA1_Q2.5_ka3_结果.json")
ASM_JSON = os.path.join(BASE, "2026-09-15_ORd三通道α化组装_结果.json")
FEDIDA_CSV = os.path.join(BASE, "2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv")
B3_SS = os.path.join(BASE, "StageB_B3稳态_CL1000.json")'''
NEW_PATHS = '''ANCHOR = r"D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/04_细胞线4/_归档/α模型/CiPA官方锚"
CIPA_CSV = os.path.join(ANCHOR, "newCiPA.csv")
_PKG = (r"D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
        r"alpha-ion-channel-models/04_code/code_package_v03/08_StageB_ORd_host")
KA3_JSON = os.path.join(_PKG, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")
B0_SS = os.path.join(_PKG, "StageB_B0稳态_CL1000.json")
B1_JSON = os.path.join(_PKG, "2026-09-14_α模型_StageB_B1_官方静态臂_24药_结果.json")
ASM_JSON = os.path.join(_PKG, "2026-09-15_ORd三通道α化组装_结果.json")
FEDIDA_CSV = os.path.join(_PKG, "2026-09-15_Kα2_Fedida2024_Fig3B_wtEQ_数字化.csv")
_OUTDIR = os.path.dirname(os.path.abspath(__file__))
TAU_SWAP = os.environ.get("TAU_SWAP", "orig")     # orig / full / act
BUDGET_S = float(os.environ.get("BUDGET_S", "0"))  # 0 = 不限时
# R 线对照口径：orig 对照封卷 B2；full/act 对照同配置换表 B2（单变量原则）
B2_JSON = (os.path.join(_PKG, "2026-09-14_α模型_StageB_B2_α替换_24药_tauA1_Q2.5_ka3_结果.json")
           if TAU_SWAP == "orig" else
           os.path.join(_OUTDIR, f"2026-11-09_StageB_B2_α替换_24药_tauA1_Q2.5_ka3_结果_换表{TAU_SWAP}.json"))
B3_SS = os.path.join(_OUTDIR, f"StageB_B3稳态_CL1000_换表{TAU_SWAP}.json")
B0_FIXED = os.path.join(_OUTDIR, "2026-11-09_StageB_B0_路径修复.py")
ARCH_PACK = (r"D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/"
             r"2026-10-16_清理/04_细胞线4/_归档/结果/hERG本地运行包_历史版本/"
             r"hERG本地运行包_v1.1/本地运行包")
# Channelpedia Kv11.1 τ_m QC 中位（35°C 测量 ×3.445 折回 21.5°C，秒）；与原 α 表同位点替换
_FAC215 = 2.5 ** 1.35
TM_ORIG = {-130: 0.03033214284254048, -120: 0.03033214284254048,
           -110: 0.049199997974127406, -100: 0.08577126043784405,
           -90: 0.072, -80: 0.24, -70: 0.8127, -60: 1.20549, -50: 10.14073,
           -40: 24.0539, -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
           20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}
CP_DEACT = {-70: 37.7, -60: 43.4, -50: 56.7, -40: 80.0, -30: 85.1}
CP_ACT = {-20: 280.7, -10: 323.0, 0: 268.0, 10: 150.6, 20: 90.2,
          30: 64.1, 40: 57.5, 50: 71.5, 60: 105.6}
TM_CP_FULL = dict(TM_ORIG); TM_CP_FULL.update({v: ms / 1000 * _FAC215 for v, ms in CP_DEACT.items()})
TM_CP_FULL.update({v: ms / 1000 * _FAC215 for v, ms in CP_ACT.items()})
TM_CP_ACT = dict(TM_ORIG); TM_CP_ACT.update({v: ms / 1000 * _FAC215 for v, ms in CP_ACT.items()})'''
rep(OLD_PATHS, NEW_PATHS)

# ---------------------------------------------------------------- A2 build_tabs_kr 换表钩子
OLD = '''    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)

    def scaled(tab):'''
NEW = '''    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)
    if TAU_SWAP != "orig":
        _table = TM_CP_FULL if TAU_SWAP == "full" else TM_CP_ACT
        tabs["tau_m"] = eng.Tab({float(k): float(v) for k, v in _table.items()})

    def scaled(tab):'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A3 load_engine_patched + ckpt 设施（插在 load_cipa 前）
OLD = '''def load_cipa(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["drug"].strip()] = row
    return rows'''
NEW = '''def load_cipa(path):
    rows = {}
    with open(path, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["drug"].strip()] = row
    return rows


def load_engine_patched():
    """原 α 引擎 + DATA 重定向到归档运行包（与 B2 换表版同一实现）。"""
    fp = (r"D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
          r"alpha-ion-channel-models/04_code/code_package_v03/07_drug_grammar_CiPA/"
          "2026-09-14_α模型_正式组装_前向引擎.py")
    src_ = open(fp, encoding="utf-8").read()
    for line in src_.splitlines():
        if line.startswith("DATA = "):
            src_ = src_.replace(line, 'DATA = r"' + ARCH_PACK + '"', 1)
            break
    spec_ = importlib.util.spec_from_loader("alpha_engine_patched", loader=None)
    mod_ = importlib.util.module_from_spec(spec_)
    mod_.__dict__["__file__"] = fp
    exec(compile(src_, fp, "exec"), mod_.__dict__)
    return mod_


CKPT = {}
CKPT_FP = None


def ckpt_save():
    if CKPT_FP:
        json.dump(CKPT, open(CKPT_FP, "w", encoding="utf-8"), ensure_ascii=False, default=float)'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A4 b0 加载换路径修复版（one_drug + main 共两处）
rep('    b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")',
    '    b0 = load_mod("b0_engine", B0_FIXED)', count=2)

# ---------------------------------------------------------------- A5 引擎加载换 patched（one_drug + main 共两处）
rep('    eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")',
    '    eng = load_engine_patched()', count=2)

# ---------------------------------------------------------------- A6 one_drug 的 run_beats 改分段缓存
OLD = '''    G = {"Kr": G_Kr, "Ks": G_Ks * mult_ks}
    rhs = build_b3(b0, P_d, tabs, G, kon, koff, C)
    apd, apa, qnet, _ = run_beats(b0, rhs, y0, N_DRUG, label=f"{drug}")
    nd = int(np.sum(~np.isnan(apd)))'''
NEW = '''    G = {"Kr": G_Kr, "Ks": G_Ks * mult_ks}
    rhs = build_b3(b0, P_d, tabs, G, kon, koff, C)
    import hashlib as _hl
    _cdir = os.path.join(_OUTDIR, f"_b3task_cache_换表{TAU_SWAP}" + ("_冒烟" if SMOKE else ""))
    os.makedirs(_cdir, exist_ok=True)
    _cfp = os.path.join(_cdir, _hl.md5(f"{drug}|{C_scale}".encode()).hexdigest() + ".json")
    try:
        _st = json.load(open(_cfp)) if os.path.exists(_cfp) else {}
    except Exception:
        _st = {}   # 被杀毒点写坏的缓存，重来
    _y_cur = [float(x) for x in _st.get("y", y0)]
    _apd_l = [float(x) for x in _st.get("apd", [])]
    _qn_l = [float(x) for x in _st.get("qnet", [])]
    while len(_apd_l) < N_DRUG:
        _nb = min(100, N_DRUG - len(_apd_l))
        _a, _ap, _q, _o = run_beats(b0, rhs, _y_cur, _nb, label=f"{drug}@{len(_apd_l)}")
        _apd_l += [float(x) for x in _a]
        _qn_l += [float(x) for x in _q]
        _y_cur = [float(x) for x in _o["y_end"]]
        json.dump({"y": _y_cur, "apd": _apd_l, "qnet": _qn_l}, open(_cfp + ".tmp", "w"))
        os.replace(_cfp + ".tmp", _cfp)
    apd = np.array(_apd_l)
    qnet = np.array(_qn_l)
    nd = int(np.sum(~np.isnan(apd)))'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A7 main 头部：CKPT 初始化 + tag 带换表后缀
OLD = '''def main():
    t_start = time.time()
    tag = "_冒烟" if SMOKE else "_结果"'''
NEW = '''def main():
    global CKPT, CKPT_FP
    t_start = time.time()
    tag = ("_冒烟" if SMOKE else "_结果") + f"_换表{TAU_SWAP}"
    CKPT_FP = os.path.join(_OUTDIR, f"2026-11-09_StageB_B3_ckpt_换表{TAU_SWAP}"
                                    + ("_冒烟" if SMOKE else "") + ".json")
    if os.path.exists(CKPT_FP):
        CKPT.update(json.load(open(CKPT_FP, encoding="utf-8")))
        print(f"[ckpt] 恢复 {CKPT_FP}：已含 {list(CKPT.keys())}", flush=True)
    PHASES = set(os.environ.get("PHASES", "ctrl,c2c3,drugs").split(","))'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A8 control 段改分段缓存 + ckpt 恢复
OLD = '''    # ---------- control: steady-state rebuild + C1 + save B3 steady state ----------
    print(f"\\n[control] steady-state rebuild {N_CTRL} beats (from B0 steady state + alpha-state init) ...", flush=True)
    G0 = {"Kr": G_Kr, "Ks": G_Ks}
    rhs_c = build_b3(b0, P_ctrl, tabs, G0, 0.0, 0.0, 0.0)
    apd_c, apa_c, qn_c, out_c = run_beats(b0, rhs_c, y0, N_CTRL, label="control",
                                          keep_currents=True)
    APD_ctrl, qNet_ctrl = last_valid(apd_c), last_valid(qn_c)'''
NEW = '''    # ---------- control: steady-state rebuild + C1 + save B3 steady state ----------
    print(f"\\n[control] steady-state rebuild {N_CTRL} beats (from B0 steady state + alpha-state init) ...", flush=True)
    G0 = {"Kr": G_Kr, "Ks": G_Ks}
    rhs_c = build_b3(b0, P_ctrl, tabs, G0, 0.0, 0.0, 0.0)
    out_c = None
    if "control" in CKPT:
        _cc = CKPT["control"]
        APD_ctrl, qNet_ctrl = _cc["APD"], _cc["qNet"]
        apd_c = np.array(CKPT["control_apd"])
        qn_c = np.array(CKPT["control_qnet"])
        print(f"[control] ckpt 命中: APD90={APD_ctrl:.3f} qNet={qNet_ctrl:.4f}", flush=True)
    else:
        _seg = 100
        _nseg = (N_CTRL + _seg - 1) // _seg
        _sdone = CKPT.get("control_segs_done", 0)
        _apd_all = list(CKPT.get("control_apd", []))
        _qn_all = list(CKPT.get("control_qnet", []))
        _y = [float(x) for x in CKPT["control_y"]] if "control_y" in CKPT else list(y0)
        for _s in range(_sdone, _nseg):
            _nb = min(_seg, N_CTRL - _s * _seg)
            _last = (_s == _nseg - 1)
            _a, _ap, _q, _o = run_beats(b0, rhs_c, _y, _nb,
                                        label=f"control seg{_s + 1}/{_nseg}",
                                        keep_currents=_last)
            _apd_all += [float(x) for x in _a]
            _qn_all += [float(x) for x in _q]
            _y = [float(x) for x in _o["y_end"]]
            if _last:
                out_c = _o
            CKPT["control_segs_done"] = _s + 1
            CKPT["control_apd"] = _apd_all
            CKPT["control_qnet"] = _qn_all
            CKPT["control_y"] = _y
            ckpt_save()
        apd_c = np.array(_apd_all)
        qn_c = np.array(_qn_all)
        CKPT["control"] = {"APD": float(apd_c[~np.isnan(apd_c)][-1]),
                           "qNet": float(_qn_all[-1]), "y_end": _y}
        ckpt_save()
        APD_ctrl, qNet_ctrl = CKPT["control"]["APD"], CKPT["control"]["qNet"]'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A9 ss 保存 + integ 段（ckpt 兼容）
OLD = '''    ss_file = B3_SS if not SMOKE else B3_SS.replace(".json", "_冒烟.json")
    with open(ss_file, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"from": "B0 steady state + alpha-state init", "beats": N_CTRL,
                            "G_Kr": G_Kr, "G_Ks": G_Ks, "APD90": APD_ctrl,
                            "steady_range_ms": r_c, "C1_rel": c1, "smoke": SMOKE},
                   "states56": [float(x) for x in out_c["y_end"]]}, fh, ensure_ascii=False)
    print(f"  B3 steady state saved: {ss_file}", flush=True)

    # ---------- last-beat six-current integrals (piggyback: refill assembly verdict card second-review B1/B4) ----------
    print("\\n[integral registry] official reference arm 2 beats (from B0 steady state) vs B3 main-arm last beat ...", flush=True)
    P_ref = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
    rhs_r0 = b0.build_model(P_ref)
    rhs_r = lambda t, y: rhs_r0(t, y)
    apd_r, apa_r, qn_r, out_r = run_beats(b0, rhs_r, y_ss, 2, label="ref-arm", keep_currents=True)
    integ = {}
    for i, nm in enumerate(("INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1")):
        q_ref = float(np.trapezoid(out_r["cur"][i], out_r["t"]))
        q_b3 = float(np.trapezoid(out_c["cur"][i], out_c["t"]))
        integ[nm] = {"q_ref_pAms": q_ref, "q_B3_pAms": q_b3,
                     "ratio": (q_b3 / q_ref if abs(q_ref) > 1e-12 else None)}
    print(f"  IKr charge: ref {integ['IKr']['q_ref_pAms']:.2f} vs B3 {integ['IKr']['q_B3_pAms']:.2f}"
          f" pA/pF*ms (ratio {integ['IKr']['ratio']:.3f})", flush=True)
    print(f"  IKs charge: ref {integ['IKs']['q_ref_pAms']:.2f} vs B3 {integ['IKs']['q_B3_pAms']:.2f}"
          f" pA/pF*ms (ratio {integ['IKs']['ratio']:.3f})", flush=True)'''
NEW = '''    ss_file = B3_SS if not SMOKE else B3_SS.replace(".json", "_冒烟.json")
    if not os.path.exists(ss_file):
        with open(ss_file, "w", encoding="utf-8") as fh:
            json.dump({"meta": {"from": "B0 steady state + alpha-state init", "beats": N_CTRL,
                                "G_Kr": G_Kr, "G_Ks": G_Ks, "APD90": APD_ctrl,
                                "steady_range_ms": r_c, "C1_rel": c1, "smoke": SMOKE,
                                "tau_swap": TAU_SWAP},
                       "states56": [float(x) for x in CKPT["control"]["y_end"]]}, fh,
                      ensure_ascii=False)
        print(f"  B3 steady state saved: {ss_file}", flush=True)
    else:
        print(f"  B3 steady state 已存在（沿用）: {ss_file}", flush=True)

    # ---------- last-beat six-current integrals (piggyback: refill assembly verdict card second-review B1/B4) ----------
    if "integ" in CKPT:
        integ = CKPT["integ"]
        print("\\n[integral registry] ckpt 命中（沿用首轮实跑值）", flush=True)
    else:
        print("\\n[integral registry] official reference arm 2 beats (from B0 steady state) vs B3 main-arm last beat ...", flush=True)
        P_ref = b0.load_named_values(b0.PARS_FILE, b0.PARS_NAMES)
        rhs_r0 = b0.build_model(P_ref)
        rhs_r = lambda t, y: rhs_r0(t, y)
        apd_r, apa_r, qn_r, out_r = run_beats(b0, rhs_r, y_ss, 2, label="ref-arm", keep_currents=True)
        if out_c is None:
            # control 为 ckpt 命中、本轮无电流记录：用稳态重放 1 拍补记
            _yc = [float(x) for x in CKPT["control"]["y_end"]]
            _a1, _p1, _q1, out_c = run_beats(b0, rhs_c, _yc, 1, label="control-补记", keep_currents=True)
        integ = {}
        for i, nm in enumerate(("INa", "INaL", "Ito", "ICaL", "IKr", "IKs", "IK1")):
            q_ref = float(np.trapezoid(out_r["cur"][i], out_r["t"]))
            q_b3 = float(np.trapezoid(out_c["cur"][i], out_c["t"]))
            integ[nm] = {"q_ref_pAms": q_ref, "q_B3_pAms": q_b3,
                         "ratio": (q_b3 / q_ref if abs(q_ref) > 1e-12 else None)}
        CKPT["integ"] = integ
        ckpt_save()
        print(f"  IKr charge: ref {integ['IKr']['q_ref_pAms']:.2f} vs B3 {integ['IKr']['q_B3_pAms']:.2f}"
              f" pA/pF*ms (ratio {integ['IKr']['ratio']:.3f})", flush=True)
        print(f"  IKs charge: ref {integ['IKs']['q_ref_pAms']:.2f} vs B3 {integ['IKs']['q_B3_pAms']:.2f}"
              f" pA/pF*ms (ratio {integ['IKs']['ratio']:.3f})", flush=True)'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A10 C2/C3 缓存 + PHASES 门控
OLD = '''    # ---------- C2 / C3 self-checks ----------
    rows = load_cipa(CIPA_CSV)
    checks = {}
    for tagc, drugc in [("C2", "dofetilide"), ("C3", "nifedipine")]:
        rec = one_drug((drugc, 100.0, G_Kr, G_Ks, ss_file))
        r_apd = rec["APD90_ms"] / APD_ctrl
        r_q = rec["qNet_uC_uF"] / qNet_ctrl
        ok = bool(r_apd > 1.40 and r_q < 1.0) if tagc == "C2" else bool(r_apd < 1.0)
        checks[tagc] = {"drug": drugc, "C": "100xCmax", "APD90_ratio": r_apd,
                        "qNet_ratio": r_q, "pass": ok}
        print(f"  [{tagc}] {drugc} 100×Cmax: APD90 {r_apd:.3f}×  qNet {r_q:.3f}×  "
              f"-> {'pass' if ok else 'fail'}", flush=True)'''
NEW = '''    # ---------- C2 / C3 self-checks ----------
    rows = load_cipa(CIPA_CSV)
    checks = CKPT.get("checks", {})
    if "c2c3" in PHASES:
        for tagc, drugc in [("C2", "dofetilide"), ("C3", "nifedipine")]:
            if tagc in checks:
                print(f"  [{tagc}] ckpt 命中", flush=True)
                continue
            rec = one_drug((drugc, 100.0, G_Kr, G_Ks, ss_file))
            r_apd = rec["APD90_ms"] / APD_ctrl
            r_q = rec["qNet_uC_uF"] / qNet_ctrl
            ok = bool(r_apd > 1.40 and r_q < 1.0) if tagc == "C2" else bool(r_apd < 1.0)
            checks[tagc] = {"drug": drugc, "C": "100xCmax", "APD90_ratio": r_apd,
                            "qNet_ratio": r_q, "pass": ok}
            CKPT["checks"] = checks
            ckpt_save()
            print(f"  [{tagc}] {drugc} 100×Cmax: APD90 {r_apd:.3f}×  qNet {r_q:.3f}×  "
                  f"-> {'pass' if ok else 'fail'}", flush=True)'''
rep(OLD, NEW)

# ---------------------------------------------------------------- A11 drugs 段 done_map + BUDGET_S
OLD = '''    # ---------- 24 drugs ----------
    drug_list = SMOKE_DRUGS if SMOKE else DRUGS_24
    tasks = [(d, 1.0, G_Kr, G_Ks, ss_file) for d in drug_list]
    print(f"\\n[{len(tasks)} drugs] {N_DRUG} beats each, NWORKERS={NWORKERS}", flush=True)
    results = []
    if NWORKERS > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            for rec in ex.map(one_drug, tasks):
                results.append(rec)
                print(f"  [done] {rec['drug']} APD90={rec['APD90_ms']:.2f}"
                      f"  qNet={rec['qNet_uC_uF']:.4f}", flush=True)
    else:
        for i, tk in enumerate(tasks):
            t0 = time.time()
            rec = one_drug(tk)
            results.append(rec)
            print(f"  [{i + 1}/{len(tasks)}] {rec['drug']:15s} CiPA={rec['CiPA']}"
                  f"  APD90={rec['APD90_ms']:8.2f}  qNet={rec['qNet_uC_uF']:.4f}"
                  f"  steady {rec['steady_range_ms']:.2f}{'v' if rec['steady_ok'] else 'x'}"
                  f"  {time.time() - t0:.0f}s", flush=True)'''
NEW = '''    # ---------- 24 drugs ----------
    if "drugs" not in PHASES:
        print("[阶段] 跳过药物段", flush=True)
        raise SystemExit(2)
    drug_list = SMOKE_DRUGS if SMOKE else DRUGS_24
    tasks = [(d, 1.0, G_Kr, G_Ks, ss_file) for d in drug_list]
    print(f"\\n[{len(tasks)} drugs] {N_DRUG} beats each, NWORKERS={NWORKERS}", flush=True)
    done_map = CKPT.get("tasks", {})
    todo = [tk for tk in tasks if tk[0] not in done_map]
    print(f"[ckpt] 任务 {len(tasks) - len(todo)}/{len(tasks)} 已完成，本轮待跑 {len(todo)}", flush=True)
    if NWORKERS > 1 and todo:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            futs = {ex.submit(one_drug, tk): tk for tk in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                done_map[rec["drug"]] = rec
                CKPT["tasks"] = done_map
                ckpt_save()
                el = time.time() - t_start
                print(f"  [done {len(done_map)}/{len(tasks)}] {rec['drug']} APD90={rec['APD90_ms']:.2f}"
                      f"  qNet={rec['qNet_uC_uF']:.4f}  t={el:.0f}s", flush=True)
                if BUDGET_S and el > BUDGET_S:
                    print(f"[预算] 超过 {BUDGET_S}s，本轮收手（已落盘 {len(done_map)}/{len(tasks)}）", flush=True)
                    for f2 in futs:
                        f2.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    os._exit(2)   # 硬退出：不等在跑任务，防止挂过 Bash 时限
    else:
        for i, tk in enumerate(todo):
            t0 = time.time()
            rec = one_drug(tk)
            done_map[rec["drug"]] = rec
            CKPT["tasks"] = done_map
            ckpt_save()
            print(f"  [{len(done_map)}/{len(tasks)}] {rec['drug']:15s} CiPA={rec['CiPA']}"
                  f"  APD90={rec['APD90_ms']:8.2f}  qNet={rec['qNet_uC_uF']:.4f}"
                  f"  steady {rec['steady_range_ms']:.2f}{'v' if rec['steady_ok'] else 'x'}"
                  f"  {time.time() - t0:.0f}s", flush=True)
            if BUDGET_S and time.time() - t_start > BUDGET_S:
                print(f"[预算] 收手（{len(done_map)}/{len(tasks)}）", flush=True)
                raise SystemExit(2)
    results = [done_map[tk[0]] for tk in tasks]'''
rep(OLD, NEW)

# ---------------------------------------------------------------- 写盘
open(DST, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print(f"手术完成: {n0} -> {len(src)} chars, ast.parse 通过")
print(f"输出: {DST}")
