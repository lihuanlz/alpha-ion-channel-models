# make_b2_swap_2026-11-09.py
# 生成 B2 换表版：文本手术（每处补丁带断言，断一处即报错不出文件）
import os

SRC = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/alpha-ion-channel-models/"
       "04_code/code_package_v03/08_StageB_ORd_host/2026-09-14_alpha_StageB_B2_alpha_substitution_24drugs.py")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026-11-09_StageB_B2_换表.py")

ARCH = "D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/04_细胞线4/_归档"
ANCHOR_DIR = ARCH + "/α模型/CiPA官方锚"
PKG08 = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
         "alpha-ion-channel-models/04_code/code_package_v03/08_StageB_ORd_host")
B0_PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026-11-09_StageB_B0_路径修复.py")

src = open(SRC, encoding="utf-8").read()
n0 = len(src)

def rep(old, new, cnt=1):
    global src
    assert src.count(old) >= cnt, f"锚点未找到: {old[:60]}..."
    src = src.replace(old, new, cnt)

# 1) 输入路径重定向
rep('ANCHOR = os.path.join(BASE, "CiPA官方锚")', f'ANCHOR = r"{ANCHOR_DIR}"')
rep('KA3_JSON = os.path.join(BASE, "2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json")',
    f'KA3_JSON = r"{PKG08}/2026-09-14_α模型_药物卡3_全药库全景参数表_结果.json"')
rep('B0_SS = os.path.join(BASE, "StageB_B0稳态_CL1000.json")',
    f'B0_SS = r"{PKG08}/StageB_B0稳态_CL1000.json"')
rep('B1_JSON = os.path.join(BASE, "2026-09-14_α模型_StageB_B1_官方静态臂_24药_结果.json")',
    f'B1_JSON = r"{PKG08}/2026-09-14_α模型_StageB_B1_官方静态臂_24药_结果.json"')

# 2) 换表开关 + CP 表（插在 NWORKERS 行后）
rep('NWORKERS = int(os.environ.get("NWORKERS", "1"))',
    '''NWORKERS = int(os.environ.get("NWORKERS", "1"))
TAU_SWAP = os.environ.get("TAU_SWAP", "orig")     # orig / full / act
BUDGET_S = float(os.environ.get("BUDGET_S", "0"))  # 0 = 不限时
_OUTDIR = os.path.dirname(os.path.abspath(__file__))
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
TM_CP_ACT = dict(TM_ORIG); TM_CP_ACT.update({v: ms / 1000 * _FAC215 for v, ms in CP_ACT.items()})


def load_engine_patched():
    """原 α 引擎 + DATA 重定向到归档运行包。"""
    fp = (r"D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
          r"alpha-ion-channel-models/04_code/code_package_v03/07_drug_grammar_CiPA/"
          "2026-09-14_α模型_正式组装_前向引擎.py")
    src_ = open(fp, encoding="utf-8").read()
    for line in src_.splitlines():
        if line.startswith("DATA = "):
            src_ = src_.replace(line, 'DATA = r"' + ARCH_PACK + '"', 1)
            break
    import importlib.util as _ilu
    spec_ = _ilu.spec_from_loader("alpha_engine_patched", loader=None)
    mod_ = _ilu.module_from_spec(spec_)
    mod_.__dict__["__file__"] = fp
    exec(compile(src_, fp, "exec"), mod_.__dict__)
    return mod_''')

# 3) ARCH_PACK 常量（供引擎补丁用）
rep('CL = 1000.0', f'ARCH_PACK = r"{ARCH}/结果/hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包"\nCL = 1000.0')

# 4) 引擎与 B0 加载重定向（各 2 处）
rep('b0 = load_mod("b0_engine", "2026-09-14_α模型_StageB_B0_ORd移植地基复核.py")',
    f'b0 = load_mod("b0_engine", r"{B0_PATCHED}")'.replace("os.path.join(BASE, filename)", "filename"), cnt=2)
# 上面只换了文件名参数，但 load_mod 内部仍 join(BASE,...) —— 直接改 load_mod 支持绝对路径
rep('''def load_mod(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod''',
    '''def load_mod(name, filename):
    fp = filename if os.path.isabs(filename) else os.path.join(BASE, filename)
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod''')
rep('eng = load_mod("alpha_engine", "2026-09-14_α模型_正式组装_前向引擎.py")',
    'eng = load_engine_patched()', cnt=2)

# 5) build_alpha_tabs 注入换表
rep('''    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)

    def scaled(tab, arm=None):''',
    '''    tabs = eng.build_tabs("16713003", amp, hook, inact, hss, Ii, Vi)
    if TAU_SWAP != "orig":
        _table = TM_CP_FULL if TAU_SWAP == "full" else TM_CP_ACT
        tabs["tau_m"] = eng.Tab({float(k): float(v) for k, v in _table.items()})

    def scaled(tab, arm=None):''')

# 6) 输出重定向到本目录 + 换表标签
rep('tag = "_冒烟" if SMOKE else "_结果"',
    'tag = ("_冒烟" if SMOKE else "_结果") + f"_换表{TAU_SWAP}"')
rep('fjson = os.path.join(BASE, f"2026-09-14_α模型_StageB_B2_α替换_24药_{arm_tag}{tag}.json")',
    'fjson = os.path.join(_OUTDIR, f"2026-11-09_StageB_B2_α替换_24药_{arm_tag}{tag}.json")')
rep('fcsv = os.path.join(BASE, f"2026-09-14_α模型_StageB_B2_α替换_24药_{arm_tag}{tag}.csv")',
    'fcsv = os.path.join(_OUTDIR, f"2026-09-14_StageB_B2_α替换_24药_{arm_tag}{tag}.csv")')
rep('fpng = os.path.join(BASE, f"2026-09-14_α模型_StageB_B2_α替换_24药_{arm_tag}{tag}.png")',
    'fpng = os.path.join(_OUTDIR, f"2026-11-09_StageB_B2_α替换_24药_{arm_tag}{tag}.png")')

# 7) 断点续跑：校准/控制/C2C3/药物任务 四层 ckpt + 预算退出
rep('''def main():
    t_start = time.time()''',
    '''CKPT = {}
CKPT_FP = None

def ckpt_save():
    if CKPT_FP:
        json.dump(CKPT, open(CKPT_FP, "w", encoding="utf-8"), ensure_ascii=False, default=float)


def main():
    global CKPT, CKPT_FP
    t_start = time.time()
    _arm_tag0 = f"tauA{TAU_ARM:g}_Q{Q10:g}_{IC50_SRC}"
    CKPT_FP = os.path.join(_OUTDIR, f"2026-11-09_StageB_B2_ckpt_{_arm_tag0}_换表{TAU_SWAP}"
                                     + ("_冒烟" if SMOKE else "") + ".json")
    if os.path.exists(CKPT_FP):
        CKPT.update(json.load(open(CKPT_FP, encoding="utf-8")))
        print(f"[ckpt] 恢复 {CKPT_FP}：已含 {list(CKPT.keys())}", flush=True)''')

# 校准 + 控制段包 ckpt
rep('''    G_alpha, apd_cal, cal_ok = calibrate_G(b0, atabs, P_ctrl, y0)
    print(f"[G_alpha calib] G_alpha={G_alpha:.4g}  calibration last-beat APD90={apd_cal:.2f}ms  "
          f"C1(<=2%) -> {'pass' if cal_ok else 'borderline/fail - registered'}", flush=True)''',
    '''    if "calib" in CKPT:
        G_alpha, apd_cal, cal_ok = CKPT["calib"]
        print(f"[G_alpha calib] ckpt 命中: G_alpha={G_alpha:.4g} apd90={apd_cal:.2f}", flush=True)
    else:
        G_alpha, apd_cal, cal_ok = calibrate_G(b0, atabs, P_ctrl, y0)
        CKPT["calib"] = [G_alpha, apd_cal, cal_ok]; ckpt_save()
        print(f"[G_alpha calib] G_alpha={G_alpha:.4g}  calibration last-beat APD90={apd_cal:.2f}ms  "
              f"C1(<=2%) -> {'pass' if cal_ok else 'borderline/fail - registered'}", flush=True)''')

rep('''    apd_c, qnet_c, _, v_c, _ = run_beats(b0, rhs_c, y0, N_CTRL, label="control")''',
    '''    if "control" in CKPT:
        _cc = CKPT["control"]
        APD_ctrl, qNet_ctrl = _cc["APD"], _cc["qNet"]
        apd_c = np.array([APD_ctrl] * 11)  # steady_flag 仅需末段
        v_c = np.array(_cc["v_c"]) if _cc.get("v_c") else None
        r_c, ok_c = 0.0, True
        print(f"[control] ckpt 命中: APD90={APD_ctrl:.3f} qNet={qNet_ctrl:.4f}", flush=True)
    else:
        apd_c, qnet_c, _, v_c, _ = run_beats(b0, rhs_c, y0, N_CTRL, label="control")
        CKPT["control"] = {"APD": float(apd_c[~np.isnan(apd_c)][-1]),
                           "qNet": float(qnet_c[~np.isnan(qnet_c)][-1]),
                           "v_c": v_c.tolist() if v_c is not None else None}
        ckpt_save()''')

# C2/C3 包 ckpt
rep('''    checks = {}
    for tagc, drugc, expect in [("C2", "dofetilide", "APD90 +40% and qNet decrease"),
                                ("C3", "nifedipine", "APD90 shortening")]:''',
    '''    checks = CKPT.get("checks", {})
    for tagc, drugc, expect in [("C2", "dofetilide", "APD90 +40% and qNet decrease"),
                                ("C3", "nifedipine", "APD90 shortening")]:
        if tagc in checks:
            print(f"  [{tagc}] ckpt 命中", flush=True)
            continue''')

rep('''        checks[tagc] = {"drug": drugc, "C": "100xCmax", "APD90_ratio": r_apd,
                        "qNet_ratio": r_q, "pass": ok, "expect": expect}''',
    '''        checks[tagc] = {"drug": drugc, "C": "100xCmax", "APD90_ratio": r_apd,
                        "qNet_ratio": r_q, "pass": ok, "expect": expect}
        CKPT["checks"] = checks; ckpt_save()''')

# 药物任务循环 → 波次 + 预算 + 逐任务落盘
rep('''    results = []
    if NWORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            for rec in ex.map(one_drug, tasks):
                results.append(rec)
                print(f"  [done] {rec['drug']}/{rec['mode']} APD90={rec['APD90_ms']:.2f}  "
                      f"qNet={rec['qNet_uC_uF']:.4f}", flush=True)
    else:
        for i, tk in enumerate(tasks):
            t0 = time.time()
            rec = one_drug(tk)
            results.append(rec)
            print(f"  [{i + 1}/{len(tasks)}] {rec['drug']:15s}/{rec['mode']:4s} "
                  f"CiPA={rec['CiPA']} C={rec['C_nM']:9.3f}nM  "
                  f"APD90={rec['APD90_ms']:8.2f}  qNet={rec['qNet_uC_uF']:.4f}  "
                  f"steady {rec['steady_range_ms']:.2f}ms{'v' if rec['steady_ok'] else 'x'}  "
                  f"{time.time() - t0:.0f}s", flush=True)''',
    '''    done_map = CKPT.get("tasks", {})
    todo = [tk for tk in tasks if f"{tk[0]}|{tk[1]}" not in done_map]
    print(f"[ckpt] 任务 {len(tasks) - len(todo)}/{len(tasks)} 已完成，本轮待跑 {len(todo)}", flush=True)
    if NWORKERS > 1 and todo:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=NWORKERS) as ex:
            futs = {ex.submit(one_drug, tk): tk for tk in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                done_map[f"{rec['drug']}|{rec['mode']}"] = rec
                CKPT["tasks"] = done_map; ckpt_save()
                el = time.time() - t_start
                print(f"  [done {len(done_map)}/{len(tasks)}] {rec['drug']}/{rec['mode']} "
                      f"APD90={rec['APD90_ms']:.2f} qNet={rec['qNet_uC_uF']:.4f}  t={el:.0f}s", flush=True)
                if BUDGET_S and el > BUDGET_S:
                    print(f"[预算] 超过 {BUDGET_S}s，本轮收手（已落盘 {len(done_map)}/{len(tasks)}）", flush=True)
                    for f2 in futs:
                        f2.cancel()
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise SystemExit(2)
    else:
        for i, tk in enumerate(todo):
            t0 = time.time()
            rec = one_drug(tk)
            done_map[f"{rec['drug']}|{rec['mode']}"] = rec
            CKPT["tasks"] = done_map; ckpt_save()
            print(f"  [{len(done_map)}/{len(tasks)}] {rec['drug']:15s}/{rec['mode']:4s} "
                  f"APD90={rec['APD90_ms']:8.2f}  qNet={rec['qNet_uC_uF']:.4f}  {time.time() - t0:.0f}s", flush=True)
            if BUDGET_S and time.time() - t_start > BUDGET_S:
                print(f"[预算] 收手（{len(done_map)}/{len(tasks)}）", flush=True)
                raise SystemExit(2)
    results = [done_map[f"{tk[0]}|{tk[1]}"] for tk in tasks]''')

open(DST, "w", encoding="utf-8").write(src)
print(f"patched B2 written: {DST}")
print(f"size {n0} -> {len(src)}")
import ast
ast.parse(src)
print("ast.parse OK")
