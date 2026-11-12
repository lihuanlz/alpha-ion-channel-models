# cp_swap_lib_2026-11-09.py
# ============================================================================
# Channelpedia τ_m 换表共享库（供药物卡4 / StageB B2 B3 换表版引用）
#
# 内容：
#   1. load_engine()      —— 加载原 α 引擎，运行时把 DATA 重定向到归档运行包
#   2. TM_ORIG / TM_CP_FULL / TM_CP_ACT —— τ_m 锚表（秒，21.5°C 等效基准）
#   3. swap_tau_m()       —— 在引擎 build_tabs 产物上替换 τ_m 表
#
# 换表映射（QC: flag=active ∧ 剖面CI成功 ∧ r2>=0.9；35°C 中位 ×3.445 折回 21.5°C）：
#   CP_full: -70..-30 ← τ_deact 35°C；-20..+60 ← τ_act 35°C
#   CP_act : 仅 -20..+60 ← τ_act 35°C
# 其余锚点（m_ss/h_ss/τ_h、E_rev、G）一律不动 —— 单变量替换。
# ============================================================================
import os, json, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENG_07 = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/_github/"
          "alpha-ion-channel-models/04_code/code_package_v03/07_drug_grammar_CiPA/"
          "2026-09-14_α模型_正式组装_前向引擎.py")
ARCH_PACK = ("D:/Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911/09_历史归档/2026-10-16_清理/"
             "04_细胞线4/_归档/结果/hERG本地运行包_历史版本/hERG本地运行包_v1.1/本地运行包")

TM_ORIG = {-130: 0.03033214284254048, -120: 0.03033214284254048,
           -110: 0.049199997974127406, -100: 0.08577126043784405,
           -90: 0.072, -80: 0.24,
           -70: 0.8127, -60: 1.20549, -50: 10.14073, -40: 24.0539,
           -30: 9.8, -20: 4.0, -10: 2.8, 0: 2.0, 10: 1.2,
           20: 0.71, 30: 0.45, 40: 0.29, 50: 0.29, 60: 0.30}

_FAC = 2.5 ** 1.35
CP_DEACT = {-70: 37.7, -60: 43.4, -50: 56.7, -40: 80.0, -30: 85.1}      # ms @35°C
CP_ACT = {-20: 280.7, -10: 323.0, 0: 268.0, 10: 150.6, 20: 90.2,
          30: 64.1, 40: 57.5, 50: 71.5, 60: 105.6}                      # ms @35°C

TM_CP_FULL = dict(TM_ORIG)
TM_CP_FULL.update({v: ms / 1000 * _FAC for v, ms in CP_DEACT.items()})
TM_CP_FULL.update({v: ms / 1000 * _FAC for v, ms in CP_ACT.items()})
TM_CP_ACT = dict(TM_ORIG)
TM_CP_ACT.update({v: ms / 1000 * _FAC for v, ms in CP_ACT.items()})


def load_engine():
    """加载原引擎并把 DATA 重定向到归档运行包（协议与细胞 .mat 均在归档内完好）。"""
    src = open(ENG_07, encoding="utf-8").read()
    old_data = None
    for line in src.splitlines():
        if line.startswith('DATA = '):
            old_data = line
            break
    assert old_data, "engine DATA line not found"
    src = src.replace(old_data, f'DATA = r"{ARCH_PACK}"', 1)
    spec = importlib.util.spec_from_loader("alpha_engine_patched", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = ENG_07
    exec(compile(src, ENG_07, "exec"), mod.__dict__)
    return mod


def anchor_tabs(eng, cell="16713003"):
    """复现引擎的逐细胞锚表（需要该细胞的失活协议 .mat）。"""
    amp = json.load(open(eng.F_AMP, encoding="utf-8"))
    hook = json.load(open(eng.F_HOOK, encoding="utf-8"))
    inact = json.load(open(eng.F_INACT, encoding="utf-8"))
    hss = json.load(open(eng.F_HSS, encoding="utf-8"))
    Vi, Ii = eng.load_mat("inactivation_protocol.mat", cell, "inactivation")
    return eng.build_tabs(cell, amp, hook, inact, hss, Ii, Vi)


def swap_tau_m(tabs, eng, which):
    """which: 'orig' 不动；'full' → TM_CP_FULL；'act' → TM_CP_ACT。返回新 tabs dict。"""
    if which == "orig":
        return tabs
    table = TM_CP_FULL if which == "full" else TM_CP_ACT
    out = dict(tabs)
    out["tau_m"] = eng.Tab({float(k): float(v) for k, v in table.items()})
    return out
