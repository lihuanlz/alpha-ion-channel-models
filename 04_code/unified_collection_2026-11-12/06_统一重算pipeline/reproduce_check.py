# -*- coding: utf-8 -*-
"""
reproduce_check.py — 第三方一键核对入口（2026-11-12）

只读本目录下的三个产物文件，重算论文全部头条数字并打印 PASS/FAIL：
  1. master_unified_v1.csv                     (统一总表, 487,940 行)
  2. cipa2_cells.csv                           (CiPA 细胞级 τ 比, ss 最高浓度块口径)
  3. ../features/robust_verdict_all_profile_2026-11-08.csv  (四口径判决冻结表)

用法:
  python reproduce_check.py                # 默认从脚本所在目录读
  python reproduce_check.py --root <dir>   # 指定数据目录(解压 Zenodo 包后的情形)

不依赖任何原始波形; 原始波形的重提取见 REPRODUCE.md 第二节。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def check(name, got, want, tol=0.0):
    ok = (abs(got - want) <= tol) if isinstance(want, float) else (got == want)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {want}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE))
    root = Path(ap.parse_args().root)
    ok = True

    # ---- 1. 统一总表规模 ----
    m = pd.read_csv(root / "master_unified_v1.csv", low_memory=False)
    ok &= check("master rows", len(m), 487940)
    ok &= check("master columns", len(m.columns), 22)

    # ---- 2. 剖面 CI 校准(τ 类, qc ok, 有区间, 正值) ----
    tau = m[(m.parameter.astype(str).str.startswith("tau"))
            & (m.qc_flag == "ok") & m.prof_lo.notna() & (m.value > 0)]
    cov = ((tau.prof_lo <= tau.value) & (tau.value <= tau.prof_hi)).mean()
    ok &= check("profile CI coverage (n=%d)" % len(tau), round(float(cov), 3), 0.962, tol=0.005)

    # ---- 3. 逐源行数 ----
    want_rows = {"channelpedia": 276039, "cipa_5lab": 192055, "lei2019": 10203,
                 "beattie9": 207, "nav15_27193878": 371, "cav12_ren2022": 562,
                 "iks_fedida": 8147, "cipa_official": 250,
                 "channelpedia_groupstats": 76, "icgenealogy": 30}
    got_rows = m.source_big.value_counts().to_dict()
    for k, v in want_rows.items():
        ok &= check(f"rows[{k}]", int(got_rows.get(k, -1)), v)

    # ---- 4. 判决冻结表( headline D 口径 ) ----
    v = pd.read_csv(HERE.parent / "features" / "robust_verdict_all_profile_2026-11-08.csv")
    ok &= check("verdict rows", len(v), 212)
    want_verdict = {"kv_main": (64, 63, 2.47), "tau_deact": (48, 43, 2.82),
                    "tau_inact": (35, 28, 1.33), "rec": (42, 39, 2.15),
                    "hcn": (5, 5, 8.03)}
    for tb, (n_tot, n_pass, med) in want_verdict.items():
        g = v[(v.table == tb) & v.ratio_D.notna()]
        ok &= check(f"verdict[{tb}] evaluable", len(g), n_tot)
        ok &= check(f"verdict[{tb}] pass", int((g.ratio_D > 1).sum()), n_pass)
        ok &= check(f"verdict[{tb}] median", round(float(g.ratio_D.median()), 2), med, tol=0.01)

    # ---- 5. CiPA τ 比(ss 口径) ----
    cc = pd.read_csv(root / "cipa2_cells.csv")
    tr = cc["tau_ratio"].dropna()
    tr = tr[(tr > 0) & (tr < 10)]
    ok &= check("cipa tau_ratio n", len(tr), 970)
    ok &= check("cipa tau_ratio median", round(float(tr.median()), 3), 0.948, tol=0.005)
    lab_med = {l: round(float(cc[cc.lab == l]["tau_ratio"].dropna().pipe(lambda s: s[(s > 0) & (s < 10)]).median()), 2)
               for l in ["lab_1", "lab_2", "lab_3", "lab_4", "lab_5"]}
    for l, w in zip(["lab_1", "lab_2", "lab_3", "lab_4", "lab_5"], [1.05, 0.99, 0.81, 0.92, 1.17]):
        ok &= check(f"cipa {l} median", lab_med[l], w, tol=0.01)
    b = cc[cc.blind == True]["tau_ratio"].dropna(); b = b[(b > 0) & (b < 10)]
    ok &= check("cipa blinded median (n=%d)" % len(b), round(float(b.median()), 2), 0.87, tol=0.01)

    # ---- 6. Beattie 九细胞锚 ----
    b9 = m[(m.source_big == "beattie9") & (m.parameter == "Vhalf_act") & (m.qc_flag == "ok")]
    ok &= check("beattie Vhalf_act median mV", round(float(b9.value.median()), 2), -8.97, tol=0.05)

    print("\n== %s ==" % ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
