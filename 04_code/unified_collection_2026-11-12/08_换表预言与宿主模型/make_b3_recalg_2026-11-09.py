# -*- coding: utf-8 -*-
"""
2026-11-09 · 手术生成 B3 换表重校 G_Kr 版
==========================================
以 2026-11-09_StageB_B3_换表.py 为底，加 RECAL_G=1 臂：
换表后 G_Kr 重校准（log 网格 + 二分，探针缓存于 ckpt），G_Ks 保持冻结。
用途：B3 full C1 fail（APD90 −7.25%）后，验证"重校增益能否恢复绝对 APD + 判决是否仍不变"。
输出文件/缓存/稳态名均带 _recalG 后缀，与冻结臂完全隔离。
"""
import os

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026-11-09_StageB_B3_换表.py")
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2026-11-09_StageB_B3_换表重校G.py")

src = open(SRC, encoding="utf-8").read()
n0 = len(src)


def rep(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, f"锚点出现 {n} 次（预期 {count}）: {old[:60]!r}..."
    src = src.replace(old, new)


# A1 RECAL_G 开关
rep('TAU_SWAP = os.environ.get("TAU_SWAP", "orig")     # orig / full / act',
    'TAU_SWAP = os.environ.get("TAU_SWAP", "orig")     # orig / full / act\n'
    'RECAL_G = os.environ.get("RECAL_G", "0") == "1"   # 1 = 换表后重校 G_Kr（G_Ks 冻结）\n'
    '_RG = "_recalG" if RECAL_G else ""')

# A2 稳态文件名隔离
rep('B3_SS = os.path.join(_OUTDIR, f"StageB_B3稳态_CL1000_换表{TAU_SWAP}.json")',
    'B3_SS = os.path.join(_OUTDIR, f"StageB_B3稳态_CL1000_换表{TAU_SWAP}{_RG}.json")')

# A3 任务缓存目录隔离
rep('_cdir = os.path.join(_OUTDIR, f"_b3task_cache_换表{TAU_SWAP}" + ("_冒烟" if SMOKE else ""))',
    '_cdir = os.path.join(_OUTDIR, f"_b3task_cache_换表{TAU_SWAP}{_RG}" + ("_冒烟" if SMOKE else ""))')

# A4 ckpt 文件名隔离
rep('CKPT_FP = os.path.join(_OUTDIR, f"2026-11-09_StageB_B3_ckpt_换表{TAU_SWAP}"',
    'CKPT_FP = os.path.join(_OUTDIR, f"2026-11-09_StageB_B3_ckpt_换表{TAU_SWAP}{_RG}"')

# A5 输出 tag 隔离
rep('tag = ("_冒烟" if SMOKE else "_结果") + f"_换表{TAU_SWAP}"',
    'tag = ("_冒烟" if SMOKE else "_结果") + f"_换表{TAU_SWAP}{_RG}"')

# A6 重校块（插在 control 段前；y0/tabs/P_ctrl/b0 此时已就绪）
OLD = '''    # ---------- control: steady-state rebuild + C1 + save B3 steady state ----------'''
NEW = '''    # ---------- RECAL_G arm: recalibrate G_Kr so control APD90 returns to the B0 target ----------
    if RECAL_G:
        _target = B0_FORMAL_APD90
        _tol = 0.02 * B0_FORMAL_APD90
        _probes = CKPT.setdefault("grecal_probes", {})

        def _probe_gkr(lg):
            key = f"{lg:.6f}"
            if key in _probes:
                _a, _p = _probes[key]
                print(f"  [recalG ckpt] G_Kr=10^{lg:+.3f} APD90={_a:.2f}", flush=True)
                return _a, _p
            _G = {"Kr": 10.0 ** lg, "Ks": G_Ks}
            _rhs = build_b3(b0, P_ctrl, tabs, _G, 0.0, 0.0, 0.0)
            _a, _p, _q, _o = run_beats(b0, _rhs, y0, 100, label="recalG")
            _oka = _a[~np.isnan(_a)]
            _okp = _p[~np.isnan(_p)]
            apd90 = float(np.median(_oka[-10:])) if len(_oka) else float("nan")
            apa90 = float(np.median(_okp[-10:])) if len(_okp) else float("nan")
            _probes[key] = [apd90, apa90]
            ckpt_save()
            return apd90, apa90

        _lg0 = math.log10(G_Kr)
        _pts = []
        for _lg in (_lg0 - 0.3, _lg0, _lg0 + 0.3):
            _a, _p = _probe_gkr(_lg)
            _pts.append((_lg, _a, _p))
            print(f"  [recalG grid] G_Kr=10^{_lg:+.3f}={10 ** _lg:.4g} APD90={_a:.2f} APA={_p:.1f}", flush=True)
        _healthy = [(lg, a) for lg, a, p in _pts if p >= 90.0 and not math.isnan(a)]
        _br = None
        for (l1, a1), (l2, a2) in zip(_healthy, _healthy[1:]):
            if (a1 - _target) * (a2 - _target) <= 0 and a1 != a2:
                _br = (l1, l2)
                break
        if _br is None:
            _srt = sorted(_healthy, key=lambda t: t[0])
            if _srt and all(a < _target for _, a in _srt):
                _br = (_srt[0][0] - 0.3, _srt[0][0])
                _a, _p = _probe_gkr(_br[0])
                print(f"  [recalG extend-lo] G_Kr=10^{_br[0]:+.3f} APD90={_a:.2f} APA={_p:.1f}", flush=True)
            elif _srt and all(a > _target for _, a in _srt):
                _br = (_srt[-1][0], _srt[-1][0] + 0.3)
                _a, _p = _probe_gkr(_br[1])
                print(f"  [recalG extend-hi] G_Kr=10^{_br[1]:+.3f} APD90={_a:.2f} APA={_p:.1f}", flush=True)
        if _br:
            _lo, _hi = _br
            for _it in range(10):
                _mid = 0.5 * (_lo + _hi)
                _a, _p = _probe_gkr(_mid)
                print(f"  [recalG bisect {_it + 1}/10] G_Kr=10^{_mid:.3f}={10 ** _mid:.4g}"
                      f" APD90={_a:.2f} APA={_p:.1f}", flush=True)
                if _p < 90.0 or math.isnan(_a) or _a > _target:
                    _lo = _mid      # APD 过长/不健康 -> G_Kr 偏小，上调下界
                else:
                    _hi = _mid
                if _p >= 90.0 and not math.isnan(_a) and abs(_a - _target) <= _tol:
                    break
            G_Kr = 10.0 ** (0.5 * (_lo + _hi))
            CKPT["grecal_G_Kr"] = G_Kr
            ckpt_save()
            print(f"[recalG] G_Kr 重校完成: {G_Kr:.5f}（冻结值 0.32266）", flush=True)
        else:
            print("  [recalG] 网格无括号区间，保留冻结值并登记", flush=True)

    # ---------- control: steady-state rebuild + C1 + save B3 steady state ----------'''
rep(OLD, NEW)

# A7 meta 记录 recal 标记
rep('    out = {"meta": {"smoke": SMOKE, "prereg": "v7 §11.9 + v7.1 §11.10",',
    '    out = {"meta": {"smoke": SMOKE, "recal_g": RECAL_G, "prereg": "v7 §11.9 + v7.1 §11.10",')

open(DST, "w", encoding="utf-8").write(src)
import ast
ast.parse(src)
print(f"手术完成: {n0} -> {len(src)} chars, ast.parse 通过")
print(f"输出: {DST}")
