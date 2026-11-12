# -*- coding: utf-8 -*-
"""
Stage5c raw层漏电流审计
========================
动机：lab_2 系统性偏移、lab_5 rundown 是否有漏电解释？
方法：对 raw∩subtracted 交集细胞，文本快读电流列：
  - I_hold_raw/sub: trace_#1(对照sweep) 前500点(100ms保持段)中位数
  - leak_hold: 保持段 raw-sub 差
  - leak_max_abs: 全trace |raw-sub| 最大值(step/ramp段漏电幅度)
  - hold_drift_raw/sub: 前20个sweep保持电流的线性斜率(pA/sweep)
断点续跑: stage5c_ckpt.jsonl; --budget 秒数自停。
"""
import os, sys, glob, json, time, zipfile, argparse
import numpy as np
from multiprocessing import Pool

BASE = r'D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia'
WARE = os.path.join(BASE, '数据总库', '01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t')
OUT = os.path.join(BASE, '分析_全药池_2026-11-10')
CKPT = os.path.join(OUT, 'stage5c_ckpt.jsonl')
CSV = os.path.join(OUT, 'stage5c_leak_audit.csv')

NSWEEP_DRIFT = 20   # 前20个sweep算保持漂移
NHOLD = 500         # 保持段点数(100ms @5kHz)

def read_cols(zp, ncols):
    """文本快读前 ncols 个电流列。返回 (list of np.array, 单位换算到pA的因子uf)。
    单位坑修复(2026-11-11审计)：lab_2 电流后缀为 _current_A(安培)，必须×1e12。"""
    cols = [[] for _ in range(ncols)]
    with zipfile.ZipFile(zp) as z:
        with z.open(z.namelist()[0]) as fh:
            header = fh.readline().decode('ascii', 'ignore')
            suffix = header.split(',')[1].rsplit('_current_', 1)[-1].strip().lower() if '_current_' in header else 'pa'
            uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(suffix)
            for i, line in enumerate(fh):
                p = line.split(b',', ncols + 1)
                if len(p) < ncols + 1:
                    continue
                for k in range(ncols):
                    try:
                        cols[k].append(float(p[k + 1]))
                    except ValueError:
                        cols[k].append(np.nan)
    if uf is None:
        uf = np.nan
    return [np.array(c) * uf for c in cols], uf

def proc_one(task):
    """task = (cell, lab, drug, raw_zip, sub_zip)"""
    cell, lab, drug, rz, sz = task
    try:
        raw, uf_r = read_cols(rz, NSWEEP_DRIFT)
        sub, uf_s = read_cols(sz, NSWEEP_DRIFT)
        n = min(len(raw[0]), len(sub[0]))
        if n < NHOLD:
            return {'cell': cell, 'lab': lab, 'drug': drug, 'err': 'short'}
        # trace_#1
        r1, s1 = raw[0][:n], sub[0][:n]
        I_hold_raw = float(np.nanmedian(r1[:NHOLD]))
        I_hold_sub = float(np.nanmedian(s1[:NHOLD]))
        leak = r1 - s1
        leak_hold = float(np.nanmedian(leak[:NHOLD]))
        leak_max = float(np.nanmax(np.abs(leak)))
        # 前20 sweep 保持漂移
        def drift(cols):
            holds = [np.nanmedian(c[:NHOLD]) for c in cols if len(c) >= NHOLD]
            if len(holds) < 5:
                return np.nan
            x = np.arange(len(holds))
            return float(np.polyfit(x, holds, 1)[0])
        return {'cell': cell, 'lab': lab, 'drug': drug,
                'I_hold_raw': I_hold_raw, 'I_hold_sub': I_hold_sub,
                'leak_hold': leak_hold, 'leak_max_abs': leak_max,
                'hold_drift_raw': drift(raw), 'hold_drift_sub': drift(sub),
                'uf_raw': uf_r, 'uf_sub': uf_s}
    except Exception as e:
        return {'cell': cell, 'lab': lab, 'drug': drug, 'err': str(e)[:80]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=240)
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()

    # 任务清单: raw∩sub 交集
    tasks = []
    for z in glob.glob(os.path.join(WARE, 'lab_*', '*', '*', 'raw', 'ted', '*.csv.zip')):
        parts = os.path.normpath(z).split(os.sep)
        i = parts.index('01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t')
        lab, drug = parts[i + 1], parts[i + 3]
        cell = os.path.basename(z)[:-len('.csv.zip')]
        sz = z.replace(os.sep + 'raw' + os.sep, os.sep + 'subtracted' + os.sep)
        if os.path.exists(sz):
            tasks.append((cell, lab, drug, z, sz))
    print('交集任务:', len(tasks))

    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['cell'], r['lab'], r['drug']))
                except Exception:
                    pass
    todo = [t for t in tasks if (t[0], t[1], t[2]) not in done]
    print('已完成:', len(done), ' 待跑:', len(todo))
    if not todo:
        finalize()
        return

    fout = open(CKPT, 'a', encoding='utf-8')
    n_done = 0
    pool = Pool(args.workers)
    try:
        for r in pool.imap_unordered(proc_one, todo, chunksize=8):
            fout.write(json.dumps(r, ensure_ascii=False) + '\n')
            n_done += 1
            if n_done % 200 == 0:
                fout.flush()
                print(f'  {n_done}/{len(todo)}  elapsed={time.time()-t0:.0f}s', flush=True)
            if time.time() - t0 > args.budget:
                print('budget 自停, 已跑', n_done)
                break
    finally:
        pool.terminate()
        fout.close()
    finalize()

def finalize():
    import pandas as pd
    rows = []
    with open(CKPT, encoding='utf-8') as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows).drop_duplicates(subset=['cell', 'lab', 'drug'], keep='last')
    df.to_csv(CSV, index=False)
    ok = df[df['err'].isna()] if 'err' in df.columns else df
    print('finalize:', len(df), '行, ok:', len(ok))
    for lab in sorted(ok['lab'].unique()):
        s = ok[ok['lab'] == lab]
        print(f'{lab}: n={len(s)}  I_hold_raw中位={s["I_hold_raw"].median():.1f}pA  leak_hold中位={s["leak_hold"].median():.1f}pA  leak_max中位={s["leak_max_abs"].median():.0f}pA  drift_raw中位={s["hold_drift_raw"].median():.3f}pA/sweep')

if __name__ == '__main__':
    main()
