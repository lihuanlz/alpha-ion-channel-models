# -*- coding: utf-8 -*-
# stage5a_全段提取.py
# ============================================================================
# Stage5a：全段提取（中间浓度段 + E-4031 残余 + 分段斜率）——用官方 ted.xlsx
# LiquidAdditions 的 FIRST/LAST/ANL 索引定位每段，替代幅度启发式分段。
#
# 口径（冻结）：
#   幅度 = stage2d 同款自适应峰（排除开头 5% 后 argmax）
#   段参考 = 该段 ANL 列出的分析 sweep（缺 ANL 则取段内最后 5 条）
#   block_seg = 1 - median(A_seg) / median(A_vehicle_ref)
#   seg_slope = 段内末 25% sweep 的线性斜率（frac/sweep，相对该细胞参考幅度）
#   E-4031 残余 = median(A_e4031) / median(A_vehicle_ref)
#   伪迹剔除、单位换算、列过滤同 stage2d（电流列必须含 _current_）
# ============================================================================
import os, json, glob, time, argparse, zipfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "数据总库", "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
CKPT = os.path.join(HERE, "stage5a_ckpt.jsonl")

def peak_amp(y):
    i0 = int(0.05 * len(y))
    return float(y[i0:].max())

def proc_cell(args):
    zpath, segs_json, cell, lab, drug = args
    try:
        z = zipfile.ZipFile(zpath)
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("trace_#") and "_current_" in c]
        if len(cols) < 8:
            return dict(ok=False, cell=cell, err=f"traces={len(cols)}")
        uf = {"pa": 1.0, "na": 1e3, "a": 1e12, "ma": 1e15, "ua": 1e6, "µa": 1e6}.get(
            cols[0].rsplit("_current_", 1)[-1].lower())
        if uf is None:
            return dict(ok=False, cell=cell, err="单位未知")
        amps = {}
        for c in cols:
            tr = int(c.split("#")[1].split("_")[0])
            y = df[c].to_numpy(float) * uf
            if np.ptp(y) > 4000 or abs(np.max(y)) > 3000:
                continue
            amps[tr] = peak_amp(y)
        segs = json.loads(segs_json)
        out = dict(ok=True, cell=cell, lab=lab, drug=drug, segments=[])
        # 参考：vehicle 段（CTLFL=Y）的 ANL 或末 5
        veh = [s for s in segs if s["ctl"]]
        if not veh:
            return dict(ok=False, cell=cell, err="无vehicle段")
        v = veh[0]
        va = [amps[t] for t in v["anl"][-5:] if t in amps] or [amps[t] for t in range(v["last"] - 4, v["last"] + 1) if t in amps]
        if len(va) < 3:
            return dict(ok=False, cell=cell, err="veh参考不足")
        ref = float(np.median(va))
        if ref <= 0:
            return dict(ok=False, cell=cell, err="ref<=0")
        for s in segs:
            if s["ctl"]:
                continue
            trs = [t for t in range(s["first"], s["last"] + 1) if t in amps]
            if len(trs) < 5:
                continue
            aa = np.array([amps[t] for t in trs])
            tail = trs[max(0, int(len(trs) * 0.75)):]
            tail_a = np.array([amps[t] for t in tail])
            if len(tail) >= 3:
                sl = np.polyfit(np.arange(len(tail)), tail_a / ref, 1)[0]
            else:
                sl = np.nan
            last5 = [amps[t] for t in (s["anl"][-5:] if s["anl"] else trs[-5:]) if t in amps]
            if len(last5) < 2:
                last5 = [amps[t] for t in trs[-5:]]
            out["segments"].append(dict(
                liquid=s["liquid"], conc_uM=s["conc_uM"], e4031=s["e4031"],
                first=s["first"], last=s["last"], n_sweeps=len(trs),
                A_seg=float(np.median(last5)), block_seg=float(1 - np.median(last5) / ref),
                seg_slope_per_sweep=float(sl)))
        out["ref_A"] = ref
        return out
    except Exception as e:
        return dict(ok=False, cell=cell, err=f"{type(e).__name__}: {e}")

UNIT = {"um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3, "pm": 1e-6}

def build_jobs():
    jobs = []
    for ted in glob.glob(os.path.join(DB, "*", "*", "*", "subtracted", "ted", "ted.xlsx")):
        rel = os.path.relpath(ted, DB).replace("\\", "/").split("/")
        lab, drugdir = rel[0], rel[2]
        try:
            la = pd.ExcelFile(ted).parse("LiquidAdditions")
        except Exception as e:
            print(f"  [ted读取失败] {lab}/{drugdir}: {e}", flush=True)
            continue
        for cid, g in la.groupby("CELLID"):
            segs = []
            srczip = None
            for _, r in g.iterrows():
                src = str(r.get("SRCXFN", ""))
                if ".zip" in src:
                    srczip = src.split(":")[0]
                anl = [int(x) for x in str(r.get("ANL", "")).split(";") if x.strip().isdigit()]
                conc = r.get("CONC", np.nan)
                u = UNIT.get(str(r.get("CONCU", "")).lower(), np.nan)
                segs.append(dict(liquid=str(r.get("LIQUID", "")),
                                 conc_uM=float(conc) * u if np.isfinite(conc) and np.isfinite(u) else np.nan,
                                 ctl=str(r.get("CTLFL", "")).upper() == "Y",
                                 e4031=str(r.get("SBFL", "")).upper() == "Y",
                                 first=int(r["FIRST"]), last=int(r["LAST"]), anl=anl))
            if not srczip:
                continue
            zpath = os.path.join(os.path.dirname(ted), srczip)
            if not os.path.exists(zpath):
                continue
            cell = f"{lab}/{drugdir}/{os.path.basename(srczip).replace('.csv.zip', '')}"
            jobs.append((zpath, json.dumps(segs), cell, lab, drugdir))
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    jobs = build_jobs()
    print(f"官方索引驱动任务：{len(jobs)} 细胞", flush=True)
    done = set()
    if os.path.exists(CKPT):
        with open(CKPT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cell"])
                except Exception:
                    pass
    todo = [j for j in jobs if j[2] not in done]
    print(f"断点：已完成 {len(done)}，本轮 {len(todo)}", flush=True)
    t0 = time.time()
    with open(CKPT, "a", encoding="utf-8") as out:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for i, res in enumerate(ex.map(proc_cell, todo, chunksize=8)):
                out.write(json.dumps(res, ensure_ascii=False) + "\n")
                out.flush()
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
                if time.time() - t0 > a.budget:
                    print("  [budget] 到点自停", flush=True)
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
    print(f"本轮耗时 {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
