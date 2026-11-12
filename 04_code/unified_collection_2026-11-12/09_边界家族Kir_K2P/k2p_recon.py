# -*- coding: utf-8 -*-
"""
K2P 漏通道候选不变量侦察 —— 2026-11-08
理念平移: VSD 通道的定律是"τ/k 守恒、V½/幅度漂移"。
漏通道无门控 -> 候选不变量换成:
  (a) 反转电位 E_rev (应钉在 E_K)
  (b) 归一化 I-V 形状 (GHK 开放整流形状 = 硬件; 幅度 = 细胞态)
  (c) 瞬时性 (leak 应无时间依赖: I_early/I_late ≈ 1)
判决: shape_disp << amp_disp 且 E_rev 散布小
产出: icg_out/../features/k2p_recon_2026-11-08.csv + fig_k2p_recon.png
"""
import re, sys
import numpy as np, pandas as pd, h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent
NWB = ROOT / "nwb_full"
XI = 1e-4
ACT_V = list(range(-160, 81, 15))  # 17

def parse_cell(cid, channel, temp):
    fp = NWB / f"c{cid}.nwb"
    try:
        with h5py.File(fp, "r") as f:
            ts = f.get("acquisition/timeseries")
            if ts is None or "Activation" not in ts: return None
            reps = ts["Activation/repetitions"]
            rep = "repetition1" if "repetition1" in reps else list(reps.keys())[0]
            d = ts[f"Activation/repetitions/{rep}/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T  # (sweeps, samples)
            if d.shape[0] < 17 or d.shape[1] < 6000: return None
            base = d[:, int(0.05/XI):int(0.09/XI)].mean(axis=1)
            d0 = d - base[:, None]
            i_late = d0[:, int(0.55/XI):int(0.59/XI)].mean(axis=1)   # 脉冲末
            i_early = d0[:, int(0.11/XI):int(0.12/XI)].mean(axis=1)  # 脉冲初
            V = np.array(ACT_V[:17], float)
            # E_rev: 零点线性插值
            sgn = np.sign(i_late)
            e_rev = np.nan
            for i in range(len(V)-1):
                if sgn[i] == 0 or sgn[i]*sgn[i+1] < 0:
                    e_rev = V[i] - i_late[i]*(V[i+1]-V[i])/(i_late[i+1]-i_late[i]); break
            i80 = i_late[-1]
            if abs(i80) < 0.02: return None  # 无电流
            shape = list(i_late / i80)[:17] + [np.nan]*(17-len(i_late))
            inst = np.where(np.abs(i_late) > 0.01, i_early/np.where(i_late==0,np.nan,i_late), np.nan)
            return dict(cell_id=cid, channel=channel, temp=temp, e_rev=e_rev,
                        i80=i80, shape=list(shape), inst_med=float(np.nanmedian(inst[3:15])))
    except Exception:
        return None

def job(a): return parse_cell(*a)

def main():
    cells = pd.read_csv(ROOT/"features"/"channelpedia_all_cells.csv")
    have = set(int(re.match(r"c(\d+)\.nwb", f.name).group(1)) for f in NWB.glob("c*.nwb"))
    k2p = cells[cells.channel.str.startswith(("TA","TR","TW","TH")) &
                (cells.flag=="active") & cells.cell_id.isin(have)].drop_duplicates("cell_id")
    todo = [(int(r.cell_id), r.channel, None) for r in k2p.itertuples()]
    print(f"K2P 目标: {len(todo)}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, a) for a in todo]
        for fut in as_completed(futs):
            r = fut.result()
            if r: rows.append(r)
    df = pd.DataFrame(rows)
    print(f"提取成功: {len(df)}")
    # 汇总 per subtype
    out = []
    for ch, g in df.groupby("channel"):
        n = len(g)
        S = np.array(g["shape"].tolist())   # (n,17)
        amp_cv = float(np.abs(g.i80).std()/np.abs(g.i80).mean())
        shape_sd = np.nanstd(S, axis=0)     # 逐电压形状散布(线性空间)
        shape_sd_med = float(np.median(shape_sd[3:14]))
        erev = g.e_rev.dropna()
        out.append(dict(channel=ch, n=n, amp_CV=round(amp_cv,3),
                        shape_SD_med=round(shape_sd_med,3),
                        Erev_med=round(float(g.e_rev.median()),1),
                        Erev_SD=round(float(g.e_rev.std()),2),
                        inst_med=round(float(g.inst_med.median()),3)))
        # 保存逐细胞形状
    res = pd.DataFrame(out)
    res.to_csv(ROOT/"features"/"k2p_recon_2026-11-08.csv", index=False)
    df.to_pickle(ROOT/"features"/"k2p_recon_cells_2026-11-08.pkl")
    print(res.to_string(index=False))

    # 图
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot; setup_plot()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    V = np.array(ACT_V, float)
    for ax, (ch, g) in zip(axes.ravel(), df.groupby("channel")):
        S = np.array(g["shape"].tolist())
        for s in S: ax.plot(V, s, color="#2471a3", alpha=0.15, lw=0.7)
        ax.plot(V, np.nanmedian(S, axis=0), color="k", lw=2)
        ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
        ax.set_title(f"{ch} (n={len(g)})", fontsize=10)
        ax.set_ylim(-0.6, 1.3)
    for ax in axes[-1]: ax.set_xlabel("V (mV)")
    for ax in axes[:,0]: ax.set_ylabel("I / I(+80 mV)")
    fig.suptitle("K2P normalized I-V shapes across cells (Channelpedia active cells)")
    fig.tight_layout()
    fig.savefig(ROOT/"features"/"fig_k2p_recon.png", dpi=200, bbox_inches="tight")
    print("fig saved")

if __name__ == "__main__":
    main()
