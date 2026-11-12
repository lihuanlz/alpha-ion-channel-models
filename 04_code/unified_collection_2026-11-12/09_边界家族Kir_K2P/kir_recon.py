# -*- coding: utf-8 -*-
"""
Kir 救援提取管线 —— 2026-11-08
背景: Channelpedia 官方 QC 把 791/807 Kir 细胞判为 "silent"，但协议 #5 实测为
      0..-170 mV step -10（500ms, 0.1-0.6s, 18 sweeps）——内向导通区完整覆盖。
      官方判 silent 是因电流绝对值小（Kv 导向阈值），不是无信号。
提取: i_late (500-590ms) I-V; E_rev (跨零插值); 内向斜率电导; 归一化形状; 瞬时性
判决: 真Kir子集(E_rev<-60mV 且 |I(-170)|>=0.1nA) 上:
      (a) E_rev 钉 E_K (-93mV): 散布应小
      (b) 归一化 I-V 形状 SD << 幅度 CV   (与 K2P 同构的"形状守恒 vs 幅度漂移")
      (c) 瞬时性 i_early/i_late ≈ 1 (Kir 整流近瞬时)
产出: features/kir_recon_2026-11-08.csv + kir_recon_cells pkl + fig_kir_recon.png
"""
import re, sys
import numpy as np, pandas as pd, h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).parent
NWB = ROOT / "nwb_full"
XI = 1e-4
KIR_V = list(range(0, -171, -10))   # 协议#5: 0..-170, 18 sweeps

def parse_cell(cid, channel):
    fp = NWB / f"c{cid}.nwb"
    try:
        with h5py.File(fp, "r") as f:
            ts = f.get("acquisition/timeseries")
            if ts is None or "Activation" not in ts: return None
            reps = ts["Activation/repetitions"]
            rep = "repetition1" if "repetition1" in reps else list(reps.keys())[0]
            d = ts[f"Activation/repetitions/{rep}/data"][:]
            if d.ndim == 1: d = d[:, None]
            d = d.T
            if d.shape[0] < 18 or d.shape[1] < 6000: return None   # 非协议5
            base = d[:, 500:900].mean(axis=1)
            d0 = d - base[:, None]
            i_late = d0[:, 5000:5900].mean(axis=1)[:18]
            i_early = d0[:, 1100:1200].mean(axis=1)[:18]
            V = np.array(KIR_V, float)
            # E_rev 跨零插值
            sgn = np.sign(i_late); e_rev = np.nan
            for i in range(17):
                if sgn[i] == 0 or sgn[i] * sgn[i+1] < 0:
                    e_rev = V[i] - i_late[i] * (V[i+1]-V[i]) / (i_late[i+1]-i_late[i]); break
            i170 = i_late[-1]
            if abs(i170) < 0.05: return None                      # 连漏电流都测不出
            shape = i_late / i170
            # 内向斜率电导 (nS): -140..-100 段线性
            m_in = (V >= -140) & (V <= -100)
            gin = np.nan
            if m_in.sum() >= 4:
                p = np.polyfit(V[m_in], i_late[m_in], 1)
                gin = float(p[0])                                  # nA/mV = µS → ×1e3 nS
            # 瞬时性: 深超极化 sweeps 的 i_early/i_late
            deep = np.abs(i_late[10:18]) > 0.02
            inst = float(np.nanmedian((i_early[10:18][deep]) / i_late[10:18][deep])) if deep.any() else np.nan
            return dict(cell_id=cid, channel=channel, e_rev=e_rev, i170=i170,
                        gin_uS=gin, shape=list(shape), inst_deep=inst)
    except Exception:
        return None

def job(a): return parse_cell(*a)

def main():
    cells = pd.read_csv(ROOT / "features" / "channelpedia_all_cells.csv", low_memory=False)
    have = {int(re.match(r"c(\d+)\.nwb", f.name).group(1)) for f in NWB.glob("c*.nwb")}
    kir = cells[cells.channel.str.startswith("Kir") & cells.cell_id.isin(have)].drop_duplicates("cell_id")
    todo = [(int(r.cell_id), r.channel) for r in kir.itertuples()]
    print(f"Kir 目标: {len(todo)}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(job, a) for a in todo]
        for i, fut in enumerate(as_completed(futs)):
            r = fut.result()
            if r: rows.append(r)
            if (i+1) % 200 == 0: print(f"  {i+1}/{len(todo)}", flush=True)
    df = pd.DataFrame(rows)
    print(f"提取成功(有电流): {len(df)}")

    # 分类: 真Kir vs 漏
    df["kind"] = np.where((df.e_rev < -60) & (df.i170.abs() >= 0.1), "kir",
                 np.where(df.e_rev.abs() < 30, "leak", "other"))
    print(df["kind"].value_counts().to_dict())

    # 判决: 真Kir子集, 全族 + 分亚型(只报 n>=8)
    out = []
    for grp, g in [("ALL", df[df.kind == "kir"])] + list(df[df.kind == "kir"].groupby("channel")):
        n = len(g)
        if n < 8: continue
        S = np.array(g["shape"].tolist())
        amp_cv = float(np.abs(g.i170).std() / np.abs(g.i170).mean())
        ssd = np.nanstd(S, axis=0)
        shape_sd_med = float(np.median(ssd[3:17]))
        out.append(dict(group=grp, n=n,
                        Erev_med=round(float(g.e_rev.median()), 1),
                        Erev_SD=round(float(g.e_rev.std()), 2),
                        amp_CV=round(amp_cv, 3),
                        shape_SD_med=round(shape_sd_med, 3),
                        ratio_amp_over_shape=round(amp_cv / shape_sd_med, 2) if shape_sd_med > 0 else np.nan,
                        inst_med=round(float(g.inst_deep.median()), 3)))
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "features" / "kir_recon_2026-11-08.csv", index=False)
    df.to_pickle(ROOT / "features" / "kir_recon_cells_2026-11-08.pkl")
    print(res.to_string(index=False))

    # 图: 真Kir vs 漏的归一化 I-V
    sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
    from daimon_runtime import setup_plot; setup_plot()
    import matplotlib.pyplot as plt
    V = np.array(KIR_V, float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for kind, col, ax in [("kir", "#c0392b", axes[0]), ("leak", "#7f8c8d", axes[1])]:
        g = df[df.kind == kind]
        S = np.array(g["shape"].tolist()) if len(g) else np.empty((0, 18))
        for s in S: ax.plot(V, s, color=col, alpha=0.12, lw=0.7)
        if len(S): ax.plot(V, np.nanmedian(S, axis=0), color="k", lw=2)
        ax.axvline(-93, color="blue", ls=":", lw=1)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{kind} (n={len(g)})  norm. I-V")
        ax.set_xlabel("V (mV)"); ax.set_ylabel("I / I(-170 mV)")
    g = df[df.kind == "kir"]
    axes[2].hist(g.e_rev.dropna(), bins=np.arange(-120, 20, 5), color="#c0392b", alpha=0.8, edgecolor="k", lw=0.4)
    axes[2].axvline(-93, color="blue", ls="--", lw=1.2, label="E_K ≈ -93")
    axes[2].set_title(f"真Kir E_rev 分布 (n={g.e_rev.notna().sum()})")
    axes[2].set_xlabel("E_rev (mV)"); axes[2].legend()
    fig.suptitle("Kir rescue: Channelpedia 'silent' cells re-extracted (protocol #5: 0..-170 mV)", fontsize=11)
    fig.tight_layout()
    fig.savefig(ROOT / "features" / "fig_kir_recon.png", dpi=200, bbox_inches="tight")
    print("fig saved")

if __name__ == "__main__":
    main()
