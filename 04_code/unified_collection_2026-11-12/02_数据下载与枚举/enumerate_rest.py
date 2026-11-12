# -*- coding: utf-8 -*-
"""枚举 silent/bad/excl 三类细胞 ID（公开页），并入总清单"""
import re, time
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
BASE = "https://channelpedia.epfl.ch"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (academic research; Huan Li)"})

cen = pd.read_csv(ROOT / "features" / "channelpedia_census.csv")
combos = cen[["channel", "combo", "n_silent", "n_bad", "n_excl"]].drop_duplicates()

rows = []
t0 = time.time()
for i, r in combos.iterrows():
    for flag in ["silent", "bad", "excl"]:
        n = r.get(f"n_{flag}", 0)
        if not n or n <= 0:
            continue
        n_pages = int(n // 30) + 2
        got = set()
        for pg in range(1, n_pages + 1):
            try:
                h = S.get(f"{BASE}/expdata", params={"icORhc": "Ionchannel", "icname": r.channel,
                                                     "sht": r.combo, "goodFlag": flag, "page": pg},
                          timeout=60).text
                ids = set(re.findall(r'/expdata/details/(\d+)', h))
                new = ids - got
                got |= ids
                for cid in new:
                    rows.append({"channel": r.channel, "combo": r.combo, "cell_id": int(cid), "flag": flag})
                if not ids:  # 没有更多页
                    break
                time.sleep(0.12)
            except Exception as e:
                print(f"retry {r.channel}/{flag}/p{pg}: {e}", flush=True)
                time.sleep(2)
    if (i + 1) % 50 == 0:
        print(f"{i+1}/{len(combos)} combos, {time.time()-t0:.0f}s, cells={len(rows)}", flush=True)

df = pd.DataFrame(rows).drop_duplicates("cell_id") if rows else pd.DataFrame(columns=["channel","combo","cell_id","flag"])
df.to_csv(ROOT / "features" / "channelpedia_other_cells.csv", index=False)
print(f"DONE 其他类细胞: {len(df)}", flush=True)
