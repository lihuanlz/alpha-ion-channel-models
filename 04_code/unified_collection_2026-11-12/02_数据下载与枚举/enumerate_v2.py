# -*- coding: utf-8 -*-
"""
Channelpedia 全集枚举器 v2（修正分页解析）
- 枚举全部 84 通道 × 组合 × 4 种 goodFlag 的细胞 ID
- 分页正则兼容 &amp;page= HTML 转义
产出:
  features/channelpedia_census_v2.csv
  features/channelpedia_all_cells.csv  (channel, combo, flag, cell_id)
"""
import re, time, json
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
BASE = "https://channelpedia.epfl.ch"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (academic research; Huan Li)"})
FLAGS = ["active", "silent", "bad", "excl"]

def get(params):
    for a in range(4):
        try:
            r = S.get(f"{BASE}/expdata", params=params, timeout=60)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            time.sleep(1.5 * (a + 1))
    return None

def n_pages_of(html):
    ps = [int(p) for p in re.findall(r'page=(\d+)', html)]
    return max(ps) if ps else 1

def cell_ids(html):
    return set(int(x) for x in re.findall(r'/expdata/details/(\d+)', html))

def main():
    html = get({})
    ch_list = [c for c in re.findall(r'value="([^"]+)"',
               re.findall(r'<select name="icname".*?>(.*?)</select>', html, re.S)[0])
               if c and not c.startswith("Select")]
    print(f"通道数: {len(ch_list)}", flush=True)
    cen_rows, cell_rows = [], []
    t0 = time.time()
    for ci, ch in enumerate(ch_list):
        h = get({"icORhc": "Ionchannel", "icname": ch})
        if not h:
            continue
        m = re.search(r'<select name="sht".*?>(.*?)</select>', h, re.S)
        if not m:
            continue
        shts = [s for s in re.findall(r'value="([^"]+)"', m.group(1)) if s and not s.startswith("Select")]
        for sht in shts:
            h2 = get({"icORhc": "Ionchannel", "icname": ch, "sht": sht, "goodFlag": "active"})
            if not h2:
                continue
            mg = re.search(r'<select name="goodFlag".*?>(.*?)</select>', h2, re.S)
            counts = {}
            if mg:
                for val, label in re.findall(r'value="([^"]*)"[^>]*>([^<]*)', mg.group(1)):
                    mc = re.search(r"(\d+)\s+(\w+)\s+cells", label)
                    if mc:
                        counts[mc.group(2)] = int(mc.group(1))
            cen_rows.append({"channel": ch, "combo": sht,
                             "n_active": counts.get("active", 0), "n_silent": counts.get("silent", 0),
                             "n_bad": counts.get("bad", 0), "n_excl": counts.get("excluded", 0)})
            for flag in FLAGS:
                key = "excl" if flag == "excl" else flag
                if counts.get({"excl": "excluded"}.get(flag, flag), 0) == 0:
                    continue
                hf = h2 if flag == "active" else get({"icORhc": "Ionchannel", "icname": ch, "sht": sht, "goodFlag": flag})
                if not hf:
                    continue
                npages = n_pages_of(hf)
                got = set()
                for cid in cell_ids(hf):
                    got.add(cid)
                for pg in range(2, npages + 1):
                    hp = get({"icORhc": "Ionchannel", "icname": ch, "sht": sht, "goodFlag": flag, "page": pg})
                    if not hp:
                        continue
                    got |= cell_ids(hp)
                    time.sleep(0.08)
                for cid in got:
                    cell_rows.append({"channel": ch, "combo": sht, "flag": flag, "cell_id": cid})
                time.sleep(0.08)
        if (ci + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{ci+1}/{len(ch_list)} 通道 {el:.0f}s combos={len(cen_rows)} cells={len(cell_rows)}", flush=True)

    cen = pd.DataFrame(cen_rows)
    cen.to_csv(ROOT / "features" / "channelpedia_census_v2.csv", index=False)
    cells = pd.DataFrame(cell_rows).drop_duplicates(["cell_id", "flag"]) if cell_rows else pd.DataFrame()
    cells.to_csv(ROOT / "features" / "channelpedia_all_cells.csv", index=False)
    summ = {
        "combos": len(cen),
        "counts_sum": {k: int(cen[f"n_{k}"].sum()) for k in ["active", "silent", "bad", "excl"]},
        "unique_cells_by_flag": cells.groupby("flag").cell_id.nunique().to_dict() if len(cells) else {},
        "unique_cells_total": int(cells.cell_id.nunique()) if len(cells) else 0,
    }
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
