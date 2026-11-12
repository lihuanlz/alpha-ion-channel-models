# -*- coding: utf-8 -*-
"""
Channelpedia 全集枚举器（公开页，无需登录）
产出：
  features/channelpedia_census.csv   —— 每个 通道×宿主组合×温度×goodFlag 的细胞计数
  features/channelpedia_active_cells.csv —— 所有 active 细胞的 cell_id 清单（下载目标）
"""
import re, time, json, sys
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
BASE = "https://channelpedia.epfl.ch"
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (academic research; Huan Li)"})

def get(url, **kw):
    for attempt in range(4):
        try:
            r = S.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            print(f"  retry {attempt+1}: {e}", flush=True)
        time_wait = 2 * (attempt + 1)
        time.sleep(time_wait)
    return None

# 1) 拿通道清单
html = get(f"{BASE}/expdata")
channels = re.findall(r'<select name="icname".*?>(.*?)</select>', html, re.S)[0]
ch_list = re.findall(r'value="([^"]+)"', channels)
ch_list = [c for c in ch_list if c and not c.startswith("Select")]
print(f"通道数: {len(ch_list)}", flush=True)

rows = []
cell_rows = []
t0 = time.time()
for ci, ch in enumerate(ch_list):
    h = get(f"{BASE}/expdata", params={"icORhc": "Ionchannel", "icname": ch})
    if h is None:
        print(f"{ch}: 拉 sht 失败", flush=True); continue
    m = re.search(r'<select name="sht".*?>(.*?)</select>', h, re.S)
    if not m:
        continue
    shts = [s for s in re.findall(r'value="([^"]+)"', m.group(1)) if s and not s.startswith("Select")]
    for sht in shts:
        h2 = get(f"{BASE}/expdata", params={"icORhc": "Ionchannel", "icname": ch, "sht": sht, "goodFlag": "active"})
        if h2 is None:
            continue
        mg = re.search(r'<select name="goodFlag".*?>(.*?)</select>', h2, re.S)
        counts = {}
        if mg:
            for val, label in re.findall(r'value="([^"]*)"[^>]*>([^<]*)', mg.group(1)):
                mc = re.search(r"(\d+)\s+(\w+)\s+cells", label)
                if mc:
                    counts[mc.group(2)] = int(mc.group(1))
        n_active = counts.get("active", 0)
        rows.append({"channel": ch, "combo": sht,
                     "n_active": n_active,
                     "n_silent": counts.get("silent", 0),
                     "n_bad": counts.get("bad", 0),
                     "n_excl": counts.get("excluded", 0)})
        # 抓 active 细胞 ID（分页）
        if n_active > 0:
            pages = [int(p) for p in re.findall(r'[?&]page=(\d+)', h2)]
            n_pages = max(pages) if pages else 1
            ids = re.findall(r'/expdata/details/(\d+)', h2)
            for cid in set(ids):
                cell_rows.append({"channel": ch, "combo": sht, "cell_id": int(cid)})
            for pg in range(2, n_pages + 1):
                hp = get(f"{BASE}/expdata", params={"icORhc": "Ionchannel", "icname": ch,
                                                    "sht": sht, "goodFlag": "active", "page": pg})
                if hp is None:
                    continue
                for cid in set(re.findall(r'/expdata/details/(\d+)', hp)):
                    cell_rows.append({"channel": ch, "combo": sht, "cell_id": int(cid)})
                time.sleep(0.15)
        time.sleep(0.15)
    if (ci + 1) % 10 == 0:
        print(f"{ci+1}/{len(ch_list)} 通道, {time.time()-t0:.0f}s, combos={len(rows)}, active cells={len(cell_rows)}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "features" / "channelpedia_census.csv", index=False)
dc = pd.DataFrame(cell_rows).drop_duplicates("cell_id")
dc.to_csv(ROOT / "features" / "channelpedia_active_cells.csv", index=False)
print(json.dumps({
    "channels": len(ch_list),
    "combos": len(rows),
    "total_active": int(df.n_active.sum()),
    "total_silent": int(df.n_silent.sum()),
    "total_bad": int(df.n_bad.sum()),
    "total_excl": int(df.n_excl.sum()),
    "unique_active_cells_listed": len(dc),
}, ensure_ascii=False), flush=True)
print("DONE", flush=True)
