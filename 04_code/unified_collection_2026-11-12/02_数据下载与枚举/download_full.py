# -*- coding: utf-8 -*-
"""
Channelpedia 批量下载器 v4
- 每线程独立 session；token 线程内复用
- 302 = 文件不存在，记 missing 不重试
- 所有分支关闭响应，防连接泄漏死锁
- 硬截止防 drip-feed；断点续传
- 下载范围：channelpedia_all_cells.csv 全部唯一细胞（NWB + JSON 都试）
产出: nwb_full/c<id>.nwb, acell_full/c<id>.json, features/download_report.json
"""
import re, time, sys, json, threading
import requests
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).parent
BASE = "https://channelpedia.epfl.ch"
NWB_DIR = ROOT / "nwb_full"
JSON_DIR = ROOT / "acell_full"
NWB_DIR.mkdir(exist_ok=True)
JSON_DIR.mkdir(exist_ok=True)
REPORT = ROOT / "features" / "download_report.json"
USERNAME = "lihaunlz"
PASSWORD = "lop09ihj"

_tls = threading.local()

def get_client():
    c = getattr(_tls, "client", None)
    if c is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "Mozilla/5.0 (academic research; Huan Li)"})
        r = s.get(f"{BASE}/login", timeout=(10, 30))
        tok = re.search(r'authenticity_token" value="([^"]+)"', r.text).group(1)
        s.post(f"{BASE}/login", data={"utf8": "✓", "authenticity_token": tok,
               "authentication[name]": USERNAME, "authentication[password]": PASSWORD,
               "commit": "Login"}, timeout=(10, 30))
        r2 = s.get(f"{BASE}/expdata/download/29326", timeout=(10, 30))
        if "Please login" in r2.text:
            raise RuntimeError("login failed")
        tok2 = re.search(r'authenticity_token" value="([^"]+)"', r2.text).group(1)
        c = {"session": s, "token": tok2}
        _tls.client = c
    return c

def reset_client():
    old = getattr(_tls, "client", None)
    if old:
        try: old["session"].close()
        except Exception: pass
    _tls.client = None

def fetch(cid, kind, out, expect):
    """返回 ok/skip/missing/fail"""
    if out.exists() and out.stat().st_size > 8000:
        return "skip"
    for attempt in range(3):
        r = None
        try:
            c = get_client()
            r = c["session"].post(f"{BASE}/expdata/download/{cid}/{kind}",
                                  data={"utf8": "✓", "authenticity_token": c["token"]},
                                  timeout=(10, 30), stream=True, allow_redirects=False)
            if r.status_code == 302:
                r.close()
                return "missing"
            ct = r.headers.get("Content-Type", "").lower()
            if r.status_code == 200 and (expect in ct or "octet-stream" in ct):
                tmp = out.with_suffix(out.suffix + ".part")
                deadline = time.time() + 150
                ok = True
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
                        if time.time() > deadline:
                            ok = False
                            break
                r.close()
                if ok and tmp.stat().st_size > 8000:
                    tmp.replace(out)
                    return "ok"
                tmp.unlink(missing_ok=True)
            else:
                r.close()
                reset_client()
                time.sleep(1 + attempt)
        except Exception:
            try:
                if r is not None: r.close()
            except Exception: pass
            time.sleep(1.5 * (attempt + 1))
    return "fail"

def job(cid):
    r1 = fetch(cid, "rCell/nwb", NWB_DIR / f"c{cid}.nwb", "octet-stream")
    r2 = fetch(cid, "aCell/json", JSON_DIR / f"c{cid}.json", "json")
    return cid, r1, r2

def main():
    cells = pd.read_csv(ROOT / "features" / "channelpedia_all_cells.csv")
    ids = sorted(int(x) for x in cells.cell_id.unique())
    print(f"唯一细胞总数: {len(ids)}", flush=True)
    get_client()
    print("线程0 登录 OK", flush=True)
    n_ok = n_skip = n_missing = n_fail = 0
    t0 = time.time()
    done = 0
    rep = {"missing": [], "fail": []}
    if REPORT.exists():
        old = json.loads(REPORT.read_text())
        rep["missing"] = old.get("missing", [])
        rep["fail"] = old.get("fail", [])
    missing_set = set(rep["missing"])
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(job, cid) for cid in ids]
        for fut in as_completed(futs):
            cid, r1, r2 = fut.result()
            if r1 == "ok": n_ok += 1
            elif r1 == "skip": n_skip += 1
            elif r1 == "missing":
                n_missing += 1
                missing_set.add(cid)
            else: n_fail += 1
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                real = n_ok + n_missing + n_fail
                rate = real / max(el, 1) * 60
                eta = (len(ids) - done) / max(real / max(el, 1), 0.001) / 60
                rep["missing"] = sorted(missing_set)
                REPORT.write_text(json.dumps(rep))
                print(f"{done}/{len(ids)} new={n_ok} miss={n_missing} fail={n_fail} {rate:.0f}/min ETA {eta:.0f}min", flush=True)
    rep["missing"] = sorted(missing_set)
    REPORT.write_text(json.dumps(rep))
    print(f"DONE new={n_ok} skip={n_skip} miss={n_missing} fail={n_fail}", flush=True)

if __name__ == "__main__":
    main()
