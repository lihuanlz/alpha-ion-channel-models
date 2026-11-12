# -*- coding: utf-8 -*-
# CiPA 全量终验：独立于下载时缓存，对 14,973 个文件重算 sha256 对拍 OSF 官方清单
import json, os, hashlib, time, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "CiPA全药库_全量_2026-11-09")
MANIFEST = os.path.join(ROOT, "osf_full_manifest_2026-11-09.json")
REPORT = os.path.join(ROOT, "_终验报告_2026-11-09.json")

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def check(f):
    dst = os.path.join(ROOT, f["path"].replace("/", os.sep))
    if not os.path.exists(dst):
        return ("missing", f["path"])
    if f["size"] and os.path.getsize(dst) != f["size"]:
        return ("size_mismatch", f["path"])
    if f["sha256"]:
        try:
            if sha256_of(dst) != f["sha256"]:
                return ("hash_mismatch", f["path"])
        except Exception as e:
            return ("read_error", f["path"] + " | " + str(e))
        return ("ok", None)
    return ("ok_nohash", None)

mf = json.load(open(MANIFEST, encoding="utf-8"))
files = mf["files"]
print(f"终验开始: {len(files)} 文件", flush=True)
t0 = time.time()
stat = {"ok": 0, "ok_nohash": 0}
bad = {"missing": [], "size_mismatch": [], "hash_mismatch": [], "read_error": []}
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for res, path in ex.map(check, files):
        done += 1
        if res in ("ok", "ok_nohash"):
            stat[res] += 1
        else:
            bad[res].append(path)
        if done % 2000 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(files)}  {el:.0f}s  预计剩余 {el/done*(len(files)-done):.0f}s", flush=True)

el = time.time() - t0
verdict = "PASS" if not any(bad.values()) else "FAIL"
rep = {
    "date": time.strftime("%Y-%m-%d %H:%M"),
    "node": mf.get("node"),
    "manifest": os.path.basename(MANIFEST),
    "files": len(files),
    "ok": stat["ok"],
    "ok_nohash_size_only": stat["ok_nohash"],
    "missing": len(bad["missing"]),
    "size_mismatch": len(bad["size_mismatch"]),
    "hash_mismatch": len(bad["hash_mismatch"]),
    "read_error": len(bad["read_error"]),
    "bad_lists": bad,
    "elapsed_s": round(el, 1),
    "verdict": verdict,
}
json.dump(rep, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n终验 {verdict}: ok={stat['ok']} 无哈希仅尺寸={stat['ok_nohash']} "
      f"缺失={len(bad['missing'])} 尺寸不符={len(bad['size_mismatch'])} "
      f"哈希不符={len(bad['hash_mismatch'])} 读错={len(bad['read_error'])}  用时 {el:.0f}s", flush=True)
print(f"报告: {REPORT}", flush=True)
for k, v in bad.items():
    for p in v[:10]:
        print(f"  BAD[{k}] {p}", flush=True)
