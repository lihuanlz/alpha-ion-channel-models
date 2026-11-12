# -*- coding: utf-8 -*-
"""
数据归拢到数据总库（同卷 os.rename 秒级移动 + mklink /J junction 回指，旧脚本零改动）
幂等：已移动的跳过，已建 junction 的跳过。可重复跑。
"""
import os, sys, json, shutil, subprocess, datetime

LAW = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\04_细胞线4\_rev_2026-10-21\law_channelpedia"
HUB = os.path.join(LAW, "数据总库")
os.makedirs(HUB, exist_ok=True)

report = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "moves": [], "junctions": [], "copies": [], "verify": [], "errors": []}

def is_junction(p):
    return os.path.isdir(p) and (os.stat(p).st_file_attributes & 0x400) if hasattr(os.stat(p), 'st_file_attributes') else os.path.islink(p)

def move_dir(src, dst, tag):
    if os.path.isdir(dst):
        report["moves"].append({"tag": tag, "src": src, "dst": dst, "status": "已存在(跳过)"}); return True
    if not os.path.isdir(src):
        report["errors"].append({"tag": tag, "src": src, "err": "源不存在"}); return False
    os.rename(src, dst)
    report["moves"].append({"tag": tag, "src": src, "dst": dst, "status": "moved"}); return True

def move_files(srcdir, names, dstdir, tag):
    os.makedirs(dstdir, exist_ok=True)
    ok = True
    for n in names:
        s = os.path.join(srcdir, n); d = os.path.join(dstdir, n)
        if os.path.exists(d):
            continue
        if not os.path.exists(s):
            report["errors"].append({"tag": tag, "src": s, "err": "源不存在"}); ok = False; continue
        os.rename(s, d)
    report["moves"].append({"tag": tag, "src": srcdir, "dst": dstdir, "files": len(names), "status": "moved" if ok else "部分缺失"})
    return ok

def make_junction(link, target, tag):
    if os.path.isdir(link):
        report["junctions"].append({"tag": tag, "link": link, "target": target, "status": "已存在(跳过)"}); return
    if not os.path.isdir(target):
        report["errors"].append({"tag": tag, "err": f"junction 目标不存在: {target}"}); return
    r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True, text=True)
    if r.returncode == 0:
        report["junctions"].append({"tag": tag, "link": link, "target": target, "status": "created"})
    else:
        report["errors"].append({"tag": tag, "link": link, "err": (r.stdout + r.stderr).strip()})

def verify(link, expect_min_files, tag):
    try:
        n = sum(len(files) for _, _, files in os.walk(link))
        ok = n >= expect_min_files
        report["verify"].append({"tag": tag, "link": link, "files_seen": n, "ok": ok})
        return ok
    except Exception as e:
        report["verify"].append({"tag": tag, "link": link, "err": str(e), "ok": False}); return False

# ---------- 01 CiPA 全药库 ----------
d1 = os.path.join(HUB, "01_CiPA_hERG_5lab_30药_62GB_OSFa6k5t")
if move_dir(os.path.join(LAW, "CiPA全药库_全量_2026-11-09"), d1, "01_CiPA"):
    make_junction(os.path.join(LAW, "CiPA全药库_全量_2026-11-09"), d1, "01_CiPA")
verify(os.path.join(LAW, "CiPA全药库_全量_2026-11-09"), 14000, "01_CiPA")

# ---------- 02 nwb_full ----------
d2 = os.path.join(HUB, "02_Channelpedia_nwb_19344细胞_72GB")
if move_dir(os.path.join(LAW, "nwb_full"), d2, "02_nwb"):
    make_junction(os.path.join(LAW, "nwb_full"), d2, "02_nwb")
verify(os.path.join(LAW, "nwb_full"), 19000, "02_nwb")

# ---------- 03 data\cells ----------
d3 = os.path.join(HUB, "03_Channelpedia_精选细胞nwb_3171_12GB")
if move_dir(os.path.join(LAW, "data", "cells"), d3, "03_cells"):
    make_junction(os.path.join(LAW, "data", "cells"), d3, "03_cells")
verify(os.path.join(LAW, "data", "cells"), 3000, "03_cells")

# ---------- 03b kv 压缩包备份 ----------
d3b = os.path.join(HUB, "03b_kv压缩包备份_原始下载归档")
parts = ["kv_cells.7z"] + [f"part_{i}" for i in range(12)] + ["extract7z.py", "extract.log", "extract.err", "readme.txt"]
move_files(os.path.join(LAW, "data"), parts, d3b, "03b_kv压缩包")

# ---------- 04 acell_full ----------
d4 = os.path.join(HUB, "04_Channelpedia_逐细胞JSON_19282")
if move_dir(os.path.join(LAW, "acell_full"), d4, "04_acell"):
    make_junction(os.path.join(LAW, "acell_full"), d4, "04_acell")
verify(os.path.join(LAW, "acell_full"), 19000, "04_acell")

# ---------- 05 features ----------
d5 = os.path.join(HUB, "05_提取常数表_features")
if move_dir(os.path.join(LAW, "features"), d5, "05_features"):
    make_junction(os.path.join(LAW, "features"), d5, "05_features")
verify(os.path.join(LAW, "features"), 10, "05_features")

# ---------- 06 CiPA 官方锚（复制不移动） ----------
src_anchor = r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档\2026-10-16_清理\04_细胞线4\_归档\α模型\CiPA官方锚"
d6 = os.path.join(HUB, "06_CiPA官方锚_7通道IC50_29药")
if os.path.isdir(src_anchor):
    if not os.path.isdir(d6):
        shutil.copytree(src_anchor, d6)
    n = sum(len(files) for _, _, files in os.walk(d6))
    report["copies"].append({"tag": "06_CiPA官方锚", "src": src_anchor, "dst": d6, "files": n, "has_newCiPA": os.path.exists(os.path.join(d6, "newCiPA.csv"))})
else:
    # 备选：搜索 newCiPA.csv
    found = []
    for root, dirs, files in os.walk(r"D:\Kimi_Agent_细胞仿真工具包扩展以及具身智能20260911\09_历史归档"):
        if "newCiPA.csv" in files:
            found.append(root)
            break
    report["errors"].append({"tag": "06_CiPA官方锚", "err": "预期路径不存在", "found_candidates": found})

with open(os.path.join(HUB, "_归拢报告_2026-11-10.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=1)

print("=== 移动 ===")
for m in report["moves"]: print(" ", m["tag"], m["status"])
print("=== junction ===")
for j in report["junctions"]: print(" ", j["tag"], j["status"])
print("=== 穿透验证 ===")
for v in report["verify"]: print(" ", v["tag"], "files=", v.get("files_seen"), "OK" if v.get("ok") else "FAIL")
print("=== 复制 ===")
for c in report["copies"]: print(" ", c["tag"], "files=", c["files"], "newCiPA:", c["has_newCiPA"])
print("=== 错误 ===")
for e in report["errors"]: print(" ", e)
print("DONE")
