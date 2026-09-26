# 2026-09-14_CiPA全药库_下载器.py
# ============================================================================
# CiPA a6k5t 全药库下载器（OSF 源站直取，断点续传 + sha256 核验）
#
# 用法（Spyder 或命令行）：
#   第一步 爬清单（只读，约几分钟）：
#     %runfile '...下载器.py' -- crawl
#   第二步 按档位下载：
#     %runfile '...下载器.py' -- light        # 轻量骨架：全部 ted.xlsx + tables（≈百MB级）
#     %runfile '...下载器.py' -- full         # 全量（含 csv.zip，数十 GB，先看清单再定）
#     %runfile '...下载器.py' -- drug cisapride        # 只下某药（全实验室）
#     %runfile '...下载器.py' -- lab 3 drug cisapride  # 只下某实验室某药
#   加 --figures 连 figures png 一起下（light 默认不含）
#
# 落盘：本脚本同目录 CiPA全药库\lab_N\<phase>\<drug>\...（保持 OSF 目录结构）
# 清单：CiPA全药库\osf_full_manifest_2026-09-14.json（路径/大小/sha256/下载链接）
# 核验：每个文件下完即算 sha256 对清单；已有文件尺寸+哈希一致则跳过（断点续传）。
# ============================================================================
import json, os, sys, hashlib, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "CiPA全药库")
MANIFEST = os.path.join(ROOT, "osf_full_manifest_2026-09-14.json")
NODE = "a6k5t"
API = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/"
DATA_FID = "689b65f0e0339f07a513c4d6"   # data/ 顶层（2026-09-14 核实在案）
UA = {"User-Agent": "cipa-fetch/1.0"}


def get_json(url, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                wait = float(ra) if ra else 10.0 + 5 * k
                time.sleep(wait)
            elif k == tries - 1:
                raise
            else:
                time.sleep(2 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise IOError(f"429 限流未缓解: {url}")


def children(fid):
    url = API + fid + "/"
    out = []
    while url:
        d = get_json(url)
        for f in d["data"]:
            a = f["attributes"]
            out.append({"id": f["id"], "name": a["name"], "kind": a["kind"],
                        "size": a.get("size"),
                        "sha256": (a.get("extra") or {}).get("hashes", {}).get("sha256"),
                        "dl": (f.get("links") or {}).get("download")})
        url = (d["links"] or {}).get("next")
    return out


def crawl_drug(lab, phase, drug_name, drug_fid):
    """递归枚举一个药物目录的全部文件。"""
    files = []
    stack = [(f"{lab}/{phase}/{drug_name}", drug_fid)]
    while stack:
        path, did = stack.pop()
        for it in children(did):
            p = f"{path}/{it['name']}"
            if it["kind"] == "folder":
                stack.append((p, it["id"]))
            else:
                files.append({"path": p, "size": it["size"], "sha256": it["sha256"], "dl": it["dl"]})
    # 药物目录旁的同名 html 概要（在 phase 层），由调用方补
    return files


def crawl_all():
    os.makedirs(ROOT, exist_ok=True)
    done, all_files = set(), []
    if os.path.exists(MANIFEST):  # 断点续爬：已入库的药物目录跳过
        old = json.load(open(MANIFEST, encoding="utf-8"))
        all_files = old.get("files", [])
        done = {"/".join(f["path"].split("/")[:3]) for f in all_files}
        print(f"续爬：已有 {len(done)} 药物目录、{len(all_files)} 文件", flush=True)
    labs = children(DATA_FID)
    jobs = []
    for lb in labs:
        if lb["kind"] != "folder":
            continue
        for ph in children(lb["id"]):
            if ph["kind"] != "folder":
                continue
            for dr in children(ph["id"]):
                if dr["kind"] == "folder":
                    key = f"{lb['name']}/{ph['name']}/{dr['name']}"
                    if key not in done:
                        jobs.append((lb["name"], ph["name"], dr["name"], dr["id"]))
    print(f"待枚举药物目录: {len(jobs)}（跳过已完成 {len(done)}）", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(crawl_drug, *j): j for j in jobs}
        for i, fu in enumerate(futs):
            j = futs[fu]
            try:
                fs = fu.result()
                all_files.extend(fs)
                print(f"  [{i + 1}/{len(jobs)}] {'/'.join(j[:3])}: {len(fs)} 文件", flush=True)
            except Exception as e:
                print(f"  [{i + 1}/{len(jobs)}] {'/'.join(j[:3])}: 失败 {e}", flush=True)
            json.dump({"node": NODE, "crawled": time.strftime("%Y-%m-%d %H:%M"),
                       "files": all_files},
                      open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(f["size"] or 0 for f in all_files)
    print(f"清单落盘: {MANIFEST}\n  文件 {len(all_files)}，合计 {tot / 1e9:.2f} GB", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def want(path, tier, figures):
    """light: ted.xlsx + tables/*.xlsx；full: 全部。--figures 追加 png。"""
    base = os.path.basename(path).lower()
    if tier == "full":
        return True
    if base == "ted.xlsx" or ("/tables/" in path.replace("\\", "/") and base.endswith(".xlsx")):
        return True
    if figures and base.endswith(".png"):
        return True
    return False


def download(tier, drug=None, lab=None, figures=False):
    mf = json.load(open(MANIFEST, encoding="utf-8"))
    todo = []
    for f in mf["files"]:
        p = f["path"]
        if drug and f"/{drug}/" not in p.replace("\\", "/"):
            continue
        if lab and not p.startswith(f"lab_{lab}/"):
            continue
        if not want(p, tier, figures):
            continue
        todo.append(f)
    tot = sum(f["size"] or 0 for f in todo)
    print(f"待下 {len(todo)} 文件，合计 {tot / 1e9:.2f} GB", flush=True)
    ok = skip = fail = 0
    for i, f in enumerate(todo):
        dst = os.path.join(ROOT, f["path"].replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and f["size"] and os.path.getsize(dst) == f["size"]:
            if not f["sha256"] or sha256_of(dst) == f["sha256"]:
                skip += 1
                continue
        url = f["dl"]
        good = False
        for k in range(4):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "wb") as w:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        w.write(chunk)
                if f["sha256"] and sha256_of(dst + ".part") != f["sha256"]:
                    raise IOError("sha256 不符")
                os.replace(dst + ".part", dst)
                good = True
                break
            except Exception as e:
                print(f"    重试{k + 1} {f['path']}: {e}", flush=True)
                time.sleep(2 * (k + 1))
        if good:
            ok += 1
        else:
            fail += 1
            print(f"  !! 失败 {f['path']}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  进度 {i + 1}/{len(todo)}（新下 {ok} 跳过 {skip} 失败 {fail}）", flush=True)
    print(f"完成: 新下 {ok} 跳过 {skip} 失败 {fail}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "crawl":
        crawl_all()
    else:
        tier = args[0]
        drug = None
        lab = None
        figures = "--figures" in args
        if "drug" in args:
            drug = args[args.index("drug") + 1]
        if "lab" in args:
            lab = args[args.index("lab") + 1]
        download(tier, drug, lab, figures)
