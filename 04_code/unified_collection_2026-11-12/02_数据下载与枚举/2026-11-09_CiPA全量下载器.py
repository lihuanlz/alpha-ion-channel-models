# 2026-11-09_CiPA全量下载器.py
# ============================================================================
# CiPA a6k5t 全节点下载器 v2（OSF 源站直取，断点续传 + sha256 核验 + 预算自停）
#
# 与 2026-09-14 旧下载器的差异：
#   - 全新清单 osf_full_manifest_2026-11-09.json（旧清单只爬到 lab_3，弃用）
#   - 下载改为 12 线程并发（旧版串行）
#   - 支持 --budget N：N 秒后停止派生新任务，在途任务收尾后干净退出，
#     适配前台 ~290s 硬限的分段续跑（重跑即续传：尺寸+sha256 一致则跳过）
#
# 用法：
#   python 2026-11-09_CiPA全量下载器.py crawl [--budget N] [--only lab_3/hERG/dofetilide]
#   python 2026-11-09_CiPA全量下载器.py download [--tier full|light] [--lab 3] [--drug dofetilide] [--budget N]
#   python 2026-11-09_CiPA全量下载器.py status
#
# 落盘：本脚本同目录 CiPA全药库_全量_2026-11-09\lab_N\<phase>\<drug>\...（保持 OSF 目录结构）
# ============================================================================
import json, os, sys, hashlib, time, threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "CiPA全药库_全量_2026-11-09")
MANIFEST = os.path.join(ROOT, "osf_full_manifest_2026-11-09.json")
NODE = "a6k5t"
DATA_FID = "689b65f0e0339f07a513c4d6"   # data/ 顶层（2026-09-14 核实在案，2026-11-09 复核不变）
API = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/"
UA = {"User-Agent": "cipa-fetch/2.0"}

T0 = time.time()
BUDGET = None  # 秒；None=不限
ABORT_OK = False  # 仅 worker 阶段允许 BudgetAbort；前置枚举阶段不中止

def over_budget():
    return BUDGET is not None and (time.time() - T0) > BUDGET

def get_json(url, tries=10):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if ABORT_OK and over_budget() and k >= 1:
                raise BudgetAbort()
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                time.sleep(float(ra) if ra else 10.0 + 5 * k)
            elif k == tries - 1:
                raise
            else:
                time.sleep(2 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise IOError(f"不可恢复: {url}")

def children(fid):
    url = API + fid + "/?page%5Bsize%5D=200"
    out = []
    while url:
        d = get_json(url)
        for f in d["data"]:
            a = f["attributes"]
            out.append({"id": f["id"], "name": a["name"], "kind": a["kind"],
                        "size": a.get("size"),
                        "sha256": (a.get("extra") or {}).get("hashes", {}).get("sha256"),
                        "dl": (f.get("links") or {}).get("download")})
        url = (d.get("links") or {}).get("next")
    return out

class BudgetAbort(Exception):
    pass

def crawl_drug(lab, phase, drug_name, drug_fid):
    files = []
    stack = [(f"{lab}/{phase}/{drug_name}", drug_fid)]
    while stack:
        if over_budget():
            raise BudgetAbort()
        path, did = stack.pop()
        for it in children(did):
            p = f"{path}/{it['name']}"
            if it["kind"] == "folder":
                stack.append((p, it["id"]))
            else:
                files.append({"path": p, "size": it["size"], "sha256": it["sha256"], "dl": it["dl"]})
    return files

_lock = threading.Lock()

def crawl(only=None):
    os.makedirs(ROOT, exist_ok=True)
    all_files, done = [], set()
    if os.path.exists(MANIFEST):
        old = json.load(open(MANIFEST, encoding="utf-8"))
        all_files = old.get("files", [])
        done = {"/".join(f["path"].split("/")[:3]) for f in all_files}
        print(f"续爬：已有 {len(done)} 药物目录、{len(all_files)} 文件", flush=True)
    DIRS_CACHE = os.path.join(ROOT, "osf_dirs_cache_2026-11-09.json")
    if os.path.exists(DIRS_CACHE):
        dirs = [tuple(x) for x in json.load(open(DIRS_CACHE, encoding="utf-8"))]
    else:
        dirs = []
        for lb in children(DATA_FID):
            if lb["kind"] != "folder":
                continue
            for ph in children(lb["id"]):
                if ph["kind"] != "folder":
                    continue
                for dr in children(ph["id"]):
                    if dr["kind"] == "folder":
                        dirs.append((lb["name"], ph["name"], dr["name"], dr["id"]))
        json.dump(dirs, open(DIRS_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    jobs = [j for j in dirs
            if "/".join(j[:3]) not in done and (not only or "/".join(j[:3]) == only)]
    print(f"待枚举药物目录: {len(jobs)}（跳过已完成 {len(done)}）", flush=True)
    global ABORT_OK
    ABORT_OK = True
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {}
        for j in jobs:
            if over_budget():
                print("预算到点，停止派生新爬取任务", flush=True)
                break
            futs[ex.submit(crawl_drug, *j)] = j
        for i, fu in enumerate(futs):
            j = futs[fu]
            try:
                fs = fu.result()
                with _lock:
                    all_files.extend(fs)
                print(f"  [{i + 1}/{len(futs)}] {'/'.join(j[:3])}: {len(fs)} 文件", flush=True)
            except BudgetAbort:
                print(f"  [{i + 1}/{len(futs)}] {'/'.join(j[:3])}: 预算中止（下段重爬）", flush=True)
            except Exception as e:
                print(f"  [{i + 1}/{len(futs)}] {'/'.join(j[:3])}: 失败 {e}", flush=True)
            json.dump({"node": NODE, "crawled": time.strftime("%Y-%m-%d %H:%M"),
                       "files": all_files},
                      open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(f["size"] or 0 for f in all_files)
    print(f"清单落盘: {MANIFEST}\n  累计 {len(all_files)} 文件，合计 {tot / 1e9:.2f} GB", flush=True)

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def want(path, tier):
    if tier == "full":
        return True
    base = os.path.basename(path).lower()
    return base == "ted.xlsx" or ("/tables/" in path.replace("\\", "/") and base.endswith(".xlsx"))

_stat = {"ok": 0, "skip": 0, "fail": 0, "abort": 0, "bytes": 0}
VERIFIED = {}
VCACHE = os.path.join(ROOT, "_verified.json")

def load_verified():
    global VERIFIED
    if os.path.exists(VCACHE):
        try:
            VERIFIED = json.load(open(VCACHE, encoding="utf-8"))
        except Exception:
            VERIFIED = {}

def save_verified():
    try:
        json.dump(VERIFIED, open(VCACHE, "w", encoding="utf-8"))
    except Exception:
        pass

def fetch_one(f):
    if over_budget():
        return "budget"
    dst = os.path.join(ROOT, f["path"].replace("/", os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if f["path"] in VERIFIED and os.path.exists(dst) and os.path.getsize(dst) == (f["size"] or 0):
        with _lock:
            _stat["skip"] += 1
        return "skip"
    if os.path.exists(dst) and f["size"] and os.path.getsize(dst) == f["size"]:
        if not f["sha256"] or sha256_of(dst) == f["sha256"]:
            VERIFIED[f["path"]] = f["size"] or 0
            with _lock:
                _stat["skip"] += 1
            return "skip"
        os.remove(dst)
    part = dst + ".part"
    for k in range(8):
        if over_budget():
            with _lock:
                _stat["abort"] += 1
            return "abort"
        try:
            pos = os.path.getsize(part) if os.path.exists(part) else 0
            if f["size"] and pos >= f["size"]:
                pass  # 已完整，直接进哈希校验
            else:
                headers = dict(UA)
                mode = "wb"
                if pos:
                    headers["Range"] = f"bytes={pos}-"
                    mode = "ab"
                req = urllib.request.Request(f["dl"], headers=headers)
                with urllib.request.urlopen(req, timeout=120) as r:
                    if pos and getattr(r, "status", 200) != 206:
                        mode = "wb"  # 服务器忽略 Range，整文件重来
                    with open(part, mode) as w:
                        while True:
                            if over_budget():
                                raise BudgetAbort()
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            w.write(chunk)
            if f["sha256"] and sha256_of(part) != f["sha256"]:
                try: os.remove(part)
                except OSError: pass
                raise IOError("sha256 不符（已删残缺件）")
            os.replace(part, dst)
            VERIFIED[f["path"]] = f["size"] or 0
            with _lock:
                _stat["ok"] += 1
                _stat["bytes"] += f["size"] or 0
            return "ok"
        except BudgetAbort:
            with _lock:
                _stat["abort"] += 1
            return "abort"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = e.headers.get("Retry-After")
                wait = min(90.0, float(ra) if ra else 15.0 + 10 * k)
                print(f"    429限流 {wait:.0f}s后重试 {f['path']}", flush=True)
                time.sleep(wait)
            else:
                print(f"    重试{k + 1} {f['path']}: {e}", flush=True)
                time.sleep(3 * (k + 1))
        except Exception as e:
            print(f"    重试{k + 1} {f['path']}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    with _lock:
        _stat["fail"] += 1
    print(f"  !! 失败 {f['path']}", flush=True)
    return "fail"

def autodetect_proxy():
    """探测本机代理（v2rayN 10809 / Clash 7890 等），通则启用。返回代理地址或 None。"""
    for px in ("http://127.0.0.1:10809", "http://127.0.0.1:10808",
               "http://127.0.0.1:7890", "http://127.0.0.1:7897"):
        try:
            req = urllib.request.Request("https://api.osf.io/v2/", headers=UA)
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler({"http": px, "https": px}))
            with opener.open(req, timeout=5) as r:
                if r.status == 200:
                    return px
        except Exception:
            pass
    return None

def download(tier="full", lab=None, drug=None, workers=32, until_done=False, proxy=None):
    mf = json.load(open(MANIFEST, encoding="utf-8"))
    load_verified()
    px = proxy or autodetect_proxy()
    if px:
        os.environ["HTTP_PROXY"] = px
        os.environ["HTTPS_PROXY"] = px
        print(f"代理: {px}", flush=True)
    else:
        print("代理: 无（直连）", flush=True)
    todo = []
    for f in mf["files"]:
        p = f["path"].replace("\\", "/")
        if drug and f"/{drug}/" not in p:
            continue
        if lab and not p.startswith(f"lab_{lab}/"):
            continue
        if not want(p, tier):
            continue
        todo.append(f)
    tot = sum(f["size"] or 0 for f in todo)
    print(f"目标 {len(todo)} 文件，合计 {tot / 1e9:.2f} GB（tier={tier}，并发 {workers}）", flush=True)
    for pass_no in range(1, 61):
        _stat.update({"ok": 0, "skip": 0, "fail": 0, "abort": 0, "bytes": 0})
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = []
            for f in todo:
                if over_budget():
                    break
                futs.append(ex.submit(fetch_one, f))
            for i, fu in enumerate(futs):
                try:
                    fu.result()
                except BudgetAbort:
                    pass
                except Exception as e:
                    print(f"  !! 任务异常: {e}", flush=True)
                n = i + 1
                if n % 200 == 0 or n == len(futs):
                    save_verified()
                    el = time.time() - t0
                    mb = _stat["bytes"] / 1e6
                    print(f"  进度 {n}/{len(futs)}（新下 {_stat['ok']} 跳过 {_stat['skip']} "
                          f"失败 {_stat['fail']} 预算中止 {_stat['abort']}）"
                          f" 本段 {mb:.0f} MB / {el:.0f}s = {mb/el:.1f} MB/s", flush=True)
        save_verified()
        print(f"第 {pass_no} 趟结束: 新下 {_stat['ok']} 跳过 {_stat['skip']} 失败 {_stat['fail']} "
              f"预算中止 {_stat['abort']}，新下字节 {_stat['bytes']/1e9:.2f} GB", flush=True)
        if not until_done:
            break
        if _stat["fail"] == 0 and _stat["abort"] == 0 and _stat["ok"] == 0:
            print("全部在盘，收工。", flush=True)
            break
        if over_budget():
            break
        if px is None:
            px2 = autodetect_proxy()
            if px2:
                os.environ["HTTP_PROXY"] = px2
                os.environ["HTTPS_PROXY"] = px2
                px = px2
                print(f"检测到代理上线: {px}", flush=True)
        print(f"10s 后补传下一趟（剩余失败 {_stat['fail']}）", flush=True)
        time.sleep(10)

def status():
    mf = json.load(open(MANIFEST, encoding="utf-8"))
    files = mf["files"]
    tot = sum(f["size"] or 0 for f in files)
    have = have_b = 0
    bad = []
    for f in files:
        dst = os.path.join(ROOT, f["path"].replace("/", os.sep))
        if os.path.exists(dst) and f["size"] and os.path.getsize(dst) == f["size"]:
            have += 1
            have_b += f["size"] or 0
        else:
            bad.append(f["path"])
    print(f"清单 {len(files)} 文件 / {tot/1e9:.2f} GB")
    print(f"已在盘 {have} 文件 / {have_b/1e9:.2f} GB（{have/len(files)*100:.1f}%）")
    print(f"缺 {len(bad)} 文件")
    # 按实验室分账
    import collections
    per = collections.Counter(); perb = collections.Counter()
    for f in files:
        dst = os.path.join(ROOT, f["path"].replace("/", os.sep))
        ok = os.path.exists(dst) and f["size"] and os.path.getsize(dst) == f["size"]
        lb = f["path"].split("/")[0]
        if ok:
            per[lb] += 1; perb[lb] += f["size"] or 0
    for lb in sorted(set(x["path"].split("/")[0] for x in files)):
        n_all = sum(1 for x in files if x["path"].startswith(lb + "/"))
        print(f"  {lb}: {per[lb]}/{n_all} 文件, {perb[lb]/1e9:.2f} GB")

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--budget" in args:
        BUDGET = float(args[args.index("--budget") + 1])
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]
    if not args or args[0] == "crawl":
        crawl(only=only)
    elif args[0] == "download":
        tier = "full"
        if "--tier" in args:
            tier = args[args.index("--tier") + 1]
        lab = args[args.index("--lab") + 1] if "--lab" in args else None
        drug = args[args.index("--drug") + 1] if "--drug" in args else None
        workers = int(args[args.index("--workers") + 1]) if "--workers" in args else 32
        until_done = "--until-done" in args
        proxy = None
        if "--proxy" in args:
            proxy = args[args.index("--proxy") + 1]
            os.environ.pop("HTTP_PROXY", None)
            os.environ.pop("HTTPS_PROXY", None)
        download(tier=tier, lab=lab, drug=drug, workers=workers,
                 until_done=until_done, proxy=proxy)
    elif args[0] == "status":
        status()
