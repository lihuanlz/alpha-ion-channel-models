# 2026-09-14_CiPA_full_drug_library_downloader.py
# ============================================================================
# CiPA a6k5t full drug-library downloader (direct from the OSF source, resume + sha256 verification)
#
# Usage (Spyder or command line):
#   Step 1 crawl the manifest (read-only, a few minutes):
#     %runfile '...downloader.py' -- crawl
#   Step 2 download by tier:
#     %runfile '...downloader.py' -- light        # light skeleton: all ted.xlsx + tables (~hundreds of MB)
#     %runfile '...downloader.py' -- full         # everything (incl. csv.zip, tens of GB; check the manifest first)
#     %runfile '...downloader.py' -- drug cisapride        # one drug only (all labs)
#     %runfile '...downloader.py' -- lab 3 drug cisapride  # one drug at one lab only
#   add --figures to also download figures png (excluded from light by default)
#
# Saved to: CiPA全药库\lab_N\<phase>\<drug>\... next to this script (preserves the OSF directory structure)
# Manifest: CiPA全药库\osf_full_manifest_2026-09-14.json (path/size/sha256/download link)
# Verification: sha256 computed against the manifest right after each file; existing files with matching size+hash are skipped (resume).
# ============================================================================
import json, os, sys, hashlib, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "CiPA全药库")
MANIFEST = os.path.join(ROOT, "osf_full_manifest_2026-09-14.json")
NODE = "a6k5t"
API = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/"
DATA_FID = "689b65f0e0339f07a513c4d6"   # data/ top level (verified on 2026-09-14)
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
    raise IOError(f"429 rate limit not relieved: {url}")


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
    """Recursively enumerate all files of one drug directory."""
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
    # same-named html summary next to the drug directory (at phase level), added by the caller
    return files


def crawl_all():
    os.makedirs(ROOT, exist_ok=True)
    done, all_files = set(), []
    if os.path.exists(MANIFEST):  # resume crawl: skip drug directories already in the manifest
        old = json.load(open(MANIFEST, encoding="utf-8"))
        all_files = old.get("files", [])
        done = {"/".join(f["path"].split("/")[:3]) for f in all_files}
        print(f"resuming crawl: {len(done)} drug directories, {len(all_files)} files already present", flush=True)
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
    print(f"drug directories to enumerate: {len(jobs)} (skipping {len(done)} completed)", flush=True)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(crawl_drug, *j): j for j in jobs}
        for i, fu in enumerate(futs):
            j = futs[fu]
            try:
                fs = fu.result()
                all_files.extend(fs)
                print(f"  [{i + 1}/{len(jobs)}] {'/'.join(j[:3])}: {len(fs)} files", flush=True)
            except Exception as e:
                print(f"  [{i + 1}/{len(jobs)}] {'/'.join(j[:3])}: failed {e}", flush=True)
            json.dump({"node": NODE, "crawled": time.strftime("%Y-%m-%d %H:%M"),
                       "files": all_files},
                      open(MANIFEST, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tot = sum(f["size"] or 0 for f in all_files)
    print(f"manifest saved: {MANIFEST}\n  files {len(all_files)}, total {tot / 1e9:.2f} GB", flush=True)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def want(path, tier, figures):
    """light: ted.xlsx + tables/*.xlsx; full: everything. --figures appends png."""
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
    print(f"to download {len(todo)} files, total {tot / 1e9:.2f} GB", flush=True)
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
                    raise IOError("sha256 mismatch")
                os.replace(dst + ".part", dst)
                good = True
                break
            except Exception as e:
                print(f"    retry {k + 1} {f['path']}: {e}", flush=True)
                time.sleep(2 * (k + 1))
        if good:
            ok += 1
        else:
            fail += 1
            print(f"  !! failed {f['path']}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  progress {i + 1}/{len(todo)} (new {ok} skipped {skip} failed {fail})", flush=True)
    print(f"done: new {ok} skipped {skip} failed {fail}", flush=True)


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
