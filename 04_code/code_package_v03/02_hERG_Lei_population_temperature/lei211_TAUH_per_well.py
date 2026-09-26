# lei211_TAUH_逐孔.py
# 与 lei211_TAUH_群体表.py 同一 DoE 逻辑，但输出逐孔 τ_rec 值（供跨批合并）
# 用法: BATCH=<batch> python lei211_TAUH_逐孔.py
import os, json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "数据", "Lei全量")
DT_I = 2e-4
SIN_V = [-140, -120, -100, -80, -60, -40, -20]

BATCH = os.environ.get("BATCH", "herg37oc3")
WELLS = [l.strip() for l in open(os.path.join(DATA, "qc", f"selected-{BATCH}.txt"), encoding="utf-8")
         if l.strip() and not l.startswith("#")]

d = np.genfromtxt(os.path.join(DATA, "protocol", "protocol-sinactiv.csv"),
                  delimiter=",", skip_header=1)
V10 = d[:, 1:]
def segments(v10):
    r = np.round(v10).astype(int)
    edges = np.where(np.diff(r) != 0)[0] + 1
    return [(int(r[s[0]]), s[0], s[-1] + 1) for s in np.split(np.arange(len(v10)), edges)]
TEST_IDX = {}
for sw, V in enumerate(SIN_V):
    for vv, a, b in segments(V10[:, sw]):
        if vv == V and (b - a) * 1e-4 >= 0.4:
            TEST_IDX[sw] = (a // 2, b // 2)
            break

tt = np.arange(750) * DT_I
GRID = []
for tr in np.exp(np.linspace(np.log(0.001), np.log(0.08), 22)):
    for td in np.exp(np.linspace(np.log(max(0.004, tr * 1.5)), np.log(2.0), 26)):
        GRID.append((tr, td))
PINV = []
for tr, td in GRID:
    x = np.exp(-tt / td) - np.exp(-tt / tr)
    X = np.column_stack([np.ones(750), x])
    PINV.append(np.linalg.pinv(X))
PINV = np.array(PINV)

per_cell = {}
for w in WELLS:
    In = np.genfromtxt(os.path.join(DATA, "sinactiv" if BATCH == "herg25oc1" else os.path.join(BATCH, "sinactiv"), f"{BATCH}-sinactiv-{w}.csv"),
                       delimiter=",", skip_header=1)
    if In.ndim == 1:
        In = In[:, None]
    cell = {}
    for sw, V in enumerate(SIN_V):
        if sw not in TEST_IDX:
            continue
        a, b = TEST_IDX[sw]
        seg = In[a: a + 750, sw]
        if len(seg) < 750:
            continue
        sig = float(np.std(In[: int(0.09 / DT_I), sw]))
        best = None
        for gi, (tr, td) in enumerate(GRID):
            sol = PINV[gi] @ seg
            pred_x = sol[1] * (np.exp(-tt / td) - np.exp(-tt / tr)) + sol[0]
            sse = float(np.sum((seg - pred_x) ** 2))
            if best is None or sse < best[0]:
                best = (sse, tr, td, sol[1])
        if best is not None and abs(best[3]) >= 4 * sig:
            cell[str(V)] = best[1]
    per_cell[w] = cell
    print(w, {k: round(v * 1000, 2) for k, v in cell.items()}, flush=True)

fp = os.path.join(ROOT, f"lei211_tauh_逐孔_{BATCH}.json")
json.dump(per_cell, open(fp, "w", encoding="utf-8"), ensure_ascii=False)
print("saved", fp)
