"""Well-grouped bootstrap on LIVECell test (reviewer check: images of one well are correlated, so resampling single
images may give too narrow intervals). Same statistics as bootstrap.py (F1@0.5, F1@0.75, mean Cellpose AP@0.5) and
fusion_ci.py (fusion gains), but each resample draws whole wells (cell line + well from the file name, e.g.
A172_Phase_C7_1_... -> A172_C7) with replacement. Writes results/bootstrap_grouped.json; prints image- vs well-level CIs."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, re, numpy as np
from bootstrap import load, stats, ORDER
R = f"{WORK}"; B = 2000
def well(fn): m = re.match(r"^([A-Za-z0-9]+)_Phase_([A-Z0-9]+)_", fn); return f"{m.group(1)}_{m.group(2)}"
data = {m: load("livecell", m) for m in ORDER}
keys = sorted(set.intersection(*[set(data[m]) for m in ORDER])); W = np.array([well(k) for k in keys]); wl = np.unique(W)
members = [np.nonzero(W == w)[0] for w in wl]; rng = np.random.default_rng(0)
idx = [np.concatenate([members[j] for j in rng.integers(0, len(wl), len(wl))]) for _ in range(B)]
arr = {m: (np.array([data[m][k][0] for k in keys]), np.array([data[m][k][1] for k in keys])) for m in ORDER}
img_ci = json.load(open(f"{R}/results/bootstrap.json"))["LIVECell"]
out = {"n_img": len(keys), "n_wells": int(len(wl)), "models": {}, "paired": {}, "fusion": {}}
boots = {}
for m in ORDER:
    T, C = arr[m]; pt = stats(T, C); bs = np.array([stats(T[i], C[i]) for i in idx]); boots[m] = bs
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    out["models"][m] = {k: [float(pt[i]), float(lo[i]), float(hi[i])] for i, k in enumerate(["F1_50", "F1_75", "CPAP50"])}
    print(f"{m:13s} F1@.5 {pt[0]:.3f}  image-CI ±{(img_ci[m]['F1_50'][2]-img_ci[m]['F1_50'][1])/2:.4f}  well-CI ±{(hi[0]-lo[0])/2:.4f}")
for a, b in [("anchor_based", "vista2d"), ("anchor_based", "cpsam"), ("anchor_based", "livecell"), ("anchor_free", "vista2d"), ("cpsam", "microsam"), ("livecell", "cyto3")]:
    d = boots[a] - boots[b]; pt = stats(*arr[a]) - stats(*arr[b])
    out["paired"][f"{a}-{b}"] = {k: {"diff": float(pt[i]), "ci": [float(np.percentile(d[:, i], 2.5)), float(np.percentile(d[:, i], 97.5))]} for i, k in enumerate(["F1_50", "F1_75", "CPAP50"])}
    p = out["paired"][f"{a}-{b}"]["F1_50"]; q = img_ci["paired"][f"{a}-{b}"]["F1_50"]
    print(f"{a}-{b}: F1@.5 {p['diff']:+.4f} image-CI {np.round(q['ci'], 4)} well-CI {np.round(p['ci'], 4)}")
# fusion gains: per-image arrays are in sorted image-id order of the test GT
G = json.load(open(f"{LC_ANN}/livecell_coco_test.json"))
fn = {im["id"]: im["file_name"] for im in G["images"]}; Wf = np.array([well(fn[i]) for i in sorted(fn)])
wlf = np.unique(Wf); memf = [np.nonzero(Wf == w)[0] for w in wlf]
idxf = [np.concatenate([memf[j] for j in rng.integers(0, len(wlf), len(wlf))]) for _ in range(B)]
for tag in ["cpsam+anchor_based", "cpsam+anchor_free", "livecell+anchor_based", "vista2d+anchor_free"]:
    z = np.load(f"{R}/results/fusion/test__{tag}_perimg.npz"); T0, T1, ng = z["tff_flow"], z["tff_fused"], z["n_gt"]
    assert len(ng) == len(Wf), (tag, len(ng), len(Wf))
    o0, o1, at = z["outcome_flow"], z["outcome"], z["attr"]; im = np.repeat(np.arange(len(ng)), ng); cr = at[:, 0] >= 5
    m0 = np.bincount(im, (o0 == 4) & cr, len(ng)); m1 = np.bincount(im, (o1 == 4) & cr, len(ng)); nc = np.bincount(im, cr, len(ng))
    def f1(T, k): S = T.sum(0)[k]; return 2 * S[0] / (2 * S[0] + S[1] + S[2])
    d50 = np.array([f1(T1[b], 0) - f1(T0[b], 0) for b in idxf]); dc = np.array([(m1[b].sum() - m0[b].sum()) / max(nc[b].sum(), 1) * 100 for b in idxf])
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out["fusion"][tag] = {"dF1_50": float(f1(T1, 0) - f1(T0, 0)), "dF1_50_ci": ci(d50), "d_crowd_miss_ci": ci(dc)}
    print(f"fusion {tag}: dF1@.5 {out['fusion'][tag]['dF1_50']:+.4f} well-CI {np.round(ci(d50), 4)}  d crowded-miss well-CI {np.round(ci(dc), 1)}")
print(len(keys), "images in", len(wl), "wells")
json.dump(out, open(f"{R}/results/bootstrap_grouped.json", "w"), indent=1)
