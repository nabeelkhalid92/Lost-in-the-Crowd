"""Fusion baselines on the LIVECell nuclear-count benchmark (reviewer check: is the gain specific to the coverage rule?).
Same data, cutoffs and well-level bootstrap as fusion_count.py; one row per (cell line, flow, detector, rule):
  cov   : add a detection (score >= t) if at most tau = 0.5 of its visible area is covered by flow instances (the paper's rule)
  nms   : add a detection if its mask IoU with every flow instance is < 0.5 (plain union + mask NMS, flow instances first)
  union : add every detection >= t (no check against the flow instances)
Cutoffs: t from the LIVECell validation grid of each rule (results/fusion/val__livecell+<det>[__rule].json).
usage: fusion_count_baselines.py [flow ...]   (default cpsam cyto3) -> results/fusion/count_fusion_baselines.json"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import glob, json, os, sys
import numpy as np
import scipy.ndimage as ndi
R = f"{WORK}"; FL = f"{FC_WORK}/counting_data"
SET = {("anchor_based", "cov"): 0.3, ("anchor_free", "cov"): 0.6, ("anchor_based", "nms"): 0.4, ("anchor_free", "nms"): 0.6,
       ("anchor_based", "union"): 0.7, ("anchor_free", "union"): 0.7}
FLOWS = sys.argv[1:] or ["cpsam", "cyto3"]

def load(m, cell, key):
    z = np.load(f"{R}/preds/count/{m}/{cell}__{key}.npz"); return z["lab"].astype(np.int32), z["scores"].astype(np.float32)

def n_added(fl, dl, ds, t, rule):
    n = dl.max()
    if n == 0: return 0
    ids = np.arange(1, n + 1); area = ndi.sum(np.ones_like(dl), dl, ids)
    sc = np.zeros(n, np.float32); sc[:min(n, len(ds))] = ds[:min(n, len(ds))]
    ok = (sc >= t) & (area >= 20)
    if rule == "cov":
        ok &= ndi.sum(fl > 0, dl, ids) / np.maximum(area, 1) <= 0.5
    elif rule == "nms":
        nf = fl.max(); m = (dl > 0) & (fl > 0)
        inter = np.bincount(dl[m].astype(np.int64) * (nf + 1) + fl[m], minlength=(n + 1) * (nf + 1)).reshape(n + 1, nf + 1)[1:, 1:]
        fa = np.bincount(fl.ravel(), minlength=nf + 1)[1:]
        iou = inter / np.maximum(area[:, None] + fa[None, :] - inter, 1) if nf else np.zeros((n, 1))
        ok &= iou.max(1) < 0.5
    return int(ok.sum())

out = {}
for cell in ["A172", "A549"]:
    ref = {k[:-4] if k.endswith(".png") else k: float(v) for k, v in json.load(open(f"{FL}/{cell}_counts.json")).items()}
    keys = sorted(os.path.basename(f).split("__", 1)[1][:-4] for f in glob.glob(f"{R}/preds/count/cpsam/{cell}__*.npz"))
    keys = [k for k in keys if k in ref]
    well = np.array([k.split("-")[0] for k in keys]); wl = np.unique(well)
    for flow in FLOWS:
        F = {k: load(flow, cell, k)[0] for k in keys}
        for (det, rule), t in SET.items():
            rows = []
            for k in keys:
                fl = F[k]; dl, ds = load(det, cell, k); nf = len(np.unique(fl)) - 1
                rows.append((ref[k], nf, nf + n_added(fl, dl, ds, t, rule)))
            a = np.array(rows, float); g = a[:, 0]; e0, e1 = (a[:, 1] - g) / g, (a[:, 2] - g) / g
            q = np.quantile(g, [1 / 3, 2 / 3]); b = np.digitize(g, q); rng = np.random.RandomState(0); d = []
            for _ in range(2000):
                idx = np.concatenate([np.nonzero(well == w)[0] for w in rng.choice(wl, len(wl))])
                d.append((np.abs(e1[idx]).mean() - np.abs(e0[idx]).mean()) * 100)
            r = {"t": t, "n": len(a), "n_wells": int(len(wl)), "MAPE_flow": float(np.abs(e0).mean() * 100), "MAPE_fused": float(np.abs(e1).mean() * 100),
                 "bias_flow": float(e0.mean() * 100), "bias_fused": float(e1.mean() * 100),
                 "dense_bias_flow": float(e0[b == 2].mean() * 100), "dense_bias_fused": float(e1[b == 2].mean() * 100),
                 "dMAPE_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
            out[f"{cell}/{flow}+{det}/{rule}"] = r
            print(f"{cell} {flow:8s}+{det:12s} {rule:5s} t={t}  MAPE {r['MAPE_flow']:5.1f} -> {r['MAPE_fused']:5.1f} {np.round(r['dMAPE_ci'], 1)}  bias {r['bias_flow']:+6.1f} -> {r['bias_fused']:+6.1f}  dense {r['dense_bias_flow']:+5.0f} -> {r['dense_bias_fused']:+5.0f}", flush=True)
tag = "" if FLOWS == ["cpsam", "cyto3"] else "_" + "_".join(FLOWS)
json.dump(out, open(f"{R}/results/fusion/count_fusion_baselines{tag}.json", "w"), indent=1)
