"""Bootstrap 95% CIs (over images, 2000 resamples) for pooled F1@0.5/0.75 and mean Cellpose-style AP@0.5 on LIVECell
and per CTC dataset, plus paired bootstrap tests for key comparisons (same image resamples for both models)."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, numpy as np
R = f"{WORK}"; rng = np.random.default_rng(0); B = 2000
cut = json.load(open(f"{R}/results/val_cutoffs.json"))
ORDER = ["cyto", "cyto2", "cyto3", "cpsam", "vista2d", "livecell", "microsam", "anchor_based", "anchor_free"]
CTC = ["PhC-C2DL-PSC", "PhC-C2DH-U373", "DIC-C2DH-HeLa", "BF-C2DL-HSC", "BF-C2DL-MuSC"]
def key(m): return f"raw_{cut[m]}" if m in cut else "flat_None"
def load(ds, m):
    if ds == "livecell" and m in cut and os.path.exists(f"{R}/results/analysis/livecell_cap3000__{m}__{key(m)}.npz"): ds = "livecell_cap3000"   # 3000-cap rerun
    z = np.load(f"{R}/results/analysis/{ds}__{m}__{key(m)}.npz", allow_pickle=True)
    return {str(k): (t, c) for k, t, c in zip(z["keys"], z["tff_img"], z["cpap_img"])}
def stats(T, C):   # T: [n, IoU, 3] per-image TP/FP/FN ; C: [n, IoU] per-image CP-AP
    S = T.sum(0); f = 2 * S[:, 0] / np.maximum(2 * S[:, 0] + S[:, 1] + S[:, 2], 1)
    return np.array([f[0], f[5], C[:, 0].mean()])
out = {}
for ds, groups in [("livecell", {"LIVECell": None}), ("ctc", {d: d for d in CTC})]:
    data = {m: load(ds, m) for m in ORDER}
    for gname, pref in groups.items():
        keys = sorted(set.intersection(*[set(k for k in data[m] if pref is None or k.startswith(pref + "_")) for m in ORDER]))
        n = len(keys); idx = rng.integers(0, n, (B, n))
        arr = {m: (np.array([data[m][k][0] for k in keys]), np.array([data[m][k][1] for k in keys])) for m in ORDER}
        boots = {}
        for m in ORDER:
            T, C = arr[m]; pt = stats(T, C)
            bs = np.array([stats(T[i], C[i]) for i in idx]); boots[m] = bs
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            out.setdefault(gname, {})[m] = {"n_img": n, "F1_50": [pt[0], lo[0], hi[0]], "F1_75": [pt[1], lo[1], hi[1]], "CPAP50": [pt[2], lo[2], hi[2]]}
        comps = [("anchor_based", "vista2d"), ("anchor_based", "cpsam"), ("anchor_based", "livecell"), ("anchor_free", "vista2d"), ("cpsam", "microsam"), ("livecell", "cyto3")]
        for a, b in comps:
            d = boots[a] - boots[b]; pt = stats(*arr[a]) - stats(*arr[b])
            out[gname].setdefault("paired", {})[f"{a}-{b}"] = {k: {"diff": float(pt[i]), "ci": [float(np.percentile(d[:, i], 2.5)), float(np.percentile(d[:, i], 97.5))],
                                                                    "p_two_sided": float(min(1, 2 * min((d[:, i] <= 0).mean(), (d[:, i] >= 0).mean())))}
                                                                for i, k in enumerate(["F1_50", "F1_75", "CPAP50"])}
        print(gname, n, "images", {m: [round(x, 3) for x in out[gname][m]["F1_50"]] for m in ORDER}, flush=True)
json.dump(out, open(f"{R}/results/bootstrap.json", "w"), indent=1, default=float)
for g in out:
    for k, v in out[g]["paired"].items(): print(g, k, {m: (round(x["diff"], 3), [round(c, 3) for c in x["ci"]], x["p_two_sided"]) for m, x in v.items()})
