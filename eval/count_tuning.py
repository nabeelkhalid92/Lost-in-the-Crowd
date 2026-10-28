"""Fair comparison of counting remedies on the LIVECell nuclear-count benchmark. Each remedy's free parameter is tuned
for COUNT (minimum |mean bias|) on a small tuning set, the first row of wells (one well per seeding density), and all
remedies are scored on the remaining wells: MAPE, and mean signed error in the sparsest and densest third of the
reference counts (does the remedy remove the density-dependent slope or only shift the offset?).
Remedies: Cellpose-SAM default; Cellpose-SAM with the flow threshold tuned (0.4-0.8); CenterMask2 with the cutoff tuned
(0.30-0.70); CenterMask2 at cutoff 0.5 times a scale factor fitted on the tuning wells; fusion (Cellpose-SAM + Cascade,
settings from LIVECell val); micro-SAM. Writes results/count_tuning.json."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import glob, json, os
import numpy as np
R = f"{WORK}"; FL = f"{FC_WORK}/counting_data"
FT = {0.4: "cpsam", 0.5: "cpsam_ft0.5", 0.6: "cpsam_ft0.6", 0.7: "cpsam_ft0.7", 0.8: "cpsam_relaxed"}
def flow_counts(m, cell):
    return {os.path.basename(f).split("__", 1)[1][:-4]: int(np.load(f)["lab"].max()) for f in glob.glob(f"{R}/preds/count/{m}/{cell}__*.npz")}
def det_counts(m, cell, t):
    out = {}
    for f in glob.glob(f"{R}/preds/count/{m}/{cell}__*.npz"):
        z = np.load(f); lab = z["lab"]; sc = z["scores"]; n = int(lab.max()); s_ = np.zeros(n); s_[:min(n, len(sc))] = sc[:min(n, len(sc))]
        present = np.zeros(n + 1, bool); present[np.unique(lab)] = True
        out[os.path.basename(f).split("__", 1)[1][:-4]] = int(((s_ >= t) & present[1:]).sum())
    return out
fus = json.load(open(f"{R}/results/fusion/count_fusion.json")); res = {}
for cell in ["A172", "A549"]:
    ref = {k[:-4] if k.endswith(".png") else k: float(v) for k, v in json.load(open(f"{FL}/{cell}_counts.json")).items()}
    well = lambda k: k.split("-")[0]; first = min(well(k)[0] for k in ref); tune = lambda k: well(k)[0] == first   # first row of wells: one per seeding density
    def score(C):
        ks = [k for k in C if k in ref and not tune(k)]; g = np.array([ref[k] for k in ks]); e = (np.array([C[k] for k in ks]) - g) / g * 100
        q = np.quantile(g, [1 / 3, 2 / 3]); b = np.digitize(g, q)
        return {"MAPE": float(np.abs(e).mean()), "bias": float(e.mean()), "sparse": float(e[b == 0].mean()), "dense": float(e[b == 2].mean()), "n_img": len(ks)}
    def tbias(C): ks = [k for k in C if k in ref and tune(k)]; return abs(np.mean([(C[k] - ref[k]) / ref[k] for k in ks])) if ks else 9e9
    rows = {}
    rows["Cellpose-SAM (default)"] = score(flow_counts("cpsam", cell))
    avail = {t: flow_counts(m, cell) for t, m in FT.items() if os.path.isdir(f"{R}/preds/count/{m}")}
    tb = min(avail, key=lambda t: tbias(avail[t])); rows[f"Cellpose-SAM (flow thr. {tb}, tuned)"] = score(avail[tb])
    cm = {t: det_counts("anchor_free", cell, t) for t in np.round(np.arange(0.30, 0.701, 0.02), 2)}
    tc = min(cm, key=lambda t: tbias(cm[t])); rows[f"CenterMask2 (cutoff {tc}, tuned)"] = score(cm[tc])
    c5 = cm[0.5]; kt = [k for k in c5 if k in ref and tune(k)]; f = sum(ref[k] for k in kt) / sum(c5[k] for k in kt)
    rows[f"CenterMask2 0.5 x {f:.2f} (scale fitted)"] = score({k: v * f for k, v in c5.items()})
    pi = fus[f"{cell}/cpsam+anchor_based"]["per_image"]; rows["fusion (Cellpose-SAM + Cascade)"] = score({k: c for k, _, _, c in pi})
    rows["micro-SAM"] = score(flow_counts("microsam", cell))
    res[cell] = rows; print(f"\n{cell} (held-out wells; tuning wells = row {first})")
    for k, r in rows.items(): print(f"  {k:40s} MAPE {r['MAPE']:5.1f}  bias {r['bias']:+6.1f}  sparse third {r['sparse']:+6.1f}  dense third {r['dense']:+6.1f}  (n={r['n_img']})")
json.dump(res, open(f"{R}/results/count_tuning.json", "w"), indent=1)
