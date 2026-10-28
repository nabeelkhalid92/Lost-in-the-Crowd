"""Crowded-cell miss rate against false positives per 1,000 cells on the full LIVECell test set, for every operating
point of each model: Cellpose-SAM over the flow-consistency threshold (0 = check off), CenterMask2 and Cascade Mask R-CNN
over the score cutoff. Defaults (flow threshold 0.4; validation-selected cutoffs) are starred. Also a paired bootstrap
(over images) of the crowded-miss difference at matched false-positive rates. Writes figures/oppoint_curve.pdf and
results/oppoint_curve.json."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = f"{WORK}"; A = f"{R}/results/analysis"
def pt(ds, m, k):
    s = json.load(open(f"{A}/{ds}__{m}.json"))[k]; z = np.load(f"{A}/{ds}__{m}__{k}.npz", allow_pickle=True)
    o, cr, cnt = z["outcome"], z["attr"][:, 0] >= 5, z["counts"].astype(int)
    img = np.repeat(np.arange(len(cnt)), cnt[:, 1])
    return {"fp": s["per_1000_gt"]["false_pos"], "miss": 100 * float((o[cr] == 4).mean()), "F1": s["F1"][0],
            "m_img": np.bincount(img, (o == 4) & cr, len(cnt)), "n_img": np.bincount(img, cr, len(cnt))}
C = {}
cp = [(0.2, ("livecell_curve_ft0.2", "flat_None")), (0.3, ("livecell_curve_ft0.3", "flat_None")), (0.4, ("livecell", "flat_None")),
      (0.5, ("livecell_curve_ft0.5", "flat_None")), (0.6, ("livecell_curve_ft0.6", "flat_None")), (0.7, ("livecell_curve_ft0.7", "flat_None")),
      (0.8, ("livecell_sweep", "flat_None")), (1.0, ("livecell_curve_ft1.0", "flat_None")), ("off", ("livecell_curve_ft0.0", "flat_None"))]
C["Cellpose-SAM"] = [(t, pt(ds, "cpsam", k)) for t, (ds, k) in cp if os.path.exists(f"{A}/{ds}__cpsam.json")]
for m, name in [("anchor_free", "CenterMask2"), ("anchor_based", "Cascade Mask R-CNN")]:
    C[name] = [(t, pt("livecell_cap3000", m, f"raw_{t}")) for t in [0.3, 0.4, 0.5, 0.6, 0.7]]
DEF = {"Cellpose-SAM": 0.4, "CenterMask2": 0.5, "Cascade Mask R-CNN": 0.3}
COL = {"Cellpose-SAM": "#3f6fb0", "CenterMask2": "#d62728", "Cascade Mask R-CNN": "#ff7f0e"}
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for name, pts in C.items():
    x = [p["fp"] for _, p in pts]; y = [p["miss"] for _, p in pts]; o = np.argsort(x)
    ax.plot(np.array(x)[o], np.array(y)[o], "-o", ms=3, lw=1.2, color=COL[name], label=name)
    for t, p in pts:
        if t == DEF[name]: ax.plot(p["fp"], p["miss"], "*", ms=11, color=COL[name], mec="k", mew=0.5)
        if name == "Cellpose-SAM" and t in (0.4, 0.8): ax.annotate(f"{t}", (p["fp"], p["miss"]), textcoords="offset points", xytext=(-12, -9) if t == 0.4 else (4, 3), fontsize=6, color=COL[name])
ax.set_xscale("log"); ax.set_xlabel("false positives per 1,000 cells", fontsize=8); ax.set_ylabel("crowded cells missed (%)", fontsize=8)
ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, frameon=False); ax.grid(alpha=0.3, lw=0.5)
fig.tight_layout(); fig.savefig(f"{R}/paper/draft/figures/oppoint_curve.pdf"); fig.savefig(f"{R}/paper/draft/figures/oppoint_curve.png", dpi=200)
# paired bootstrap over images: Cellpose-SAM default vs the detectors at their closest false-positive rate
def boot(a, b, B=2000):
    rng = np.random.RandomState(0); n = len(a["m_img"]); d = []
    for _ in range(B):
        i = rng.randint(0, n, n); d.append(100 * (a["m_img"][i].sum() / a["n_img"][i].sum() - b["m_img"][i].sum() / b["n_img"][i].sum()))
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
cpd = dict(C["Cellpose-SAM"])[0.4]; out = {"points": {n: [(str(t), p["fp"], p["miss"], p["F1"]) for t, p in v] for n, v in C.items()}, "paired": {}}
for name, t in [("CenterMask2", 0.6), ("Cascade Mask R-CNN", 0.4)]:
    p = dict(C[name])[t]; ci = boot(cpd, p)
    out["paired"][f"Cellpose-SAM default - {name} {t}"] = {"diff": cpd["miss"] - p["miss"], "ci": ci, "fp": [cpd["fp"], p["fp"]]}
    print(f"crowded missed: Cellpose-SAM default {cpd['miss']:.1f}% (FP {cpd['fp']:.0f}) - {name} {t} {p['miss']:.1f}% (FP {p['fp']:.0f}) = {cpd['miss'] - p['miss']:+.1f} {np.round(ci, 1)}")
for name, v in C.items():
    print(name, [(t, round(p["fp"]), round(p["miss"], 1), round(p["F1"], 3)) for t, p in v])
json.dump(out, open(f"{R}/results/oppoint_curve.json", "w"), indent=1)
