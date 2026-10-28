"""Runtime vs cell density from the per-image timings of the full LIVECell-test inference runs
(flow_comparison/logs, RTX 2080 Ti, batch 1, includes post-processing). One source for every method."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import re, json, numpy as np
from collections import Counter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L = f"{FC_WORK}/logs"; R = f"{WORK}"
LOGS = {"Cellpose v1": "cyto_inference.log", "Cellpose 2": "cyto2_inference.log", "Cellpose3": "cyto3_inference.log",
        "Cellpose LIVECell": "livecell_inference.log", "Cellpose-SAM": "cpsam_inference.log", "VISTA-2D": "vista2d_inference.log",
        "Cascade Mask R-CNN": "anchor_based_inference.log", "CenterMask2": "anchor_free_inference.log"}
COL = {"Cellpose v1": "#c7d4e8", "Cellpose 2": "#9fb6d8", "Cellpose3": "#6f93c6", "Cellpose-SAM": "#3f6fb0", "VISTA-2D": "#1f4e8c",
       "Cellpose LIVECell": "#7a5fa8", "Cascade Mask R-CNN": "#d1453b", "CenterMask2": "#e58a2e"}
gt = json.load(open(f"{LC_ANN}/livecell_coco_test.json"))
fid = {im["id"]: im["file_name"] for im in gt["images"]}; n = Counter(fid[a["image_id"]] for a in gt["annotations"])
rx = re.compile(r"\]\s+(\S+\.tif)\s.*?\s([\d.]+)s\s+ETA")
edges = [0, 100, 300, 700, 1500, 10**6]; lab = ["<100", "100-300", "300-700", "700-1500", ">=1500"]
out = {}
for m, f in LOGS.items():
    t = {}
    for line in open(f"{L}/{f}"):
        x = rx.search(line)
        if x: t[x.group(1)] = float(x.group(2))
    ks = [k for k in t if k in n]; tt = np.array([t[k] for k in ks]); nn = np.array([n[k] for k in ks]); b = np.digitize(nn, edges) - 1
    out[m] = {"n": len(ks), "mean": float(tt.mean()), "median": float(np.median(tt)), "binned_mean": [float(tt[b == i].mean()) for i in range(5)], "bin_n": [int((b == i).sum()) for i in range(5)]}
    print(f"{m:20s} n={len(ks)} mean {tt.mean():.2f}s  " + " ".join(f"{v:.2f}" for v in out[m]["binned_mean"]))
json.dump(out, open(f"{R}/results/speed.json", "w"), indent=1)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.4, 3.0))
for m, v in out.items():
    ax.plot(range(5), v["binned_mean"], "-o", ms=2.5, lw=1.3, color=COL[m], ls="--" if m in ("Cascade Mask R-CNN", "CenterMask2") else "-", label=m)
ax.set_yscale("log"); ax.set_xticks(range(5)); ax.set_xticklabels(lab, fontsize=6.5); ax.set_xlabel("ground-truth cells per image"); ax.set_ylabel("s / image (log)")
ax.grid(alpha=.25, which="both"); fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=5.8, frameon=False, columnspacing=1.0, handlelength=2.2)
fig.tight_layout(rect=[0, 0.2, 1, 1]); fig.savefig(f"{R}/paper/draft/figures/speed.pdf"); fig.savefig(f"{R}/paper/draft/figures/speed.png", dpi=200)
