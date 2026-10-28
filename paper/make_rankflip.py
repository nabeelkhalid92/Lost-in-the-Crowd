"""RQ1 figure: model ranking on LIVECell under COCO mask AP vs under F1 at IoU 0.5 (same predictions). micro-SAM is left out:
it provides no instance scores, so its COCO AP is not comparable (protocol section). Writes figures/rankflip.pdf/.png."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = f"{WORK}"; S = json.load(open(f"{R}/results/summary.json"))["livecell"]
NAME = {"cyto3": "Cellpose3", "cpsam": "Cellpose-SAM", "vista2d": "VISTA-2D", "livecell": "Cellpose LIVECell",
        "anchor_based": "Cascade Mask R-CNN", "anchor_free": "CenterMask2"}
COL = {"cyto3": "#6f93c6", "cpsam": "#3f6fb0", "vista2d": "#1f4e8c", "livecell": "#7a5fa8", "microsam": "#3a9a6a", "anchor_based": "#d1453b", "anchor_free": "#e58a2e"}
ms = [m for m in NAME if m in S]
rA = {m: i for i, m in enumerate(sorted(ms, key=lambda m: -S[m]["AP"]), 1)}
rF = {m: i for i, m in enumerate(sorted(ms, key=lambda m: -S[m]["F1_50"]), 1)}
plt.rcParams.update({"font.size": 7.5})
fig, ax = plt.subplots(figsize=(3.4, 2.6))
for m in ms:
    ax.plot([0, 1], [rA[m], rF[m]], "-o", color=COL[m], lw=2 if m.startswith("anchor") else 1.3, ms=4)
    ax.text(-0.05, rA[m], f"{NAME[m]} ({S[m]['AP']:.1f})", ha="right", va="center", fontsize=6.5, color=COL[m])
    ax.text(1.05, rF[m], f"({S[m]['F1_50']:.3f}) {NAME[m]}", ha="left", va="center", fontsize=6.5, color=COL[m])
ax.set_xlim(-1.1, 2.1); ax.invert_yaxis(); ax.axis("off")
ax.text(-0.05, 0.1, "rank by COCO mask AP", ha="right", fontsize=7, weight="bold"); ax.text(1.05, 0.1, "rank by F1 @ IoU 0.5", ha="left", fontsize=7, weight="bold")
fig.savefig(f"{R}/paper/draft/figures/rankflip.pdf", bbox_inches="tight", pad_inches=0.03); fig.savefig(f"{R}/paper/draft/figures/rankflip.png", dpi=150, bbox_inches="tight", pad_inches=0.03); print("ok")
