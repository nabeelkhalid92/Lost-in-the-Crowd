"""Direct test of the overlap limitation: of the pixels that belong to two or more annotated cells (overlap pixels),
which share is covered by two or more predicted masks? A model with one label per pixel cannot exceed 0. Also the share
of overlap pixels covered by at least one prediction, and, for annotated cells with >= 15% overlap that a prediction
matches (IoU >= 0.5), the share of the cell's overlap region included in its matched mask (completeness).
LIVECell test, every 3rd image. Writes results/overlap_pixels.json."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
from pycocotools.coco import COCO
R = f"{WORK}"
G = COCO(f"{LC_ANN}/livecell_coco_test.json"); ids = sorted(G.imgs)[::3]
RUNS = [("Cellpose-SAM default", f"{R}/preds/livecell_cpsam.json", None), ("Cellpose-SAM relaxed", f"{R}/preds/livecell_sweep_cpsam.json", None),
        ("CenterMask2 0.5", f"{R}/preds/livecell_cap3000_anchor_free.json", 0.5), ("CenterMask2 0.6", f"{R}/preds/livecell_cap3000_anchor_free.json", 0.6),
        ("Cascade 0.3", f"{R}/preds/livecell_cap3000_anchor_based.json", 0.3), ("Cascade 0.4", f"{R}/preds/livecell_cap3000_anchor_based.json", 0.4)]
dec = lambda r: MU.decode(r).astype(bool)
gtm = {}
for i in ids:
    anns = G.loadAnns(G.getAnnIds(imgIds=[i]))
    if len(anns) >= 2: gtm[i] = anns
out, cache = {}, {}
for name, f, t in RUNS:
    if f not in cache:
        P = defaultdict(list)
        for d in json.load(open(f)):
            if d["image_id"] in gtm: P[d["image_id"]].append(d)
        cache = {f: P}
    P = cache[f]; c = defaultdict(float)
    for i, anns in gtm.items():
        gm = np.stack([G.annToMask(a).astype(bool) for a in anns]); cnt = gm.sum(0); ovp = cnt >= 2
        if not ovp.any(): continue
        ps = [d for d in P.get(i, []) if t is None or d["score"] >= t]
        rl = [{"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]} for d in ps]
        pc = np.zeros(cnt.shape, np.int16)
        for r in rl: pc += dec(r)
        c["ov_px"] += ovp.sum(); c["ov_px_any"] += (ovp & (pc >= 1)).sum(); c["ov_px_double"] += (ovp & (pc >= 2)).sum()
        if not rl: continue
        grl = [G.annToRLE(a) for a in anns]; iou = MU.iou(grl, rl, [0] * len(rl))
        for g in range(len(anns)):
            own_ov = gm[g] & ovp; frac = own_ov.sum() / max(gm[g].sum(), 1)
            if frac < 0.15: continue
            c["ov_cells"] += 1; j = iou[g].argmax()
            if iou[g, j] < 0.5: continue
            c["ov_cells_found"] += 1; c["ov_region"] += own_ov.sum(); c["ov_region_in_mask"] += (own_ov & dec(rl[j])).sum()
    r = {k: float(v) for k, v in c.items()}
    r["double_cover_pct"] = 100 * r["ov_px_double"] / r["ov_px"]; r["any_cover_pct"] = 100 * r["ov_px_any"] / r["ov_px"]
    r["overlap_completeness_pct"] = 100 * r["ov_region_in_mask"] / max(r["ov_region"], 1); out[name] = r
    print(f"{name:22s} overlap pixels covered twice {r['double_cover_pct']:5.1f}% | covered at all {r['any_cover_pct']:5.1f}% | found overlapping cells: own overlap region inside its mask {r['overlap_completeness_pct']:5.1f}%", flush=True)
json.dump(out, open(f"{R}/results/overlap_pixels.json", "w"), indent=1)
