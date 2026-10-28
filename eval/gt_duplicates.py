"""Near-duplicate annotations in LIVECell test GT: pairs of instances of the same image with mask IoU > 0.5 (two
different cells essentially never overlap that much). For each image, greedily keep one instance per duplicate group
(the first in annotation order). Writes results/gt_duplicates.json with per-line counts and the kept annotation ids."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, numpy as np
from collections import Counter, defaultdict
from pycocotools.coco import COCO
from pycocotools import mask as MU
R = f"{WORK}"
G = COCO(f"{LC_ANN}/livecell_coco_test.json")
stats = defaultdict(Counter); drop = []
for iid in G.imgs:
    aids = G.getAnnIds(imgIds=[iid]); line = G.imgs[iid]["file_name"].split("_")[0]; stats[line]["cells"] += len(aids)
    if len(aids) < 2: continue
    r = [G.annToRLE(a) for a in G.loadAnns(aids)]
    iou = MU.iou(r, r, [0] * len(r)); np.fill_diagonal(iou, 0)
    for thr, k in [(0.5, "pairs_iou_gt_0.5"), (0.9, "pairs_iou_gt_0.9")]: stats[line][k] += int((np.triu(iou) > thr).sum())
    removed = np.zeros(len(aids), bool)
    for i in range(len(aids)):
        if removed[i]: continue
        dup = np.nonzero((iou[i] > 0.5) & ~removed)[0]; dup = dup[dup > i]; removed[dup] = True
    stats[line]["removed"] += int(removed.sum()); stats[line]["images_with_dups"] += int(removed.any())
    drop += [int(a) for a, rm in zip(aids, removed) if rm]
tot = Counter()
for v in stats.values(): tot.update(v)
out = {"per_line": {k: dict(v) for k, v in sorted(stats.items())}, "total": dict(tot), "dropped_ann_ids": drop}
json.dump(out, open(f"{R}/results/gt_duplicates.json", "w"))
for k, v in sorted(stats.items()): print(k, dict(v), f"removed {100*v['removed']/v['cells']:.2f}%")
print("TOTAL", dict(tot), f"removed {100*tot['removed']/tot['cells']:.2f}%")
