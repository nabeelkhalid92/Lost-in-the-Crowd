"""Split the MISSED annotated cells of one analysis run (analyze.py outcome 4, the same definition as all miss rates in
the paper) into 'absorbed' (>= 50% of the cell covered by predicted masks, i.e. swallowed by a neighbor's mask) and
'empty' (no prediction there), for crowded cells (>= 5 neighbors) and all cells. Full LIVECell test set, GT order as in
analyze.py (sorted image ids, all annotations). Also returns per-image counts for paired bootstraps.
usage: absorbed_split.py <analysis_prefix (dataset__method__key)> <preds.json> <cutoff or none> <name>
writes results/absorbed/<name>.json"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
from pycocotools.coco import COCO
R = f"{WORK}"; PFX, P, T, NAME = sys.argv[1:5]; T = None if T == "none" else float(T)
z = np.load(f"{R}/results/analysis/{PFX}.npz", allow_pickle=True); out_, att, cnt, keys = z["outcome"], z["attr"], z["counts"].astype(int), z["keys"]
G = COCO(f"{LC_ANN}/livecell_coco_test.json"); fn2id = {G.imgs[i]["file_name"]: i for i in G.imgs}
PR = defaultdict(list)
for d in json.load(open(P)):
    if T is None or d["score"] >= T: PR[d["image_id"]].append(d)
off = 0; per = []
ids = sorted(G.imgs); assert len(ids) == len(keys)   # analyze.py iterates sorted image ids (52 file names occur twice)
for iid, k, (npred, ngt) in zip(ids, keys, cnt):
    assert G.imgs[iid]["file_name"] == str(k); o = out_[off:off + ngt]; cr = att[off:off + ngt, 0] >= 5; off += ngt
    gt = [G.annToRLE(a) for a in G.loadAnns(G.getAnnIds(imgIds=[iid]))]; assert len(gt) == ngt, (k, len(gt), ngt)
    miss = np.nonzero(o == 4)[0]; ab = np.zeros(ngt, bool)
    if len(miss) and PR[iid]:
        U = MU.merge([{"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]} for d in PR[iid]], intersect=False)
        for g in miss: ab[g] = MU.area(MU.merge([gt[g], U], intersect=True)) >= 0.5 * MU.area(gt[g])
    per.append([int(cr.sum()), int(((o == 4) & cr).sum()), int((ab & cr).sum()), int(ngt), int((o == 4).sum()), int(ab.sum())])
a = np.array(per); r = {"crowded_missed": 100 * a[:, 1].sum() / a[:, 0].sum(), "crowded_absorbed": 100 * a[:, 2].sum() / a[:, 0].sum(),
                        "crowded_empty": 100 * (a[:, 1] - a[:, 2]).sum() / a[:, 0].sum(), "all_missed": 100 * a[:, 4].sum() / a[:, 3].sum(),
                        "all_absorbed": 100 * a[:, 5].sum() / a[:, 3].sum(), "per_image": per, "keys": [str(k) for k in keys]}
print(f"{NAME:22s} crowded: missed {r['crowded_missed']:5.1f}% = absorbed {r['crowded_absorbed']:5.1f}% + empty {r['crowded_empty']:5.1f}%", flush=True)
os.makedirs(f"{R}/results/absorbed", exist_ok=True); json.dump(r, open(f"{R}/results/absorbed/{NAME}.json", "w"), default=float)
