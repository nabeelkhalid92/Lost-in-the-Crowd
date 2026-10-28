"""Best achievable result for any one-label-per-pixel model on ISBI 2014: the annotation itself, flattened so that each
overlap pixel belongs to one cell (rule 'small': the smaller cell on top, as used for training Cellpose-SAM; rule
'large': the larger cell on top). Written as a prediction file (score 1) and scored with eval_isbi.py like the models.
usage: flat_bound.py <data_dir> <split> <rule> <out.json>"""
import json, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
D, SPLIT, RULE, OUT = sys.argv[1:5]
J = json.load(open(f"{D}/{SPLIT}.json")); A = defaultdict(list)
for a in J["annotations"]: A[a["image_id"]].append(a)
res = []
for im in J["images"]:
    ms = [MU.decode({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode()}).astype(bool) for a in A[im["id"]]]
    lab = np.zeros(ms[0].shape, np.int32)
    order = np.argsort([-m.sum() for m in ms]) if RULE == "small" else np.argsort([m.sum() for m in ms])   # painted last = on top
    for j in order: lab[ms[j]] = j + 1
    for j in range(len(ms)):
        m = lab == j + 1
        if not m.any(): continue
        r = MU.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode()
        res.append({"image_id": im["id"], "category_id": 1, "segmentation": r, "score": 1.0, "bbox": [float(x) for x in MU.toBbox(r)]})
json.dump(res, open(OUT, "w")); print(RULE, len(res), "instances")
