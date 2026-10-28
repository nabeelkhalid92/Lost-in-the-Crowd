"""Flattened version of a detector's LIVECell COCO predictions: paint all detections (score-ascending, highest on
top) into one label image per image, re-extract the visible instances with their original scores, write COCO json.
Used to measure how much COCO AP a detector loses when its overlapping masks are flattened like a flow output."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
import os
M = sys.argv[1]; src = os.environ.get("SRC", f"{FC_WORK}/predictions/full/{M}.json")
P = defaultdict(list)
for d in json.load(open(src)): P[d["image_id"]].append(d)
out = []
for n, (iid, ds) in enumerate(sorted(P.items()), 1):
    h, w = ds[0]["segmentation"]["size"]; ds = sorted(ds, key=lambda d: d["score"])
    lab = np.zeros((h, w), np.int32)
    for k, d in enumerate(ds, 1):
        r = dict(d["segmentation"]); r["counts"] = r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]
        lab[MU.decode(r).astype(bool)] = k
    for k in np.unique(lab)[1:]:
        r = MU.encode(np.asfortranarray((lab == k).astype(np.uint8))); r["counts"] = r["counts"].decode()
        out.append({"image_id": iid, "category_id": 1, "segmentation": r, "bbox": [float(x) for x in MU.toBbox(r)], "score": ds[k - 1]["score"]})
    if n % 200 == 0: print(n, len(out), flush=True)
json.dump(out, open(os.environ.get("DST", f"{WORK}/preds/livecell_{M}_flat.json"), "w")); print("done", M, len(out))
