"""Convert preds/livecell/<method>/*.npz label maps into a COCO prediction json (same format as flow_comparison)."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import glob, json, sys, numpy as np, scipy.ndimage as ndi
from pycocotools import mask as MU
M = sys.argv[1]; DS = sys.argv[2] if len(sys.argv) > 2 else "livecell"; out = []
CUTS = __import__("json").load(open(f"{WORK}/results/val_cutoffs.json"))
for f in sorted(glob.glob(f"{WORK}/preds/{DS}/{M}/*.npz")):
    z = np.load(f); lab = z["lab"].astype(np.int32); sc = z["scores"]; iid = int(f.rsplit("__", 1)[1][:-4])
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        m = np.zeros(lab.shape, np.uint8, order="F"); m[sl] = lab[sl] == k
        r = MU.encode(m); r["counts"] = r["counts"].decode()
        out.append({"image_id": iid, "category_id": 1, "segmentation": r, "bbox": [float(x) for x in MU.toBbox(r)],
                    "score": float(sc[k - 1]) if len(sc) >= k else 1.0})
json.dump(out, open(f"{WORK}/preds/{DS}_{M}.json", "w")); print(M, len(out), "instances")
