"""Direct evidence for the 'self-inflicted loss' claim: from the stored network outputs, compute the masks the default
post-processing discards (flow check on at 0.4 vs off), and classify each discarded mask as a real cell (IoU >= 0.5 with
an annotated cell that the default output misses), a duplicate/partial of a found cell (IoU >= 0.5 with a cell that
the default output already matches), or junk (no annotated cell). Also stores the crops used in the qualitative figure.
usage: discarded_masks.py METHOD   (env BENCH, LC_GT)"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, sys
import numpy as np, torch
from pycocotools import mask as MU
from cellpose import dynamics
R = os.environ.get("BENCH", f"{WORK}"); M = sys.argv[1]; dev = torch.device("cuda:0")
GT = os.environ.get("LC_GT", f"{LC_ANN}/livecell_coco_test.json"); J = json.load(open(GT))
from collections import defaultdict
anns = defaultdict(list); hw = {im["id"]: (im["height"], im["width"]) for im in J["images"]}
for a in J["annotations"]: anns[a["image_id"]].append(a)
CROPS = {"BV2_Phase_A4_1_02d16h00m_2.tif": (270, 90), "SHSY5Y_Phase_A10_1_03d08h00m_4.tif": (120, 540), "MCF7_Phase_H4_2_03d08h00m_1.tif": (300, 330)}
def rles(lab):
    ids = np.unique(lab); ids = ids[ids > 0]; return ids, [MU.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in ids]
def match(iou, thr=0.5):   # greedy one-to-one, returns dict pred->gt
    g, p = np.nonzero(iou >= thr); o = np.argsort(-iou[g, p]); ug, up, res = set(), set(), {}
    for k in o:
        if g[k] in ug or p[k] in up: continue
        ug.add(g[k]); up.add(p[k]); res[p[k]] = g[k]
    return res
tot = {"discarded": 0, "real_missed": 0, "dup_of_found": 0, "junk": 0, "default_n": 0, "gt_n": 0, "default_missed": 0}; n = 0; crops = {}
for im in J["images"]:
    f = f"{R}/preds/flows_test/{M}/{im['id']}.npz"
    if not os.path.exists(f): continue
    z = np.load(f); dP, cp = z["dP"].astype(np.float32), z["cellprob"].astype(np.float32)
    kw = dict(niter=200, cellprob_threshold=0.0, min_size=15)
    lab0 = np.asarray(dynamics.compute_masks(dP, cp, flow_threshold=0.0, device=dev, **kw)).astype(np.int32)   # check off
    err = np.asarray(dynamics.flow_error(lab0, dP, device=dev)[0]) if lab0.max() else np.zeros(0)
    keep = np.ones(lab0.max() + 1, bool); keep[0] = False; keep[1:][err > 0.4] = False          # default: discard error > 0.4
    g = [MU.frPyObjects(a["segmentation"], *hw[im["id"]]) for a in anns[im["id"]]]; g = [MU.merge(r) if isinstance(r, list) else r for r in g]
    ids, pr = rles(lab0)
    if len(pr) and len(g):
        iou = MU.iou(pr, g, [0] * len(g)).T   # [G, P]
        kept_idx = [k for k, i in enumerate(ids) if keep[i]]; disc_idx = [k for k, i in enumerate(ids) if not keep[i]]
        m_def = match(iou[:, kept_idx]) if kept_idx else {}; found_gt = set(m_def.values())
        cls = {}
        for k in disc_idx:
            gi = int(np.argmax(iou[:, k])); v = iou[gi, k]
            cls[k] = "junk" if v < 0.5 else ("dup_of_found" if gi in found_gt else "real_missed")
        for k, c in cls.items(): tot[c] += 1
        tot["discarded"] += len(disc_idx); tot["default_n"] += len(kept_idx); tot["gt_n"] += len(g); tot["default_missed"] += len(g) - len(found_gt)
        if im["file_name"] in CROPS:
            y, x = CROPS[im["file_name"]]; W = 150
            dl = np.zeros_like(lab0); cl = np.zeros_like(lab0)
            for k, c in cls.items(): dl[lab0 == ids[k]] = {"real_missed": 1, "dup_of_found": 2, "junk": 3}[c]
            crops[im["file_name"]] = {"lab_default": np.where(keep[lab0], lab0, 0)[y:y+W, x:x+W], "discard_class": dl[y:y+W, x:x+W], "lab_all": lab0[y:y+W, x:x+W]}
    n += 1
    if n % 200 == 0: print(n, tot, flush=True)
tot["n_img"] = n; os.makedirs(f"{R}/results/discarded", exist_ok=True)
json.dump(tot, open(f"{R}/results/discarded/{M}.json", "w"), indent=1)
np.savez_compressed(f"{R}/results/discarded/{M}_crops.npz", **{f"{k}__{kk}": v for k, d in crops.items() for kk, v in d.items()})
print("FINAL", M, tot, "real_missed/discarded=%.3f" % (tot["real_missed"] / max(tot["discarded"], 1)), "real_missed/default_missed=%.3f" % (tot["real_missed"] / max(tot["default_missed"], 1)))
