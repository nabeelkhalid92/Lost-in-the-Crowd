"""COCO mask AP (LIVECell protocol: maxDets 2000, useCats 0, LIVECell area ranges) on any COCO-format test set,
overall and per group (image field cell_line / group / difficulty when present). Reuses evaluate_partition of
flow_comparison/code/score_coco_v3.py. usage: score_coco_generic.py --pred P --gt G --tag T"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, json, os, sys
from collections import defaultdict
sys.path.insert(0, f"{REPO}"); sys.argv_backup = sys.argv
import score_coco_v3 as V3
from pycocotools.coco import COCO
ap = argparse.ArgumentParser(); ap.add_argument("--pred", required=True); ap.add_argument("--gt", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--out", default=f"{WORK}/results/coco_v3"); a = ap.parse_args()
G = COCO(a.gt); preds = json.load(open(a.pred))
for p in preds: p.pop("bbox", None)
ids = sorted(G.imgs); res = {"tag": a.tag, "overall": V3.evaluate_partition(G, preds, ids, "overall"), "per_group": {}}
print(f"overall mAP={res['overall']['mAP']:.4f} AP50={res['overall']['AP50']:.4f} AP75={res['overall']['AP75']:.4f} n_img={len(ids)}", flush=True)
for field in ["cell_line", "group", "difficulty"]:
    by = defaultdict(list)
    for iid in ids:
        v = G.imgs[iid].get(field)
        if v is not None: by[str(v)].append(iid)
    if len(by) < 2: continue
    for g, gi in sorted(by.items()):
        s = V3.evaluate_partition(G, preds, gi, g); res["per_group"][f"{field}:{g}"] = s
        print(f"  {field}:{g:<16} mAP={s['mAP']:.4f} AP50={s['AP50']:.4f} n_img={len(gi)} n_gt={s['n_gt']}", flush=True)
os.makedirs(a.out, exist_ok=True); json.dump(res, open(f"{a.out}/{a.tag}_scores.json", "w"), indent=1); print("wrote", f"{a.out}/{a.tag}_scores.json")
