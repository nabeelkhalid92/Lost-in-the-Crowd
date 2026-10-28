"""
Per-cell-type fresh COCOeval on filtered GT + filtered preds.
Each per-cell-type eval is small (50 images), so fast.
Overall is one big eval — slow for large pred sets but unavoidable.

For models with very large pred sets, --skip-overall lets us only compute
per-cell-type and aggregate later.
"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, json, os, re
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

SUBSET_GT = f"{FC_WORK}/subset/livecell_test_subset_coco.json"
RESULTS_DIR = os.environ.get("RESULTS_DIR", f"{FC_WORK}/results")
LINE_RE = re.compile(r"^([A-Za-z0-9]+)_Phase_")
CELL_TYPES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]

def cfg(coco_eval):
    coco_eval.params.catIds = [1]
    coco_eval.params.useCats = 0
    coco_eval.params.maxDets = [100, 500, 2000]
    coco_eval.params.areaRng = [[0,1e10],[0,324],[324,961],[961,1e10]]
    coco_eval.params.areaRngLbl = ["all","small","medium","large"]

def summary(ce):
    p = ce.params
    prec = ce.eval["precision"]
    if prec.size == 0: return {"mAP":-1,"AP50":-1,"AP75":-1,"AP_small":-1,"AP_medium":-1,"AP_large":-1}
    M = p.maxDets.index(2000)
    def _m(s): s = s[s>-1]; return float(np.mean(s)) if s.size else -1.0
    def _ap(t=None, lbl="all"):
        a = p.areaRngLbl.index(lbl); s = prec
        if t is not None: s = s[np.where(p.iouThrs==t)[0]]
        return _m(s[..., a, M])
    return {"mAP":_ap(None,"all"),"AP50":_ap(0.5,"all"),"AP75":_ap(0.75,"all"),
            "AP_small":_ap(None,"small"),"AP_medium":_ap(None,"medium"),"AP_large":_ap(None,"large")}

def make_partition_coco_objs(gt_full, preds_full, image_ids):
    """Build a fresh COCO_gt and COCO_dt restricted to image_ids."""
    sel = set(image_ids)
    sub_imgs = [im for im in gt_full.dataset["images"]      if im["id"]      in sel]
    sub_anns = [a  for a  in gt_full.dataset["annotations"] if a["image_id"] in sel]
    sub_data = {"images": sub_imgs, "annotations": sub_anns, "categories": gt_full.dataset["categories"]}
    sub_gt = COCO(); sub_gt.dataset = sub_data; sub_gt.createIndex()
    sub_preds = [p for p in preds_full if p["image_id"] in sel]
    if not sub_preds: return sub_gt, None, sub_anns, sub_preds
    sub_dt = sub_gt.loadRes(sub_preds)
    return sub_gt, sub_dt, sub_anns, sub_preds

def evaluate_partition(gt_full, preds_full, image_ids, label):
    sub_gt, sub_dt, sub_anns, sub_preds = make_partition_coco_objs(gt_full, preds_full, image_ids)
    if sub_dt is None:
        return {"mAP":0.0,"AP50":0.0,"AP75":0.0,"AP_small":-1,"AP_medium":-1,"AP_large":-1,
                "n_images":len(image_ids),"n_preds":0,"n_gt":len(sub_anns)}
    ce = COCOeval(sub_gt, sub_dt, "segm")
    cfg(ce)
    ce.params.imgIds = list(image_ids)
    ce.evaluate(); ce.accumulate()
    s = summary(ce)
    s.update({"n_images":len(image_ids),"n_preds":len(sub_preds),"n_gt":len(sub_anns)})
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag",  required=True)
    ap.add_argument("--gt",   default=SUBSET_GT)
    ap.add_argument("--skip-overall", action="store_true",
                    help="Skip overall eval (slow); aggregate per-cell-type weighted by image count")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    coco_gt = COCO(args.gt)
    preds = json.load(open(args.pred))
    for p in preds: p.pop("bbox", None)
    print(f"Loaded {len(preds)} predictions covering {len(set(p['image_id'] for p in preds))} unique images.", flush=True)

    id2line = {im["id"]: LINE_RE.match(im["file_name"]).group(1)
               for im in coco_gt.dataset["images"]
               if LINE_RE.match(im["file_name"])}
    by_line = defaultdict(list)
    for iid, line in id2line.items(): by_line[line].append(iid)
    all_ids = list(id2line.keys())

    results = {"tag": args.tag, "overall": {}, "per_cell_type": {}}

    print(f"\n=== Per cell type ===", flush=True)
    for line in CELL_TYPES:
        ids = by_line[line]
        print(f"  scoring {line} ({len(ids)} imgs)...", flush=True)
        s = evaluate_partition(coco_gt, preds, ids, line)
        results["per_cell_type"][line] = s
        print(f"    {line:<7} mAP={s['mAP']:.4f}  AP50={s['AP50']:.4f}  AP75={s['AP75']:.4f}  n_preds={s['n_preds']:>6} n_gt={s['n_gt']:>6}", flush=True)

    if not args.skip_overall:
        print(f"\n=== Overall ({len(all_ids)} imgs) ===", flush=True)
        s = evaluate_partition(coco_gt, preds, all_ids, "overall")
        results["overall"] = s
        print(f"  mAP={s['mAP']:.4f}  AP50={s['AP50']:.4f}  AP75={s['AP75']:.4f}", flush=True)
    else:
        # Aggregate per-cell-type weighted by image count
        per = results["per_cell_type"]
        n = sum(per[ct]["n_images"] for ct in CELL_TYPES)
        agg = {}
        for k in ["mAP","AP50","AP75"]:
            vals = [(per[ct][k], per[ct]["n_images"]) for ct in CELL_TYPES if per[ct][k] >= 0]
            if vals:
                agg[k] = sum(v*w for v,w in vals) / sum(w for _,w in vals)
            else:
                agg[k] = -1
        agg["n_images"] = n
        agg["n_preds"]  = sum(per[ct]["n_preds"] for ct in CELL_TYPES)
        agg["n_gt"]     = sum(per[ct]["n_gt"] for ct in CELL_TYPES)
        agg["aggregated_from_per_celltype"] = True
        results["overall"] = agg
        print(f"\n=== Overall (aggregated from per-cell-type) ===", flush=True)
        print(f"  mAP={agg['mAP']:.4f}  AP50={agg['AP50']:.4f}  AP75={agg['AP75']:.4f}", flush=True)

    out = os.path.join(RESULTS_DIR, f"{args.tag}_scores.json")
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nWrote {out}", flush=True)

if __name__ == "__main__":
    main()
