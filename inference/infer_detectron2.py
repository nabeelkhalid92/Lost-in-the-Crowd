"""
Inference for anchor-based (Cascade Mask R-CNN, ResNeSt-200) and
anchor-free (CenterMask2, VoVNet-99) detection models on the same
400-image LIVECell stratified subset, using the same COCO output
format as the Cellpose wrapper for direct comparison.

Run with (detectron2 + detectron2-ResNeSt on PYTHONPATH):
  PYTHONPATH=<detectron2-ResNeSt checkout> \\
    python infer_detectron2.py --model anchor_based --out predictions/anchor_based.json

CenterMask2 (anchor-free) requires the centermask2 module — handled by
trying multiple PYTHONPATH candidates.
"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, json, os, re, sys, time
import numpy as np
import skimage.io
from pycocotools import mask as maskUtils

SUBSET_GT = f"{FC_WORK}/subset/livecell_test_subset_coco.json"
FULL_GT   = f"{LC_ANN}/livecell_coco_test.json"
IMG_DIR   = f"{LC_IMG}/livecell_test_images"

CONFIG = {
    "anchor_based": {
        "config":  f"{LC_MODELS}/anchor_based/livecell_config.yaml",
        "weights": f"{LC_MODELS}/anchor_based/LIVECell_anchor_based_model.pth",
    },
    "anchor_free": {
        "config":  f"{LC_MODELS}/anchor_free/livecell_config.yaml",
        "weights": f"{LC_MODELS}/anchor_free/LIVECell_anchor_free_model.pth",
    },
}

def encode_rle(binmask):
    rle = maskUtils.encode(np.asfortranarray(binmask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def masks_to_coco(masks_np, scores_np, image_id):
    preds = []
    for k in range(masks_np.shape[0]):
        m = masks_np[k].astype(bool)
        if not m.any(): continue
        rle = encode_rle(m)
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
        preds.append({
            "image_id": int(image_id),
            "category_id": 1,
            "segmentation": {"size": list(rle["size"]), "counts": rle["counts"]},
            "bbox": [float(b) for b in bbox],
            "score": float(scores_np[k]),
        })
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIG))
    ap.add_argument("--out",   required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--score-thresh", type=float, default=0.05)  # low threshold; mAP rewards ranked recall
    ap.add_argument("--max-dets", type=int, default=3000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--gt", default=SUBSET_GT)
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.full:
        args.gt = FULL_GT

    spec = CONFIG[args.model]

    # Try centermask2 import for anchor_free if needed
    if args.model == "anchor_free":
        try:
            from centermask.config import get_cfg
        except Exception:
            for cand in [f"{CENTERMASK2}", f"{CENTERMASK2}",
                         f"{CENTERMASK2}",
                         f"{CENTERMASK2}"]:
                if os.path.isdir(cand):
                    sys.path.insert(0, cand)
                    break
            from centermask.config import get_cfg
    else:
        from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(spec["config"])
    cfg.MODEL.WEIGHTS = spec["weights"]
    cfg.MODEL.DEVICE  = "cuda"
    if hasattr(cfg.MODEL, "ROI_HEADS"):
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    if args.model == "anchor_free":
        # CenterMask uses FCOS NMS thresholds
        if hasattr(cfg.MODEL, "FCOS"):
            cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.score_thresh
            cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = max(args.max_dets, 5000)
            cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = args.max_dets
    cfg.TEST.DETECTIONS_PER_IMAGE = args.max_dets

    print(f"Model: {args.model}  score_thresh: {args.score_thresh}  max_dets: {args.max_dets}", flush=True)
    predictor = DefaultPredictor(cfg)
    print("predictor loaded", flush=True)

    coco = json.load(open(args.gt))
    images = coco["images"]
    if args.limit: images = images[:args.limit]
    img_dir = args.img_dir

    # Resume support
    all_preds = []
    done_ids = set()
    if args.resume and os.path.exists(args.out):
        try:
            all_preds = json.load(open(args.out))
            done_ids = {p["image_id"] for p in all_preds}
            print(f"  resuming: {len(done_ids)} images already done", flush=True)
        except Exception as e:
            print(f"  resume failed ({e})", flush=True)
    images = [im for im in images if im["id"] not in done_ids]
    print(f"  remaining: {len(images)}", flush=True)
    t_start = time.time()
    for i, im in enumerate(images):
        path = os.path.join(img_dir, im["file_name"])
        img = skimage.io.imread(path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] != 3:
            img = img[..., :3]
        img = img.astype(np.uint8)

        t1 = time.time()
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
            out = predictor(img)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  [OOM] {im['file_name']}: empty cache + retry...", flush=True)
                _torch.cuda.empty_cache()
                try:
                    out = predictor(img)
                except RuntimeError:
                    print(f"  [SKIP] {im['file_name']}: OOM persists", flush=True)
                    continue
            else:
                raise
        instances = out["instances"]
        K = len(instances)
        if K == 0:
            elapsed = time.time() - t1
            rate = (i+1) / max(time.time()-t_start, 1e-9)
            eta = (len(images) - i - 1) / max(rate, 1e-9)
            print(f"  [{i+1}/{len(images)}] {im['file_name']}  K=0  {elapsed:.2f}s  ETA={eta/60:.1f}min", flush=True)
            continue
        masks = instances.pred_masks.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        preds = masks_to_coco(masks, scores, im["id"])
        all_preds.extend(preds)

        elapsed = time.time() - t1
        rate = (i+1) / max(time.time()-t_start, 1e-9)
        eta = (len(images) - i - 1) / max(rate, 1e-9)
        print(f"  [{i+1}/{len(images)}] {im['file_name']}  K={K}  {elapsed:.2f}s  ETA={eta/60:.1f}min", flush=True)

        if (i+1) % 50 == 0:
            tmp = args.out + ".tmp"
            with open(tmp, "w") as f: json.dump(all_preds, f)
            os.replace(tmp, args.out)
            print(f"    [checkpoint] {len(all_preds)} preds saved", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f: json.dump(all_preds, f)
    os.replace(tmp, args.out)
    print(f"\nDONE. {len(all_preds)} predictions across {len(set(p['image_id'] for p in all_preds))} images.", flush=True)
    print(f"Wallclock: {(time.time()-t_start)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
