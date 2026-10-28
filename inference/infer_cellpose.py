"""
Uniform Cellpose inference wrapper for {cyto2, cyto3, livecell, cpsam}.

Handles cellpose 2.x and 4.x API differences.
- Per-cell-type fixed diameter from $FC_WORK/subset/diameters.json
- Per-instance score = mean(cellprob) inside mask, after sigmoid if needed
- Streams COCO predictions to disk every N images (resumable for cpsam CPU run)

Usage:
  python infer_cellpose.py --model cyto2 --out predictions/cyto2.json
  python infer_cellpose.py --model cyto3 --out predictions/cyto3.json
  python infer_cellpose.py --model livecell --out predictions/livecell.json
  python infer_cellpose.py --model cpsam --out predictions/cpsam.json [--cpu]

  --resume  : continue from existing partial predictions file
  --limit N : process only first N images (debug)
"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, json, os, re, sys, time
import numpy as np
import skimage.io
from pycocotools import mask as maskUtils

SUBSET_GT  = f"{FC_WORK}/subset/livecell_test_subset_coco.json"
FULL_GT    = f"{LC_ANN}/livecell_coco_test.json"
DIAM_PATH  = f"{FC_WORK}/subset/diameters.json"
DIAM_FULL  = f"{FC_WORK}/subset/diameters_full.json"
IMG_DIR    = f"{LC_IMG}/livecell_test_images"
LINE_RE    = re.compile(r"^([A-Za-z0-9]+)_Phase_")

MODEL_REGISTRY = {
    # name: kwargs to pass to CellposeModel(...)
    # NOTE: cellpose 2.2.3 has a bug where pretrained_model='cyto2' silently loads cyto.
    # Workaround: use model_type='cyto2' (cellpose 2.x) OR explicit path (both versions).
    "cyto":     {"model_type": "cyto",     "diam_mean": 30.0},
    "cyto2":    {"model_type": "cyto2",    "diam_mean": 30.0},
    "livecell": {"model_type": "livecell", "diam_mean": 30.0},
    # cellpose 3.x:
    "cyto3":    {"pretrained_model": "cyto3"},
    "livecell_cp3": {"pretrained_model": "livecell_cp3"},
    # cellpose 4.x only:
    "cpsam":    {"pretrained_model": "cpsam", "use_bfloat16": False},  # bf16 unsupported on RTX 2080 Ti / cu102
    "cpsam_v2": {"pretrained_model": "cpsam_v2", "use_bfloat16": False},  # cellpose 4.2.1 default (newer checkpoint)
    "cpdino":   {"pretrained_model": "cpdino", "use_bfloat16": False},    # CellposeDINO (DINOv3 ViT-L), cellpose 4.2.1, needs python>=3.11 + dinov3
}

def parse_line(fn):
    m = LINE_RE.match(fn)
    return m.group(1) if m else None

def encode_rle(binmask):
    """Encode HxW boolean -> COCO RLE dict with str counts."""
    rle = maskUtils.encode(np.asfortranarray(binmask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def per_instance_scores(label_mask, cellprob):
    """For each label k=1..K, score = mean cellprob inside mask k.
    cellprob shape == label_mask shape; sigmoid if it looks like logits.
    """
    if cellprob.shape != label_mask.shape:
        # cellpose sometimes returns flows at different scale; resize via simple repeat
        from scipy.ndimage import zoom
        zy = label_mask.shape[0] / cellprob.shape[0]
        zx = label_mask.shape[1] / cellprob.shape[1]
        cellprob = zoom(cellprob, (zy, zx), order=1)
    # If max > 1.5 or min < -0.5, assume logits and sigmoid.
    cmin, cmax = float(cellprob.min()), float(cellprob.max())
    if cmax > 1.5 or cmin < -0.5:
        cellprob = 1.0 / (1.0 + np.exp(-cellprob))
    K = int(label_mask.max())
    scores = np.zeros(K, dtype=np.float32)
    for k in range(1, K + 1):
        m = (label_mask == k)
        if m.any():
            scores[k - 1] = float(cellprob[m].mean())
        else:
            scores[k - 1] = 0.0
    return scores

def label_to_coco_preds(label_mask, scores, image_id):
    """Convert HxW int label mask -> list of COCO segm prediction dicts."""
    preds = []
    K = int(label_mask.max())
    for k in range(1, K + 1):
        m = (label_mask == k)
        if not m.any():
            continue
        rle = encode_rle(m)
        ys, xs = np.where(m)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
        preds.append({
            "image_id": int(image_id),
            "category_id": 1,
            "segmentation": {"size": list(rle["size"]), "counts": rle["counts"]},
            "bbox": [float(b) for b in bbox],
            "score": float(scores[k - 1]),
        })
    return preds

def load_model(model_name, use_gpu):
    from cellpose import models, version as cp_version
    print(f"  cellpose version: {cp_version}", flush=True)
    spec = dict(MODEL_REGISTRY[model_name])
    major = int(str(cp_version).split(".")[0])
    # cellpose 4.x: drop diam_mean (not accepted), prefer pretrained_model.
    # cellpose 2.x: model_type for built-in names is the only reliable path
    # because pretrained_model='cyto2' silently falls back to cyto (bug in 2.2.3).
    if major >= 4:
        if "model_type" in spec:
            spec["pretrained_model"] = spec.pop("model_type")
        spec.pop("diam_mean", None)
        m = models.CellposeModel(gpu=use_gpu, **spec)
    elif major == 3:
        # 3.x accepts either; pretrained_model='cyto3' resolves via get_model_params.
        spec.pop("diam_mean", None)
        spec.pop("use_bfloat16", None)
        # 3.x: pretrained_model dominant
        if "model_type" in spec and "pretrained_model" not in spec:
            spec["pretrained_model"] = spec.pop("model_type")
        m = models.CellposeModel(gpu=use_gpu, **spec)
        loaded = getattr(m, "pretrained_model", None)
        print(f"  loaded weights: {loaded}", flush=True)
    else:
        # 2.x bug workaround
        if "pretrained_model" in spec and "model_type" not in spec:
            spec["model_type"] = spec.pop("pretrained_model")
        m = models.CellposeModel(gpu=use_gpu, **spec)
        loaded = getattr(m, "pretrained_model", None)
        print(f"  loaded weights: {loaded}", flush=True)
        if loaded and isinstance(loaded, list):
            expected_substr = spec.get("model_type", "")
            if expected_substr and not any(expected_substr in p for p in loaded):
                raise RuntimeError(f"cellpose loaded {loaded} but expected '{expected_substr}'")
    return m, cp_version

def run_inference(model, img, diameter, cp_version):
    """Run cellpose; return (label_mask, cellprob_2d).
    Handles 2.x vs 4.x return signature differences.
    """
    major = int(str(cp_version).split(".")[0])
    ft = float(os.environ.get("FLOW_THR", "0.4"))   # flow-consistency threshold (Cellpose default 0.4; relaxed setting 0.8)
    if major >= 4:
        out = model.eval(img, diameter=diameter, flow_threshold=ft)
    elif major == 3:
        out = model.eval(img, diameter=diameter, channels=[0, 0])
    else:
        out = model.eval(img, diameter=diameter, channels=[0, 0])
    if len(out) == 4:
        masks, flows, styles, diams = out
    elif len(out) == 3:
        masks, flows, styles = out
    else:
        raise RuntimeError(f"unexpected eval() return tuple length: {len(out)}")
    # flows is a list; flows[2] is cellprob in 2.x; in 4.x cpsam similar.
    cellprob = None
    if isinstance(flows, list) and len(flows) >= 3:
        cp = flows[2]
        if cp is not None:
            cellprob = np.asarray(cp).astype(np.float32)
    if cellprob is None:
        cellprob = np.ones_like(masks, dtype=np.float32) * 0.5
    return masks, cellprob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--out",    required=True)
    ap.add_argument("--cpu",    action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit",  type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--gt",   default=SUBSET_GT, help="COCO GT JSON (defaults to stratified subset)")
    ap.add_argument("--diam", default=None, help="path to per-cell-type diameter json")
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--full", action="store_true", help="shortcut: use full LIVECell test set")
    args = ap.parse_args()
    if args.full:
        args.gt   = FULL_GT
        args.diam = DIAM_FULL
    if args.diam is None:
        args.diam = DIAM_PATH

    use_gpu = not args.cpu
    print(f"Model: {args.model}  GPU: {use_gpu}  Out: {args.out}", flush=True)

    coco = json.load(open(args.gt))
    diams = json.load(open(args.diam))
    img_dir = args.img_dir

    # Determine work list
    images = coco["images"]
    if args.limit:
        images = images[:args.limit]

    done_ids = set()
    preds_so_far = []
    if args.resume and os.path.exists(args.out):
        try:
            existing = json.load(open(args.out))
            preds_so_far = existing
            done_ids = {p["image_id"] for p in existing}
            print(f"  resuming: {len(done_ids)} images already done, {len(preds_so_far)} preds in file", flush=True)
        except Exception as e:
            print(f"  resume failed ({e}); starting fresh", flush=True)
            preds_so_far = []

    work = [im for im in images if im["id"] not in done_ids]
    print(f"  remaining work: {len(work)} / {len(images)}", flush=True)

    print("  loading model...", flush=True)
    t0 = time.time()
    model, cp_version = load_model(args.model, use_gpu)
    print(f"  model loaded in {time.time()-t0:.1f}s", flush=True)

    t_start = time.time()
    n_done_this_run = 0
    for i, im in enumerate(work):
        fn = im["file_name"]
        line = parse_line(fn)
        diameter = float(diams.get(line, 25.0))
        path = os.path.join(img_dir, fn)
        try:
            img = skimage.io.imread(path)
        except Exception as e:
            print(f"  [skip] {fn}: imread failed: {e}", flush=True)
            continue
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            img = img.mean(axis=-1).astype(np.uint8)
        elif img.ndim == 3:
            img = img[..., 0]
        img = img.astype(np.uint8)

        t1 = time.time()
        try:
            masks, cellprob = run_inference(model, img, diameter, cp_version)
        except Exception as e:
            print(f"  [error] {fn}: {e}", flush=True)
            continue
        K = int(masks.max())
        scores = per_instance_scores(masks, cellprob) if K > 0 else np.zeros(0, dtype=np.float32)
        preds = label_to_coco_preds(masks, scores, im["id"])
        preds_so_far.extend(preds)
        n_done_this_run += 1
        elapsed = time.time() - t1

        # Progress
        rate = n_done_this_run / max(time.time() - t_start, 1e-9)
        eta = (len(work) - n_done_this_run) / max(rate, 1e-9)
        print(f"  [{i+1}/{len(work)}] {fn}  line={line} diam={diameter:.1f}  K={K}  {elapsed:.2f}s  ETA={eta/60:.1f}min", flush=True)

        # Periodic checkpoint
        if (n_done_this_run % args.ckpt_every) == 0:
            tmp = args.out + ".tmp"
            with open(tmp, "w") as f:
                json.dump(preds_so_far, f)
            os.replace(tmp, args.out)
            print(f"    [checkpoint] {len(preds_so_far)} preds saved", flush=True)

    # Final save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(preds_so_far, f)
    os.replace(tmp, args.out)
    print(f"\nDONE. {len(preds_so_far)} predictions across {len(set(p['image_id'] for p in preds_so_far))} images.", flush=True)
    print(f"Wallclock this run: {(time.time()-t_start)/60:.1f} min", flush=True)

if __name__ == "__main__":
    main()
