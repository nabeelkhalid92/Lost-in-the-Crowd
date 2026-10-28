"""
VISTA-2D inference on the LIVECell stratified subset.

CellSamWrapper (SAM ViT-B encoder + decoder) outputs 3 channels:
  ch0 = cellprob (logit), ch1, ch2 = vector flow components (dy, dx).
Postprocessing uses cellpose dynamics.compute_masks (cellprob_thresh=0.4, flow_thresh=0.4).

Per-instance score = sigmoid(mean cellprob) inside mask.

Run:
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    $VISTA2D_PYTHON code/infer_vista2d.py \
        --out predictions/vista2d.json
"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, json, os, re, sys, time
import numpy as np
import skimage.io
import torch
from pycocotools import mask as maskUtils

SUBSET_GT = f"{FC_WORK}/subset/livecell_test_subset_coco.json"
FULL_GT   = f"{LC_ANN}/livecell_coco_test.json"
IMG_DIR   = f"{LC_IMG}/livecell_test_images"

CKPT_DIR  = f"{VISTA2D_BUNDLE}/vista2d/models"
SAM_PATH  = CKPT_DIR + "/sam_vit_b_01ec64.pth"
MODEL_PT  = CKPT_DIR + "/model.pt"

LINE_RE = re.compile(r"^([A-Za-z0-9]+)_Phase_")

def encode_rle(binmask):
    rle = maskUtils.encode(np.asfortranarray(binmask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def label_to_coco(label_mask, cellprob_logit, image_id):
    K = int(label_mask.max())
    if K == 0: return []
    # sigmoid the cellprob logit
    cp = 1.0 / (1.0 + np.exp(-cellprob_logit))
    preds = []
    for k in range(1, K + 1):
        m = (label_mask == k)
        if not m.any(): continue
        rle = encode_rle(m)
        ys, xs = np.where(m)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
        score = float(cp[m].mean())
        preds.append({
            "image_id": int(image_id),
            "category_id": 1,
            "segmentation": {"size": list(rle["size"]), "counts": rle["counts"]},
            "bbox": [float(b) for b in bbox],
            "score": score,
        })
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt",  default=SUBSET_GT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--img-dir", default=IMG_DIR)
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.full:
        args.gt = FULL_GT

    from monai.networks.nets.cell_sam_wrapper import CellSamWrapper
    from monai.inferers import SlidingWindowInfererAdapt
    from cellpose.dynamics import compute_masks

    device = torch.device("cuda:0")
    print("Loading CellSamWrapper...", flush=True)
    t0 = time.time()
    net = CellSamWrapper(checkpoint=SAM_PATH).to(device)
    ck = torch.load(MODEL_PT, map_location="cpu")
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    res = net.load_state_dict(sd, strict=False)
    print(f"  loaded in {time.time()-t0:.1f}s, missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    net.eval()
    inferer = SlidingWindowInfererAdapt(roi_size=(256,256), sw_batch_size=1, overlap=0.625, mode="gaussian")

    coco = json.load(open(args.gt))
    images = coco["images"]
    if args.limit: images = images[:args.limit]

    all_preds = []
    done_ids = set()
    if args.resume and os.path.exists(args.out):
        try:
            all_preds = json.load(open(args.out))
            done_ids = {p["image_id"] for p in all_preds}
            print(f"  resuming: {len(done_ids)} done", flush=True)
        except Exception:
            all_preds = []
    images = [im for im in images if im["id"] not in done_ids]
    print(f"  remaining: {len(images)}", flush=True)

    t_start = time.time()
    for i, im in enumerate(images):
        path = os.path.join(args.img_dir, im["file_name"])
        img = skimage.io.imread(path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] != 3:
            img = img[..., :3]
        # MONAI bundle scales to 0..1 with percentile clipping
        x = img.astype(np.float32)
        # per-channel min-max normalize
        for c in range(3):
            ch = x[..., c]
            cmin, cmax = ch.min(), ch.max()
            if cmax > cmin:
                x[..., c] = (ch - cmin) / (cmax - cmin)
        # 1..99 percentile clipping
        for c in range(3):
            ch = x[..., c]
            lo, hi = np.percentile(ch, [1, 99])
            x[..., c] = np.clip((ch - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        img_t = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).to(device)
        t1 = time.time()
        try:
            with torch.no_grad():
                pred = inferer(inputs=img_t, network=net)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                with torch.no_grad():
                    pred = inferer(inputs=img_t, network=net)
            else:
                raise
        torch.cuda.empty_cache()
        logits = pred[0].float().cpu().numpy()  # (3, H, W)
        dp = logits[1:]
        cellprob_logit = logits[0]
        try:
            label = compute_masks(dp, cellprob_logit, niter=200,
                                      cellprob_threshold=0.4, flow_threshold=0.4,
                                      interp=True, device=device)
        except RuntimeError:
            label = compute_masks(dp, cellprob_logit, niter=200,
                                      cellprob_threshold=0.4, flow_threshold=0.4,
                                      interp=True, device=None)
        K = int(label.max())
        preds = label_to_coco(label, cellprob_logit, im["id"])
        all_preds.extend(preds)
        elapsed = time.time() - t1
        rate = (i+1) / max(time.time() - t_start, 1e-9)
        eta = (len(images) - i - 1) / max(rate, 1e-9)
        print(f"  [{i+1}/{len(images)}] {im['file_name']}  K={K}  {elapsed:.2f}s  ETA={eta/60:.1f}min", flush=True)
        if (i+1) % args.ckpt_every == 0:
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
