"""Run a flow model on LIVECell val/test with the oracle diameter and store its raw outputs (flow field dP, cell
probability logits, and the masks at the model's default thresholds) so that post-processing thresholds can be swept
offline (sweep_thresholds.py). usage: infer_flows.py METHOD SPLIT   (env: BENCH, FLOWCODE, LC_IMG_VAL, LC_IMG_TEST, VISTA_DIR)
Output: $BENCH/preds/flows_<split>/<method>/<image_id>.npz with dP (2,H,W) float16, cellprob (H,W) float16, lab uint16."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, sys, time
import numpy as np, skimage.io
ROOT = os.environ.get("BENCH", f"{WORK}"); M, SPLIT = sys.argv[1], sys.argv[2]
GT = {"val": os.environ.get("LC_GT_VAL", f"{LC_ANN}/livecell_coco_val.json"),
      "test": os.environ.get("LC_GT", f"{LC_ANN}/livecell_coco_test.json")}[SPLIT]
IMG = {"val": os.environ.get("LC_IMG_VAL", f"{LC_IMG}/livecell_train_val_images"),
       "test": os.environ.get("LC_IMG", f"{LC_IMG}/livecell_test_images")}[SPLIT]
D = json.load(open(f"{ROOT}/data/livecell_diameters_full.json"))
OUT = f"{ROOT}/preds/flows_{SPLIT}/{M}"; os.makedirs(OUT, exist_ok=True)
ims = json.load(open(GT))["images"]
if M in ("cyto3", "cpsam", "livecell", "cyto2", "cyto"):
    sys.path.insert(0, os.environ.get("FLOWCODE", f"{REPO}"))
    try: import pycocotools.mask  # noqa: F401 (infer_cellpose imports it; not needed here)
    except ImportError:
        import types; pc = types.ModuleType("pycocotools"); pc.mask = types.ModuleType("pycocotools.mask"); sys.modules["pycocotools"] = pc; sys.modules["pycocotools.mask"] = pc.mask
    from infer_cellpose import load_model
    model, cpv = load_model(M, True); major = int(str(cpv).split(".")[0])
    def run(img, d):
        out = model.eval(img, diameter=d) if major >= 4 else model.eval(img, diameter=d, channels=[0, 0])
        masks, flows = out[0], out[1]
        return np.asarray(masks), np.asarray(flows[1], np.float32), np.asarray(flows[2], np.float32)
elif M == "vista2d":
    import torch
    from monai.networks.nets.cell_sam_wrapper import CellSamWrapper
    from monai.inferers import SlidingWindowInfererAdapt
    from cellpose.dynamics import compute_masks
    CK = os.environ.get("VISTA_DIR", f"{VISTA2D_BUNDLE}/vista2d/models")
    dev = torch.device("cuda:0"); net = CellSamWrapper(checkpoint=f"{CK}/sam_vit_b_01ec64.pth").to(dev)
    ck = torch.load(f"{CK}/model.pt", map_location="cpu"); net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck, strict=False); net.eval()
    inferer = SlidingWindowInfererAdapt(roi_size=(256, 256), sw_batch_size=1, overlap=0.625, mode="gaussian")
    def run(img, d):   # same normalisation and defaults as infer_frames.py (cellprob 0.4, flow 0.4)
        x = img.astype(np.float32); x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
        lo, hi = np.percentile(x, [1, 99]); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        t = torch.from_numpy(np.stack([x] * 3)).unsqueeze(0).to(dev)
        with torch.no_grad(): lg = inferer(inputs=t, network=net)[0].float().cpu().numpy()
        torch.cuda.empty_cache()
        lab = compute_masks(lg[1:], lg[0], niter=200, cellprob_threshold=0.4, flow_threshold=0.4, interp=True, device=dev)
        lab = np.asarray(lab[0] if isinstance(lab, tuple) else lab)
        return lab, lg[1:], lg[0]
else:
    raise SystemExit(M)
t0 = time.time(); n = 0
for im in ims:
    o = f"{OUT}/{im['id']}.npz"
    if os.path.exists(o): continue
    img = skimage.io.imread(f"{IMG}/{im['file_name']}")
    if img.ndim == 3: img = img[..., 0]
    lab, dP, cp = run(img, D[im["file_name"].split("_Phase_")[0]])
    np.savez_compressed(o, lab=lab.astype(np.uint16), dP=dP.astype(np.float16), cellprob=cp.astype(np.float16))
    n += 1
    if n % 50 == 0: print(f"[{M}/{SPLIT}] {n} done {(time.time()-t0)/n:.2f}s/img", flush=True)
print(f"[{M}/{SPLIT}] DONE {n} in {(time.time()-t0)/60:.1f} min", flush=True)
