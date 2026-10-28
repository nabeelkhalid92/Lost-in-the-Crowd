"""Run one segmentation method on a frame list and save per-frame label masks + per-label scores (.npz).
Datasets:  ctc   -> data/ctc_manifest.json (union of seg + count frames per sequence)
           count -> LIVECell nuclear-count benchmark (A172, A549), 1408x1040 mosaics at LIVECell scale
Scale protocol (identical information for every method): Cellpose-family gets diameter=d (oracle median
diameter); all other methods get the image rescaled by TARGET_D/d so cells match the LIVECell training scale,
and masks are resized back (nearest) to native resolution.
Methods: cyto cyto2 livecell (cellpose 2.x) | cyto3 (cellpose 3.x) | cpsam (cellpose 4.x)
         anchor_based anchor_free (LIVECell-paper detectron2 models) | microsam (micro-SAM vit_b_lm AIS)"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, glob, json, os, sys, time
import numpy as np, tifffile, skimage.io, skimage.transform
ROOT = os.environ.get("BENCH", f"{WORK}")
TARGET_D = 25.0
ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--dataset", required=True, choices=["ctc", "count", "livecell", "coco"])
ap.add_argument("--only", default=""); a = ap.parse_args()
# LIVECell test + nuclear-count images are already at LIVECell scale -> non-Cellpose methods run natively there
NATIVE = a.dataset in ("livecell", "count") or os.environ.get("NATIVE") == "1"   # NATIVE=1: no rescaling, raw uint8 passed through

def norm8(img):
    img = img.astype(np.float32)
    if img.ndim == 3: img = img[..., 0] if img.shape[-1] in (3, 4) else img[0]
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.8)
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

CELLPOSE = a.method in ("cyto", "cyto2", "livecell", "cyto3", "cpsam", "cpsam_v2", "cpdino", "vista2d") or a.method.startswith("cpcustom")  # vista2d normalises internally too
LC_MEAN, LC_STD = 128.0, 11.4   # LIVECell test-set intensity statistics (uint8 phase contrast)
def prep(img):
    """Cellpose normalises internally -> 8-bit image as is. LIVECell-scale data (LIVECell test, nuclear-count
    benchmark) is raw LIVECell-style uint8 -> passed unchanged. CTC frames for LIVECell-trained / other methods are
    standardised to LIVECell intensity statistics (same mean/std), the minimal domain alignment for a fixed model."""
    if img.ndim == 3: img = img[..., 0] if img.shape[-1] in (3, 4) else img[0]
    if CELLPOSE: return img if img.dtype == np.uint8 else norm8(img)
    if NATIVE and img.dtype == np.uint8: return img
    if os.environ.get("NO_STD") == "1": return img if img.dtype == np.uint8 else norm8(img)   # rescale only, no intensity standardisation
    if os.environ.get("PREP") == "lcpaper":   # LIVECell-paper cross-dataset pipeline: median->128, flat-field (51x51 median), clip [70,180]
        import scipy.ndimage as ndi
        x = img.astype(np.float32); x = x * 128.0 / max(np.median(x), 1e-6)
        bg = ndi.median_filter(x, size=51, mode="reflect"); x = x / np.maximum(bg, 1e-6) * 128.0
        return np.clip(x, 70, 180).astype(np.uint8)
    x = img.astype(np.float32); x = LC_MEAN + (x - x.mean()) / max(x.std(), 1e-6) * LC_STD
    return np.clip(x, 0, 255).astype(np.uint8)

def jobs():
    if a.dataset == "ctc":
        man = json.loads(open(f"{ROOT}/data/ctc_manifest.json").read().replace(f"{CTC_ROOT}", os.environ.get("CTC_ROOT", f"{CTC_ROOT}")))
        for seq, m in man.items():
            if a.only and a.only not in seq: continue
            cf = sorted(int(t) for t in m["count_frames"]); cf = cf[::int(os.environ.get("CTC_COUNT_STRIDE", "1"))]
            if os.environ.get("RAW_DUMP"): cf = []   # raw dump: gold-segmentation frames only
            frames = sorted(set(int(t) for t in m["seg_gold"]) | set(cf))
            for t in frames:
                yield seq.replace("/", "_"), f"{m['img_dir']}/t{t:0{m['ndig']}d}.tif", f"t{t:04d}", m["diam_median"]
    elif a.dataset == "livecell":
        gt = json.load(open(os.environ.get("LC_GT", f"{LC_ANN}/livecell_coco_test.json")))
        D = json.load(open(f"{ROOT}/data/livecell_diameters_full.json"))
        idir = os.environ.get("LC_IMG", f"{LC_IMG}/livecell_test_images")
        for im in gt["images"]:
            line = im["file_name"].split("_Phase_")[0]
            yield "livecell", f"{idir}/{im['file_name']}", f"{im['id']}", D[line]
    elif a.dataset == "coco":
        # any COCO-format test set (wsbench conversions: EVICAN, Omnipose bact_phase, yeast). Oracle diameter = median
        # equivalent diameter of the annotated cells (category COCO_CAT) per group (cell line / species) -- as for CTC
        from collections import defaultdict
        J = json.load(open(os.environ["COCO_JSON"])); idir = os.environ["COCO_IMG"]; cat = int(os.environ.get("COCO_CAT", "1"))
        gid = {im["id"]: str(im.get("cell_line") or im.get("group") or "all").replace("/", "_") for im in J["images"]}
        dia = defaultdict(list)
        for an in J["annotations"]:
            if an["category_id"] == cat and not an.get("iscrowd"): dia[gid[an["image_id"]]].append(2 * np.sqrt(an["area"] / np.pi))
        for im in J["images"]:
            g = gid[im["id"]]; yield g, f"{idir}/{im['file_name']}", f"{im['id']}", float(np.median(dia[g]))
    else:
        D = json.load(open(f"{ROOT}/results/count_diam_sizemodel.json"))
        for cell, root in [("A172", os.environ.get("A172", f"{COUNT_ROOT}/A172")),
                           ("A549", os.environ.get("A549", f"{COUNT_ROOT}/A549"))]:
            for f in sorted(glob.glob(f"{root}/*/*.png")):
                yield cell, f, os.path.basename(f)[:-4], D[cell]["median"]

M = a.method
if M in ("cyto", "cyto2", "livecell", "cyto3", "cpsam", "cpsam_v2", "cpdino") or M.startswith("cpcustom"):
    sys.path.insert(0, os.environ.get("FLOWCODE", f"{REPO}"))
    try:
        import pycocotools.mask  # noqa: F401  (infer_cellpose imports it; not needed here)
    except ImportError:
        import types; pc = types.ModuleType("pycocotools"); pc.mask = types.ModuleType("pycocotools.mask")
        sys.modules["pycocotools"] = pc; sys.modules["pycocotools.mask"] = pc.mask
    from infer_cellpose import load_model, run_inference, per_instance_scores
    if M.startswith("cpcustom"):   # a Cellpose model trained in this study (path in CP_MODEL)
        from cellpose import models as _cpm, version as cpv
        model = _cpm.CellposeModel(gpu=True, pretrained_model=os.environ["CP_MODEL"])
    else:
        model, cpv = load_model(M, True)
    import scipy.ndimage as ndi
    def per_instance_scores(lab, cp):  # vectorised: mean cellprob per label (sigmoid if logits)
        cp = np.asarray(cp, np.float32)
        if cp.shape != lab.shape: cp = skimage.transform.resize(cp, lab.shape, order=1, preserve_range=True)
        if cp.max() > 1.5 or cp.min() < -0.5: cp = 1 / (1 + np.exp(-cp))
        return np.asarray(ndi.mean(cp, lab, np.arange(1, lab.max() + 1)), np.float32)
    DIAM_MODE = os.environ.get("DIAM_MODE", "oracle")   # oracle | auto (Cellpose size model / model default)
    def predict(img8, d):
        lab, cp = run_inference(model, img8, None if DIAM_MODE == "auto" else d, cpv)
        sc = per_instance_scores(lab, cp) if lab.max() > 0 else np.zeros(0, np.float32)
        return lab.astype(np.int32), sc
elif M in ("anchor_based", "anchor_free"):
    import torch
    if M == "anchor_free":
        for cand in [f"{CENTERMASK2}", f"{CENTERMASK2}", f"{CENTERMASK2}",
                     f"{CENTERMASK2}"]:
            if os.path.isdir(cand): sys.path.insert(0, cand); break
        from centermask.config import get_cfg
    else:
        from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    W = f"{LC_MODELS}"
    cfg = get_cfg(); cfg.set_new_allowed(True)
    cfg.merge_from_file(f"{W}/{M}/livecell_config.yaml")
    cfg.MODEL.WEIGHTS = f"{W}/{M}/LIVECell_{M}_model.pth"; cfg.MODEL.DEVICE = "cuda"
    FLOOR = 0.3 if a.dataset == "count" else 0.05
    if hasattr(cfg.MODEL, "ROI_HEADS"): cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = FLOOR
    if hasattr(cfg.MODEL, "FCOS"):
        cfg.MODEL.FCOS.INFERENCE_TH_TEST = FLOOR; cfg.MODEL.FCOS.PRE_NMS_TOPK_TEST = 5000; cfg.MODEL.FCOS.POST_NMS_TOPK_TEST = int(os.environ.get("DET_MAX", 3000))
    cfg.TEST.DETECTIONS_PER_IMAGE = int(os.environ.get("DET_MAX", 3000))
    pred = DefaultPredictor(cfg)
    def predict(img8, d):
        # nuclear-count images are 2x2 mosaics of LIVECell-size fields: run each 520x704 quadrant separately so the
        # detector keeps its native proposal/detection budget per field (same setting as the LIVECell test images)
        if a.dataset == "count" and os.environ.get("DET_TILE", "1") == "1" and img8.shape == (1040, 1408):
            lab = np.zeros(img8.shape, np.int32); scs = []; off = 0
            for y0 in (0, 520):
                for x0 in (0, 704):
                    l, s_ = predict_one(img8[y0:y0 + 520, x0:x0 + 704], d)
                    lab[y0:y0 + 520, x0:x0 + 704] = np.where(l > 0, l + off, 0); off += len(s_); scs.append(s_)
            return lab, np.concatenate(scs).astype(np.float32)
        return predict_one(img8, d)
    def predict_one(img8, d):
        s = 1.0 if NATIVE else TARGET_D / d; H, W_ = img8.shape
        im = img8 if abs(s - 1) < 1e-3 else (skimage.transform.rescale(img8, s, order=1, preserve_range=True, anti_aliasing=s < 1)).astype(np.uint8)
        pad = None
        if os.environ.get("PAD") == "1" and (im.shape[0] < 520 or im.shape[1] < 704):   # small crops: embed in a LIVECell-size field
            ph, pw = max(520, im.shape[0]), max(704, im.shape[1]); canvas = np.full((ph, pw), int(np.median(im)), np.uint8)
            canvas[:im.shape[0], :im.shape[1]] = im; pad = im.shape; im = canvas
        # detectron2 resizes to the config's test size; keep the native-scale input instead
        tg = getattr(pred, 'aug', None) or getattr(pred, 'transform_gen'); tg.short_edge_length = (min(im.shape), min(im.shape)); tg.max_size = max(im.shape) + 1
        torch.cuda.empty_cache()
        with torch.no_grad(): inst = pred(np.stack([im] * 3, -1))["instances"].to("cpu")
        sc = inst.scores.numpy(); ms = inst.pred_masks.numpy()
        if pad is not None: ms = ms[:, :pad[0], :pad[1]]; keep = ms.reshape(len(ms), -1).any(1); ms, sc = ms[keep], sc[keep]
        if os.environ.get("RAW_DUMP"):
            from pycocotools import mask as MU
            raw = []
            for i in np.argsort(-sc):
                mm = ms[i] if ms[i].shape == (H, W_) else skimage.transform.resize(ms[i].astype(np.uint8), (H, W_), order=0, preserve_range=True, anti_aliasing=False).astype(bool)
                r = MU.encode(np.asfortranarray(mm.astype(np.uint8))); r["counts"] = r["counts"].decode()
                raw.append({"segmentation": r, "score": float(sc[i])})
            RAW[0] = raw
        order = np.argsort(sc)  # paint low->high so the highest score wins overlaps
        lab = np.zeros(ms.shape[1:] if len(ms) else (pad or im.shape), np.int32); keep = []
        for k, i in enumerate(order, 1):
            lab[ms[i]] = k; keep.append(sc[i])
        # drop labels fully overwritten
        ids = np.unique(lab); ids = ids[ids > 0]
        remap = np.zeros(len(keep) + 1, np.int32); remap[ids] = np.arange(1, len(ids) + 1)
        lab = remap[lab]; sc2 = np.array([keep[i - 1] for i in ids], np.float32)
        if lab.shape != (H, W_): lab = skimage.transform.resize(lab, (H, W_), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
        return lab, sc2
elif M == "microsam":
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
    P, S = get_predictor_and_segmenter(model_type="vit_b_lm", segmentation_mode="ais", is_tiled=False)
    PT, ST = get_predictor_and_segmenter(model_type="vit_b_lm", segmentation_mode="ais", is_tiled=True)
    def predict(img8, d):
        s = 1.0 if NATIVE else TARGET_D / d; H, W_ = img8.shape
        im = img8 if abs(s - 1) < 1e-3 else (skimage.transform.rescale(img8, s, order=1, preserve_range=True, anti_aliasing=s < 1)).astype(np.uint8)
        # micro-SAM's documented usage: tiled AIS for images larger than the 1024-px SAM input
        if max(im.shape) > 1024:
            lab = automatic_instance_segmentation(predictor=PT, segmenter=ST, input_path=im, ndim=2, tile_shape=(512, 512), halo=(64, 64), verbose=False)
        else:
            lab = automatic_instance_segmentation(predictor=P, segmenter=S, input_path=im, ndim=2, verbose=False)
        lab = np.asarray(lab).astype(np.int32)
        if lab.shape != (H, W_): lab = skimage.transform.resize(lab, (H, W_), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
        n = int(lab.max()); return lab, np.ones(n, np.float32)
elif M == "vista2d":
    # Same model, normalisation and post-processing as flow_comparison/code/infer_vista2d.py (MONAI VISTA-2D bundle)
    import torch
    from monai.networks.nets.cell_sam_wrapper import CellSamWrapper
    from monai.inferers import SlidingWindowInfererAdapt
    from cellpose.dynamics import compute_masks
    CK = os.environ.get("VISTA_DIR", f"{VISTA2D_BUNDLE}/vista2d/models")
    dev = torch.device("cuda:0"); net = CellSamWrapper(checkpoint=f"{CK}/sam_vit_b_01ec64.pth").to(dev)
    ck = torch.load(f"{CK}/model.pt", map_location="cpu"); net.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck, strict=False); net.eval()
    inferer = SlidingWindowInfererAdapt(roi_size=(256, 256), sw_batch_size=1, overlap=0.625, mode="gaussian")
    def predict(img8, d):
        s = 1.0 if NATIVE else TARGET_D / d; H, W_ = img8.shape
        im = img8 if abs(s - 1) < 1e-3 else (skimage.transform.rescale(img8, s, order=1, preserve_range=True, anti_aliasing=s < 1)).astype(np.uint8)
        x = im.astype(np.float32); x = (x - x.min()) / max(x.max() - x.min(), 1e-6)
        lo, hi = np.percentile(x, [1, 99]); x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        t = torch.from_numpy(np.stack([x] * 3)).unsqueeze(0).to(dev)
        with torch.no_grad(): lg = inferer(inputs=t, network=net)[0].float().cpu().numpy()
        torch.cuda.empty_cache()
        lab = compute_masks(lg[1:], lg[0], niter=200, cellprob_threshold=0.4, flow_threshold=0.4, interp=True, device=dev)
        lab = np.asarray(lab[0] if isinstance(lab, tuple) else lab).astype(np.int32)
        import scipy.ndimage as ndi
        cp = 1 / (1 + np.exp(-lg[0])); sc = np.asarray(ndi.mean(cp, lab, np.arange(1, lab.max() + 1)), np.float32) if lab.max() else np.zeros(0, np.float32)
        if lab.shape != (H, W_): lab = skimage.transform.resize(lab, (H, W_), order=0, preserve_range=True, anti_aliasing=False).astype(np.int32)
        return lab, sc
else:
    raise SystemExit(f"unknown method {M}")

RAW = [None]
OUT = f"{ROOT}/preds/{os.environ.get('OUT_NAME', a.dataset)}{'_raw' if os.environ.get('RAW_DUMP') else ''}/{M}"; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); n = 0; times = {}
JOBS = list(jobs())
if os.environ.get("REVERSE"): JOBS = JOBS[::-1]   # second worker on the same output dir (existing files are skipped)
for grp, path, key, d in JOBS:
    o = f"{OUT}/{grp}__{key}.npz"
    if os.path.exists(o): continue
    img = tifffile.imread(path) if path.endswith(".tif") else skimage.io.imread(path)
    img8 = prep(img)
    t1 = time.time(); lab, sc = predict(img8, d); dt = time.time() - t1
    np.savez_compressed(o, lab=lab.astype(np.uint16), scores=sc.astype(np.float32), sec=np.float32(dt), shape=np.array(img8.shape))
    if RAW[0] is not None: json.dump(RAW[0], open(o[:-4] + "_raw.json", "w")); RAW[0] = None
    n += 1
    if n % 25 == 0: print(f"[{M}/{a.dataset}] {n} done, last {grp} {key} K={int(lab.max())} {dt:.2f}s", flush=True)
print(f"[{M}/{a.dataset}] DONE {n} new frames in {(time.time()-t0)/60:.1f} min", flush=True)
