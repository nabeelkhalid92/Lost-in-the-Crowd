"""Predictions on ISBI 2014 val/test as COCO json (overlapping masks allowed).
  cellpose: CP_MODEL (path or 'cpsam'), flow threshold FLOW_THR (default 0.4); one label per pixel; score = mean cell
            probability; no diameter (native 512x512, cells 40-150 px, within Cellpose-SAM's training range)
  maskrcnn: detectron2 run dir (model_final.pth + config.yaml); every detection with its score (cutoff applied later)
usage: infer_isbi.py cellpose <data_dir> <split> <out.json>   |   infer_isbi.py maskrcnn <data_dir> <split> <out.json> <run_dir>"""
import json, os, sys
import numpy as np, skimage.io
from pycocotools import mask as MU
KIND, D, SPLIT, OUT = sys.argv[1:5]
J = json.load(open(f"{D}/{SPLIT}.json")); res = []
def enc(m):
    r = MU.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode(); return r
if KIND == "cellpose":
    import scipy.ndimage as ndi
    from cellpose import models
    model = models.CellposeModel(gpu=True, pretrained_model=os.environ.get("CP_MODEL", "cpsam")); ft = float(os.environ.get("FLOW_THR", "0.4"))
    for im in J["images"]:
        img = skimage.io.imread(f"{D}/{SPLIT}/images/{im['file_name']}")
        out = model.eval(img, flow_threshold=ft); lab = np.asarray(out[0]).astype(np.int32); cp = 1 / (1 + np.exp(-np.asarray(out[1][2], np.float32)))
        if lab.max() == 0: continue
        sc = ndi.mean(cp, lab, np.arange(1, lab.max() + 1))
        for k in range(1, lab.max() + 1):
            m = lab == k
            if m.sum() < 30: continue
            r = enc(m); res.append({"image_id": im["id"], "category_id": 1, "segmentation": r, "score": float(sc[k - 1]), "bbox": [float(x) for x in MU.toBbox(r)]})
else:
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg(); cfg.merge_from_file(f"{sys.argv[5]}/config.yaml"); cfg.MODEL.WEIGHTS = f"{sys.argv[5]}/model_final.pth"; cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    pred = DefaultPredictor(cfg)
    for im in J["images"]:
        img = skimage.io.imread(f"{D}/{SPLIT}/images/{im['file_name']}")
        bgr = np.stack([img] * 3, -1)[:, :, ::-1] if img.ndim == 2 else img[:, :, :3][:, :, ::-1]
        o = pred(np.ascontiguousarray(bgr))["instances"].to("cpu")
        for m, s in zip(o.pred_masks.numpy(), o.scores.numpy()):
            if m.sum() < 30: continue
            r = enc(m); res.append({"image_id": im["id"], "category_id": 1, "segmentation": r, "score": float(s), "bbox": [float(x) for x in MU.toBbox(r)]})
json.dump(res, open(OUT, "w")); print("DONE", len(res), "instances on", len(J["images"]), "images")
