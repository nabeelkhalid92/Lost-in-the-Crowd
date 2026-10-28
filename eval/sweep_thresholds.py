"""Solution 1: choose a flow model's post-processing thresholds on the LIVECell validation split instead of using the
defaults. Cellpose keeps a pixel if cellprob > cellprob_threshold and removes every mask whose predicted flows are
inconsistent (flow error > flow_threshold, default 0.4; 0 disables the check). In crowded regions these removed masks
are exactly the lost cells. Uses the stored raw outputs of infer_flows.py.
  sweep_thresholds.py METHOD val            -> grid on validation, results/sweep/val__METHOD.json (+ chosen setting)
  sweep_thresholds.py METHOD test [C F]     -> masks at the chosen (or given) thresholds on test; writes a COCO-style
                                               prediction json preds/livecell_sweep_METHOD.json for analyze.py / COCO
Grid: cellprob in {-1, 0, 1} (x) flow in {0.4 (default), 0.8, 0 (off)}; masks are computed once per cellprob with
the flow check off, then the per-mask flow error is thresholded offline (same operations as cellpose.compute_masks)."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, sys, time
import numpy as np
from pycocotools import mask as MU
R = os.environ.get("BENCH", f"{WORK}"); M, SPLIT = sys.argv[1], sys.argv[2]
import torch
from cellpose import dynamics, utils, version as cpv
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); major = int(str(cpv).split(".")[0])
CP = [-1.0, 0.0, 1.0]; FT = [0.4, 0.8, 0.0]
GT = {"val": os.environ.get("LC_GT_VAL", f"{LC_ANN}/livecell_coco_val.json"),
      "test": os.environ.get("LC_GT", f"{LC_ANN}/livecell_coco_test.json")}[SPLIT]
J = json.load(open(GT)); ims = J["images"]

def masks_at(dP, cp, c):
    """label image at cellprob threshold c with the flow check off (min_size 15, holes filled as in cellpose)."""
    kw = dict(niter=200, cellprob_threshold=c, flow_threshold=0.0, min_size=15)
    out = dynamics.compute_masks(dP, cp, device=dev, **kw) if major >= 4 else dynamics.compute_masks(dP, cp, interp=True, use_gpu=dev.type == "cuda", device=dev, **kw)
    return np.asarray(out[0] if isinstance(out, tuple) else out).astype(np.int32)

def flow_errors(lab, dP):
    if lab.max() == 0: return np.zeros(0)
    e = dynamics.flow_error(lab, dP, device=dev) if major >= 4 else dynamics.flow_error(lab, dP, use_gpu=dev.type == "cuda", device=dev)
    return np.asarray(e[0] if isinstance(e, tuple) else e)

def apply_ft(lab, err, ft):
    if ft <= 0 or lab.max() == 0: return lab
    bad = np.nonzero(err > ft)[0] + 1; keep = np.ones(lab.max() + 1, bool); keep[bad] = False; keep[0] = False
    idx = np.zeros(lab.max() + 1, np.int32); idx[keep] = np.arange(1, keep.sum() + 1); return idx[lab]

def gt_rles(iid, anns):
    out = []
    for a in anns.get(iid, []):
        s = a["segmentation"]; h, w = a["_h"], a["_w"]
        r = MU.frPyObjects(s, h, w); out.append(MU.merge(r) if isinstance(r, list) else r)
    return out

if SPLIT == "val":
    from collections import defaultdict
    anns = defaultdict(list); hw = {im["id"]: (im["height"], im["width"]) for im in ims}
    for a in J["annotations"]:
        a["_h"], a["_w"] = hw[a["image_id"]]; anns[a["image_id"]].append(a)
    T = {(c, f): np.zeros(3, np.int64) for c in CP for f in FT}; CNT = {(c, f): [] for c in CP for f in FT}; n = 0; t0 = time.time()
    for im in ims:
        f = f"{R}/preds/flows_val/{M}/{im['id']}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); dP, cp = z["dP"].astype(np.float32), z["cellprob"].astype(np.float32); g = gt_rles(im["id"], anns)
        for c in CP:
            lab0 = masks_at(dP, cp, c); err = flow_errors(lab0, dP)
            for ft in FT:
                lab = apply_ft(lab0, err, ft); ids = np.unique(lab); ids = ids[ids > 0]
                pr = [MU.encode(np.asfortranarray((lab == i).astype(np.uint8))) for i in ids]
                if g and pr:
                    iou = MU.iou(pr, g, [0] * len(g)); tp = 0; used = set()
                    gi, pi = np.nonzero(iou.T >= 0.5); order = np.argsort(-iou.T[gi, pi]); ug = set()
                    for k in order:
                        if gi[k] in ug or pi[k] in used: continue
                        ug.add(gi[k]); used.add(pi[k]); tp += 1
                else: tp = 0
                T[(c, ft)] += (tp, len(pr) - tp, len(g) - tp); CNT[(c, ft)].append((len(pr), len(g)))
        n += 1
        if n % 50 == 0: print(f"[{M}] {n} val images {(time.time()-t0)/n:.1f}s/img", flush=True)
    res = {}
    for (c, ft), t in T.items():
        cnt = np.array(CNT[(c, ft)], float); ok = cnt[:, 1] >= 10
        res[f"{c}_{ft}"] = {"cellprob": c, "flow": ft, "F1_50": float(2 * t[0] / max(2 * t[0] + t[1] + t[2], 1)), "TP": int(t[0]), "FP": int(t[1]), "FN": int(t[2]),
                            "count_MAPE": float(np.mean(np.abs(cnt[ok, 0] - cnt[ok, 1]) / cnt[ok, 1]) * 100), "n_img": n}
    best = max(res.values(), key=lambda r: r["F1_50"]); dflt = res["0.0_0.4"] if M != "vista2d" else res.get("0.0_0.4")
    os.makedirs(f"{R}/results/sweep", exist_ok=True)
    json.dump({"method": M, "grid": res, "chosen": best}, open(f"{R}/results/sweep/val__{M}.json", "w"), indent=1)
    for k, r in sorted(res.items(), key=lambda kv: -kv[1]["F1_50"]): print(f"{M} cellprob {r['cellprob']:+.0f} flow {r['flow']:.1f}: F1@.5 {r['F1_50']:.4f} FN {r['FN']} FP {r['FP']} count {r['count_MAPE']:.1f}%")
    print("chosen:", best["cellprob"], best["flow"], "vs default", dflt and dflt["F1_50"])
else:
    if len(sys.argv) > 4: c, ft = float(sys.argv[3]), float(sys.argv[4])
    else: ch = json.load(open(f"{R}/results/sweep/val__{M}.json"))["chosen"]; c, ft = ch["cellprob"], ch["flow"]
    print(f"[{M}] test at cellprob {c} flow {ft}", flush=True); out = []; n = 0; t0 = time.time()
    import scipy.ndimage as ndi
    for im in ims:
        f = f"{R}/preds/flows_test/{M}/{im['id']}.npz"
        if not os.path.exists(f): continue
        z = np.load(f); dP, cp = z["dP"].astype(np.float32), z["cellprob"].astype(np.float32)
        lab = apply_ft(*(lambda l: (l, flow_errors(l, dP)))(masks_at(dP, cp, c)), ft); ids = np.unique(lab); ids = ids[ids > 0]
        prob = 1 / (1 + np.exp(-cp)); sc = ndi.mean(prob, lab, ids) if len(ids) else []
        for i, s in zip(ids, sc):
            r = MU.encode(np.asfortranarray((lab == i).astype(np.uint8))); r["counts"] = r["counts"].decode()
            out.append({"image_id": im["id"], "category_id": 1, "segmentation": r, "score": float(s), "area": float(MU.area(r)), "bbox": [float(x) for x in MU.toBbox(r)]})
        n += 1
        if n % 100 == 0: print(f"[{M}] {n} test images {(time.time()-t0)/n:.1f}s/img", flush=True)
    json.dump(out, open(os.environ.get("OUT_JSON", f"{R}/preds/livecell_sweep_{M}.json"), "w")); print(f"[{M}] wrote {len(out)} instances for {n} images")
