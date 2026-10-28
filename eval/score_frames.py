"""Score frame-level predictions (preds/<dataset>/<method>/*.npz).
ctc  : per dataset (both sequences pooled)
       SEG    = CTC SEG measure vs GOLD masks (GT object matched if pred covers >50% of it; mean Jaccard, unmatched=0)
       AP     = COCO segm AP (0.50:0.95) vs complete SILVER masks; AP50 likewise
       CPAP50 = Cellpose-style AP@0.5 = TP/(TP+FP+FN) vs SILVER, at the method's operating point
       count  = instance count at operating point vs number of TRA markers (all count frames)
count: LIVECell nuclear-count benchmark, counts vs fluorescence nuclei counts
Operating point: detectors keep score>=0.5; flow/micro-SAM keep all output instances."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np, tifffile
from pycocotools import mask as MU
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ROOT = f"{WORK}"
DET = {"anchor_based", "anchor_free"}
ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--methods", nargs="+", required=True)
ap.add_argument("--thr", type=float, default=0.5); ap.add_argument("--cutoffs", default="")
ap.add_argument("--common", default="", help="file with frame keys shared by all methods (CTC counting)"); a = ap.parse_args()
CUT = json.load(open(a.cutoffs)) if a.cutoffs else {}
COMMON = set(open(a.common).read().split()) if a.common else None

def load(p):
    z = np.load(p); return z["lab"].astype(np.int32), z["scores"], float(z["sec"])

def op_filter(lab, sc, method):
    if method not in DET or lab.max() == 0: return lab, sc
    thr = CUT.get(method, a.thr)
    n = lab.max(); s_ = np.zeros(n, np.float32); s_[:min(n, len(sc))] = sc[:min(n, len(sc))]
    keep = np.zeros(n + 1, bool); keep[1:] = s_ >= thr
    idx = np.zeros(lab.max() + 1, np.int32); idx[keep] = np.arange(1, keep.sum() + 1)
    return idx[lab], s_[s_ >= thr]

def iou_match(gt, pr):
    """TP count at IoU>0.5 between label images (matches are unique at IoU>0.5)."""
    if gt.max() == 0 or pr.max() == 0: return 0, int(gt.max() and len(np.unique(gt)) - 1), int(pr.max() and len(np.unique(pr)) - 1)
    g = gt.ravel(); p = pr.ravel(); m = (g > 0) & (p > 0)
    pair, inter = np.unique(np.stack([g[m], p[m]]), axis=1, return_counts=True)
    ga = np.bincount(g, minlength=g.max() + 1); pa = np.bincount(p, minlength=p.max() + 1)
    iou = inter / (ga[pair[0]] + pa[pair[1]] - inter)
    tp = int((iou > 0.5).sum()); ng = int((ga[1:] > 0).sum()); npred = int((pa[1:] > 0).sum())
    return tp, ng, npred

def ctc_seg(gt, pr):
    """CTC SEG: list of Jaccard per GT object."""
    out = []
    for g in np.unique(gt)[1:]:
        gm = gt == g; ids, c = np.unique(pr[gm], return_counts=True)
        j = 0.0
        for i, n in zip(ids, c):
            if i > 0 and n > 0.5 * gm.sum():
                pm = pr == i; j = (gm & pm).sum() / (gm | pm).sum()
        out.append(j)
    return out

import scipy.ndimage as ndi
def rles(lab):
    """[(label_id, RLE)] for every non-empty label (label ids may have gaps, e.g. CTC silver truth = track ids)."""
    out = []; H, W = lab.shape
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        m = np.zeros((H, W), np.uint8, order="F"); m[sl] = (lab[sl] == k)
        if m.any(): out.append((k, MU.encode(m)))
    return out

def count_metrics(pairs):
    p = np.array([x[0] for x in pairs], float); g = np.array([x[1] for x in pairs], float)
    return {"n": len(p), "MAE": float(np.mean(np.abs(p - g))), "MAPE": float(np.mean(np.abs(p - g) / np.maximum(g, 1)) * 100),
            "bias_pct": float(np.mean((p - g) / np.maximum(g, 1)) * 100), "r": float(np.corrcoef(p, g)[0, 1]) if len(p) > 2 else None}

res = {}
if a.dataset == "ctc":
    man = json.load(open(f"{ROOT}/data/ctc_manifest.json"))
    datasets = sorted(set(k.split("/")[0] for k in man))
    for M in a.methods:
        res[M] = {}
        for D in datasets:
            seg, cnt, cntc, tp = [], [], [], [0, 0, 0]; secs = []
            gt_anns, dt_anns, imgs = [], [], []; aid = 1
            for seq in [s for s in man if s.startswith(D + "/")]:
                m = man[seq]; g = seq.replace("/", "_")
                for t, gp in m["seg_gold"].items():
                    f = f"{ROOT}/preds/ctc/{M}/{g}__t{int(t):04d}.npz"
                    if not os.path.exists(f): continue
                    lab, sc, sec = load(f); secs.append(sec)
                    # COCO AP vs silver uses ALL scored instances (ranked); other metrics use the operating point
                    sil = tifffile.imread(m["seg_silver"][t]).astype(np.int32)
                    iid = len(imgs) + 1; imgs.append({"id": iid, "height": sil.shape[0], "width": sil.shape[1]})
                    for _, r in rles(sil):
                        gt_anns.append({"id": aid, "image_id": iid, "category_id": 1, "segmentation": r, "area": float(MU.area(r)), "bbox": list(MU.toBbox(r)), "iscrowd": 0}); aid += 1
                    for k, r in rles(lab):
                        dt_anns.append({"image_id": iid, "category_id": 1, "segmentation": r, "score": float(sc[k - 1]) if len(sc) >= k else 1.0})
                    labo, _ = op_filter(lab, sc, M)
                    seg += ctc_seg(tifffile.imread(gp).astype(np.int32), labo)
                    x = iou_match(sil, labo); tp = [tp[i] + x[i] for i in range(3)]
                for t, c in m["count_frames"].items():
                    f = f"{ROOT}/preds/ctc/{M}/{g}__t{int(t):04d}.npz"
                    if not os.path.exists(f): continue
                    lab, sc, sec = load(f); labo, _ = op_filter(lab, sc, M)
                    cnt.append((len(np.unique(labo)) - 1, c))
                    if COMMON is not None and f"{g}__t{int(t):04d}" in COMMON: cntc.append((len(np.unique(labo)) - 1, c))
            if not imgs: continue
            for r in gt_anns + dt_anns: r["segmentation"]["counts"] = r["segmentation"]["counts"].decode() if isinstance(r["segmentation"]["counts"], bytes) else r["segmentation"]["counts"]
            G = COCO(); G.dataset = {"images": imgs, "annotations": gt_anns, "categories": [{"id": 1}]}; G.createIndex()
            AP = AP50 = AP75 = 0.0
            if dt_anns:
                Dt = G.loadRes(dt_anns); E = COCOeval(G, Dt, "segm"); E.params.maxDets = [100, 500, 3000]; E.params.useCats = 0
                E.evaluate(); E.accumulate()
                pr = E.eval["precision"][:, :, 0, 0, -1]; AP = float(np.mean(pr[pr > -1])) * 100
                p50 = E.eval["precision"][0, :, 0, 0, -1]; AP50 = float(np.mean(p50[p50 > -1])) * 100
                p75 = E.eval["precision"][5, :, 0, 0, -1]; AP75 = float(np.mean(p75[p75 > -1])) * 100
            res[M][D] = {"SEG": float(np.mean(seg)) if seg else None, "n_gold_obj": len(seg), "AP": AP, "AP50": AP50, "AP75": AP75,
                         "CPAP50": tp[0] / max(tp[1] + tp[2] - tp[0], 1), "count": count_metrics(cnt) if cnt else None,
                         "count_common": count_metrics(cntc) if len(cntc) > 2 else None,
                         "sec_per_img": float(np.mean(secs)) if secs else None, "n_seg_frames": len(imgs)}
            r = res[M][D]
            print(f"{M:13s} {D:14s} SEG {r['SEG']:.3f}  AP {AP:5.1f}  AP50 {AP50:5.1f}  CPAP50 {r['CPAP50']:.3f}  "
                  f"count MAPE {r['count']['MAPE'] if r['count'] else float('nan'):6.1f}%  bias {r['count']['bias_pct'] if r['count'] else float('nan'):+6.1f}%", flush=True)
elif a.dataset == "count":
    FL = f"{FC_WORK}/counting_data"
    DENS = {"A172": {"8": "2.5k", "7": "5k", "6": "10k"}, "A549": {"6": "0.5k", "5": "1k", "4": "2k", "3": "4k", "2": "8k"}}
    for M in a.methods:
        res[M] = {}
        for cell in ["A172", "A549"]:
            gt = {k: float(v) for k, v in json.load(open(f"{FL}/{cell}_counts.json")).items()}
            pairs = []; byd = defaultdict(list)
            for fn, g in gt.items():
                stem = fn[:-4]; f = f"{ROOT}/preds/count/{M}/{cell}__{stem}.npz"
                if os.path.exists(f):
                    lab, sc, _ = load(f); labo, _ = op_filter(lab, sc, M); p = len(np.unique(labo)) - 1
                else:
                    cj = f"{ROOT}/results/counting/{cell}_counts_{M}.json"
                    if not os.path.exists(cj): continue
                    c = json.load(open(cj))
                    if fn not in c: continue
                    p = c[fn]
                pairs.append((p, g)); col = re.search(r"[A-Z](\d)-1_", fn).group(1); byd[DENS[cell].get(col, "?")].append((p, g))
            os.makedirs(f"{ROOT}/results/counting_pairs", exist_ok=True); json.dump(pairs, open(f"{ROOT}/results/counting_pairs/{cell}_{M}.json", "w"))
            if pairs:
                res[M][cell] = {"all": count_metrics(pairs), "by_density": {k: count_metrics(v) for k, v in byd.items()}}
                r = res[M][cell]["all"]
                print(f"{M:13s} {cell} n={r['n']:4d} MAE {r['MAE']:7.1f}  MAPE {r['MAPE']:5.1f}%  bias {r['bias_pct']:+6.1f}%  r {r['r']:.3f}", flush=True)
os.makedirs(f"{ROOT}/results/scores", exist_ok=True)
for M in res: json.dump(res[M], open(f"{ROOT}/results/scores/{a.dataset}_{M}.json", "w"), indent=1)
