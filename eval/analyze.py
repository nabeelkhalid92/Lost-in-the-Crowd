"""Instance-level equal-treatment analysis for one (dataset, method).

Datasets
  livecell  : LIVECell test (GT = manual COCO masks, may overlap)
  livecell_val : LIVECell validation (detectors only; used to choose the score cutoff)
  ctc       : CTC gold-segmentation frames, GT = complete silver-truth masks

For each image we build GT instances and predicted instances (RLE) and compute, at every IoU threshold
0.50:0.05:0.95, one-to-one greedy matches -> TP / FP / FN, pooled F1 and the per-image Cellpose-style AP
TP/(TP+FP+FN). Detectors are scored
  raw       : every detection is its own instance (overlaps allowed)
  flattened : detections painted into one label image (higher score on top), then re-extracted
at each score cutoff in CUTOFFS; flow / SAM methods output label images, so raw == flattened for them.
At IoU 0.5 every GT cell gets one outcome: tight (IoU>=0.75), loose (0.5<=IoU<0.75), merged, split, missed;
every unmatched prediction is a merger, a split fragment or a false positive. Outcomes are stored per GT cell
together with its local density (other GT centroids within 1.5 x its own equivalent diameter) and, for
LIVECell, the fraction of its area overlapped by other GT cells.
Writes results/analysis/<dataset>__<method>.json (summary) and .npz (per-GT outcome table)."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, glob, json, os, re, sys, time
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
import scipy.ndimage as ndi

ROOT = os.environ.get("BENCH", f"{WORK}")
IOUS = np.round(np.arange(0.50, 0.951, 0.05), 2)
DET = {"anchor_based", "anchor_free"}
CUTOFFS = [float(c) for c in os.environ["CUTOFFS"].split(",")] if os.environ.get("CUTOFFS") else [0.3, 0.4, 0.5, 0.6, 0.7]
LC_PRED = {  # LIVECell COCO prediction files (flow_comparison runs + this benchmark)
    "cyto": f"{FC_WORK}/predictions/full/cyto.json",
    "cyto2": f"{FC_WORK}/predictions/full/cyto2.json",
    "cyto3": f"{FC_WORK}/predictions/full/cyto3.json",
    "livecell": f"{FC_WORK}/predictions/full/livecell.json",
    "vista2d": f"{FC_WORK}/predictions/full/vista2d.json",
    "anchor_based": f"{FC_WORK}/predictions/full/anchor_based.json",
    "anchor_free": f"{FC_WORK}/predictions/full/anchor_free.json",
    "cpsam": f"{ROOT}/preds/livecell_cpsam.json",
    "microsam": f"{ROOT}/preds/livecell_microsam.json",
}
LC_GT = {"livecell": os.environ.get("LC_GT", f"{LC_ANN}/livecell_coco_test.json"),
         "livecell_val": os.environ.get("LC_GT_VAL", f"{LC_ANN}/livecell_coco_val.json")}


def rle_of_label(lab):
    out = []
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        m = np.zeros(lab.shape, np.uint8, order="F"); m[sl] = lab[sl] == k
        out.append(MU.encode(m))
    return out


def flatten(rles, scores, shape):
    """Paint instances (ascending score -> highest on top) into a label image and re-extract them."""
    lab = np.zeros(shape, np.int32); order = np.argsort(scores)
    for n, i in enumerate(order, 1):
        lab[MU.decode(rles[i]).astype(bool)] = n
    ids = np.unique(lab); ids = ids[ids > 0]
    out_r, out_s = [], []
    for i in ids:
        out_r.append(MU.encode(np.asfortranarray((lab == i).astype(np.uint8)))); out_s.append(scores[order[i - 1]])
    return out_r, np.array(out_s, np.float32)


def match(iou, thr):
    """Greedy one-to-one matching by descending IoU; returns list of (g, p, iou)."""
    g, p = np.nonzero(iou >= thr)
    if len(g) == 0: return []
    o = np.argsort(-iou[g, p]); ug, up, res = set(), set(), []
    for k in o:
        a, b = g[k], p[k]
        if a in ug or b in up: continue
        ug.add(a); up.add(b); res.append((a, b, iou[a, b]))
    return res


def evaluate_image(gt_r, pr_r, gt_attr):
    """Return per-IoU (TP, FP, FN), per-GT outcome codes and per-pred class at IoU 0.5."""
    nG, nP = len(gt_r), len(pr_r)
    if nG and nP:
        iou = MU.iou(pr_r, gt_r, [0] * nG).T  # [G, P]
        ag = MU.area(gt_r).astype(np.float64); ap_ = MU.area(pr_r).astype(np.float64)
        inter = iou * (ag[:, None] + ap_[None, :]) / (1 + iou)
    else:
        iou = np.zeros((nG, nP)); inter = iou; ag = MU.area(gt_r) if nG else np.zeros(0); ap_ = MU.area(pr_r) if nP else np.zeros(0)
    tpfpfn = []
    for t in IOUS:
        m = match(iou, t); tpfpfn.append((len(m), nP - len(m), nG - len(m)))
    # outcome taxonomy at IoU 0.5
    m = match(iou, 0.5); out = np.full(nG, 4, np.int8)  # 0 tight 1 loose 2 merged 3 split 4 missed
    mg = {a: (b, v) for a, b, v in m}; mp = {b for a, b, v in m}
    for a, (b, v) in mg.items(): out[a] = 0 if v >= 0.75 else 1
    pcls = np.zeros(nP, np.int8)  # 0 matched 1 merger 2 split fragment 3 FP
    if nG and nP:
        cov_g = inter / np.maximum(ag[:, None], 1)      # fraction of GT covered by pred
        cov_p = inter / np.maximum(ap_[None, :], 1)     # fraction of pred inside GT
        for b in range(nP):
            if b in mp: continue
            gs = np.nonzero(cov_g[:, b] > 0.5)[0]
            if len(gs) >= 2:
                pcls[b] = 1
                for a in gs:
                    if a not in mg: out[a] = 2
        for a in range(nG):
            if out[a] != 4: continue
            frag = np.nonzero((cov_p[a] > 0.5) & (inter[a] > 0))[0]
            if len(frag) >= 2:
                out[a] = 3
                for b in frag:
                    if b not in mp and pcls[b] == 0: pcls[b] = 2
        for b in range(nP):
            if b not in mp and pcls[b] == 0: pcls[b] = 3
    elif nP:
        pcls[:] = 3
    return np.array(tpfpfn), out, pcls


def gt_attributes(gt_r, overlaps=True):
    """Local density (other centroids within 1.5 x own diameter) and overlap fraction."""
    n = len(gt_r)
    if n == 0: return np.zeros((0, 2))
    bb = MU.toBbox(gt_r); c = bb[:, :2] + bb[:, 2:] / 2; a = MU.area(gt_r).astype(np.float64)
    d = 2 * np.sqrt(a / np.pi)
    dist = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
    dens = (dist < 1.5 * d[:, None]).sum(1)
    ov = np.zeros(n)
    if overlaps and n > 1:
        iou = MU.iou(gt_r, gt_r, [0] * n); np.fill_diagonal(iou, 0)
        inter = iou * (a[:, None] + a[None, :]) / (1 + iou)
        ov = np.minimum(inter.sum(1) / np.maximum(a, 1), 1.0)
    return np.stack([dens, ov], 1)


# optional: drop near-duplicate GT annotations (results/gt_duplicates.json, from gt_duplicates.py)
GT_DROP = set(json.load(open(os.environ["GT_DROP"]))["dropped_ann_ids"]) if os.environ.get("GT_DROP") else set()


def images(dataset, method):
    """Yield (image_key, gt_rles, pred_rles, pred_scores, shape)."""
    if dataset in ("livecell", "livecell_val"):
        from pycocotools.coco import COCO
        G = COCO(LC_GT[dataset])
        pf = os.environ.get("PRED_JSON") or (LC_PRED.get(method, f"{ROOT}/preds/weaksup/{method}.json") if dataset == "livecell" else f"{ROOT}/preds/livecell_val_{method}.json")
        P = defaultdict(list)
        for d in json.load(open(pf)): P[d["image_id"]].append(d)
        for iid in sorted(G.imgs):
            im = G.imgs[iid]; h, w = im["height"], im["width"]
            gt = [G.annToRLE(a) for a in G.loadAnns([x for x in G.getAnnIds(imgIds=[iid]) if x not in GT_DROP])]
            ps = P.get(iid, [])
            pr = [{"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]} for d in ps]
            yield (im["file_name"], gt, pr, np.array([d["score"] for d in ps], np.float32), (h, w))
    elif dataset == "coco":
        from pycocotools.coco import COCO
        G = COCO(os.environ["COCO_JSON"]); cat = int(os.environ.get("COCO_CAT", "1")); P = defaultdict(list)
        for d in json.load(open(os.environ["PRED_JSON"])): P[d["image_id"]].append(d)
        for iid in sorted(G.imgs):
            im = G.imgs[iid]; h, w = im["height"], im["width"]
            gt = [G.annToRLE(a) for a in G.loadAnns(G.getAnnIds(imgIds=[iid], catIds=[cat], iscrowd=False))]
            ps = P.get(iid, [])
            pr = [{"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]} for d in ps]
            yield (str(iid), gt, pr, np.array([d["score"] for d in ps], np.float32), (h, w))
    else:
        import tifffile
        man = json.load(open(f"{ROOT}/data/ctc_manifest.json"))
        for seq, m in man.items():
            g = seq.replace("/", "_")
            for t in m["seg_gold"]:
                key = f"{g}__t{int(t):04d}"
                sil = tifffile.imread(m["seg_silver"][t]).astype(np.int32)
                if method in DET:
                    f = f"{ROOT}/preds/ctc_raw/{method}/{key}_raw.json"
                    if not os.path.exists(f): continue
                    raw = json.load(open(f))
                    pr = [{"size": r["segmentation"]["size"], "counts": r["segmentation"]["counts"].encode()} for r in raw]
                    sc = np.array([r["score"] for r in raw], np.float32)
                else:
                    f = f"{ROOT}/preds/ctc/{method}/{key}.npz"
                    if not os.path.exists(f): continue
                    z = np.load(f); lab = z["lab"].astype(np.int32)
                    pr = rle_of_label(lab); sc = np.ones(len(pr), np.float32)
                yield (key, rle_of_label(sil), pr, sc, sil.shape)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--method", required=True)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    global OUT_DS; OUT_DS = os.environ.get("OUT_DS", a.dataset)
    det = a.method in DET or a.method.startswith(("PACE", "Point2Mask", "BoxSup", "flowprompt"))
    modes = [("raw", c) for c in CUTOFFS] + [("flat", c) for c in CUTOFFS] if det else [("flat", None)]
    acc = {f"{mo}_{c}": {"tpfpfn": np.zeros((len(IOUS), 3), np.int64), "cpap_img": [], "tff_img": [], "outcomes": [], "pcls": [], "attr": [],
                         "counts": []} for mo, c in modes}
    t0 = time.time(); n = 0
    for key, gt, pr, sc, shape in images(a.dataset, a.method):
        attr = gt_attributes(gt, overlaps=a.dataset.startswith("livecell") or a.dataset == "coco")
        for mo, c in modes:
            if det:
                keep = np.nonzero(sc >= c)[0]; prc = [pr[i] for i in keep]; scc = sc[keep]
                if mo == "flat" and len(prc): prc, scc = flatten(prc, scc, shape)
            else:
                prc = pr
            tff, out, pcls = evaluate_image(gt, prc, attr)
            A = acc[f"{mo}_{c}"]; A["tpfpfn"] += tff
            A["cpap_img"].append(tff[:, 0] / np.maximum(tff[:, 0] + tff[:, 1] + tff[:, 2], 1)); A["tff_img"].append(tff)
            A["outcomes"].append(out); A["pcls"].append(pcls); A["attr"].append(attr); A["counts"].append((len(prc), len(gt), key))
        n += 1
        if n % 100 == 0: print(f"[{a.dataset}/{a.method}] {n} images {(time.time()-t0)/n:.2f}s/img", flush=True)
        if a.limit and n >= a.limit: break
    os.makedirs(f"{ROOT}/results/analysis", exist_ok=True)
    summ = {}
    for k, A in acc.items():
        T = A["tpfpfn"]; tp, fp, fn = T[:, 0], T[:, 1], T[:, 2]
        out = np.concatenate(A["outcomes"]) if A["outcomes"] else np.zeros(0, np.int8)
        pc = np.concatenate(A["pcls"]) if A["pcls"] else np.zeros(0, np.int8)
        nG = max(len(out), 1)
        cnt = np.array([(x[0], x[1]) for x in A["counts"]], float)
        summ[k] = {"iou": IOUS.tolist(), "F1": (2 * tp / np.maximum(2 * tp + fp + fn, 1)).tolist(),
                   "CPAP_mean_img": np.mean(A["cpap_img"], 0).tolist(), "CPAP_pooled": (tp / np.maximum(tp + fp + fn, 1)).tolist(),
                   "n_gt": int(len(out)), "n_pred": int(len(pc)),
                   "per_1000_gt": {"tight": 1000 * float((out == 0).sum()) / nG, "loose": 1000 * float((out == 1).sum()) / nG,
                                   "merged_gt": 1000 * float((out == 2).sum()) / nG, "split_gt": 1000 * float((out == 3).sum()) / nG,
                                   "missed": 1000 * float((out == 4).sum()) / nG, "merger_preds": 1000 * float((pc == 1).sum()) / nG,
                                   "split_fragments": 1000 * float((pc == 2).sum()) / nG, "false_pos": 1000 * float((pc == 3).sum()) / nG},
                   "count_MAPE_vs_gt": float(np.mean(np.abs(cnt[:, 0] - cnt[:, 1]) / np.maximum(cnt[:, 1], 1)) * 100) if len(cnt) else None,
                   "count_bias_vs_gt": float(np.mean((cnt[:, 0] - cnt[:, 1]) / np.maximum(cnt[:, 1], 1)) * 100) if len(cnt) else None}
        np.savez_compressed(f"{ROOT}/results/analysis/{OUT_DS}__{a.method}__{k}.npz", outcome=out,
                            attr=np.concatenate(A["attr"]) if A["attr"] else np.zeros((0, 2)), pcls=pc,
                            counts=cnt, keys=np.array([x[2] for x in A["counts"]]),
                            cpap_img=np.array(A["cpap_img"]), tff_img=np.array(A["tff_img"]))
    json.dump(summ, open(f"{ROOT}/results/analysis/{OUT_DS}__{a.method}.json", "w"), indent=1)
    for k, s in summ.items():
        print(f"{a.dataset} {a.method} {k}: F1@.5 {s['F1'][0]:.3f} F1@.75 {s['F1'][5]:.3f} CPAP@.5 {s['CPAP_mean_img'][0]:.3f} "
              f"miss {s['per_1000_gt']['missed']:.0f} merged {s['per_1000_gt']['merged_gt']:.0f} split {s['per_1000_gt']['split_gt']:.0f} "
              f"FP {s['per_1000_gt']['false_pos']:.0f} loose {s['per_1000_gt']['loose']:.0f}", flush=True)


if __name__ == "__main__":
    main()
