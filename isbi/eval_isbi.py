"""Evaluate ISBI 2014 predictions, overall and per overlap bin of the test images (OverlapRatio of the synthetic set).
ISBI challenge conventions (Lu et al.): an annotated cell is detected if a prediction overlaps it with Dice > 0.7
(greedy one-to-one, highest Dice first); FNo = share of undetected cells; Dice = mean Dice of detected cells. Plus:
F1 at IoU 0.5; overlap pixels (inside >= 2 annotated cells) covered by >= 2 predictions; hidden-part recovery = for
detected cells with overlap, share of the cell's overlapped pixels inside its matched mask.
Operating point: predictions with score >= cutoff (detectors: chosen on val by F1@0.5; flow models: all instances).
usage: eval_isbi.py <data_dir> <split> <preds.json> <cutoff> <name>"""
import json, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
D, SPLIT, P, CUT, NAME = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
J = json.load(open(f"{D}/{SPLIT}.json")); A = defaultdict(list); PR = defaultdict(list)
for a in J["annotations"]: A[a["image_id"]].append({"size": a["segmentation"]["size"], "counts": a["segmentation"]["counts"].encode()})
for p in json.load(open(P)):
    if p["score"] >= CUT: PR[p["image_id"]].append({"size": p["segmentation"]["size"], "counts": p["segmentation"]["counts"].encode()})
bins = defaultdict(lambda: defaultdict(float))
for im in J["images"]:
    gt, pr = A[im["id"]], PR[im["id"]]; b = f"{im.get('overlap_lo', 0):.1f}-{im.get('overlap_hi', 0):.1f}"
    gm = np.stack([MU.decode(g).astype(bool) for g in gt]); ovp = gm.sum(0) >= 2
    pm = np.stack([MU.decode(p).astype(bool) for p in pr]) if pr else np.zeros((0,) + gm.shape[1:], bool)
    inter = np.einsum("ghw,phw->gp", gm.astype(np.float32), pm.astype(np.float32)) if len(pr) else np.zeros((len(gt), 0))
    ga, pa = gm.sum((1, 2))[:, None], pm.sum((1, 2))[None]
    dice = 2 * inter / np.maximum(ga + pa, 1); iou = inter / np.maximum(ga + pa - inter, 1)
    def greedy(S, thr):
        pairs, ug, up = [], set(), set()
        for g, p in sorted(zip(*np.nonzero(S > thr)), key=lambda t: -S[t]):
            if g in ug or p in up: continue
            pairs.append((g, p)); ug.add(g); up.add(p)
        return pairs
    md = greedy(dice, 0.7); mi = greedy(iou, 0.5)
    for key in ["all", b]:
        c = bins[key]; c["n_gt"] += len(gt); c["n_pr"] += len(pr); c["det"] += len(md); c["dice_sum"] += sum(dice[g, p] for g, p in md); c["tp50"] += len(mi)
        c["ov_px"] += ovp.sum(); c["ov_px2"] += (ovp & (pm.sum(0) >= 2)).sum() if len(pr) else 0
        for g, p in md:
            hid = gm[g] & ovp
            if hid.sum(): c["hid_px"] += hid.sum(); c["hid_in"] += (hid & pm[p]).sum()
out = {}
for k in sorted(bins, key=lambda s: (s != "all", s)):
    c = bins[k]; r = {"n_cells": int(c["n_gt"]), "FNo": 100 * (1 - c["det"] / c["n_gt"]), "Dice": c["dice_sum"] / max(c["det"], 1),
                      "F1_50": 2 * c["tp50"] / max(c["n_gt"] + c["n_pr"], 1), "overlap_px_double": 100 * c["ov_px2"] / max(c["ov_px"], 1),
                      "hidden_recovered": 100 * c["hid_in"] / max(c["hid_px"], 1)}
    out[k] = r
    print(f"{NAME:28s} {k:8s} n={r['n_cells']:5d} FNo {r['FNo']:5.1f}%  Dice {r['Dice']:.3f}  F1@.5 {r['F1_50']:.3f}  overlap px x2 {r['overlap_px_double']:5.1f}%  hidden part recovered {r['hidden_recovered']:5.1f}%")
json.dump(out, open(P.replace(".json", f"_eval_{CUT}.json"), "w"), indent=1)
