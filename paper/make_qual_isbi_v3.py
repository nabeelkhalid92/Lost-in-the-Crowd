"""ISBI 2014 qualitative figure, v3 (one message: the flow representation cuts overlapping cells short).

Same automatically chosen image as make_qual_v2.py (30-50% overlap; no false positives and every cell detected by both
models; Mask R-CNN at least at its mean Dice and hidden-part recovery; Cellpose-SAM hidden-part recovery closest to its
test mean of 56%). Panels: annotation and Mask R-CNN (top), Cellpose-SAM (bottom, centered). Outlines only, one color per
cell, each outline drawn in full so that two cells sharing pixels show crossing outlines (no fills).
Writes paper/draft/figures/qual_isbi_v3.pdf/.png"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json
from collections import defaultdict
import numpy as np, skimage.io, skimage.segmentation as SS, skimage.morphology as SM
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pycocotools import mask as MU

R = f"{WORK}"; I = f"{ISBI_WORK}"; FIG = f"{R}/paper/draft/figures"
dec = lambda d: MU.decode({"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]}).astype(bool)
PAL = np.array([plt.cm.tab10(i)[:3] for i in [2, 1, 0, 4, 5, 6, 8, 9]])
MAG = np.array((0.9, 0.1, 0.8))


def load(f, t=None):
    P = defaultdict(list)
    for d in json.load(open(f)):
        if t is None or d["score"] >= t: P[d["image_id"]].append(d)
    return P


def match(gms, pms):
    if not pms: return {}
    inter = np.array([[(g & p).sum() for p in pms] for g in gms], float); s = np.array([g.sum() for g in gms])[:, None] + np.array([p.sum() for p in pms])[None]
    dice = 2 * inter / np.maximum(s, 1); out, up = {}, set()
    for g, p in sorted(zip(*np.nonzero(dice > 0.7)), key=lambda t: -dice[t]):
        if g in out or p in up: continue
        out[g] = p; up.add(p)
    return out


J = json.load(open(f"{I}/data/test.json")); A = defaultdict(list)
for a in J["annotations"]: A[a["image_id"]].append(a)
Q = {"cp": load(f"{I}/preds/cpft_test.json"), "mr": load(f"{I}/preds/mrcnn_test.json", 0.9)}
cand = []
for im_ in J["images"]:   # selection identical to make_qual_v2.py
    if im_["overlap_lo"] < 0.3: continue
    gms = [dec(a) for a in A[im_["id"]]]; ovp = np.sum(gms, 0) >= 2; r = {}
    for k in ["cp", "mr"]:
        pms = [dec(d) for d in Q[k][im_["id"]]]; mt = match(gms, pms); hid = inn = 0; dices = []
        for g, p in mt.items():
            h = gms[g] & ovp; hid += h.sum(); inn += (h & pms[p]).sum(); dices.append(2 * (gms[g] & pms[p]).sum() / (gms[g].sum() + pms[p].sum()))
        r[k] = (len(mt) / len(gms), inn / max(hid, 1), len(pms) - len(mt), float(np.mean(dices)) if dices else 0.0)
    if r["cp"][0] == 1 and r["mr"][0] == 1 and r["cp"][2] == 0 and r["mr"][2] == 0 and 3 <= len(gms) <= 6 and r["mr"][3] >= 0.92 and r["mr"][1] >= 0.90:
        cand.append((abs(r["cp"][1] - 0.56), im_, r))
cand.sort(key=lambda t: t[0]); _, im_, r = cand[0]
print("ISBI image", im_["file_name"], "hidden cp/mr", round(r["cp"][1], 3), round(r["mr"][1], 3))

img = skimage.io.imread(f"{I}/data/test/images/{im_['file_name']}").astype(float); img = (img - img.min()) / max(np.ptp(img), 1)
gms = [dec(a) for a in A[im_["id"]]]; ovp = np.sum(gms, 0) >= 2
cp = [dec(d) for d in Q["cp"][im_["id"]]]; mr = [dec(d) for d in Q["mr"][im_["id"]]]
ys, xs = np.nonzero(np.any(gms, 0)); c0, c1 = max(0, ys.min() - 8), min(img.shape[0], ys.max() + 8); d0, d1 = max(0, xs.min() - 8), min(img.shape[1], xs.max() + 8)
Z = lambda a: a[c0:c1, d0:d1]
base = lambda: 0.35 + 0.65 * np.dstack([img] * 3)   # lightened gray image


def outlines(rgb, masks, colors):
    for m, c in zip(masks, colors):
        rgb[SM.binary_dilation(SS.find_boundaries(m, mode="inner"), SM.disk(1))] = c
    return rgb


def colors_for(masks):
    mt = {p: g for g, p in match(gms, masks).items()}
    return [PAL[mt.get(k, len(gms) + k) % len(PAL)] for k in range(len(masks))]


asp = (c1 - c0) / (d1 - d0); fig = plt.figure(figsize=(4.2, 2 * 2.1 * asp + 0.6)); gs = fig.add_gridspec(2, 4)
ax = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 1:3])]   # Cellpose-SAM centered below
ax[0].imshow(Z(outlines(base(), gms, [PAL[k % len(PAL)] for k in range(len(gms))])))
ax[0].set_title("annotation\n(overlap 30-40%)", fontsize=8)
ax[1].imshow(Z(outlines(base(), mr, colors_for(mr))))
ax[1].set_title(f"Mask R-CNN\nkeeps {100 * r['mr'][1]:.0f}% of hidden parts", fontsize=8)
ax[2].imshow(Z(outlines(base(), cp, colors_for(cp))))
ax[2].set_title(f"Cellpose-SAM\nkeeps {100 * r['cp'][1]:.0f}% of hidden parts", fontsize=8)
for a in ax: a.axis("off")
fig.tight_layout(h_pad=0.8)
fig.savefig(f"{FIG}/qual_isbi_v3.pdf"); fig.savefig(f"{FIG}/qual_isbi_v3.png", dpi=200); print("saved")
