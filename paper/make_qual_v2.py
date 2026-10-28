"""Two qualitative figures for RQ2, one message each (examples chosen automatically, not by hand):
  qual_livecell: crowding gap is set by the operating point;  qual_isbi: the flow representation cuts overlapping cells short.
row 1  LIVECell test: the 256x256 crop with the most crowded cells (>= 5 neighbors) that default Cellpose-SAM misses
       and relaxed Cellpose-SAM finds. Annotated cells filled by outcome at IoU 0.5 (green found, red missed);
       cyan: predicted outlines. Panels: image, Cellpose-SAM default, relaxed, CenterMask2 (cutoff 0.5).
row 2  ISBI 2014 test, overlap 30-50%: the image where Mask R-CNN recovers the most more of the cells' hidden parts than
       Cellpose-SAM while both detect >= 80% of the cells. Panels: annotation, Cellpose-SAM, Mask R-CNN (outlines, one
       color per instance), hidden parts that Cellpose-SAM leaves out (magenta).
Writes paper/draft/figures/qualitative_v2.pdf/.png"""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, sys
from collections import defaultdict
import numpy as np, skimage.io, skimage.segmentation as SS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pycocotools import mask as MU
from pycocotools.coco import COCO
sys.argv = sys.argv[:1]; sys.path.insert(0, f"{REPO}")
from analyze import gt_attributes
R = f"{WORK}"; I = f"{ISBI_WORK}"; FIG = f"{R}/paper/draft/figures"
dec = lambda d: MU.decode({"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]}).astype(bool)
PAL = np.array([plt.cm.tab10(i)[:3] for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]] + [plt.cm.Set2(i)[:3] for i in range(8)])
# ---------------- LIVECell
G = COCO(f"{LC_ANN}/livecell_coco_test.json")
def load(f, t=None):
    P = defaultdict(list)
    for d in json.load(open(f)):
        if t is None or d["score"] >= t: P[d["image_id"]].append(d)
    return P
P0, P1 = load(f"{R}/preds/livecell_cpsam.json"), load(f"{R}/preds/livecell_sweep_cpsam.json")
rl = lambda ds: [{"size": d["segmentation"]["size"], "counts": d["segmentation"]["counts"].encode() if isinstance(d["segmentation"]["counts"], str) else d["segmentation"]["counts"]} for d in ds]
best = (-1, None)
for iid in sorted(G.imgs)[::2]:
    anns = G.loadAnns(G.getAnnIds(imgIds=[iid]))
    if len(anns) < 30: continue
    gt = [G.annToRLE(a) for a in anns]; cr = gt_attributes(gt)[:, 0] >= 5
    f0 = MU.iou(gt, rl(P0[iid]), [0] * len(P0[iid])).max(1) >= 0.5 if P0[iid] else np.zeros(len(gt), bool)
    f1 = MU.iou(gt, rl(P1[iid]), [0] * len(P1[iid])).max(1) >= 0.5 if P1[iid] else np.zeros(len(gt), bool)
    gain = cr & ~f0 & f1
    if gain.sum() < 3: continue
    bb = np.array([a["bbox"] for a in anns]); cx, cy = bb[:, 0] + bb[:, 2] / 2, bb[:, 1] + bb[:, 3] / 2
    H, W = G.imgs[iid]["height"], G.imgs[iid]["width"]
    for y0 in range(0, H - 255, 32):
        for x0 in range(0, W - 255, 32):
            n = (gain & (cx >= x0 + 20) & (cx < x0 + 236) & (cy >= y0 + 20) & (cy < y0 + 236)).sum()
            if n > best[0]: best = (n, (iid, y0, x0))
iid, y0, x0 = best[1]; print("LIVECell crop", G.imgs[iid]["file_name"], y0, x0, "gain", best[0])
PC = load(f"{R}/preds/livecell_cap3000_anchor_free.json", 0.5)
img = skimage.io.imread(f"{LC_IMG}/livecell_test_images/{G.imgs[iid]['file_name']}").astype(float)
crop = lambda a: a[y0:y0 + 256, x0:x0 + 256]
im = crop(img); im = (im - im.min()) / max(np.ptp(im), 1)
anns = G.loadAnns(G.getAnnIds(imgIds=[iid])); gt = [G.annToRLE(a) for a in anns]; gm = [crop(G.annToMask(a).astype(bool)) for a in anns]
fig, ax = plt.subplots(1, 4, figsize=(10, 3.1)); ax = ax[None]
def lc_panel(a, P, title):
    rgb = np.dstack([im] * 3)
    if P is not None:
        found = MU.iou(gt, rl(P), [0] * len(P)).max(1) >= 0.5 if P else np.zeros(len(gt), bool)
        for g, f in zip(gm, found):
            if g.any(): rgb[g] = 0.55 * rgb[g] + 0.45 * np.array((0.1, 0.8, 0.2) if f else (0.95, 0.1, 0.1))
        for d in P:
            m = crop(dec(d))
            if m.any(): rgb[SS.find_boundaries(m, mode="inner")] = (0, 0.85, 0.95)
        title += f"\n{int((~found[[g.any() for g in gm]]).sum())} of {sum(g.any() for g in gm)} cells missed"
    else:
        for k, g in enumerate(gm):
            if g.any(): rgb[SS.find_boundaries(g, mode="inner")] = PAL[k % len(PAL)]
    a.imshow(rgb); a.set_title(title, fontsize=8); a.axis("off")
lc_panel(ax[0, 0], None, "annotation (MCF7)"); lc_panel(ax[0, 1], P0[iid], "Cellpose-SAM, default")
lc_panel(ax[0, 2], P1[iid], "Cellpose-SAM, relaxed"); lc_panel(ax[0, 3], PC[iid], "CenterMask2, cutoff 0.5")
fig.tight_layout(); fig.savefig(f"{FIG}/qual_livecell.pdf"); fig.savefig(f"{FIG}/qual_livecell.png", dpi=150); plt.close(fig)
# ---------------- ISBI
J = json.load(open(f"{I}/data/test.json")); A = defaultdict(list)
for a in J["annotations"]: A[a["image_id"]].append(a)
Q = {"cp": load(f"{I}/preds/cpft_test.json"), "mr": load(f"{I}/preds/mrcnn_test.json", 0.9)}
def match(gms, pms):
    if not pms: return {}
    inter = np.array([[(g & p).sum() for p in pms] for g in gms], float); s = np.array([g.sum() for g in gms])[:, None] + np.array([p.sum() for p in pms])[None]
    dice = 2 * inter / np.maximum(s, 1); out, up = {}, set()
    for g, p in sorted(zip(*np.nonzero(dice > 0.7)), key=lambda t: -dice[t]):
        if g in out or p in up: continue
        out[g] = p; up.add(p)
    return out
MEAN_GAP = 0.92 - 0.56; cand = []   # test-set hidden-part recovery, Mask R-CNN minus Cellpose-SAM (Table isbi)
for im_ in J["images"]:
    if im_["overlap_lo"] < 0.3: continue
    gms = [dec(a) for a in A[im_["id"]]]; ovp = np.sum(gms, 0) >= 2; r = {}
    for k in ["cp", "mr"]:
        pms = [dec(d) for d in Q[k][im_["id"]]]; mt = match(gms, pms); hid = inn = 0; dices = []
        for g, p in mt.items():
            h = gms[g] & ovp; hid += h.sum(); inn += (h & pms[p]).sum(); dices.append(2 * (gms[g] & pms[p]).sum() / (gms[g].sum() + pms[p].sum()))
        r[k] = (len(mt) / len(gms), inn / max(hid, 1), len(pms) - len(mt), float(np.mean(dices)) if dices else 0.0)
    # typical behavior of both models: no false positives, every cell detected, Mask R-CNN at or above its test-set
    # mean Dice (0.924) and hidden-part recovery (92%); Cellpose-SAM hidden-part recovery closest to its mean (56%)
    if r["cp"][0] == 1 and r["mr"][0] == 1 and r["cp"][2] == 0 and r["mr"][2] == 0 and 3 <= len(gms) <= 6 and r["mr"][3] >= 0.92 and r["mr"][1] >= 0.90:
        cand.append((abs(r["cp"][1] - 0.56), im_, r))
cand.sort(key=lambda t: t[0]); d, im_, r = cand[0]; print(len(cand), "candidates without false positives"); print("ISBI image", im_["file_name"], "overlap", im_["overlap_lo"], "hidden cp/mr", r)
img = skimage.io.imread(f"{I}/data/test/images/{im_['file_name']}").astype(float); img = (img - img.min()) / max(np.ptp(img), 1)
gms = [dec(a) for a in A[im_["id"]]]; ovp = np.sum(gms, 0) >= 2
ys, xs = np.nonzero(np.any(gms, 0)); c0, c1 = max(0, ys.min() - 20), min(img.shape[0], ys.max() + 20); d0, d1 = max(0, xs.min() - 20), min(img.shape[1], xs.max() + 20)
Z = lambda a: a[c0:c1, d0:d1]
def isbi_panel(a, masks, title):
    rgb = np.dstack([img] * 3)
    mt = {p: g for g, p in match(gms, masks).items()} if masks is not gms else {k: k for k in range(len(gms))}
    for k, m in enumerate(masks):   # translucent fill, so a pixel shared by two masks shows both colors
        rgb[m] = 0.65 * rgb[m] + 0.35 * np.array(PAL[mt.get(k, len(gms) + k) % len(PAL)])
    for k, m in enumerate(masks): rgb[SS.find_boundaries(m, mode="thick")] = PAL[mt.get(k, len(gms) + k) % len(PAL)]   # color of the matched annotated cell
    a.imshow(Z(rgb)); a.set_title(title, fontsize=8); a.axis("off")
cp = [dec(dd) for dd in Q["cp"][im_["id"]]]; mr = [dec(dd) for dd in Q["mr"][im_["id"]]]
asp = (c1 - c0) / (d1 - d0); fig, ax = plt.subplots(2, 2, figsize=(4.2, 2 * 2.0 * asp + 0.85)); ax = np.vstack([ax[0], ax[1]]).reshape(2, 2)
AX = {(1, 0): ax[0, 0], (1, 1): ax[0, 1], (1, 2): ax[1, 0], (1, 3): ax[1, 1]}
isbi_panel(AX[(1, 0)], gms, f"annotation\n(overlap {int(im_['overlap_lo'] * 100)}-{int(im_['overlap_hi'] * 100)}%)")
isbi_panel(AX[(1, 1)], cp, f"Cellpose-SAM\nhidden parts kept {100 * r['cp'][1]:.0f}%")
isbi_panel(AX[(1, 2)], mr, f"Mask R-CNN\nhidden parts kept {100 * r['mr'][1]:.0f}%")
rgb = np.dstack([img] * 3); mt = match(gms, cp); lost = np.zeros(img.shape, bool)
for g, p in mt.items(): lost |= gms[g] & ovp & ~cp[p]
rgb[lost] = 0.4 * rgb[lost] + 0.6 * np.array((0.9, 0.1, 0.8))
for k, m in enumerate(gms): rgb[SS.find_boundaries(m, mode="inner")] = PAL[k % len(PAL)]
AX[(1, 3)].imshow(Z(rgb)); AX[(1, 3)].set_title("cut off by Cellpose-SAM\n(magenta)", fontsize=8); AX[(1, 3)].axis("off")
fig.tight_layout(h_pad=0.8); fig.savefig(f"{FIG}/qual_isbi.pdf"); fig.savefig(f"{FIG}/qual_isbi.png", dpi=150); print("saved", FIG)
