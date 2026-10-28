"""Introduction figure with two stacked examples, (a) an isolated pair of overlapping cells and (b) the same situation
in a crowded field: how a flow-based model (Cellpose-SAM) and a detection-based model (Cascade Mask R-CNN) process the
same LIVECell test crop, and which metric each community reports. Real outputs throughout: Cellpose-SAM flow field,
cell probability and label image (stored network outputs, identical to the benchmark predictions) and Cascade Mask
R-CNN detections above its validation-selected cutoff. Dashed yellow: annotated cells of the overlapping pair; the
counts state how many of them each model recovers at IoU >= 0.5."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, numpy as np, skimage.io
from pycocotools import mask as MU
from pycocotools.coco import COCO
from skimage.segmentation import find_boundaries
from matplotlib.colors import hsv_to_rgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

R = f"{WORK}"; D = f"{R}/data/intro_fig"
ANN = f"{LC_ANN}/livecell_coco_test.json"
IMG = f"{LC_IMG}/livecell_test_images"
FLOW, DET, ANNC, INK = "#3f6fb0", "#c0392b", "#f2c94c", "#222222"
CUT = json.load(open(f"{R}/results/val_cutoffs.json"))["anchor_based"]
G = COCO(ANN)
cols = np.vstack([plt.cm.tab10(np.array([0, 1, 2, 3, 4, 5, 6, 8, 9]))[:, :3],   # tab10 without its grey (invisible on phase contrast)
                  plt.cm.Set2(np.arange(8))[:, :3]])
EXAMPLES = [  # (image id, annotated pair bbox x0,y0,x1,y1 from results/intro_pairs.json, crop size relative to the pair)
    (1259721, [617.0, 53.0, 644.0, 75.0], 1.7),     # SkBr3_Phase_G3_2_01d04h00m_4: isolated pair
    (1285275, [398.0, 245.0, 417.0, 269.0], 2.4),   # SkBr3_Phase_G3_1_02d12h00m_3: pair in a crowded field
]

def dec(seg):
    return MU.decode({"size": seg["size"], "counts": seg["counts"].encode() if isinstance(seg["counts"], str) else seg["counts"]}).astype(bool)

def load(iid, pb, zoom):
    W = int(max(pb[2] - pb[0], pb[3] - pb[1]) * zoom)
    Y = int(min(max((pb[1] + pb[3]) / 2 - W / 2, 0), 520 - W)); X = int(min(max((pb[0] + pb[2]) / 2 - W / 2, 0), 704 - W)); sl = (slice(Y, Y + W), slice(X, X + W))
    fn = G.imgs[iid]["file_name"]; img = skimage.io.imread(f"{IMG}/{fn}").astype(float)[sl]
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5); gray = np.stack([np.clip((img - lo) / (hi - lo), 0, 1)] * 3, -1)
    z = np.load(f"{D}/flows_cpsam_{iid}.npz"); dP = z["dP"].astype(np.float32)[:, Y:Y + W, X:X + W]
    prob = 1 / (1 + np.exp(-z["cellprob"].astype(np.float32)[Y:Y + W, X:X + W])); labf = z["lab"].astype(int); lab = labf[sl]
    ang = (np.arctan2(dP[0], dP[1]) + np.pi) / (2 * np.pi); mag = np.clip(np.hypot(dP[0], dP[1]) / np.percentile(np.hypot(dP[0], dP[1]), 99), 0, 1)
    flow = hsv_to_rgb(np.stack([ang, np.full_like(ang, 0.85), mag], -1)); flow = flow * mag[..., None] + (1 - mag[..., None])
    boxes, masks = [], []
    for p in json.load(open(f"{D}/det_anchor_based_{iid}.json")):
        if p["score"] < CUT: continue
        m = dec(p["segmentation"])[sl]
        if m.sum() < 15: continue                                                   # slivers cut by the crop border
        bx, by, bw, bh = p["bbox"]; boxes.append((bx - X, by - Y, bw, bh, p["score"])); masks.append(m)
    # annotated cells: those the flow output misses but the detector recovers, and the overlapping pair they belong to
    gts = [dec(G.annToRLE(a)) for a in G.loadAnns(G.getAnnIds(imgIds=[iid]))]
    inside = [g for g in gts if g[sl].sum() >= 0.8 * g.sum()]
    def best(g, preds): return max([(g & m).sum() / (g | m).sum() for m in preds], default=0)
    flow_full = [labf == i for i in np.unique(labf) if i > 0]
    missed = [g[sl] for g in inside if best(g, flow_full) < 0.5 and best(g[sl], masks) >= 0.5]
    pair = [g[sl] for g in inside if any((g[sl] & m).sum() > 0 for m in missed)]
    return dict(W=W, gray=gray, flow=flow, prob=prob, lab=lab, boxes=boxes, masks=masks, pair=pair, missed=missed, fn=fn)

def masks_rgb(d, ms, alpha=0.38):
    W = d["W"]; acc = np.zeros((W, W, 3)); n = np.zeros((W, W))
    for k, m in enumerate(ms): acc[m] += cols[k % len(cols)]; n[m] += 1
    a = np.clip(alpha * n, 0, 0.75)[..., None]
    return d["gray"] * (1 - a) + np.where(n[..., None] > 0, acc / np.maximum(n, 1)[..., None], 1) * a

def n_recovered(d, preds):
    return sum(1 for g in d["pair"] if max([(g & m).sum() / (g | m).sum() for m in preds], default=0) >= 0.5)

# ---------------------------------------------------------------- layout (inches), two blocks stacked
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
FW, P, BH = 7.0, 0.92, 2.55                                        # width, panel size, block height
FH = 2 * BH + 0.05
fig = plt.figure(figsize=(FW, FH))
def panel(x, y): return fig.add_axes([x / FW, y / FH, P / FW, P / FH])
def frame(ax, col):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.1)
def arrow(x0, y0, x1, y1, col="0.4"):
    fig.add_artist(FancyArrowPatch((x0 / FW, y0 / FH), (x1 / FW, y1 / FH), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col))
def metric(x, y, w, h, title, formula, src, col):
    fig.add_artist(FancyBboxPatch((x / FW, y / FH), w / FW, h / FH, transform=fig.transFigure, boxstyle="round,pad=0.004,rounding_size=0.01", fc="white", ec=col, lw=1.1))
    fig.text((x + w / 2) / FW, (y + h * 0.72) / FH, title, ha="center", va="center", fontsize=6.8, weight="bold", color=col)
    fig.text((x + w / 2) / FW, (y + h * 0.45) / FH, formula, ha="center", va="center", fontsize=5.7, color=INK)
    fig.text((x + w / 2) / FW, (y + h * 0.18) / FH, src, ha="center", va="center", fontsize=5.1, color="0.4", style="italic")
def title(ax, s, col): ax.set_title(s, fontsize=6.2, pad=2.2, color=col)
def tag(ax, text, fc, fg="white"):
    ax.text(0.97, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.5, color=fg, weight="bold", bbox=dict(fc=fc, ec="none", pad=0.7))
def outlines(ax, d, ms):
    W = d["W"]
    for k, m in enumerate(ms):
        o = np.zeros((W, W, 4)); o[find_boundaries(m, mode="inner")] = (*cols[k % len(cols)], 1); ax.imshow(o)

xi, c1, c2, c3, xm, wm = 0.22, 1.55, 2.68, 3.81, 5.05, 1.8
def block(d, y0, letter, sub, show_titles):
    yt, yb = y0 + 1.42, y0 + 0.12; yi = (yt + yb) / 2; W = d["W"]
    fig.text(0.02 / FW, (y0 + BH - 0.12) / FH, f"({letter})", fontsize=8, weight="bold", color=INK, va="top")
    fig.text(0.02 / FW, (y0 + BH - 0.32) / FH, sub, fontsize=6.2, color="0.3", va="top", style="italic", wrap=True)
    ax = panel(xi, yi); ax.imshow(d["gray"]); frame(ax, "0.3"); title(ax, "input image", INK)
    for g in d["pair"]: ax.contour(g.astype(float), levels=[0.5], colors=ANNC, linewidths=1.0, linestyles="dashed")
    if d["pair"]: tag(ax, f"annotation: {len(d['pair'])} cells", ANNC, INK)
    fig.text((c1 - 0.12) / FW, (yt + P / 2) / FH, "flow-based", rotation=90, ha="center", va="center", fontsize=6.8, weight="bold", color=FLOW)
    fig.text((c1 - 0.12) / FW, (yb + P / 2) / FH, "detection-based", rotation=90, ha="center", va="center", fontsize=6.8, weight="bold", color=DET)
    ax = panel(c1, yt); ax.imshow(d["flow"]); frame(ax, FLOW); title(ax, "flow field", FLOW)
    ax = panel(c2, yt); ax.imshow(d["prob"], cmap="Greys_r", vmin=0, vmax=1); frame(ax, FLOW); title(ax, "cell probability", FLOW)
    ax = panel(c3, yt); msf = [d["lab"] == k for k in np.unique(d["lab"]) if k > 0]; ax.imshow(masks_rgb(d, msf, 0.5)); outlines(ax, d, msf); frame(ax, FLOW); title(ax, "one label per pixel", FLOW)
    if d["pair"]: n = n_recovered(d, msf); tag(ax, f"{len(d['pair'])} cells → {n} mask{'s' if n != 1 else ''}", FLOW)
    metric(xm, yt + 0.13, wm, P - 0.26, "Cellpose-style AP", "TP / (TP + FP + FN) at IoU 0.5", "reported in the Cellpose papers", FLOW)
    ax = panel(c1, yb); ax.imshow(d["gray"]); frame(ax, DET); title(ax, "detections and scores", DET); placed = []
    for k, (bx, by, bw, bh, s) in enumerate(d["boxes"]):
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=cols[k % len(cols)], lw=0.9))
        if 4 <= bx + bw / 2 < W - 4 and 4 <= by + bh / 2 < W - 4 and bw * bh > 120:
            lx, ly, va = max(bx, 0.5), by, "bottom"
            if by < 6: ly, va = by + 0.8, "top"
            for _ in range(4):
                if any(abs(lx - px) < 0.17 * W and abs(ly - py) < 0.08 * W for px, py in placed): ly, va = ly + 0.1 * W, "top"
            if any(abs(lx - px) < 0.17 * W and abs(ly - py) < 0.08 * W for px, py in placed): continue
            placed.append((lx, ly)); ax.text(lx, ly, f"{s:.2f}", fontsize=4.8, color="white", ha="left", va=va, bbox=dict(fc=cols[k % len(cols)], ec="none", pad=0.35), clip_on=True)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)
    ax = panel(c3, yb); ax.imshow(masks_rgb(d, d["masks"])); outlines(ax, d, d["masks"]); frame(ax, DET); title(ax, "one mask per cell, may overlap", DET)
    if d["pair"]: n = n_recovered(d, d["masks"]); tag(ax, f"{len(d['pair'])} cells → {n} mask{'s' if n != 1 else ''}", DET)
    metric(xm, yb + 0.13, wm, P - 0.26, "COCO mask AP", "precision averaged over IoU 0.50–0.95", "reported in the LIVECell paper", DET)
    arrow(xi + P + 0.03, yi + P * 0.62, c1 - 0.22, yt + P / 2); arrow(xi + P + 0.03, yi + P * 0.38, c1 - 0.22, yb + P / 2)
    arrow(c1 + P + 0.03, yt + P / 2, c2 - 0.03, yt + P / 2); arrow(c2 + P + 0.03, yt + P / 2, c3 - 0.03, yt + P / 2); arrow(c3 + P + 0.03, yt + P / 2, xm - 0.03, yt + P / 2)
    arrow(c1 + P + 0.03, yb + P / 2, c3 - 0.03, yb + P / 2); arrow(c3 + P + 0.03, yb + P / 2, xm - 0.03, yb + P / 2)
    fig.text((c2 + P / 2) / FW, (yb + P / 2 + 0.08) / FH, "one mask per detection\nabove a score cutoff", ha="center", va="bottom", fontsize=5.5, color="0.35")

data = [load(*e) for e in EXAMPLES]
block(data[0], BH + 0.05, "a", "isolated\noverlapping pair", True)
block(data[1], 0.0, "b", "overlapping pair\nin a crowded field", True)
fig.add_artist(plt.Line2D([0.02, 0.98], [(BH + 0.03) / FH] * 2, transform=fig.transFigure, color="0.8", lw=0.6))
fig.savefig(f"{R}/paper/draft/figures/intro_pipeline_two.pdf", dpi=300); fig.savefig(f"{R}/paper/draft/figures/intro_pipeline_two.png", dpi=220)
for d in data: print(d["fn"], "pair", len(d["pair"]), "flow recovers", n_recovered(d, [d["lab"] == k for k in np.unique(d["lab"]) if k > 0]), "detector recovers", n_recovered(d, d["masks"]))
