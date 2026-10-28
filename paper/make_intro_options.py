"""Two layout options for the introduction figure (reuses the data loading of make_intro_figure_two.py).
Option 2: the isolated pair through the whole pipeline, plus one extra column with the crowded-field outputs.
Option 3 (results panel): the crowded-field crop alone, annotation | Cellpose-SAM | Cascade Mask R-CNN (one column)."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import numpy as np
src = open(f"{REPO}/make_intro_figure_two.py").read()
exec(src[:src.index("# ---------------------------------------------------------------- layout")])   # imports, load(), helpers
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
A, B = [load(*e) for e in EXAMPLES]
flow_masks = lambda d: [d["lab"] == k for k in np.unique(d["lab"]) if k > 0]

def common(fig, FW, FH):
    def panel(x, y, P): return fig.add_axes([x / FW, y / FH, P / FW, P / FH])
    def frame(ax, col):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(col); sp.set_linewidth(1.1)
    def arrow(x0, y0, x1, y1, col="0.4"):
        fig.add_artist(FancyArrowPatch((x0 / FW, y0 / FH), (x1 / FW, y1 / FH), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=col))
    def metric(x, y, w, h, title, formula, src_, col):
        fig.add_artist(FancyBboxPatch((x / FW, y / FH), w / FW, h / FH, transform=fig.transFigure, boxstyle="round,pad=0.004,rounding_size=0.01", fc="white", ec=col, lw=1.1))
        fig.text((x + w / 2) / FW, (y + h * 0.74) / FH, title, ha="center", va="center", fontsize=6.6, weight="bold", color=col)
        fig.text((x + w / 2) / FW, (y + h * 0.46) / FH, formula, ha="center", va="center", fontsize=5.4, color=INK)
        fig.text((x + w / 2) / FW, (y + h * 0.18) / FH, src_, ha="center", va="center", fontsize=5.0, color="0.4", style="italic")
    def title(ax, s, col, fs=6.2): ax.set_title(s, fontsize=fs, pad=2.2, color=col)
    def tag(ax, text, fc, fg="white", fs=5.4):
        ax.text(0.97, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=fs, color=fg, weight="bold", bbox=dict(fc=fc, ec="none", pad=0.6))
    return panel, frame, arrow, metric, title, tag

def outlines(ax, d, ms):
    W = d["W"]
    for k, m in enumerate(ms):
        o = np.zeros((W, W, 4)); o[find_boundaries(m, mode="inner")] = (*cols[k % len(cols)], 1); ax.imshow(o)

def show_masks(ax, d, ms, alpha=0.4, ann=False):
    ax.imshow(masks_rgb(d, ms, alpha)); outlines(ax, d, ms)
    if ann:
        for g in d["pair"]: ax.contour(g.astype(float), levels=[0.5], colors=ANNC, linewidths=0.9, linestyles="dashed")

def counts(d, preds):
    n = n_recovered(d, preds); return f"{len(d['pair'])} cells → {n} mask{'s' if n != 1 else ''}"

def det_boxes(ax, d, fs=4.8):
    W = d["W"]; placed = []
    ax.imshow(d["gray"])
    for k, (bx, by, bw, bh, s) in enumerate(d["boxes"]):
        ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=cols[k % len(cols)], lw=0.9))
        if 4 <= bx + bw / 2 < W - 4 and 4 <= by + bh / 2 < W - 4 and bw * bh > 120:
            lx, ly, va = max(bx, 0.5), by, "bottom"
            if by < 6: ly, va = by + 0.8, "top"
            for _ in range(4):
                if any(abs(lx - px) < 0.17 * W and abs(ly - py) < 0.08 * W for px, py in placed): ly, va = ly + 0.1 * W, "top"
            if any(abs(lx - px) < 0.17 * W and abs(ly - py) < 0.08 * W for px, py in placed): continue
            placed.append((lx, ly)); ax.text(lx, ly, f"{s:.2f}", fontsize=fs, color="white", ha="left", va=va, bbox=dict(fc=cols[k % len(cols)], ec="none", pad=0.35), clip_on=True)
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(W - 0.5, -0.5)

# ------------------------------------------------------------ option 2
FW, FH, P, Pb = 7.0, 3.05, 0.95, 0.78
fig = plt.figure(figsize=(FW, FH)); panel, frame, arrow, metric, title, tag = common(fig, FW, FH)
xi, c1, c2, c3, c4, xm, wm = 0.03, 1.32, 2.44, 3.56, 4.80, 5.70, 1.20
yt, yb = 1.74, 0.20; yi = (yt + yb) / 2
ax = panel(xi, yi, P); ax.imshow(A["gray"]); frame(ax, "0.3"); title(ax, "input image", INK)
for g in A["pair"]: ax.contour(g.astype(float), levels=[0.5], colors=ANNC, linewidths=1.0, linestyles="dashed")
tag(ax, f"annotation: {len(A['pair'])} cells", ANNC, INK)
fig.text((c1 - 0.12) / FW, (yt + P / 2) / FH, "flow-based", rotation=90, ha="center", va="center", fontsize=6.8, weight="bold", color=FLOW)
fig.text((c1 - 0.12) / FW, (yb + P / 2) / FH, "detection-based", rotation=90, ha="center", va="center", fontsize=6.8, weight="bold", color=DET)
ax = panel(c1, yt, P); ax.imshow(A["flow"]); frame(ax, FLOW); title(ax, "flow field", FLOW)
ax = panel(c2, yt, P); ax.imshow(A["prob"], cmap="Greys_r", vmin=0, vmax=1); frame(ax, FLOW); title(ax, "cell probability", FLOW)
ax = panel(c3, yt, P); show_masks(ax, A, flow_masks(A), 0.5); frame(ax, FLOW); title(ax, "one label per pixel", FLOW); tag(ax, counts(A, flow_masks(A)), FLOW)
ax = panel(c1, yb, P); det_boxes(ax, A); frame(ax, DET); title(ax, "detections and scores", DET)
ax = panel(c3, yb, P); show_masks(ax, A, A["masks"]); frame(ax, DET); title(ax, "one mask per cell, may overlap", DET); tag(ax, counts(A, A["masks"]), DET)
# crowded-field column: outputs only, annotated pair dashed
oy = (P - Pb) / 2
ax = panel(c4, yt + oy, Pb); show_masks(ax, B, flow_masks(B), 0.5, ann=True); frame(ax, FLOW); tag(ax, counts(B, flow_masks(B)), FLOW, fs=5.0)
ax = panel(c4, yb + oy, Pb); show_masks(ax, B, B["masks"], ann=True); frame(ax, DET); tag(ax, counts(B, B["masks"]), DET, fs=5.0)
fig.text((c4 + Pb / 2) / FW, (yt + P + 0.02) / FH, "same, in a\ncrowded field", ha="center", va="bottom", fontsize=6.2, color=INK, style="italic")
fig.add_artist(plt.Line2D([(c4 - 0.11) / FW] * 2, [0.06, 0.94], transform=fig.transFigure, color="0.75", lw=0.6, ls=":"))
metric(xm, yt + 0.12, wm, P - 0.24, "Cellpose-style AP", "TP/(TP+FP+FN), IoU 0.5", "Cellpose papers", FLOW)
metric(xm, yb + 0.12, wm, P - 0.24, "COCO mask AP", "mean over IoU 0.50–0.95", "LIVECell paper", DET)
arrow(xi + P + 0.03, yi + P * 0.62, c1 - 0.22, yt + P / 2); arrow(xi + P + 0.03, yi + P * 0.38, c1 - 0.22, yb + P / 2)
for y0 in (yt,):
    arrow(c1 + P + 0.03, y0 + P / 2, c2 - 0.03, y0 + P / 2); arrow(c2 + P + 0.03, y0 + P / 2, c3 - 0.03, y0 + P / 2)
arrow(c1 + P + 0.03, yb + P / 2, c3 - 0.03, yb + P / 2)
fig.text((c2 + P / 2) / FW, (yb + P / 2 + 0.08) / FH, "one mask per detection\nabove a score cutoff", ha="center", va="bottom", fontsize=5.5, color="0.35")
for y0 in (yt, yb): arrow(c4 + Pb + 0.03, y0 + P / 2, xm - 0.03, y0 + P / 2)
fig.savefig(f"{R}/paper/draft/figures/intro_option2.pdf", dpi=300); fig.savefig(f"{R}/paper/draft/figures/intro_option2.png", dpi=220); plt.close(fig)

# ------------------------------------------------------------ option 3: crowded-field panel for the results (one column)
FW, FH, P = 3.35, 1.42, 1.02
fig = plt.figure(figsize=(FW, FH)); panel, frame, arrow, metric, title, tag = common(fig, FW, FH)
xs = [0.04, 1.16, 2.28]; y = 0.08
ax = panel(xs[0], y, P); ax.imshow(B["gray"]); frame(ax, "0.3"); title(ax, "annotation", INK)
for g in B["pair"]: ax.contour(g.astype(float), levels=[0.5], colors=ANNC, linewidths=1.0, linestyles="dashed")
tag(ax, f"{len(B['pair'])} overlapping cells", ANNC, INK, fs=5.0)
ax = panel(xs[1], y, P); show_masks(ax, B, flow_masks(B), 0.5, ann=True); frame(ax, FLOW); title(ax, "flow: Cellpose-SAM", FLOW, fs=5.8); tag(ax, counts(B, flow_masks(B)), FLOW, fs=5.0)
ax = panel(xs[2], y, P); show_masks(ax, B, B["masks"], ann=True); frame(ax, DET); title(ax, "detection: Cascade", DET, fs=5.8); tag(ax, counts(B, B["masks"]), DET, fs=5.0)
fig.savefig(f"{R}/paper/draft/figures/results_crowded_pair.pdf", dpi=300); fig.savefig(f"{R}/paper/draft/figures/results_crowded_pair.png", dpi=250); plt.close(fig)
print("ok")

# ------------------------------------------------------------ option 4: both inputs through every stage (a | b in each cell)
FW, FH, P, gp = 7.0, 2.85, 0.63, 0.03                         # thumbnail size, gap between a and b
CW = 2 * P + gp                                               # width of one stage cell
fig = plt.figure(figsize=(FW, FH)); panel, frame, arrow, metric, title, tag = common(fig, FW, FH)
xi, c1, c2, c3, xm, wm = 0.02, 1.60, 3.06, 4.52, 5.92, 0.98
yt, yb = 1.66, 0.22; yi = (yt + yb) / 2
def pair_cell(x, y, draw, col, head=None):
    for j, (d, lab) in enumerate(((A, "a"), (B, "b"))):
        ax = panel(x + j * (P + gp), y, P); draw(ax, d); frame(ax, col)
        if head: ax.set_title(f"({lab})", fontsize=5.6, pad=1.5, color="0.35")
    if head: fig.text((x + CW / 2) / FW, (y + P + 0.17) / FH, head, ha="center", va="bottom", fontsize=6.2, color=col)
def ctag(ax, text, fc, fg="white"):
    ax.text(0.5, 0.03, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=4.6, color=fg, weight="bold", bbox=dict(fc=fc, ec="none", pad=0.4))
def d_input(ax, d):
    ax.imshow(d["gray"])
    for g in d["pair"]: ax.contour(g.astype(float), levels=[0.5], colors=ANNC, linewidths=0.8, linestyles="dashed")
def d_flow(ax, d): ax.imshow(d["flow"])
def d_prob(ax, d): ax.imshow(d["prob"], cmap="Greys_r", vmin=0, vmax=1)
def d_lab(ax, d):
    show_masks(ax, d, flow_masks(d), 0.5); W = d["W"]
    for g in d["missed"]:                                        # annotated cell the flow output loses: merged into a neighbour or missed
        covered = (g & (d["lab"] > 0)).sum() / max(g.sum(), 1)
        word = "merged" if covered > 0.5 else "missed"
        yy, xx = np.nonzero(g); cy, cx = yy.mean(), xx.mean()
        tx = cx + (0.30 * W if cx < W / 2 else -0.30 * W); ty = max(cy - 0.30 * W, 0.12 * W)
        ax.annotate(word, xy=(cx, cy), xytext=(tx, ty), fontsize=5.4, color="white", weight="bold", ha="center", va="center",
                    bbox=dict(fc=FLOW, ec="none", pad=0.5, alpha=0.6),
                    arrowprops=dict(arrowstyle="-|>", color="white", alpha=0.6, lw=0.9, mutation_scale=6, shrinkA=0, shrinkB=1))
def d_det(ax, d): det_boxes(ax, d, fs=3.9)
def d_msk(ax, d): show_masks(ax, d, d["masks"])
pair_cell(xi, yi, d_input, "0.3", "input images")
fig.text((c1 - 0.13) / FW, (yt + P / 2) / FH, "flow-based", rotation=90, ha="center", va="center", fontsize=6.6, weight="bold", color=FLOW)
fig.text((c1 - 0.13) / FW, (yb + P / 2) / FH, "detection-based", rotation=90, ha="center", va="center", fontsize=6.6, weight="bold", color=DET)
pair_cell(c1, yt, d_flow, FLOW, "flow field"); pair_cell(c2, yt, d_prob, FLOW, "cell probability"); pair_cell(c3, yt, d_lab, FLOW, "one label per pixel")
pair_cell(c1, yb, d_det, DET, "detections and scores"); pair_cell(c3, yb, d_msk, DET, "one mask per cell, may overlap")
metric(xm, yt + 0.02, wm, P - 0.04, "Cellpose-style AP", "TP/(TP+FP+FN)\nat IoU 0.5", "Cellpose papers", FLOW)
metric(xm, yb + 0.02, wm, P - 0.04, "COCO mask AP", "mean over\nIoU 0.50–0.95", "LIVECell paper", DET)
arrow(xi + CW + 0.03, yi + P * 0.65, c1 - 0.22, yt + P / 2); arrow(xi + CW + 0.03, yi + P * 0.35, c1 - 0.22, yb + P / 2)
arrow(c1 + CW + 0.03, yt + P / 2, c2 - 0.03, yt + P / 2); arrow(c2 + CW + 0.03, yt + P / 2, c3 - 0.03, yt + P / 2)
arrow(c1 + CW + 0.03, yb + P / 2, c3 - 0.03, yb + P / 2)
fig.text((c2 + CW / 2) / FW, (yb + P / 2 + 0.07) / FH, "one mask per detection\nabove a score cutoff", ha="center", va="bottom", fontsize=5.4, color="0.35")
for y0 in (yt, yb): arrow(c3 + CW + 0.03, y0 + P / 2, xm - 0.03, y0 + P / 2)
fig.savefig(f"{R}/paper/draft/figures/intro_figure.pdf", dpi=300); fig.savefig(f"{R}/paper/draft/figures/intro_figure.png", dpi=220); plt.close(fig)
print("option4 ok")
