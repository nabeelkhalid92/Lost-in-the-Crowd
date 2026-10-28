"""Collect all scores -> LaTeX tables + figures for the ICAART paper, and results/summary.json.
Missing inputs are skipped (prints what is missing) so this can be re-run as results arrive."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import glob, json, os, re
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

R = f"{WORK}"; FC = f"{FC_WORK}/results/full"
TAB = f"{R}/paper/draft/tables"; FIG = f"{R}/paper/draft/figures"; os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
ORDER = ["cyto", "cyto2", "cyto3", "cpsam", "vista2d", "livecell", "microsam", "anchor_based", "anchor_free"]
NAME = {"cyto": "Cellpose v1", "cyto2": "Cellpose 2", "cyto3": "Cellpose3", "cpsam": "Cellpose-SAM", "vista2d": "VISTA-2D",
        "livecell": "Cellpose LIVECell", "microsam": "micro-SAM", "anchor_based": "Cascade Mask R-CNN", "anchor_free": "CenterMask2"}
FAM = {m: "flow" for m in ["cyto", "cyto2", "cyto3", "cpsam", "vista2d", "livecell"]}; FAM.update(microsam="SAM", anchor_based="det", anchor_free="det")
DET = {"anchor_based", "anchor_free"}
MK = {"cyto": "v", "cyto2": "^", "cyto3": "D", "cpsam": "o", "vista2d": "s", "livecell": "P", "microsam": "X", "anchor_based": "o", "anchor_free": "s"}
LS = {m: ("--" if m in {"anchor_based", "anchor_free"} else "-") for m in MK}
COL = {"cyto": "#c7d4e8", "cyto2": "#9fb6d8", "cyto3": "#6f93c6", "cpsam": "#3f6fb0", "vista2d": "#1f4e8c", "livecell": "#7a5fa8",
       "microsam": "#3a9a6a", "anchor_based": "#d1453b", "anchor_free": "#e58a2e"}
CTC = ["PhC-C2DL-PSC", "PhC-C2DH-U373", "DIC-C2DH-HeLa", "BF-C2DL-HSC", "BF-C2DL-MuSC"]
CTC_SHORT = {"PhC-C2DL-PSC": "PSC", "PhC-C2DH-U373": "U373", "DIC-C2DH-HeLa": "HeLa", "BF-C2DL-HSC": "HSC", "BF-C2DL-MuSC": "MuSC"}
S = {}  # summary


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- detector cutoff chosen on LIVECell VALIDATION
cut = {}
for m in DET:
    v = jl(f"{R}/results/analysis/livecell_val__{m}.json")
    if v:
        best = max([k for k in v if k.startswith("raw_")], key=lambda k: v[k]["F1"][0])
        cut[m] = float(best.split("_")[1]); S.setdefault("val_cutoff", {})[m] = {"cutoff": cut[m], "val_F1@0.5": v[best]["F1"][0],
                                                                                   "all": {k: v[k]["F1"][0] for k in v if k.startswith("raw_")}}
    else:
        print("missing val analysis", m); cut[m] = 0.5
print("cutoffs", cut); json.dump(cut, open(f"{R}/results/val_cutoffs.json", "w"))


def akey(m, mode="raw"):
    return f"{mode}_{cut[m]}" if m in DET else "flat_None"


# ---------------------------------------------------------------- LIVECell main table
coco_lc = {}
for m in ORDER:
    # cpsam / micro-SAM scored here with score_coco_v3 (per-line AP re-evaluated per subset); others from the full-test runs
    f = {"cpsam": f"{R}/results/coco_v3/icaart_cpsam_scores.json", "microsam": f"{R}/results/coco_v3/icaart_microsam_scores.json"}.get(m, f"{FC}/{m}_scores.json")
    if not os.path.exists(f): f = f"{FC}/icaart_{m}_scores.json"
    if m in DET and os.path.exists(f"{R}/results/coco_v3/icaart_{m}_cap3000_scores.json"): f = f"{R}/results/coco_v3/icaart_{m}_cap3000_scores.json"   # 3000-detection cap rerun
    d = jl(f) or jl(f"{FC_WORK}/results/icaart_{m}_scores.json")
    if d: coco_lc[m] = d
    else: print("missing LIVECell COCO", m)
for m in DET:
    d = jl(f"{R}/results/coco_v3/icaart_{m}_flat_cap3000_scores.json") or jl(f"{FC_WORK}/results/icaart_{m}_flat_scores.json")
    if d: coco_lc[m + "_flat"] = d
def an_ds(m, ds="livecell"):   # detectors: analysis of the 3000-cap rerun
    return "livecell_cap3000" if ds == "livecell" and m in DET and os.path.exists(f"{R}/results/analysis/livecell_cap3000__{m}.json") else ds
an_lc = {m: jl(f"{R}/results/analysis/{an_ds(m)}__{m}.json") for m in ORDER}
times = jl(f"{FC}/inference_times.json") or {}
rows = []
for m in ORDER:
    if m not in coco_lc or not an_lc.get(m): continue
    o = coco_lc[m]["overall"]; a = an_lc[m][akey(m)]
    # count error over images with >= 10 annotated cells: two test images are only partially annotated (2 annotated
    # cells where every model finds 80-970) and alone triple the plain mean
    cnt = np.load(f"{R}/results/analysis/{an_ds(m)}__{m}__{akey(m)}.npz")["counts"].astype(float); ok = cnt[:, 1] >= 10
    rows.append((m, o["mAP"] * 100, o["AP50"] * 100, o["AP75"] * 100, a["F1"][0], a["F1"][5], a["CPAP_mean_img"][0], float(np.mean(np.abs(cnt[ok, 0] - cnt[ok, 1]) / cnt[ok, 1]) * 100)))
S["livecell"] = {r[0]: dict(zip(["AP", "AP50", "AP75", "F1_50", "F1_75", "CPAP50", "countMAPE_vs_GT"], r[1:])) for r in rows}
if rows:
    NOAP = {"microsam"}   # no instance scores -> COCO AP not comparable (protocol section)
    P_ = {1: 1, 2: 1, 3: 1, 4: 3, 5: 3, 6: 3, 7: 1}
    best = {i: (max if i < 7 else min)(round(r[i], P_[i]) for r in rows if not (i <= 3 and r[0] in NOAP)) for i in range(1, 8)}
    def fmt(v, i, p, m=None):
        if i <= 3 and m in NOAP: return "--"
        s = f"{v:.{p}f}"; return f"\\textbf{{{s}}}" if abs(round(v, p) - best[i]) < 1e-9 else s
    L = ["\\begin{table*}[t]\\centering", "\\caption{LIVECell test set (1{,}564 images, 462{,}261 cells). COCO mask AP (ranked, IoU 0.50:0.95), "
         "F1 and Cellpose-style AP at the operating point (detectors: score cutoff chosen on the LIVECell validation split), and the mean absolute "
         "percentage error of the per-image instance count against the ground-truth count. Best per column in bold.}\\label{tab:livecell}",
         "\\small\\begin{tabular}{@{}llrrrrrrr@{}}\\toprule",
         "Model & Family & AP & AP$_{50}$ & AP$_{75}$ & F1$_{50}$ & F1$_{75}$ & $\\mathrm{AP}^{\\mathrm{CP}}_{50}$ & Count err.\\,(\\%) \\\\\\midrule"]
    for r in rows:
        BS = json.load(open(f"{R}/results/bootstrap.json"))["LIVECell"].get(r[0], {}) if os.path.exists(f"{R}/results/bootstrap.json") else {}
        def pm(key):                                                     # half-width of the 95% bootstrap interval
            v = BS.get(key); return f"\\tiny{{$\\pm${(v[2] - v[1]) / 2:.3f}}}".replace("0.", ".") if v else ""
        L.append(f"{NAME[r[0]]} & {FAM[r[0]]} & {fmt(r[1],1,1,r[0])} & {fmt(r[2],2,1,r[0])} & {fmt(r[3],3,1,r[0])} & {fmt(r[4],4,3)}{pm('F1_50')} & {fmt(r[5],5,3)}{pm('F1_75')} & {fmt(r[6],6,3)}{pm('CPAP50')} & {fmt(r[7],7,1)} \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table*}"]
    open(f"{TAB}/livecell.tex", "w").write("\n".join(L))

# ---------------------------------------------------------------- flattening + cutoff sensitivity (detectors)
# transposed (rows = COCO AP + cutoffs, columns = detector x mask mode) so it fits one column at the common font size
CUTS = [0.3, 0.4, 0.5, 0.6, 0.7]; S["flatten"] = {}; colv = []
for m in ["anchor_based", "anchor_free"]:
    a = an_lc.get(m)
    if not a: continue
    for mode in ["raw", "flat"]:
        ap = coco_lc.get(m if mode == "raw" else m + "_flat", {}).get("overall", {}).get("mAP")
        f1 = [a.get(f"{mode}_{c}", {}).get("F1", [np.nan])[0] for c in CUTS]
        S["flatten"][f"{m}_{mode}"] = {"AP": ap * 100 if ap else None, "F1_sweep": f1}; colv.append((m, ap, f1))
L = ["\\begin{table}[t]\\centering\\caption{Detectors with overlapping masks (ov.) and after flattening them into one label image (fl.), LIVECell test. "
     "First row: COCO mask AP (all ranked detections). Other rows: F1 at IoU 0.5 at each score cutoff; $\\star$ = cutoff selected on the validation split.}\\label{tab:flatten}",
     "\\small\\begin{tabular}{@{}lrrrr@{}}\\toprule",
     " & \\multicolumn{2}{c}{Cascade Mask R-CNN} & \\multicolumn{2}{c}{CenterMask2} \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
     " & ov. & fl. & ov. & fl. \\\\\\midrule",
     "COCO AP & " + " & ".join(f"{ap*100:.1f}" if ap else "--" for _, ap, _ in colv) + " \\\\\\midrule"]
for i, c in enumerate(CUTS):
    L.append(f"F1$_{{50}}$ @ {c:.1f} & " + " & ".join(f"{f1[i]:.3f}" + ("$^\\star$" if abs(c - cut[m]) < 1e-9 else "") for m, _, f1 in colv) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{TAB}/flatten.tex", "w").write("\n".join(L))

# ---------------------------------------------------------------- CTC table (segmentation + counting)
an_ctc = {m: jl(f"{R}/results/analysis/ctc__{m}.json") for m in ORDER}
sf_ctc = {m: jl(f"{R}/results/scores/ctc_{m}.json") for m in ORDER}
def ctc_ds_f1(m):
    """Per-dataset pooled F1 at IoU 0.5 / 0.75 from the per-image TP/FP/FN of the CTC analysis."""
    f = f"{R}/results/analysis/ctc__{m}__{akey(m)}.npz"
    if not os.path.exists(f): return None
    z = np.load(f, allow_pickle=True); out = {}
    for d in CTC:
        sel = np.array([str(k).startswith(d + "_") for k in z["keys"]])
        if sel.sum() == 0: continue
        T = z["tff_img"][sel].sum(0); out[d] = {"F1": (2 * T[:, 0] / np.maximum(2 * T[:, 0] + T[:, 1] + T[:, 2], 1)).tolist()}
    return out
an_ctc_ds = {m: ctc_ds_f1(m) for m in ORDER}
S["ctc"] = {}
L = ["\\begin{table*}[t]\\centering\\caption{Cell Tracking Challenge, label-free 2D (no model fine-tuned; out of domain for the LIVECell-trained models). SEG = official CTC measure vs gold truth; "
     "F1$_{50}$ vs complete silver truth; count error = mean absolute percentage error vs the number of tracking markers per frame "
     "(common frame set for all models). Best per column in bold.}\\label{tab:ctc}", "\\small\\setlength{\\tabcolsep}{2.4pt}\\begin{tabular}{@{}l" + "rrr" * len(CTC) + "@{}}\\toprule",
     "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{CTC_SHORT[d]}}}" for d in CTC) + " \\\\",
     " ".join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(CTC))),
     "Model & " + " & ".join(["SEG & F1$_{50}$ & Cnt\\%"] * len(CTC)) + " \\\\\\midrule"]
vals = defaultdict(dict)
for m in ORDER:
    s = sf_ctc.get(m)
    if not s: continue
    for d in CTC:
        if d not in s: continue
        f1 = None
        if an_ctc_ds.get(m) and d in an_ctc_ds[m]: f1 = an_ctc_ds[m][d]["F1"][0]
        vals[m][d] = (s[d]["SEG"], f1, (s[d].get("count_common") or s[d]["count"])["MAPE"] if s[d]["count"] else None)
        S["ctc"].setdefault(m, {})[d] = {"SEG": s[d]["SEG"], "AP": s[d]["AP"], "AP50": s[d]["AP50"], "AP75": s[d].get("AP75"), "CPAP50": s[d]["CPAP50"], "F1_50": f1,
                                         "count": s[d].get("count_common") or s[d]["count"], "count_all_frames": s[d]["count"]}
bestc = {}
for d in CTC:
    for j, fn in [(0, max), (1, max), (2, min)]:
        xs = [round(vals[m][d][j], (2, 2, 0)[j]) for m in vals if m != "microsam" and d in vals[m] and vals[m][d][j] is not None]   # micro-SAM was trained on CTC
        if xs: bestc[(d, j)] = fn(xs)
for m in ORDER:
    if m not in vals: continue
    cells = []
    for d in CTC:
        for j, p in [(0, 2), (1, 2), (2, 0)]:
            v = vals[m].get(d, (None, None, None))[j]
            if v is None: cells.append("--"); continue
            s_ = f"{v:.{p}f}"
            if m == "microsam": cells.append(f"\\textcolor{{gray}}{{{s_}}}"); continue
            cells.append(f"\\textbf{{{s_}}}" if abs(round(v, p) - bestc[(d, j)]) < 1e-9 else s_)
    L.append(f"{NAME[m]} & " + " & ".join(cells) + " \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table*}"]
open(f"{TAB}/ctc.tex", "w").write("\n".join(L))

# ---------------------------------------------------------------- nuclear-count benchmark table
cnt = {m: jl(f"{R}/results/scores/count_{m}.json") for m in ORDER}
S["count"] = {}
L = ["\\begin{table}[t]\\centering\\caption{LIVECell nuclear-count benchmark: instance count vs fluorescence nuclear count. MAPE (\\%), signed bias (\\%) and Pearson $r$.}\\label{tab:count}",
     "\\small\\setlength{\\tabcolsep}{1pt}\\begin{tabular}{@{}lrrrr@{}}\\toprule", "& \\multicolumn{2}{c}{A172 ($n$=348)} & \\multicolumn{2}{c}{A549 ($n$=443)} \\\\\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
     "Model & \\makecell[r]{MAPE\\\\(bias)} & $r$ & \\makecell[r]{MAPE\\\\(bias)} & $r$ \\\\\\midrule"]
for m in ORDER:
    c = cnt.get(m)
    if not c or "A172" not in c or "A549" not in c: continue
    S["count"][m] = c
    a, b = c["A172"]["all"], c["A549"]["all"]
    L.append(f"{NAME[m]} & {a['MAPE']:.1f}\\,({a['bias_pct']:+.0f}) & {a['r']:.2f} & {b['MAPE']:.1f}\\,({b['bias_pct']:+.0f}) & {b['r']:.2f} \\\\")
L += ["\\bottomrule\\end{tabular}\\end{table}"]
open(f"{TAB}/count.tex", "w").write("\n".join(L))

# ---------------------------------------------------------------- which metric predicts counting?
# pairs = (method, dataset): CTC datasets (seg metrics vs silver, count vs TRA markers) + A172 (LIVECell-A172 seg metrics vs nuclear count)
pairs = []
for m in ORDER:
    for d in CTC:
        c = S["ctc"].get(m, {}).get(d)
        if c and c["F1_50"] is not None and c["count"]:
            pairs.append({"m": m, "d": d, "AP": c["AP"], "AP50": c["AP50"], "AP75": c.get("AP75"), "CPAP50": c["CPAP50"], "F1_50": c["F1_50"], "err": c["count"]["MAPE"]})
    pc = coco_lc.get(m, {}).get("per_cell_type", {}).get("A172"); cc = cnt.get(m)
    if pc and cc and an_lc.get(m):
        z = np.load(f"{R}/results/analysis/{an_ds(m)}__{m}__{akey(m)}.npz", allow_pickle=True)
        sel = np.array([str(k).startswith("A172_") for k in z["keys"]]); T = z["tff_img"][sel].sum(0)
        f1 = 2 * T[0, 0] / max(2 * T[0, 0] + T[0, 1] + T[0, 2], 1); cp = z["cpap_img"][sel][:, 0].mean()
        pairs.append({"m": m, "d": "A172", "AP": pc["mAP"] * 100, "AP50": pc["AP50"] * 100, "AP75": pc["AP75"] * 100, "CPAP50": cp, "F1_50": f1,
                      "err": cc["A172"]["all"]["MAPE"]})
S["metric_vs_count_pairs"] = pairs
res = {}
if len(pairs) > 5:
    for k in ["AP", "AP50", "AP75", "CPAP50", "F1_50"]:
        xs = [(p[k], p["err"]) for p in pairs if p[k] is not None]
        rho = spearmanr([x[0] for x in xs], [x[1] for x in xs]).correlation if len(xs) > 3 else np.nan
        # within-dataset rank agreement averaged over datasets (removes dataset difficulty)
        wd = []
        for d in set(p["d"] for p in pairs):
            ps = [p for p in pairs if p["d"] == d and p[k] is not None]
            if len(ps) >= 4: wd.append(spearmanr([p[k] for p in ps], [p["err"] for p in ps]).correlation)
        res[k] = {"spearman_all_pairs": float(rho), "n_pairs": len(xs), "mean_within_dataset_spearman": float(np.nanmean(wd)) if wd else None, "n_datasets": len(wd)}
    S["metric_vs_count"] = res
    names = {"AP": "COCO AP", "AP50": "AP$_{50}$", "AP75": "AP$_{75}$", "CPAP50": "CP-AP$_{50}$", "F1_50": "F1$_{50}$"}
    L = ["\\begin{table}[t]\\centering\\caption{How well does each segmentation metric predict counting error against independent references? "
         "Spearman $\\rho$ between the metric and the count MAPE over all (model, dataset) pairs, and averaged within datasets. More negative = better predictor.}\\label{tab:metric_count}",
         "\\small\\begin{tabular}{@{}lrrr@{}}\\toprule", "Metric & $\\rho$ (all) & pairs & $\\bar\\rho$ (within) \\\\\\midrule"]
    for k, v in res.items():
        L.append(f"{names[k]} & {v['spearman_all_pairs']:+.2f} & {v['n_pairs']} & " + (f"{v['mean_within_dataset_spearman']:+.2f} ({v['n_datasets']})" if v['mean_within_dataset_spearman'] is not None else "--") + " \\\\")
    L += ["\\bottomrule\\end{tabular}\\end{table}"]
    open(f"{TAB}/metric_count.tex", "w").write("\n".join(L))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
# F1 vs IoU (LIVECell, and CTC pooled)
from matplotlib.lines import Line2D
def legend_below(fig, ms, extra=(), ncol=5, rect_bottom=0.2, fs=6.5):
    h = [Line2D([], [], color=COL[m], marker=MK[m], ms=3.5, lw=1.3, ls=LS[m], label=NAME[m]) for m in ms] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=ncol, fontsize=fs, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=2.2)
    fig.tight_layout(rect=[0, rect_bottom, 1, 1])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
for ax, src, title in [(axs[0], an_lc, "LIVECell test"), (axs[1], an_ctc, "CTC label-free (pooled)")]:
    for m in ORDER:
        a = src.get(m)
        if not a: continue
        k = akey(m); ax.plot(a[k]["iou"], a[k]["F1"], ls=LS[m], marker=MK[m], ms=3, lw=1.3, color=COL[m], label=NAME[m])
        if m in DET and f"flat_{cut[m]}" in a: ax.plot(a[k]["iou"], a[f"flat_{cut[m]}"]["F1"], ":", lw=1.1, color=COL[m])
    ax.set_title(title); ax.set_xlabel("IoU threshold"); ax.grid(alpha=.25)
axs[0].set_ylabel("F1")
legend_below(fig, [m for m in ORDER if an_lc.get(m) or an_ctc.get(m)], extra=[Line2D([], [], color="0.4", ls=":", lw=1.1, label="detector, flattened")], ncol=5, rect_bottom=0.12)
fig.savefig(f"{FIG}/f1_vs_iou.pdf"); fig.savefig(f"{FIG}/f1_vs_iou.png", dpi=200); plt.close()

# Error breakdown (LIVECell) + vs density + vs overlap
cats = [("loose", "loose match"), ("merged_gt", "merged"), ("split_gt", "split"), ("missed", "missed"), ("false_pos", "false pos.")]
ms_ = [m for m in ORDER if an_lc.get(m)]
if ms_:
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    x = np.arange(len(ms_)); w = 0.16; cc = ["#9aa5b1", "#d1453b", "#e58a2e", "#555555", "#3f6fb0"]
    for j, (k, lab) in enumerate(cats):
        axs[0].bar(x + (j - 2) * w, [an_lc[m][akey(m)]["per_1000_gt"][k] for m in ms_], w, color=cc[j], label=lab)
    axs[0].set_xticks(x); axs[0].set_xticklabels([NAME[m] for m in ms_], rotation=60, ha="right", fontsize=6); axs[0].set_ylabel("per 1000 GT cells")
    axs[0].legend(fontsize=5.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.2)
    for ax, col, bins, xl in [(axs[1], 0, [0, 1, 2, 3, 5, 100], ["0", "1", "2", "3-4", "5+"]), (axs[2], 1, [-0.01, 0.0001, 0.05, 0.15, 1.01], ["0", "<5%", "5-15%", ">15%"])]:
        for m in ms_:
            z = np.load(f"{R}/results/analysis/{an_ds(m)}__{m}__{akey(m)}.npz"); out, at = z["outcome"], z["attr"][:, col]
            b = np.digitize(at, bins) - 1; err = [(np.isin(out[b == i], [2, 3, 4])).mean() if (b == i).sum() > 50 else np.nan for i in range(len(bins) - 1)]
            ax.plot(range(len(bins) - 1), err, ls=LS[m], marker=MK[m], ms=3, color=COL[m], lw=1.2)
        ax.set_xticks(range(len(bins) - 1)); ax.set_xticklabels(xl); ax.grid(alpha=.25)
    axs[1].set_xlabel("neighbours within 1.5 diameters"); axs[1].set_ylabel("fraction missed/merged/split"); axs[1].set_title("vs local density")
    axs[2].set_xlabel("overlap with other GT cells"); axs[2].set_title("vs overlap fraction")
    legend_below(fig, ms_, ncol=5, rect_bottom=0.08, fs=6.5); fig.savefig(f"{FIG}/errors.pdf"); fig.savefig(f"{FIG}/errors.png", dpi=200); plt.close()

# metric flip scatter (LIVECell): COCO AP vs CP-AP50
if rows:
    # main axes = zoom on the cluster of competitive models; inset = all models (old Cellpose, micro-SAM)
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ZX, ZY = (38.5, 50.0), (0.700, 0.746)
    # label offsets (points) for the zoomed models; neighbours that nearly coincide go to opposite sides
    OFF = {"cpsam": (-14, 14, "right"), "livecell": (10, 16, "left"), "cyto3": (-6, -14, "right"), "vista2d": (8, 9, "left"),
           "anchor_based": (-4, 14, "right"), "anchor_free": (-8, -13, "right")}
    for r in rows:
        if ZX[0] <= r[1] <= ZX[1] and ZY[0] <= r[6] <= ZY[1]:
            ax.scatter(r[1], r[6], color=COL[r[0]], s=34, zorder=3, edgecolor="k", lw=.4)
            dx, dy, ha = OFF.get(r[0], (5, 4, "left"))
            ax.annotate(NAME[r[0]], (r[1], r[6]), xytext=(dx, dy), textcoords="offset points", fontsize=6, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", lw=.4, color="0.4", shrinkA=0, shrinkB=3))
    ax.set_xlim(*ZX); ax.set_ylim(*ZY)
    ax.set_xlabel("COCO mask AP (IoU 0.50:0.95)  $\\rightarrow$ detectors lead", fontsize=7)
    ax.set_ylabel("Cellpose-style AP$_{50}$  $\\rightarrow$ flows lead", fontsize=7); ax.tick_params(labelsize=6.5); ax.grid(alpha=.25)
    ins = ax.inset_axes([0.05, 0.07, 0.36, 0.36])
    for r in rows:
        ins.scatter(r[1], r[6], color=COL[r[0]], s=10, zorder=3, edgecolor="k", lw=.3)
    for m, (dx, dy) in {"microsam": (3, -1), "cyto": (3, -1), "cyto2": (3, 1)}.items():
        r = next((r for r in rows if r[0] == m), None)
        if r: ins.annotate(NAME[r[0]], (r[1], r[6]), xytext=(dx, dy), textcoords="offset points", fontsize=4.5, va="center")
    ins.add_patch(plt.Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ls="--", lw=.5, ec="0.3"))
    ins.set_xlim(15, 52); ins.set_ylim(0.35, 0.78); ins.tick_params(labelsize=4.5, length=1.5, pad=1); ins.set_title("all models", fontsize=5, pad=1.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/metric_flip.pdf"); fig.savefig(f"{FIG}/metric_flip.png", dpi=200); plt.close()

# counting: median signed count error per bin of the fluorescence nuclear count
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
EDG = np.array([0, 250, 500, 750, 1000, 1500, 2000, 2500, 3200]); cm_ms = []
for ax, cell in zip(axs, ["A172", "A549"]):
    for m in ORDER:
        f = f"{R}/results/counting_pairs/{cell}_{m}.json"
        if not (cnt.get(m) and cell in cnt[m] and os.path.exists(f)): continue
        pp = np.array(json.load(open(f)), float)
        if pp.ndim != 2 or len(pp) == 0: continue
        pr, ref = pp[:, 0], pp[:, 1]; b = np.digitize(ref, EDG) - 1; e = 100 * (pr - ref) / np.maximum(ref, 1)
        xs = [(EDG[i] + EDG[i + 1]) / 2 for i in range(len(EDG) - 1) if (b == i).sum() >= 5]
        ys = [np.median(e[b == i]) for i in range(len(EDG) - 1) if (b == i).sum() >= 5]
        ax.plot(xs, ys, ls=LS[m], marker=MK[m], ms=3, lw=1.3, color=COL[m])
        if m not in cm_ms: cm_ms.append(m)
    fj = f"{R}/results/fusion/count_fusion.json"; SHOW_FUSED = False   # fused curve dropped: fusion is a minor remedy (Sec. 7.3)
    if SHOW_FUSED and os.path.exists(fj) and f"{cell}/cpsam+anchor_based" in json.load(open(fj)):
        pi = np.array([x[1:] for x in json.load(open(fj))[f"{cell}/cpsam+anchor_based"]["per_image"]], float)
        ref, fused = pi[:, 0], pi[:, 2]; b = np.digitize(ref, EDG) - 1; e = 100 * (fused - ref) / np.maximum(ref, 1)
        xs = [(EDG[i] + EDG[i + 1]) / 2 for i in range(len(EDG) - 1) if (b == i).sum() >= 5]
        ys = [np.median(e[b == i]) for i in range(len(EDG) - 1) if (b == i).sum() >= 5]
        ax.plot(xs, ys, ls="-", marker="*", ms=6, lw=2.0, color="k", zorder=5)
    ax.axhline(0, color="k", lw=.7); ax.set_title(cell); ax.set_xlabel("fluorescence nuclear count per image"); ax.grid(alpha=.25)
axs[0].set_ylabel("count error (%), median per bin"); axs[0].set_ylim(-100, 60)
legend_below(fig, cm_ms, extra=[Line2D([], [], color="k", marker="*", ms=6, lw=2.0, label="Cellpose-SAM + Cascade (fused)")] if SHOW_FUSED else [], ncol=5, rect_bottom=0.14)
fig.savefig(f"{FIG}/counting.pdf"); fig.savefig(f"{FIG}/counting.png", dpi=200); plt.close()

# one font size for every table (no \\resizebox, which scales text differently per table)
import re as _re, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from captions import TABLE_CAPTIONS
for fn in os.listdir(TAB):
    if not fn.endswith(".tex"): continue
    t = open(f"{TAB}/{fn}").read()
    t = _re.sub(r"\\resizebox\{\\columnwidth\}\{!\}\{(\\begin\{tabular\})", r"\1", t).replace("\\end{tabular}}", "\\end{tabular}")
    t = t.replace("\\small", "\\footnotesize")
    lab = _re.search(r"\\label\{(tab:[^}]+)\}", t)
    if lab and lab.group(1) in TABLE_CAPTIONS:   # self-explanatory captions (code/captions.py)
        i = t.index("\\caption{") + len("\\caption{"); j = t.index("}\\label{" + lab.group(1))
        t = t[:i] + TABLE_CAPTIONS[lab.group(1)] + t[j:]
    open(f"{TAB}/{fn}", "w").write(t)
json.dump(S, open(f"{R}/results/summary.json", "w"), indent=1, default=float)
print("tables:", os.listdir(TAB)); print("figures:", os.listdir(FIG))
