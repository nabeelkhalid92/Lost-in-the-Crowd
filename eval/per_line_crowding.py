"""Per-cell-line crowding analysis (LIVECell test): share of annotated cells with >= 5 neighbors, and the miss rate of
those crowded cells per model. Cells are mapped to images through the per-image annotated counts stored with the
instance analysis. Writes results/per_line_crowding.json and paper/draft/tables/perline.tex."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, numpy as np
R = f"{WORK}"; CUT = json.load(open(f"{R}/results/val_cutoffs.json"))
M = [("cpsam", "SAM"), ("cyto3", "CP3"), ("livecell", "LC"), ("vista2d", "VIS"), ("anchor_based", "Casc"), ("anchor_free", "CM2")]
LINES = ["A172", "BT474", "BV2", "Huh7", "MCF7", "SHSY5Y", "SKOV3", "SkBr3"]
NAMES = {"A172": "A172", "BT474": "BT-474", "BV2": "BV-2", "Huh7": "Huh7", "MCF7": "MCF7", "SHSY5Y": "SH-SY5Y", "SKOV3": "SK-OV-3", "SkBr3": "SK-BR-3"}
res = {}
for m, _ in M:
    ds = "livecell_cap3000" if m in CUT else "livecell"; k = f"raw_{CUT[m]}" if m in CUT else "flat_None"
    z = np.load(f"{R}/results/analysis/{ds}__{m}__{k}.npz", allow_pickle=True)
    o, a, keys, ng = z["outcome"], z["attr"], z["keys"], z["counts"][:, 1].astype(int)
    line = np.repeat(np.array([str(x).split("_")[0] for x in keys]), ng); crowd = a[:, 0] >= 5
    for L in LINES:
        s = (line == L) & crowd
        res.setdefault(L, {"n_cells": int((line == L).sum()), "crowded_share": float(s.sum() / max((line == L).sum(), 1) * 100)})[m] = float((o[s] == 4).mean() * 100) if s.sum() else None
json.dump(res, open(f"{R}/results/per_line_crowding.json", "w"), indent=1)
Lx = ["\\begin{table}[t]\\centering",
      "\\caption{\\textbf{The flow models lose most crowded cells in MCF7 and SH-SY5Y, hardly more than the detectors in SK-OV-3 and SK-BR-3.} LIVECell test. Crowd: share (\\%) of cells with five or more neighbors; other columns: miss rate (\\%) of these cells. SAM: Cellpose-SAM; CP3: Cellpose3; LC: Cellpose LIVECell; VIS: VISTA-2D; Casc: Cascade Mask R-CNN; CM2: CenterMask2.}\\label{tab:perline}",
      "\\footnotesize\\setlength{\\tabcolsep}{2.0pt}\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule",
      "Line & Crowd & " + " & ".join(n for _, n in M) + " \\\\\\midrule"]
for L in LINES:
    r = res[L]
    Lx.append(f"{NAMES[L]} & {r['crowded_share']:.0f} & " + " & ".join("--" if r[m] is None else f"{r[m]:.0f}" for m, _ in M) + " \\\\")
Lx.append("\\bottomrule\\end{tabular}\\end{table}")
open(f"{R}/paper/draft/tables/perline.tex", "w").write("\n".join(Lx) + "\n")
for L in LINES: print(L, round(res[L]["crowded_share"]), {m: (None if res[L][m] is None else round(res[L][m], 1)) for m, _ in M})
