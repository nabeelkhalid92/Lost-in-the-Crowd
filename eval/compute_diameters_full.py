"""Same as compute_diameters.py but on the full LIVECell test set."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import json, os, re
import numpy as np
from collections import defaultdict

GT  = f"{LC_ANN}/livecell_coco_test.json"
OUT = f"{FC_WORK}/subset/diameters_full.json"
LINE_RE = re.compile(r"^([A-Za-z0-9]+)_Phase_")

coco = json.load(open(GT))
id2line = {im["id"]: LINE_RE.match(im["file_name"]).group(1) for im in coco["images"] if LINE_RE.match(im["file_name"])}
diams = defaultdict(list)
for a in coco["annotations"]:
    line = id2line.get(a["image_id"])
    if line is None: continue
    diams[line].append(2 * (a["area"] / np.pi) ** 0.5)
out = {}
print(f"{'cell':<10} {'median':>7} {'p25':>6} {'p75':>6} {'n':>7}")
for line, ds in sorted(diams.items()):
    ds = np.array(ds)
    med = float(np.median(ds))
    out[line] = round(med, 2)
    print(f"{line:<10} {med:>7.2f} {float(np.percentile(ds,25)):>6.2f} {float(np.percentile(ds,75)):>6.2f} {len(ds):>7}")
with open(OUT, "w") as f: json.dump(out, f, indent=2)
print(f"\nWrote {OUT}")
