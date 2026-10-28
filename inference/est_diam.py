from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
# Estimate per-cell-line diameter on the LIVECell nuclear-count images with Cellpose-3 SizeModel (cyto3).
import glob, json, numpy as np, skimage.io, random
from cellpose import models
m = models.Cellpose(gpu=True, model_type="cyto3")
out = {}
for cell, root in [("A172", f"{COUNT_ROOT}/A172"), ("A549", f"{COUNT_ROOT}/A549")]:
    fs = sorted(glob.glob(f"{root}/*/*.png")); random.seed(0); fs = random.sample(fs, 40)
    ds = []
    for f in fs:
        img = skimage.io.imread(f)
        if img.ndim == 3: img = img[..., 0]
        d, _ = m.sz.eval(img, channels=[0, 0])
        ds.append(float(np.atleast_1d(d)[0]))
    out[cell] = {"median": float(np.median(ds)), "p10": float(np.percentile(ds, 10)), "p90": float(np.percentile(ds, 90))}
    print(cell, out[cell], flush=True)
json.dump(out, open(f"{WORK}/results/count_diam_sizemodel.json", "w"), indent=1)
