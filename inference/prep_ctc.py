"""Build the CTC label-free manifest.
Segmentation frames = frames with gold SEG GT (scored vs gold with the CTC SEG measure, and vs the complete
silver ST masks with COCO AP). Counting frames = up to 150 evenly spaced frames per sequence, count = number of
TRA markers in that frame (exact, complete). Oracle diameter = median equivalent diameter of ST masks."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import glob, json, os, re, numpy as np, tifffile
T = f"{CTC_ROOT}"
DS = ["PhC-C2DL-PSC", "PhC-C2DH-U373", "DIC-C2DH-HeLa", "BF-C2DL-HSC", "BF-C2DL-MuSC"]
man = {}
for d in DS:
    for s in ["01", "02"]:
        imgs = sorted(glob.glob(f"{T}/{d}/{s}/t*.tif"))
        nT = len(imgs)
        seg = sorted(glob.glob(f"{T}/{d}/{s}_GT/SEG/man_seg*.tif"))
        segf = [int(re.findall(r"man_seg_?(\d+)", os.path.basename(p))[0]) for p in seg]
        step = max(1, nT // 150); cnt_frames = list(range(0, nT, step))
        counts = {}
        for t in cnt_frames:
            m = tifffile.imread(f"{T}/{d}/{s}_GT/TRA/man_track{t:0{len(os.path.basename(imgs[0]))-5}d}.tif")
            counts[t] = int(len(np.unique(m)) - 1)
        diams = []
        for t in segf[:10] or cnt_frames[:5]:
            st = tifffile.imread(f"{T}/{d}/{s}_ST/SEG/man_seg{t:0{len(os.path.basename(imgs[0]))-5}d}.tif")
            _, a = np.unique(st[st > 0], return_counts=True); diams += list(2 * np.sqrt(a / np.pi))
        nd = len(os.path.basename(imgs[0])) - 5
        man[f"{d}/{s}"] = {"n_frames": nT, "ndig": nd, "img_dir": f"{T}/{d}/{s}",
                           "seg_gold": {t: p for t, p in zip(segf, seg)},
                           "seg_silver": {t: f"{T}/{d}/{s}_ST/SEG/man_seg{t:0{nd}d}.tif" for t in segf},
                           "count_frames": counts, "diam_median": float(np.median(diams))}
        print(d, s, "frames", nT, "goldSEG", len(segf), "count frames", len(counts),
              "count range", min(counts.values()), max(counts.values()), "diam", round(float(np.median(diams)), 1), flush=True)
json.dump(man, open(f"{WORK}/data/ctc_manifest.json", "w"), indent=1)
