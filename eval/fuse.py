"""Instance-level flow + detector fusion (remedy for cells the flow model loses in crowded regions).
Keep every instance of the flow model; add a detection (score >= t) only if at most a fraction tau of its area is
already covered by flow instances; added detections are de-duplicated among themselves by greedy mask NMS (IoU 0.5).
The fused set is scored like any other method (F1 / Cellpose-AP at IoU 0.5..0.95, error taxonomy, count error vs GT).
  python fuse.py --split val  --flow livecell --det anchor_free          -> grid over (t, tau), results/fusion/val__...json
  python fuse.py --split test --flow cpsam    --det anchor_free --t 0.5 --tau 0.2   -> results/fusion/test__...json/.npz
GT: near-duplicate annotations removed when GT_DROP is set (results/gt_duplicates.json, test split only).
Baselines (--rule): "union" adds every detection >= t (NMS only among the added detections, tau ignored); "nms" adds a
detection only if its mask IoU with every flow instance is < 0.5 (plain union + mask NMS with flow instances first)."""
from _litc import *  # noqa: F401,F403  (file locations, see litc_paths.py)
import argparse, json, os, sys
from collections import defaultdict
import numpy as np
from pycocotools import mask as MU
from pycocotools.coco import COCO
sys.argv, _argv = sys.argv[:1], sys.argv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import LC_PRED, evaluate_image, gt_attributes, IOUS  # noqa: E402
sys.argv = _argv
R = f"{WORK}"
ap = argparse.ArgumentParser(); ap.add_argument("--split", required=True, choices=["val", "test"])
ap.add_argument("--flow", required=True); ap.add_argument("--det", required=True)
ap.add_argument("--t", type=float); ap.add_argument("--tau", type=float); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--rule", default="cov", choices=["cov", "union", "nms"])
a = ap.parse_args()
GT = {"val": f"{LC_ANN}/livecell_coco_val.json",
      "test": f"{LC_ANN}/livecell_coco_test.json"}[a.split]
def pred_file(m):
    if a.split == "val": return f"{R}/preds/livecell_val_{m}.json"
    if m in ("anchor_based", "anchor_free") and os.path.exists(f"{R}/preds/livecell_cap3000_{m}.json"): return f"{R}/preds/livecell_cap3000_{m}.json"
    return LC_PRED[m]
G = COCO(GT); DROP = set(json.load(open(os.environ["GT_DROP"]))["dropped_ann_ids"]) if os.environ.get("GT_DROP") else set()
P = {}
for m in (a.flow, a.det):
    d = defaultdict(list)
    for p in json.load(open(pred_file(m))): d[p["image_id"]].append(p)
    P[m] = d
def rl(p):
    s = p["segmentation"]; return {"size": s["size"], "counts": s["counts"].encode() if isinstance(s["counts"], str) else s["counts"]}
GRID = [(t, tau) for t in ([a.t] if a.t is not None else [0.3, 0.4, 0.5, 0.6, 0.7]) for tau in ([a.tau] if a.tau is not None else [0.1, 0.2, 0.3, 0.5])]
if a.rule != "cov": GRID = sorted({(t, 0.5) for t, _ in GRID})
acc = {g: {"T": np.zeros((len(IOUS), 3), np.int64), "out": [], "attr": [], "cnt": [], "added": 0, "timg": [], "pc": []} for g in GRID}
base = {"T": np.zeros((len(IOUS), 3), np.int64), "out": [], "cnt": [], "timg": [], "pc": []}
n = 0
for iid in sorted(G.imgs):
    gt = [G.annToRLE(x) for x in G.loadAnns([i for i in G.getAnnIds(imgIds=[iid]) if i not in DROP])]
    attr = gt_attributes(gt)
    fr = [rl(p) for p in P[a.flow].get(iid, [])]
    ds = sorted(P[a.det].get(iid, []), key=lambda p: -p["score"]); dr = [rl(p) for p in ds]; sc = np.array([p["score"] for p in ds])
    T0, o0, p0 = evaluate_image(gt, fr, attr); base["pc"].append(p0); base["T"] += T0; base["out"].append(o0); base["cnt"].append((len(fr), len(gt))); base["timg"].append(T0)
    if dr and fr:
        U = MU.merge(fr, intersect=False)
        cov = np.array([MU.area(MU.merge([d, U], intersect=True)) for d in dr], float) / np.maximum(MU.area(dr), 1)
    else:
        cov = np.zeros(len(dr))
    dd = MU.iou(dr, dr, [0] * len(dr)) if dr else np.zeros((0, 0))
    if a.rule == "union": cov = np.zeros(len(dr))
    elif a.rule == "nms": cov = (MU.iou(dr, fr, [0] * len(fr)).max(1) >= 0.5).astype(float) if dr and fr else np.zeros(len(dr))
    for (t, tau) in GRID:
        keep = []
        for k in np.nonzero((sc >= t) & (cov <= tau))[0]:      # score-ordered greedy NMS among the added detections
            if all(dd[k, j] < 0.5 for j in keep): keep.append(k)
        fused = fr + [dr[k] for k in keep]
        T, o, pc = evaluate_image(gt, fused, attr); A = acc[(t, tau)]; A["pc"].append(pc)
        A["T"] += T; A["out"].append(o); A["attr"].append(attr); A["cnt"].append((len(fused), len(gt))); A["added"] += len(keep); A["timg"].append(T)
    n += 1
    if n % 100 == 0: print(f"{n} images", flush=True)
    if a.limit and n >= a.limit: break
def summ(T, out, cnt, pc):
    tp, fp, fn = T[:, 0], T[:, 1], T[:, 2]; out = np.concatenate(out); c = np.array(cnt, float); pc = np.concatenate(pc)
    return {"F1": (2 * tp / np.maximum(2 * tp + fp + fn, 1)).tolist(), "CPAP_pooled": (tp / np.maximum(tp + fp + fn, 1)).tolist(),
            "missed_per_1000": 1000 * float((out == 4).mean()), "fp_per_1000": 1000 * float((pc == 3).sum()) / len(out), "n_pred": int(len(pc)), "merged_per_1000": 1000 * float((out == 2).mean()),
            "count_MAPE": float(np.mean(np.abs(c[:, 0] - c[:, 1]) / np.maximum(c[:, 1], 1)) * 100),
            "count_bias": float(np.mean((c[:, 0] - c[:, 1]) / np.maximum(c[:, 1], 1)) * 100)}
res = {"flow": a.flow, "det": a.det, "split": a.split, "n_img": n, "flow_only": summ(base["T"], base["out"], base["cnt"], base["pc"]), "fused": {}}
for (t, tau), A in acc.items():
    s = summ(A["T"], A["out"], A["cnt"], A["pc"]); s["added_per_img"] = A["added"] / max(n, 1); res["fused"][f"{t}_{tau}"] = s
os.makedirs(f"{R}/results/fusion", exist_ok=True); tag = f"{a.split}__{a.flow}+{a.det}" + ("" if a.rule == "cov" else f"__{a.rule}") + ("_dedup" if DROP else "") + os.environ.get("FUSE_SUFFIX", "")
json.dump(res, open(f"{R}/results/fusion/{tag}.json", "w"), indent=1)
if len(GRID) == 1:
    A = acc[GRID[0]]
    np.savez_compressed(f"{R}/results/fusion/{tag}_perimg.npz", outcome=np.concatenate(A["out"]), attr=np.concatenate(A["attr"]),
                        outcome_flow=np.concatenate(base["out"]), tff_fused=np.array(A["timg"]), tff_flow=np.array(base["timg"]),
                        n_gt=np.array([c[1] for c in A["cnt"]]), cnt_fused=np.array(A["cnt"]), cnt_flow=np.array(base["cnt"]))
fo = res["flow_only"]; print("flow only  F1@.5 %.3f F1@.75 %.3f miss %.0f count %.1f%%" % (fo["F1"][0], fo["F1"][5], fo["missed_per_1000"], fo["count_MAPE"]))
for k, s in res["fused"].items():
    print(f"fused {k:9s} F1@.5 {s['F1'][0]:.3f} F1@.75 {s['F1'][5]:.3f} miss {s['missed_per_1000']:.0f} merged {s['merged_per_1000']:.0f} count {s['count_MAPE']:.1f}% (+{s['added_per_img']:.1f}/img)")
