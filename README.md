# Lost in the Crowd: code

Code for *Lost in the Crowd: What Cell Segmentation Benchmarks Hide About Flows and Detectors*.
Nine released segmentation models (six flow-based, one SAM-based, two detectors) are scored under one protocol on
LIVECell, five Cell Tracking Challenge (CTC) datasets and the LIVECell nuclear-count benchmark. One model of each
family is fine-tuned on the ISBI 2014 overlapping-cell data.

## Layout

| Folder | Contents |
|---|---|
| `inference/` | Run the nine models: Cellpose v1/2/3, Cellpose-SAM, LIVECell model (`infer_cellpose.py`, `infer_frames.py`); LIVECell detectors (`infer_detectron2.py`, `infer_frames.py`); VISTA-2D (`infer_vista2d.py`); micro-SAM (`infer_frames.py`). Also the CTC frame list (`prep_ctc.py`), Cellpose size-model diameters (`est_diam.py`), and flows/probabilities for Fig. 1 (`infer_flows.py`). |
| `eval/` | The protocol. Matching, F1, Cellpose-style AP and error types per cell (`analyze.py`). COCO mask AP with the LIVECell settings (`score_coco_v3.py`, `score_coco_generic.py`). CTC SEG and counts (`score_frames.py`). Bootstrap over images and wells (`bootstrap.py`, `bootstrap_grouped.py`). Operating-point sweep (`sweep_thresholds.py`). Absorbed and empty misses (`absorbed_split.py`). Count tuning (`count_tuning.py`). Fusion (`fuse.py`, `fusion_count_baselines.py`). Smaller analyses quoted in the text. |
| `isbi/` | ISBI 2014: conversion to COCO, fine-tuning of Cellpose-SAM and Mask R-CNN, inference, evaluation, flattened-annotation reference. |
| `paper/` | One script per table or figure (map below). |
| `litc_paths.py` | All file locations. Set them with environment variables, see below. |

## Setup

The models need different environments. The versions used in the paper:
- Cellpose 2.2.3 (Cellpose v1, Cellpose 2, Cellpose LIVECell)
- Cellpose 3.0 (Cellpose3)
- Cellpose 4.2.1 (Cellpose-SAM, `cpsam_v2`, CellposeDINO)
- micro-SAM 1.8.5
- the MONAI VISTA-2D bundle
- Detectron2, with:
  - [detectron2-ResNeSt](https://github.com/zhanghang1989/detectron2-ResNeSt) for Cascade Mask R-CNN
  - [CenterMask2](https://github.com/youngwanLEE/centermask2) for CenterMask2

Evaluation and the paper scripts only need `numpy scipy scikit-image pycocotools matplotlib tifffile h5py`.

Point the code at your data:

```bash
export LITC_WORK=/path/to/work          # predictions (preds/), analyses (results/), paper/draft/{tables,figures}
export LITC_LEGACY=$LITC_WORK/legacy    # LIVECell predictions of the nine models, reference counts
export LIVECELL_ANN=/path/to/livecell/annotations   # livecell_coco_{train,val,test}.json
export LIVECELL_IMG=/path/to/livecell/images        # livecell_test_images/, livecell_train_val_images/
export LIVECELL_MODELS=/path/to/LIVECell/model      # anchor_based/, anchor_free/ (config + weights)
export CTC_ROOT=/path/to/ctc/train                  # one folder per CTC dataset
export COUNT_ROOT=/path/to/nuclear_count            # A172/, A549/
export ISBI_WORK=/path/to/isbi_work
export VISTA2D_BUNDLE=/path/to/vista2d_bundle VISTA2D_PYTHON=/path/to/vista2d/python
export CENTERMASK2=/path/to/centermask2
```

## Data and models (download from the original sources; none are redistributed here)

| What | Where |
|---|---|
| LIVECell images and annotations, nuclear-count benchmark, the two LIVECell detectors | github.com/sartorius-research/LIVECell |
| Cell Tracking Challenge 2D training sets (PSC, U373, HeLa, HSC, MuSC) | celltrackingchallenge.net/2d-datasets |
| ISBI 2014 Overlapping Cervical Cytology | cs.adelaide.edu.au/~carneiro/isbi14_challenge |
| Cellpose models | downloaded automatically by Cellpose |
| micro-SAM `vit_b_lm` | downloaded automatically by micro-SAM |
| VISTA-2D | MONAI model zoo, bundle `vista2d` |

## Trained weights

The two models fine-tuned on ISBI 2014 (Table 7, Fig. 8). Each zip contains the weights, a model card with
results and usage, and its licence.

| Model | Download | Size | Licence |
|---|---|---|---|
| Cellpose-SAM, fine-tuned on ISBI 2014 | [cellpose-sam-isbi2014.zip](https://github.com/nabeelkhalid92/Lost-in-the-Crowd/releases/download/v1.0/cellpose-sam-isbi2014.zip) | 1.1 GB | BSD 3-Clause (from Cellpose-SAM; SAM part Apache 2.0) |
| Mask R-CNN R50-FPN, fine-tuned on ISBI 2014 | [maskrcnn-r50fpn-isbi2014.zip](https://github.com/nabeelkhalid92/Lost-in-the-Crowd/releases/download/v1.0/maskrcnn-r50fpn-isbi2014.zip) | 164 MB | CC BY-SA 3.0 (from the Detectron2 model zoo) |

SHA-256: `cellpose-sam-isbi2014.zip` 11ea0e47673e94f3d59393df99f0c50485fe1406f5362eefda78c73fabcece3b,
`maskrcnn-r50fpn-isbi2014.zip` 7fc89d9dac5285212ceca3210b0cf36ac5243c812a25953629cd7dc0554b9f80.
The nine benchmarked models are not redistributed; they come from their original sources (above).

## Pipeline

1. **Inference.**
   - LIVECell test: `inference/infer_cellpose.py --model <cyto|cyto2|cyto3|cpsam|livecell> --full --out <json>`, `inference/infer_detectron2.py --model <anchor_based|anchor_free> --full --out <json>`, `inference/infer_vista2d.py --full --out <json>`.
   - CTC and nuclear-count frames, and micro-SAM: `inference/infer_frames.py --method <m> --dataset <ctc|count|livecell>`.
2. **Scoring and analysis.**
   - `eval/analyze.py --dataset <livecell|livecell_val|ctc> --method <m>` gives per-cell outcomes, F1 and the Cellpose-style AP.
   - `eval/score_coco_generic.py` gives COCO AP.
   - `eval/score_frames.py --dataset <ctc|count>` gives SEG and counts.
   - `eval/bootstrap.py` and `eval/bootstrap_grouped.py` give the intervals.
3. **Operating points and counting.** `eval/sweep_thresholds.py`, `eval/absorbed_split.py`, `eval/fuse.py`, `eval/fusion_count_baselines.py`, `eval/count_tuning.py`.
4. **Tables and figures.** The scripts in `paper/` (map below). They write to `$LITC_WORK/paper/draft/{tables,figures}`.
5. **ISBI 2014.** Run these in order:
   1. `isbi/convert_isbi.py <Dataset/Synthetic> <data_dir>`
   2. `isbi/train_cellpose.py <data_dir> <out>` and `isbi/train_maskrcnn.py <data_dir> <out>`
   3. `isbi/infer_isbi.py <cp|mr> <data_dir> test <pred.json> [<mr_run>]`
   4. `isbi/eval_isbi.py <data_dir> test <pred.json> <cutoff> <name>` (Mask R-CNN cutoff 0.9, chosen on the 9 validation images)
   5. `isbi/flat_bound.py <data_dir> test <small|large> <out.json>` for the flattened-annotation reference

## Where each result comes from

| Paper | Script | Main inputs |
|---|---|---|
| Fig. 1 | `paper/make_intro_options.py` (uses `make_intro_figure_two.py`) | LIVECell predictions, `inference/infer_flows.py` |
| Table 2, Fig. 3, Fig. 4, Table 4, Table 5, Table 6, Fig. 7 | `paper/make_report.py` | `results/analysis/`, `results/coco_v3/`, `results/scores/`, `results/bootstrap.json`, `results/counting_pairs/` |
| Fig. 2 | `paper/make_rankflip.py` | COCO and F1 results |
| Table 3 | `paper/make_report.py` (cutoffs, flattening); `eval/lc_flat_coco.py`; Cellpose size-model runs | `results/analysis/livecell_cap3000__*`, `results/coco_v3/*_flat_*`, `*_autodiam_*` |
| Fig. 5, matched-false-positive differences | `paper/make_oppoint_curve.py`, `eval/sweep_thresholds.py`, `eval/absorbed_split.py` | `results/absorbed/`, threshold sweeps |
| Fig. 6 | `paper/make_qual_v2.py` (LIVECell part) | default and relaxed Cellpose-SAM, CenterMask2 |
| Table 7, Fig. 8 | `isbi/eval_isbi.py`, `isbi/flat_bound.py`, `paper/make_qual_isbi_v3.py` | ISBI predictions |
| Table 8 | `eval/count_tuning.py` | `preds/count/`, `results/fusion/count_fusion.json`, reference counts |
| Numbers in the text | `eval/gt_duplicates.py` (2.5% near-duplicates), `eval/overlap_pixels.py` (4–11% double-covered overlap), `eval/discarded_masks.py` (41% of discarded masks), `eval/per_line_crowding.py` (per cell line), `paper/make_speed.py` (run times) | |

## Reproduction check

Starting from the stored predictions and analyses, this code regenerated the following with identical numbers:
- Tables 2, 4, 5, 6, 7 and 8;
- Fig. 5 values: 23.4 / 25.5 / 27.9% crowded missed at matched false positives; 11.8% and F1 0.816 relaxed;
- the text of Figs. 3, 4 and 7.

Running inference from scratch has not been re-run with this copy.

## Licence

Code: MIT (see `LICENSE`). Third-party data and models keep their own licences; `docs/REDISTRIBUTION.md` lists what
may be shared.
