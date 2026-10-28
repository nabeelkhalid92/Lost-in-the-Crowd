# Lost in the Crowd

**What Cell Segmentation Benchmarks Hide About Flows and Detectors**: code, evaluation protocol and trained weights.

## What this is

Two families of deep models dominate cell instance segmentation:

- **Flow-based models** (Cellpose, Cellpose-SAM, VISTA-2D) predict a flow field and give every pixel to exactly one cell.
- **Detection-based models** (Mask R-CNN, CenterMask2) propose cells with a confidence score and segment each one separately, so masks may overlap.

Published comparisons of the two families reach opposite conclusions. This repository scores **nine released models
under one protocol** to find out why, and where the families really differ. The data are LIVECell, five Cell Tracking
Challenge datasets, a fluorescence nuclear-count benchmark and the ISBI 2014 overlapping-cell set.

## The problem it addresses

The two communities evaluate differently:
- Detectors are scored with COCO mask AP, which ranks detections by confidence.
- Flow models are scored with an "average precision" defined as TP / (TP + FP + FN), computed from one fixed set of instances.

Both are called "AP", but they measure different things. Comparing across families also depends on settings that are
rarely reported: the detector's score cutoff, how overlapping masks are flattened, the cell diameter given to the flow
model, and its flow threshold. Change those, and the winner changes.

## Main findings

- **The metric decides the ranking.** On the same predictions, detectors lead by 4.7 points of COCO mask AP, while flow models lead on F1 at IoU 0.5.
- **Crowding is an operating-point effect.**
  - At default settings, flow models miss about twice as many crowded cells as the best detector, and undercount confluent wells by up to 48%.
  - At matched false-positive rates they lose no more crowded cells than the detectors.
  - One relaxed flow threshold closes the crowding gap, and a threshold tuned on a few reference counts removes the undercount.
- **Overlap is a real limit of flow models.** On ISBI 2014, a fine-tuned Cellpose-SAM keeps 56% of the part of a cell hidden under a neighbour (the annotation flattened to one label per pixel keeps 58%). A fine-tuned Mask R-CNN keeps 92%.
- **Transfer follows the training data, not the family.** Broadly trained generalists hold up on new imaging conditions; models trained on LIVECell alone fail on most of them.

## Trained weights

The two models fine-tuned on ISBI 2014 overlapping cervical cells (Table 7 and Fig. 8 of the paper). Each zip contains
the weights, a model card with results and usage code, and its licence.

| Model | Download | Size | Licence |
|---|---|---|---|
| Cellpose-SAM, fine-tuned on ISBI 2014 | [cellpose-sam-isbi2014.zip](https://github.com/nabeelkhalid92/Lost-in-the-Crowd/releases/download/v1.0/cellpose-sam-isbi2014.zip) | 1.1 GB | BSD 3-Clause (from Cellpose-SAM; SAM part Apache 2.0) |
| Mask R-CNN R50-FPN, fine-tuned on ISBI 2014 | [maskrcnn-r50fpn-isbi2014.zip](https://github.com/nabeelkhalid92/Lost-in-the-Crowd/releases/download/v1.0/maskrcnn-r50fpn-isbi2014.zip) | 164 MB | CC BY-SA 3.0 (from the Detectron2 model zoo) |

Checksums are in [SHA256SUMS.txt](https://github.com/nabeelkhalid92/Lost-in-the-Crowd/releases/download/v1.0/SHA256SUMS.txt)
(`sha256sum -c SHA256SUMS.txt`). The nine benchmarked models are not redistributed; get them from their original
sources (see [Data and models](#data-and-models)).

## Repository layout

| Folder | What it does |
|---|---|
| `inference/` | Runs the nine models on LIVECell, the CTC frames and the nuclear-count images |
| `eval/` | The evaluation protocol: matching, F1, Cellpose-style AP, COCO AP, error types per cell, operating-point sweeps, counting, bootstrap intervals |
| `isbi/` | The ISBI 2014 overlap benchmark: data conversion, fine-tuning of both models, inference, evaluation |
| `paper/` | One script per table or figure of the paper |
| `docs/REDISTRIBUTION.md` | What data, predictions and models may be shared, and under which licence |
| `litc_paths.py` | All file locations in one place |

## Installation

Evaluation and the paper scripts need only:

```bash
pip install numpy scipy scikit-image pycocotools matplotlib tifffile h5py
```

Running the models needs one environment per model family. The versions used in the paper:
- Cellpose 2.2.3 (Cellpose v1, Cellpose 2, Cellpose LIVECell)
- Cellpose 3.0 (Cellpose3)
- Cellpose 4.2.1 (Cellpose-SAM, `cpsam_v2`, CellposeDINO)
- micro-SAM 1.8.5
- the MONAI VISTA-2D bundle
- Detectron2, with [detectron2-ResNeSt](https://github.com/zhanghang1989/detectron2-ResNeSt) (Cascade Mask R-CNN) and [CenterMask2](https://github.com/youngwanLEE/centermask2) (CenterMask2)

Then point the code at your data with environment variables. Each one is described in `litc_paths.py`:

```bash
export LITC_WORK=/path/to/work                      # predictions, analyses, output tables and figures
export LITC_LEGACY=$LITC_WORK/legacy                # LIVECell predictions of the nine models, reference counts
export LIVECELL_ANN=/path/to/livecell/annotations   # livecell_coco_{train,val,test}.json
export LIVECELL_IMG=/path/to/livecell/images        # livecell_test_images/, livecell_train_val_images/
export LIVECELL_MODELS=/path/to/LIVECell/model      # anchor_based/, anchor_free/ (config + weights)
export CTC_ROOT=/path/to/ctc/train                  # one folder per CTC dataset
export COUNT_ROOT=/path/to/nuclear_count            # A172/, A549/
export ISBI_WORK=/path/to/isbi_work
export VISTA2D_BUNDLE=/path/to/vista2d_bundle VISTA2D_PYTHON=/path/to/vista2d/python
export CENTERMASK2=/path/to/centermask2
```

## Data and models

None of these are redistributed here; download them from the original sources.

| What | Where |
|---|---|
| LIVECell images and annotations, nuclear-count benchmark, the two LIVECell detectors | [github.com/sartorius-research/LIVECell](https://github.com/sartorius-research/LIVECell) |
| Cell Tracking Challenge 2D training sets (PSC, U373, HeLa, HSC, MuSC) | [celltrackingchallenge.net](https://celltrackingchallenge.net/2d-datasets/) |
| ISBI 2014 Overlapping Cervical Cytology | [cs.adelaide.edu.au/~carneiro/isbi14_challenge](https://cs.adelaide.edu.au/~carneiro/isbi14_challenge/) |
| Cellpose models, micro-SAM `vit_b_lm` | downloaded automatically by Cellpose and micro-SAM |
| VISTA-2D | MONAI model zoo, bundle `vista2d` |

## Reproducing the paper

1. **Run the models.**
   - LIVECell test:
     - `inference/infer_cellpose.py --model <cyto|cyto2|cyto3|cpsam|livecell> --full --out <json>`
     - `inference/infer_detectron2.py --model <anchor_based|anchor_free> --full --out <json>`
     - `inference/infer_vista2d.py --full --out <json>`
   - CTC and nuclear-count frames, and micro-SAM: `inference/infer_frames.py --method <m> --dataset <ctc|count|livecell>`
2. **Score them.**
   - `eval/analyze.py --dataset <livecell|livecell_val|ctc> --method <m>`: per-cell outcomes, F1, Cellpose-style AP
   - `eval/score_coco_generic.py`: COCO AP
   - `eval/score_frames.py --dataset <ctc|count>`: CTC SEG and counts
   - `eval/bootstrap.py`, `eval/bootstrap_grouped.py`: intervals
3. **Operating points and counting.** `eval/sweep_thresholds.py`, `eval/absorbed_split.py`, `eval/count_tuning.py`, `eval/fuse.py`, `eval/fusion_count_baselines.py`
4. **Tables and figures.** Run the scripts in `paper/`; they write to `$LITC_WORK/paper/draft/{tables,figures}`.
5. **ISBI 2014**, in this order:
   1. `isbi/convert_isbi.py <Dataset/Synthetic> <data_dir>`
   2. `isbi/train_cellpose.py <data_dir> <out>` and `isbi/train_maskrcnn.py <data_dir> <out>`
   3. `isbi/infer_isbi.py <cellpose|maskrcnn> <data_dir> test <pred.json> [<maskrcnn_run_dir>]`
   4. `isbi/eval_isbi.py <data_dir> test <pred.json> <cutoff> <name>` (Mask R-CNN cutoff 0.9, chosen on the 9 validation images)
   5. `isbi/flat_bound.py <data_dir> test <small|large> <out.json>` for the flattened-annotation reference

<details>
<summary><b>Which script produces which table or figure</b></summary>

| Paper | Script | Main inputs |
|---|---|---|
| Fig. 1 | `paper/make_intro_options.py` (uses `make_intro_figure_two.py`) | LIVECell predictions, `inference/infer_flows.py` |
| Fig. 2 | `paper/make_rankflip.py` | COCO and F1 results |
| Table 2; Figs. 3, 4, 7; Tables 4, 5, 6 | `paper/make_report.py` | `results/analysis/`, `results/coco_v3/`, `results/scores/`, `results/bootstrap.json`, `results/counting_pairs/` |
| Table 3 | `paper/make_report.py` (cutoffs, flattening), `eval/lc_flat_coco.py`, Cellpose size-model runs | `results/analysis/livecell_cap3000__*`, `results/coco_v3/*_flat_*`, `*_autodiam_*` |
| Fig. 5 and the matched-false-positive differences | `paper/make_oppoint_curve.py`, `eval/sweep_thresholds.py`, `eval/absorbed_split.py` | `results/absorbed/`, threshold sweeps |
| Fig. 6 | `paper/make_qual_v2.py` | default and relaxed Cellpose-SAM, CenterMask2 |
| Table 7, Fig. 8 | `isbi/eval_isbi.py`, `isbi/flat_bound.py`, `paper/make_qual_isbi_v3.py` | ISBI predictions |
| Table 8 | `eval/count_tuning.py` | `preds/count/`, `results/fusion/count_fusion.json`, reference counts |
| Numbers in the text | `eval/gt_duplicates.py`, `eval/overlap_pixels.py`, `eval/discarded_masks.py`, `eval/per_line_crowding.py`, `paper/make_speed.py` | |

Starting from the stored predictions, this code regenerated Tables 2, 4, 5, 6, 7 and 8 and the Fig. 5 values with
identical numbers. The inference step was not re-run from scratch with this copy.
</details>

## Licence

The code in this repository is MIT-licensed (`LICENSE`). Datasets and third-party models keep their own licences; the
two trained models carry the licences listed above. See [`docs/REDISTRIBUTION.md`](docs/REDISTRIBUTION.md) for details.
