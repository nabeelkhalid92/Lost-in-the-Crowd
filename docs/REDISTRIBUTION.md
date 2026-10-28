# What can be redistributed

Licences checked on 2026-10-28 from primary sources: LICENSE files, dataset pages and model cards. This is not legal
advice. Items marked *not found* had no licence statement anywhere checked.

## Summary

| Item | Release? | Licence to use / conditions |
|---|---|---|
| Our code (this repository) | **Yes** | MIT (`LICENSE`). The code only imports the libraries below and contains none of their code. |
| Original data (LIVECell, nuclear count, CTC, ISBI 2014) | **No, link only** | CTC forbids cloning ("strictly forbidden"). ISBI has no licence, so all rights are reserved. LIVECell is CC BY-NC 4.0; it could be re-hosted, but linking is cleaner. |
| Released weights of the nine models | **No, link only** | Each is downloaded from its source (table below). |
| Our LIVECell predictions (9 models) and analyses (`results/`) | **Yes, as CC BY-NC 4.0** | Derived from CC BY-NC data and partly non-commercial models (LIVECell detectors, CenterMask2, VISTA-2D, old Cellpose weights). Attribute LIVECell. |
| Our nuclear-count predictions | **Yes, as CC BY-NC 4.0** | Same terms as LIVECell. |
| Our CTC predictions | **Only with CTC permission** | CTC terms: "Any public non-CTC-related, scientific use of our datasets requires explicit permission from the challenge organizers." |
| Our ISBI 2014 predictions | **Ask first** | No licence was found. Email the organizers (Lu, Carneiro). |
| ISBI fine-tuned Cellpose-SAM | **Released (v1.0)** | BSD-3-Clause (the `cpsam` model card), plus the Apache-2.0 notice for SAM. Add "trained on CC BY-NC data, for non-commercial research": the Cellpose README says all Cellpose models are trained on CC BY-NC data. |
| ISBI fine-tuned Mask R-CNN | **Released (v1.0) as CC BY-SA 3.0** | Started from Detectron2 model-zoo COCO weights, which are CC BY-SA 3.0 (share-alike). It cannot be relicensed as MIT, Apache or NC. |

## Licences of third-party components

| Component | Licence | Source |
|---|---|---|
| Cellpose code | BSD-3-Clause (the README's "GPL v3" badge is outdated) | github.com/MouseLand/cellpose/blob/main/LICENSE |
| Cellpose-SAM weights (`cpsam`, `cpsam_v2`) | BSD-3-Clause on the model card; training data CC BY-NC | huggingface.co/mouseland/cellpose-sam |
| Cellpose v1/2/3 and LIVECell model weights | *not found*; HHMI content terms say non-commercial, educational and research use | cellpose.org |
| Segment Anything (SAM) weights | Apache-2.0 | github.com/facebookresearch/segment-anything |
| CellposeDINO (`cpdino`), via DINOv3 | DINOv3 License (custom): derivatives may only be shared under it, with its text included. It also bans military use and has trade-control terms. | github.com/facebookresearch/dinov3/blob/main/LICENSE.md |
| LIVECell images, annotations, detector weights, nuclear-count benchmark | CC BY-NC 4.0 (the code in that repo is MIT) | github.com/sartorius-research/LIVECell |
| Cell Tracking Challenge datasets | Custom terms: no cloning; non-CTC public use needs permission; cite and acknowledge the CTC | celltrackingchallenge.net/datasets |
| ISBI 2014 cervical cytology | *not found* (citation request only) | cs.adelaide.edu.au/~carneiro/isbi14_challenge |
| micro-SAM | code MIT; `vit_b_lm` weights CC BY 4.0 | github.com/computational-cell-analytics/micro-sam |
| VISTA-2D | code Apache-2.0; weights CC BY-NC-SA 4.0 | github.com/Project-MONAI/model-zoo (models/vista2d) |
| Detectron2 | code Apache-2.0; model-zoo weights CC BY-SA 3.0 | github.com/facebookresearch/detectron2 (LICENSE, MODEL_ZOO.md) |
| CenterMask2 code | CC BY-NC 4.0 | github.com/youngwanLEE/centermask2 |
| VoVNet backbone weights | *not found*; the sister repo is CC BY-NC 4.0 | github.com/youngwanLEE/vovnet-detectron2 |
| ResNeSt / detectron2-ResNeSt | code Apache-2.0; weights *not found* | github.com/zhanghang1989/ResNeSt |
| MS COCO | annotations CC BY 4.0; images under Flickr terms | cocodataset.org |

## To do before release

1. **CTC permission.** The CTC terms require explicit permission for public scientific use outside the challenge. That
   covers releasing CTC predictions and possibly the CTC results in the paper itself. Email the organizers.
2. **ISBI 2014.** The data have no explicit licence; the two fine-tuned models are released with attribution. Ask Lu and
   Carneiro before publishing ISBI predictions.
3. **Release files.**
   - Model files: the BSD-3 and Apache notices for the fine-tuned Cellpose-SAM, and the CC BY-SA 3.0 text for the fine-tuned Mask R-CNN.
   - Prediction archive: CC BY-NC 4.0, attributing LIVECell.
