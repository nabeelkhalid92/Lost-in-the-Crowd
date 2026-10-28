"""All file locations used by the code, in one place. Set the environment variables (or edit the defaults) once.

LITC_WORK         working directory: predictions (preds/), scores and analyses (results/), paper tables and figures
LITC_LEGACY       LIVECell inference outputs of the nine models (predictions/, subset/, results/, counting_data/)
LIVECELL_ANN      LIVECell COCO annotation folder (livecell_coco_{train,val,test}.json)
LIVECELL_IMG      LIVECell image folder containing livecell_test_images/ and livecell_train_val_images/
LIVECELL_MODELS   LIVECell detector release (anchor_based/, anchor_free/: config + weights)
CTC_ROOT          Cell Tracking Challenge 2D training data (one folder per dataset)
COUNT_ROOT        LIVECell nuclear-count benchmark (A172/, A549/)
ISBI_WORK         ISBI 2014 overlapping-cervical-cell work directory (data/, runs/, preds/)
VISTA2D_BUNDLE    MONAI VISTA-2D bundle folder;  VISTA2D_PYTHON: python of the environment that runs it
CENTERMASK2       checkout of github.com/youngwanLEE/centermask2 (only for the CenterMask2 detector)
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
_e = lambda k, d: os.environ.get(k, d)
WORK = _e("LITC_WORK", os.path.join(REPO, "work"))
FC_WORK = _e("LITC_LEGACY", os.path.join(WORK, "legacy"))
LC_ANN = _e("LIVECELL_ANN", os.path.join(REPO, "data", "livecell", "annotations"))
LC_IMG = _e("LIVECELL_IMG", os.path.join(REPO, "data", "livecell", "images"))
LC_MODELS = _e("LIVECELL_MODELS", os.path.join(REPO, "models", "livecell"))
CTC_ROOT = _e("CTC_ROOT", os.path.join(REPO, "data", "ctc"))
COUNT_ROOT = _e("COUNT_ROOT", os.path.join(REPO, "data", "nuclear_count"))
ISBI_WORK = _e("ISBI_WORK", os.path.join(REPO, "work_isbi"))
VISTA2D_BUNDLE = _e("VISTA2D_BUNDLE", os.path.join(REPO, "models", "vista2d_bundle"))
VISTA2D_PYTHON = _e("VISTA2D_PYTHON", "python")
CENTERMASK2 = _e("CENTERMASK2", os.path.join(REPO, "third_party", "centermask2"))
