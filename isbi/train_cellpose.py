"""Fine-tune Cellpose-SAM (cpsam) on ISBI 2014 train (36 images). Cellpose trains on one label per pixel, so the
overlapping cells are flattened (convert_isbi.py: smaller cell on top) -- the representation's constraint.
Settings: cellpose 4 defaults for fine-tuning (AdamW, lr 1e-5, weight decay 0.1), 300 epochs of 72 images,
min_train_masks 1 (ISBI images have 2-10 cells). Validation images are monitored only.
usage: train_cellpose.py <data_dir> <out_dir>"""
import glob, os, sys
import numpy as np, skimage.io
from cellpose import models, train
D, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
def load(split):
    fs = sorted(glob.glob(f"{D}/{split}/images/*.png"))
    return [skimage.io.imread(f) for f in fs], [skimage.io.imread(f"{D}/{split}/labels/{os.path.basename(f)[:-4]}_masks.png").astype(np.int32) for f in fs]
xtr, ytr = load("train"); xva, yva = load("val"); print(len(xtr), "train", len(xva), "val", xtr[0].shape)
model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
path = train.train_seg(model.net, train_data=xtr, train_labels=ytr, test_data=xva, test_labels=yva, learning_rate=1e-5, weight_decay=0.1,
                       n_epochs=300, nimg_per_epoch=72, batch_size=4, min_train_masks=1, save_path=OUT, model_name="cpsam_isbi", save_every=100)
print("SAVED", path)
