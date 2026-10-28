"""Convert the ISBI 2014 Overlapping Cervical Cytology synthetic set (Lu et al. 2015/2017) to COCO with one OVERLAPPING
mask per cell (cytoplasm annotation = whole cell). Train: 45 images (first 36 -> train, last 9 -> val, for selecting the
detector cutoff); test: 900 images with ground truth. Each image keeps its overlap-ratio bin and cell number.
Also writes Cellpose training label images (one label per pixel): where cells overlap, the pixel goes to the cell whose
mask is smaller (the cell on top keeps its full extent) -- the flattening a one-label-per-pixel model must train on.
usage (anaconda python with h5py): convert_isbi.py <extracted/Dataset/Synthetic> <out_dir>"""
import json, os, sys
import h5py, numpy as np, scipy.io as sio, skimage.io
from pycocotools import mask as MU
SRC, OUT = sys.argv[1], sys.argv[2]

def h5cells(f, key):
    h = h5py.File(f, "r"); refs = h[key][()]
    return h, [h[r] for r in refs.reshape(-1)]

def to_img(a):
    a = np.asarray(a).T if a.ndim == 2 else np.asarray(a)
    return a

def write(split, items):
    os.makedirs(f"{OUT}/{split}/images", exist_ok=True); os.makedirs(f"{OUT}/{split}/labels", exist_ok=True)
    J = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cell"}]}; aid = 1
    for k, (img, masks, ov, n) in enumerate(items):
        fn = f"{split}_{k:03d}.png"; skimage.io.imsave(f"{OUT}/{split}/images/{fn}", img.astype(np.uint8), check_contrast=False)
        J["images"].append({"id": k + 1, "file_name": fn, "height": img.shape[0], "width": img.shape[1], "overlap_lo": float(ov[0]), "overlap_hi": float(ov[1]), "cell_num": int(n)})
        lab = np.zeros(img.shape, np.uint16)
        for j in np.argsort([-m.sum() for m in masks]):                 # large first, smaller cells painted on top
            lab[masks[j]] = j + 1
        skimage.io.imsave(f"{OUT}/{split}/labels/{fn[:-4]}_masks.png", lab, check_contrast=False)
        skimage.io.imsave(f"{OUT}/{split}/images_cp/{fn}" if False else f"{OUT}/{split}/labels/{fn}", img.astype(np.uint8), check_contrast=False)
        for m in masks:
            r = MU.encode(np.asfortranarray(m.astype(np.uint8))); a = float(MU.area(r)); bb = [float(x) for x in MU.toBbox(r)]; r["counts"] = r["counts"].decode()
            J["annotations"].append({"id": aid, "image_id": k + 1, "category_id": 1, "segmentation": r, "area": a, "bbox": bb, "iscrowd": 0}); aid += 1
    json.dump(J, open(f"{OUT}/{split}.json", "w")); print(split, len(J["images"]), "images", len(J["annotations"]), "cells")

# training set: images in HDF5, GT in a v5 .mat
h, timgs = h5cells(f"{SRC}/trainset.mat", "trainset")
gt = sio.loadmat(f"{SRC}/trainset_GT.mat", simplify_cells=True)
train = []
for k in range(45):
    img = to_img(timgs[k][()]); cyto = gt["train_Cytoplasm"][k]; cyto = cyto if isinstance(cyto, (list, np.ndarray)) and np.asarray(cyto[0]).ndim == 2 else [cyto]
    masks = [np.asarray(c).astype(bool) for c in cyto]
    assert masks[0].shape == img.shape, (k, masks[0].shape, img.shape)
    train.append((img, masks, gt["OverlapRatio"][k], gt["CellNum"][k]))
write("train", train[:36]); write("val", train[36:])
# test set: images and GT both HDF5
h1, simgs = h5cells(f"{SRC}/testset.mat", "testset")
h2 = h5py.File(f"{SRC}/testset_GT.mat", "r"); crefs = h2["test_Cytoplasm"][()].reshape(-1); ov = h2["OverlapRatio"][()]; cn = h2["CellNum"][()].reshape(-1)
test = []
for k in range(900):
    img = to_img(simgs[k][()]); cell = h2[crefs[k]][()]
    if cell.dtype == object or cell.dtype.kind == "O" or cell.dtype == h5py.ref_dtype:
        masks = [to_img(h2[r][()]).astype(bool) for r in cell.reshape(-1)]
    else:
        masks = [to_img(cell).astype(bool)]
    assert masks[0].shape == img.shape, (k, masks[0].shape, img.shape)
    test.append((img, masks, ov[:, k], cn[k]))
write("test", test)
