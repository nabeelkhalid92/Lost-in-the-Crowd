"""Fine-tune Mask R-CNN (R50-FPN, COCO-pretrained, detectron2 0.6) on ISBI 2014 train (36 images) with the original
OVERLAPPING cell masks (each cell its own instance). 4000 iterations, batch 4, lr 0.005, horizontal + vertical flips,
native 512x512 input; evaluates COCO segm AP on the 9 validation images at the end.
usage: train_maskrcnn.py <data_dir> <out_dir>"""
import os, sys
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
D, OUT = sys.argv[1], sys.argv[2]
for s in ["train", "val", "test"]: register_coco_instances(f"isbi_{s}", {}, f"{D}/{s}.json", f"{D}/{s}/images")
cfg = get_cfg(); cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("isbi_train",); cfg.DATASETS.TEST = ("isbi_val",); cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MIN_SIZE_TRAIN = (512,); cfg.INPUT.MAX_SIZE_TRAIN = 512; cfg.INPUT.MIN_SIZE_TEST = 512; cfg.INPUT.MAX_SIZE_TEST = 512
cfg.SOLVER.IMS_PER_BATCH = 4; cfg.SOLVER.BASE_LR = 0.005; cfg.SOLVER.MAX_ITER = 4000; cfg.SOLVER.STEPS = (3000, 3600); cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.WARMUP_ITERS = 200; cfg.TEST.EVAL_PERIOD = 0; cfg.DATALOADER.NUM_WORKERS = 4; cfg.OUTPUT_DIR = OUT; cfg.SEED = 0
cfg.TEST.DETECTIONS_PER_IMAGE = 100; cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05; cfg.INPUT.MASK_FORMAT = "bitmask"
os.makedirs(OUT, exist_ok=True)
class Trainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        aug = [T.ResizeShortestEdge((512,), 512, "choice"), T.RandomFlip(horizontal=True), T.RandomFlip(horizontal=False, vertical=True)]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=aug))
    @classmethod
    def build_evaluator(cls, cfg, name): return COCOEvaluator(name, output_dir=cfg.OUTPUT_DIR)
open(f"{OUT}/config.yaml", "w").write(cfg.dump())
t = Trainer(cfg); t.resume_or_load(resume=False); t.train(); print(Trainer.test(cfg, t.model))
