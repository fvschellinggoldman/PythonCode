# https://colab.research.google.com/drive/1zYNprc--ZA-CO7dfK06OZTbhxbllBqZ-#scrollTo=9_FzH13EjseR
# Detectron imports
from collections import defaultdict

import torch
import torchvision
import os
import json
import cv2
import random
import numpy as np
import csv

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from DataAnnotation import label_segment_simple, load_dom_csv


def get_segment_dicts(img_dir, dom_dict):
    json_file = os.path.join(img_dir, "ground-truth.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []

    record = {}

    filename = os.path.join(img_dir, imgs_anns["id"])

    record["file_name"] = filename
    record["image_id"] = imgs_anns["id"]
    record["height"] = imgs_anns["height"]
    record["width"] = imgs_anns["width"]

    annos = imgs_anns["segmentations"]["majority-vote"]

    label_dict = {"Left": 0, "Top": 1, "Right": 2, "Bottom": 3, "Unclassified": 4}
    objs = []
    for anno in annos:
        for inner in anno:
            for poly in inner:
                # both bbox and poly might be numpy array type, make sure this isn't a problem
                # Label here
                bbox = np.concatenate((np.min(poly, axis=0), np.max(poly, axis=0)))
                label = label_segment_simple(record["height"], record["width"], bbox, dom_dict)
                obj = {
                    "bbox": bbox,
                    "segmentation": poly, # maybe needs to be flattened? np.array(poly).flatten()
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": label_dict[label],
                }
                objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
    return dataset_dicts


def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def colab_example():
    im = cv2.imread("./input.jpg")
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("test", out.get_image()[:, :, ::-1])
    cv2.imwrite('test.png', out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def train():
    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    return predictor


def runner():
    data_dir = "/home/fabio/MasterThesis/Dataset/webis-webseg-20-000000/000000"
    csv_dom_nodes = load_dom_csv(data_dir)
    get_segment_dicts(data_dir, csv_dom_nodes)


def main():
    #colab_example()

    #img_dir = "/home/fabio/MasterThesis/Dataset/webis-webseg-20-000000/000000"
    #print(get_segment_dicts(img_dir))

    #ballon_dir = "/home/fabio/MasterThesis/balloon/train"
    #print(get_balloon_dicts(ballon_dir))

    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")
    x = train()
    from detectron2.utils.visualizer import ColorMode
    dataset_dicts = get_balloon_dicts("balloon/val")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = x(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("t", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    runner()
