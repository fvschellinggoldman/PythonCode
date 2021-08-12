# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from DataAnnotation import load_dom_csv
from DetectronExperiment import get_segment_dicts, train_model, evaluate_model
from ImageComparer import imagehash_approach
from detectron2.data import MetadataCatalog, DatasetCatalog


def main(filepath, dummy_dataset=True, save_img_segments=False):
    if save_img_segments:
        imagehash_approach(filepath)

    csv_dom_nodes = load_dom_csv(filepath)
    for d in ["train", "val"]:
        DatasetCatalog.register("segmentation_" + d, lambda d=d: get_segment_dicts("balloon/" + d, dom_dict=csv_dom_nodes, partial=dummy_dataset))
        MetadataCatalog.get("segmentation_" + d).set(thing_classes=["Left", "Top", "Right", "Bottom", "Unclassified"])
    segmentation_meta_data = MetadataCatalog.get("segmentation_train")
    model = train_model()
    evaluate_model()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filepath = "../Dataset/webis-webseg-20"
    main(filepath)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
