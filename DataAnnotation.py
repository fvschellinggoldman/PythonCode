
# left, bottom, right, top
import csv
from collections import defaultdict
from bs4 import BeautifulSoup
from lxml import etree


# Returns interactable dom tree that can be queried via xpath from csv
def load_html(html_dir):
    with open(html_dir + "/dom.html") as html_doc:
        soup = BeautifulSoup(html_doc, 'html.parser')
    dom = etree.HTML(str(soup))
    return dom


def load_dom_csv(csv_dir):
    dom = defaultdict(lambda : "DOM doesn't exist")
    with open(csv_dir + "/nodes.csv") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # In this order its always lower, lower, higher, higher numerical value
            tmp_key = ",".join([row["left"], row["top"], row["right"], row["bottom"]])
            dom[tmp_key] = row['xpath']
    return dom

# This method applies labels to segments following a simple labeling scheme taken from
# Efficient Web Browsing on Small Screens by Hamad Ahmadi and Jun Kong
# https://dl.acm.org/doi/pdf/10.1145/1385569.1385576

# Input are the meta information of the image, i.e. image height and width, cutoff scores for segments
# and the polygon


def label_segment_simple(img_height, img_width, bbox, dom_dict):
    poly_string = ",".join([str(i) for i in bbox])
    # order of bbox is left, top, right, bottom

    if bbox[3] < 200 and "/A" in dom_dict[poly_string]:  # top
        return "Top"
    elif bbox[2] < 0.3 * img_width:  # left
        return "Left"
    elif bbox[0] > 0.7 * img_width:  # right
        return "Right"
    elif bbox[1] > img_height - 150:  # bottom
        return "Bottom"
    else:
        return "Unclassified"


def main():
    print("missing")


if __name__ == "__main__":
    main()