import os

import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from PIL import Image
from PIL import ImageFilter
import imagehash
import random
import itertools
import numpy as np
import pandas as pd

import seaborn as sns


def imagehash_approach(filepath):
    file_amt = len(os.listdir('../Dataset/webis-webseg-20-000000/000000/segmentation_screenshots'))
    li = list(itertools.combinations(range(file_amt), 2))
    dist_matrix = np.zeros((71, 71))
    for i in li:
        img1 = Image.open(filepath + "/segmentation_" + str(i[0]) + ".png")
        img2 = Image.open(filepath + "/segmentation_" + str(i[1]) + ".png")
        if img1.width < img2.width:
            img2 = img2.resize((img1.width, img1.height))
        else:
            img1 = img1.resize((img2.width, img2.height))
        img1 = img1.filter(ImageFilter.BoxBlur(radius=3))
        img2 = img2.filter(ImageFilter.BoxBlur(radius=3))
        phashvalue = imagehash.phash(img1) - imagehash.phash(img2)
        ahashvalue = imagehash.average_hash(img1) - imagehash.average_hash(img2)
        totalaccuracy = phashvalue + ahashvalue
        dist_matrix[i[0]][i[1]] = totalaccuracy
        dist_matrix[i[1]][i[0]] = totalaccuracy
    print(dist_matrix)
    df = pd.DataFrame(dist_matrix)
    df.to_csv("distanceMatrix.csv")
    print("Succesfully exported distance matrix")


def skimage_approach(filepath, l):
    for i in range(len(l) - 1):
        img1 = cv2.imread(filepath + "/segmentation_" + str(l[i]) + ".png")
        img2 = cv2.imread(filepath + "/segmentation_" + str(l[i + 1]) + ".png")
        print(structural_similarity(img1, img2, multichannel=True))


def main():
    filepath = "../Dataset/webis-webseg-20-000000/000000/segmentation_screenshots"
    #imagehash_approach(filepath)
    df = pd.read_csv("distanceMatrix.csv", index_col=0)
    plt.figure()
    sns.heatmap(df, cmap='OrRd')
    plt.savefig("heatmap.png")
    plt.show()


if __name__ == "__main__":
    main()