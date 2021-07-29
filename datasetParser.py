import json
import cv2
import numpy as np


def load_segmentations(filepath):
	with open(filepath + "/ground-truth.json", 'r') as f:
		data = f.read()

	obj = json.loads(data)
	return obj['segmentations']['majority-vote']


def load_image(filepath):
	image = cv2.imread(filepath + "/screenshot.png")
	return image


def draw_segmentations(image, segmentations, filepath):
	segmentation_counter = 0
	for large_areas in segmentations:
		color = [int(i) for i in list(np.random.choice(256, size=3))]
		for medium_inner_list in large_areas:
			for polygon in medium_inner_list:
				bbox = np.concatenate((np.min(polygon, axis=0), np.max(polygon, axis=0)))
				roi = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
				cv2.imwrite(filepath + "/segmentation_screenshots/segmentation_" + str(segmentation_counter), roi)
				segmentation_counter += 1
				image = cv2.polylines(image, [np.array(polygon)], isClosed=False, color=color, thickness=2)
	cv2.imshow('image', image)
	cv2.imwrite('test.png', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def main():
	filepath = "../Dataset/webis-webseg-20-000000/000000"
	seg = load_segmentations(filepath)
	im = load_image(filepath)
	draw_segmentations(im, seg, filepath)


if __name__ == "__main__":
	main()

