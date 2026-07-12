import cv2
import numpy as np
from core.clustering_segmentation import ClusteringSegmenter

image = cv2.imread('rice_images/images.jpeg')
segmenter = ClusteringSegmenter()
res = segmenter.segment(image)
print(f"Num grains: {res['num_grains']}")
cv2.imwrite('test_mask.jpg', res['steps']['kmeans_mask'])
cv2.imwrite('test_dt.jpg', res['steps']['distance_transform'])
