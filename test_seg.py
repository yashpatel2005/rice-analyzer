import cv2
import numpy as np
from core.segmentation import Segmenter

seg = Segmenter()
binary = np.zeros((100, 100), dtype=np.uint8)
binary[10:90, 10:90] = 255
res = seg.segment(binary)
print("Segmentation successful:", res.keys())
