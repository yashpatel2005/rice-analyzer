import cv2
from core.preprocessing import Preprocessor
from core.segmentation import Segmenter

img_path = 'uploads/upload_20260712_021042_475282_upload_20260711_220316_428036_test_rice.jpg'
img = cv2.imread(img_path)
pre = Preprocessor()
seg = Segmenter()

pre_res = pre.process(img)
seg_res = seg.segment(pre_res["binary"])
print(f"Num grains detected: {seg_res['num_grains']}")
