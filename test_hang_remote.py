import cv2
import time
import numpy as np

img_path = "uploads/upload_20260712_180416_777083_WhatsApp Image 2026-07-12 at 00.45.42.jpeg"
img = cv2.imread(img_path)
if img is None:
    print("Cannot read image")
    exit(1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Running kmeans...")
t0 = time.time()
pixels = gray.reshape(-1, 1).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
binary = labels.reshape(gray.shape).astype(np.uint8)
if centers[0][0] > centers[1][0]:
    binary = 1 - binary
binary = binary * 255

print("Running connected components...")
t0 = time.time()
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
print(f"Connected components found {num_labels}")

min_area = 50
max_area = int(gray.shape[0]*gray.shape[1] * 0.1)

valid_ids = []
for i in range(1, num_labels):
    if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
        valid_ids.append(i)
print(f"Valid ids: {len(valid_ids)}")
filtered = np.zeros_like(labels)
for new_id, old_id in enumerate(valid_ids, start=1):
    filtered[labels == old_id] = new_id

print("Running watershed...")
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed
filtered_binary = (filtered > 0).astype(np.uint8) * 255
dist = cv2.distanceTransform(filtered_binary, cv2.DIST_L2, 5)
total_px = gray.shape[0] * gray.shape[1]
scale = max(1.0, (total_px / (1920 * 1080)) ** 0.5)
min_dist = max(10, int(10 * scale))
coords = peak_local_max(dist, min_distance=min_dist, threshold_abs=0.5 * dist.max())
print(f"Found {len(coords)} peaks")
markers = np.zeros(filtered_binary.shape, dtype=np.int32)
for i, (y, x) in enumerate(coords, start=1):
    markers[y, x] = i
labels_ws = sk_watershed(-dist, markers, mask=filtered_binary)

print("Extracting contours...")
num_labels_final = labels_ws.max()
grains = []
for i in range(1, num_labels_final + 1):
    mask = (labels_ws == i).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        grains.append({"label": i, "contour": c, "mask": mask})
print(f"Extracted {len(grains)} grains")

print("Running measurements...")
from core.measurement import GrainMeasurer
analyzer = GrainMeasurer(pixels_per_mm=0.0)
measurements = analyzer.measure_all(grains)
print(f"Measurements done")

print("Running stats...")
from core.statistics import StatisticalAnalyzer
s_analyzer = StatisticalAnalyzer(pixels_per_mm=0.0)
stats = s_analyzer.analyze(measurements)
print("Stats done")
