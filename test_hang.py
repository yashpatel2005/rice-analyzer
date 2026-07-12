import cv2
import time
import numpy as np

img = cv2.imread("test_whatsapp.jpeg")
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

print(f"KMeans done in {time.time()-t0:.2f}s")

print("Running connected components...")
t0 = time.time()
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
print(f"Connected components done in {time.time()-t0:.2f}s, found {num_labels} components")

min_area = 50
max_area = int(gray.shape[0]*gray.shape[1] * 0.1)

print("Filtering components...")
t0 = time.time()
valid_ids = []
for i in range(1, num_labels):
    if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area:
        valid_ids.append(i)

print(f"Valid ids: {len(valid_ids)}")
filtered = np.zeros_like(labels)
for new_id, old_id in enumerate(valid_ids, start=1):
    filtered[labels == old_id] = new_id

print("Filtering done in {time.time()-t0:.2f}s")

print("Running watershed...")
t0 = time.time()
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed

filtered_binary = (filtered > 0).astype(np.uint8) * 255
dist = cv2.distanceTransform(filtered_binary, cv2.DIST_L2, 5)

total_px = gray.shape[0] * gray.shape[1]
scale = max(1.0, (total_px / (1920 * 1080)) ** 0.5)
min_dist = max(10, int(10 * scale))

coords = peak_local_max(
    dist, min_distance=min_dist,
    threshold_abs=0.5 * dist.max()
)
print(f"Found {len(coords)} peaks")
markers = np.zeros(filtered_binary.shape, dtype=np.int32)
for i, (y, x) in enumerate(coords, start=1):
    markers[y, x] = i

labels_ws = sk_watershed(-dist, markers, mask=filtered_binary)
print("Extracting contours...")
t0 = time.time()
num_labels_final = labels_ws.max()
grains = []
for i in range(1, num_labels_final + 1):
    mask = (labels_ws == i).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        grains.append({"label": i, "contour": c, "mask": mask})
print(f"Extracted {len(grains)} grains in {time.time()-t0:.2f}s")

print("Running measurements...")
t0 = time.time()
from core.measurement import GrainMeasurer
analyzer = GrainMeasurer(pixels_per_mm=0.0)
measurements = analyzer.measure_all(grains)
print(f"Measurements done in {time.time()-t0:.2f}s")
