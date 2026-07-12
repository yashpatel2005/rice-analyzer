import os
import cv2
import sys
from app import _run_pipeline

input_dir = 'rice_images'

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.avif')):
        continue
        
    path = os.path.join(input_dir, filename)
    image = cv2.imread(path)
    if image is None:
        continue
        
    print(f"\n--- {filename} ---")
    
    # Resize image for faster testing if it's huge
    h, w = image.shape[:2]
    if max(h, w) > 1024:
        scale = 1024 / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
        
    try:
        res_class = _run_pipeline(
            image, image_path=filename, use_watershed=True, ppm=0.0,
            contrast_boost=True, use_clustering=False, broken_threshold=0.75, block_size=121
        )
        print(f"Classical: {res_class['num_grains']} grains")
    except Exception as e:
        print(f"Classical Error: {e}")
        
    try:
        res_clust = _run_pipeline(
            image, image_path=filename, use_watershed=True, ppm=0.0,
            contrast_boost=True, use_clustering=True, broken_threshold=0.75, block_size=121
        )
        print(f"Clustering: {res_clust['num_grains']} grains")
    except Exception as e:
        print(f"Clustering Error: {e}")
