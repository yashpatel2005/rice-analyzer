import os
import cv2
import sys
import shutil
from app import _run_pipeline

input_dir = 'rice_images'
artifact_dir = '/Users/yashpatel/.gemini/antigravity-ide/brain/24a7bdbf-69f1-44fd-996c-fd0eb196dfa7/scratch/test_results'
os.makedirs(artifact_dir, exist_ok=True)

markdown = "# Rice Images Test Results\n\n"

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.avif')):
        continue
        
    path = os.path.join(input_dir, filename)
    image = cv2.imread(path)
    if image is None:
        continue
        
    # Copy original to artifact dir
    orig_path = os.path.join(artifact_dir, f"orig_{filename}")
    # Convert to jpg for easy display
    cv2.imwrite(orig_path + ".jpg", image)
    orig_abs = os.path.abspath(orig_path + ".jpg")
    
    try:
        result = _run_pipeline(
            image,
            image_path=filename,
            use_watershed=True,
            ppm=0.0,
            contrast_boost=True,
            use_clustering=True,
            broken_threshold=0.75,
            block_size=121
        )
        
        annotated_path = result.get('files', {}).get('annotated_image')
        if annotated_path and os.path.exists(os.path.join('outputs/images', annotated_path)):
            ann_abs = os.path.abspath(os.path.join('outputs/images', annotated_path))
            new_ann_abs = os.path.abspath(os.path.join(artifact_dir, f"ann_{filename}.jpg"))
            shutil.copy(ann_abs, new_ann_abs)
            
            markdown += f"## {filename}\n"
            markdown += f"**Grains Detected**: {result['num_grains']}\n\n"
            markdown += f"**Original**\n![Original]({orig_abs})\n\n"
            markdown += f"**Annotated**\n![Annotated]({new_ann_abs})\n\n"
            markdown += "---\n"
        else:
            markdown += f"## {filename}\n"
            markdown += f"**Grains Detected**: 0 (No annotated image generated)\n\n"
            markdown += f"**Original**\n![Original]({orig_abs})\n\n"
            markdown += "---\n"
            
    except Exception as e:
        markdown += f"## {filename}\n"
        markdown += f"**Error**: {e}\n\n"

with open(os.path.join(artifact_dir, "results.md"), "w") as f:
    f.write(markdown)

print("Done. Results in results.md")
