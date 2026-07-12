import requests
import sys
import os

url = "https://rice-api.yash-patel.in/api/analyze"
image_path = "test_mask.jpg"

if not os.path.exists(image_path):
    print("Image not found")
    exit(1)

with open(image_path, "rb") as f:
    files = {"image": f}
    data = {
        "pixels_per_mm": "0",
        "use_watershed": "true",
        "use_clustering": "true",
    }
    response = requests.post(url, files=files, data=data)

print(response.status_code)
try:
    json_resp = response.json()
    print(f"Success: {json_resp.get('success')}")
    if not json_resp.get("success"):
        print(f"Error: {json_resp.get('error')}")
except Exception as e:
    print("Response wasn't JSON:")
    print(response.text)
