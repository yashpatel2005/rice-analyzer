import requests
import sys
import os

url = "http://127.0.0.1:5050/api/analyze"
image_path = "test_whatsapp.jpeg"

with open(image_path, "rb") as f:
    files = {"image": f}
    data = {
        "pixels_per_mm": "0",
        "use_watershed": "true",
        "use_clustering": "true",
    }
    print(f"Testing {image_path}...")
    response = requests.post(url, files=files, data=data)

print(f"Status Code: {response.status_code}")
try:
    json_resp = response.json()
    print(f"Success: {json_resp.get('success')}")
    if not json_resp.get("success"):
        print(f"Error: {json_resp.get('error')}")
except Exception as e:
    print("Response wasn't JSON:")
    print(response.text)
