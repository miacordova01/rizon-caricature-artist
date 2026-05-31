#!/usr/bin/env python3
"""Download MediaPipe model files required by the vision pipeline."""

import os
import urllib.request

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "selfie_segmenter.tflite": (
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    ),
}

def download():
    dest = os.path.dirname(__file__)
    for name, url in MODELS.items():
        path = os.path.join(dest, name)
        if os.path.exists(path):
            print(f"  already exists: {name}")
            continue
        print(f"  downloading {name} …", end=" ", flush=True)
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path) / 1024
        print(f"{size:.0f} KB")

if __name__ == "__main__":
    print("Downloading MediaPipe models to models/")
    download()
    print("Done.")
