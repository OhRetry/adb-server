from typing import List

import numpy as np
import cv2

def _png_bytes_to_cv2_image(png_bytes: bytes):
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode PNG bytes to image")
    return img  # BGR numpy array

def _crop_image(img: np.ndarray, roi: List[int]) -> np.ndarray:
    h, w = img.shape[:2]
    if len(roi) != 4:
        raise ValueError("ROI must be [x1,y1,x2,y2]")
    x1, y1, x2, y2 = map(int, roi)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI coordinates invalid or zero-area")
    return img[y1:y2, x1:x2]