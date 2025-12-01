from __future__ import annotations
import numpy as np
from PIL import Image
import os

def load_image_rgb(path: str, size: int) -> np.ndarray:
    """
    讀入影像 → RGB → 0~1 float32 → resize 為 (size, size)
    回傳 shape: (1, H, W, 3)
    """
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr[None, ...]  # add batch dim

def save_image_rgb(path: str, arr01: np.ndarray) -> None:
    """
    arr01: shape (H, W, 3), 0~1
    """
    arr = np.clip(arr01 * 255.0, 0, 255).astype("uint8")
    Image.fromarray(arr).save(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
