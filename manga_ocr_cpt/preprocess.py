"""CPU preprocessing (numpy + Pillow), identical to the training input.
The crop is converted to grey, then
  fit="square" (manga-ocr-base-CPT): stretched to 224 x 224 (Pillow bilinear) = 14 x 14 patches, as the original manga-ocr;
  fit="aspect" (manga-ocr-base-CPT-flex): resized with its aspect ratio kept (Pillow bilinear) to fit a grid of at most 196
  patches of 16 x 16 px, placed at the top-left of the grid canvas, the rest filled with the median value of the resized image's
  border.
Values are normalised to [-1, 1] and cut into patches (channel, row, column order; the three channels are equal)."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

BUDGET, PATCH, FEAT = 196, 16, 768
FITS = ("square", "aspect")


def grid(height: int, width: int, budget: int = BUDGET):
    """Largest uniform scale s with ceil(H*s/16) * ceil(W*s/16) <= budget -> (s, grid h, grid w)."""
    lo, hi = 1e-4, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.ceil(height * mid / PATCH) * math.ceil(width * mid / PATCH) <= budget:
            lo = mid
        else:
            hi = mid
    return lo, math.ceil(height * lo / PATCH), math.ceil(width * lo / PATCH)


def _patches(arr: np.ndarray, gh: int, gw: int) -> np.ndarray:
    return arr.reshape(gh, PATCH, gw, PATCH, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, -1)


def preprocess(img: Image.Image, fit: str = "aspect"):
    """-> P [196, 768] float32, V [196] bool (real patches), grid h, grid w"""
    img = img.convert("L").convert("RGB")
    if fit == "square":
        arr = (np.asarray(img.resize((224, 224), Image.Resampling.BILINEAR), np.uint8).astype(np.float32) / 255.0 - 0.5) / 0.5
        return _patches(arr, 14, 14).copy(), np.ones(BUDGET, bool), 14, 14
    if fit != "aspect":
        raise ValueError(f"fit must be one of {FITS}, got {fit!r}")
    s, gh, gw = grid(img.height, img.width)
    rw, rh = max(1, min(gw * PATCH, round(img.width * s))), max(1, min(gh * PATCH, round(img.height * s)))
    small = np.asarray(img.resize((rw, rh), Image.Resampling.BILINEAR), np.uint8)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    canvas = np.empty((gh * PATCH, gw * PATCH, 3), np.uint8)
    canvas[:] = np.median(border, axis=0).astype(np.uint8)
    canvas[:rh, :rw] = small
    arr = (canvas.astype(np.float32) / 255.0 - 0.5) / 0.5
    n = gh * gw
    P = np.zeros((BUDGET, FEAT), np.float32)
    P[:n] = _patches(arr, gh, gw)
    V = np.zeros(BUDGET, bool)
    V[:n] = True
    return P, V, gh, gw
