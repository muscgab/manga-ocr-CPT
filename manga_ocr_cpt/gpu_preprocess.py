"""GPU version of preprocess.py (= the training input), bit for bit.
img.convert("L").convert("RGB") -> Pillow BILINEAR resize, either to 224 x 224 (fit="square") or into the largest gh x gw grid
<= 196 patches of 16 px with the rest of the canvas filled with the median border colour (fit="aspect") -> (x / 255 - 0.5) / 0.5
-> patches in (c, ky, kx) order. The three channels are equal after
the grey conversion, so the GPU works on one channel and repeats it. Pillow arithmetic re-implemented: RGB -> L with the L24
fixed-point weights, 8-bit resampling with 22-bit fixed-point coefficients (same scheme as release/mangaocr-nar-small
gpu_preprocess.py, here with the bilinear filter). Only the grid and the small coefficient tables are computed on the CPU."""
from __future__ import annotations

import math

import numpy as np
import torch

PRECISION_BITS = 22
BUDGET, PATCH = 196, 16


def grid(h, w):
    """= train_mocr.grid"""
    lo, hi = 1e-4, 1000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if math.ceil(h * mid / PATCH) * math.ceil(w * mid / PATCH) <= BUDGET: lo = mid
        else: hi = mid
    return lo, math.ceil(h * lo / PATCH), math.ceil(w * lo / PATCH)


def _bilinear(x):
    x = np.abs(x)
    return np.where(x < 1.0, 1.0 - x, 0.0)


def pil_coeffs(in_size: int, out_size: int):
    """-> xmin [out], fixed-point kk [out, ksize], ksize; Pillow precompute_coeffs + normalize_coeffs_8bpc, bilinear (support 1)"""
    scale = float(in_size) / out_size
    filterscale = max(scale, 1.0)
    support = 1.0 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    xx = np.arange(out_size, dtype=np.float64)
    center = (xx + 0.5) * scale
    ss = 1.0 / filterscale
    xmin = np.maximum(np.trunc(center - support + 0.5).astype(np.int64), 0)
    xmax = np.minimum(np.trunc(center + support + 0.5).astype(np.int64), in_size) - xmin
    x = np.arange(ksize, dtype=np.float64)[None, :]
    w = _bilinear((x + xmin[:, None] - center[:, None] + 0.5) * ss)
    w = np.where(x < xmax[:, None], w, 0.0)
    ww = np.zeros(out_size)
    for j in range(ksize):                                   # Pillow sums left to right
        ww = ww + w[:, j]
    w = np.where(ww[:, None] != 0.0, w / np.where(ww == 0.0, 1.0, ww)[:, None], w)
    k = np.where(w < 0, np.trunc(-0.5 + w * (1 << PRECISION_BITS)), np.trunc(0.5 + w * (1 << PRECISION_BITS))).astype(np.int64)
    return xmin, k, ksize


class GpuPrep:
    def __init__(self, device="cuda", fit="aspect"):
        if fit not in ("square", "aspect"):
            raise ValueError(f"fit must be 'square' or 'aspect', got {fit!r}")
        self.dev, self.fit = torch.device(device), fit
        self.c255 = torch.tensor([255.0], device=self.dev)   # true division like numpy (not x * (1/255))
        self.idx = torch.arange(BUDGET, device=self.dev)
        self._coef = {}

    def _pass(self, x, in_size, out_size, dim):
        """one Pillow 8-bit resample pass of a uint8 [H, W] tensor along dim (1 = horizontal, 0 = vertical)"""
        key = (in_size, out_size)
        if key not in self._coef:
            xmin, k, ksize = pil_coeffs(in_size, out_size)
            src = np.minimum(xmin[:, None] + np.arange(ksize)[None, :], in_size - 1)          # dead taps have k == 0
            self._coef[key] = (torch.from_numpy(src).to(self.dev), torch.from_numpy(k).to(self.dev))
            if len(self._coef) > 4096:
                self._coef.pop(next(iter(self._coef)))
        src, k = self._coef[key]
        xi = x.to(torch.int64)
        if dim == 1:
            acc = (xi[:, src] * k[None]).sum(2)                # [H, out]
        else:
            acc = (xi[src] * k[:, :, None]).sum(1)             # [out, W]
        acc = acc + (1 << (PRECISION_BITS - 1))
        return torch.clamp(acc >> PRECISION_BITS, 0, 255).to(torch.uint8)

    @torch.no_grad()
    def __call__(self, x, P_out, V_out):
        """x: uint8 [H, W, 3] RGB tensor (any device). Writes P_out [196, 768] fp32 and V_out [196] bool; returns (gh, gw)."""
        x = x.to(self.dev, non_blocking=True)
        H, W = int(x.shape[0]), int(x.shape[1])
        xi = x.to(torch.int64)
        g = ((xi[..., 0] * 19595 + xi[..., 1] * 38470 + xi[..., 2] * 7471 + 0x8000) >> 16).to(torch.uint8)   # Pillow RGB -> L
        if self.fit == "square":
            gh = gw = 14
            rw = rh = 224
        else:
            s, gh, gw = grid(H, W)
            rw, rh = max(1, min(gw * PATCH, round(W * s))), max(1, min(gh * PATCH, round(H * s)))
        if rw != W:
            g = self._pass(g, W, rw, 1)
        if rh != H:
            g = self._pass(g, H, rh, 0)
        if self.fit == "square":
            canvas = g
        else:
            border = torch.cat([g[0], g[-1], g[:, 0], g[:, -1]])
            srt = border.sort().values; n = srt.shape[0]
            med = srt[n // 2] if n % 2 else ((srt[n // 2 - 1].double() + srt[n // 2].double()) / 2).to(torch.uint8)
            canvas = med.view(1, 1).expand(gh * PATCH, gw * PATCH).clone()
            canvas[:rh, :rw] = g
        arr = (canvas.to(torch.float32) / self.c255 - 0.5) / 0.5
        n = gh * gw
        P_out.zero_()
        P_out[:n] = arr.view(gh, PATCH, gw, PATCH).permute(0, 2, 1, 3).reshape(n, PATCH * PATCH).repeat(1, 3)
        V_out.copy_(self.idx < n)
        return gh, gw
