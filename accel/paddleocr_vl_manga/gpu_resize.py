"""Bit-exact re-implementation of Pillow's 8-bit BICUBIC resize (libImaging/Resample.c, Pillow 12.x: precompute_coeffs +
normalize_coeffs_8bpc + ImagingResampleHorizontal/Vertical_8bpc, PRECISION_BITS 22, horizontal pass then vertical pass, each pass
rounding/clipping to uint8) in torch integer arithmetic, so the resize can run on the GPU. Coefficients are computed in float64 with the
same operation order as the C code (sequential sums via cumsum). Exactness is verified against PIL, not assumed."""
import math
import numpy as np
import torch

PREC = 22


def _bicubic(x):
    a = -0.5
    x = np.abs(x)
    out = np.zeros_like(x)
    m1 = x < 1.0
    m2 = (x >= 1.0) & (x < 2.0)
    x1 = x[m1]; out[m1] = ((a + 2.0) * x1 - (a + 3.0)) * x1 * x1 + 1
    x2 = x[m2]; out[m2] = (((x2 - 5) * x2 + 8) * x2 - 4) * a
    return out


def coeffs(in_size, out_size, support_base=2.0):
    """-> (xmin (out,), ksize, kk int64 (out, ksize)) exactly as precompute_coeffs + normalize_coeffs_8bpc (box = (0, in_size))."""
    scale = float(np.float32(in_size) - np.float32(0.0)) / out_size  # (double)(in1 - in0) / outSize, in0/in1 are C floats
    filterscale = max(scale, 1.0)
    support = support_base * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    xx = np.arange(out_size, dtype=np.float64)
    center = 0.0 + (xx + 0.5) * scale
    ss = 1.0 / filterscale
    xmin = np.trunc(center - support + 0.5).astype(np.int64); xmin = np.maximum(xmin, 0)
    xmax = np.trunc(center + support + 0.5).astype(np.int64); xmax = np.minimum(xmax, in_size); xmax = xmax - xmin
    x = np.arange(ksize, dtype=np.int64)[None, :]
    valid = x < xmax[:, None]
    arg = ((x + xmin[:, None]).astype(np.float64) - center[:, None] + 0.5) * ss
    w = np.where(valid, _bicubic(arg), 0.0)
    ww = np.cumsum(w, axis=1)[:, -1]                     # sequential sum (zeros past xmax add nothing)
    k = np.where(ww[:, None] != 0.0, w / np.where(ww == 0.0, 1.0, ww)[:, None], w)
    k = np.where(valid, k, 0.0)
    f = k * float(1 << PREC)
    kk = np.where(k < 0, np.trunc(-0.5 + f), np.trunc(0.5 + f)).astype(np.int64)
    return xmin, ksize, kk


class GpuBicubic:
    def __init__(self, device="cuda", cache=256):
        self.dev = device; self.cache = {}; self.cache_max = cache

    def _c(self, n_in, n_out):
        key = (n_in, n_out)
        c = self.cache.get(key)
        if c is None:
            xmin, ksize, kk = coeffs(n_in, n_out)
            idx = np.minimum(xmin[:, None] + np.arange(ksize)[None, :], n_in - 1)  # taps past xmax have weight 0
            c = (torch.from_numpy(idx).to(self.dev), torch.from_numpy(kk).to(self.dev), ksize)
            if len(self.cache) >= self.cache_max: self.cache.pop(next(iter(self.cache)))
            self.cache[key] = c
        return c

    def _pass(self, x, n_out, dim):
        """x int64 (H, W, C); resample along dim (1 = horizontal, 0 = vertical) -> uint8-valued int64."""
        idx, kk, ksize = self._c(x.shape[dim], n_out)
        acc = torch.full((), 1 << (PREC - 1), dtype=torch.int64, device=x.device)
        for j in range(ksize):
            if dim == 1:
                acc = acc + x[:, idx[:, j], :] * kk[None, :, j, None]
            else:
                acc = acc + x[idx[:, j], :, :] * kk[:, j, None, None]
        return torch.clamp(torch.div(acc, 1 << PREC, rounding_mode="floor"), 0, 255)

    def __call__(self, img_u8, out_h, out_w):
        """img_u8: uint8 tensor (H, W, 3) on device -> uint8 (out_h, out_w, 3), equal to PIL Image.resize((out_w, out_h), BICUBIC)."""
        H, W, _ = img_u8.shape
        if (H, W) == (out_h, out_w):
            return img_u8.clone()
        x = img_u8.to(torch.int64)
        if W != out_w:
            x = self._pass(x, out_w, 1)
        if H != out_h:
            x = self._pass(x, out_h, 0)
        return x.to(torch.uint8)
