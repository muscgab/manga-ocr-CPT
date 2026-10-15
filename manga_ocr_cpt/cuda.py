"""NVIDIA GPU runner: preprocessing on the GPU (bit-exact to preprocess.py) and greedy decoding replayed from CUDA graphs.
Per batch size in BUCKETS two graphs are captured at start-up: E = encoder + cross-attention K/V of both decoder layers, D = CHUNK
decoder steps on a static self-attention cache (the step index is a device tensor, so D is replayed chunk after chunk). After
each chunk the host checks whether every crop has produced the end token. A call with b crops runs in the smallest bucket >= b
(extra rows repeat the last crop and are dropped); larger calls are split."""
from __future__ import annotations

import math

import numpy as np
import torch

from .gpu_preprocess import GpuPrep
from .model import MAX_NEW, START, MangaOcrNet

BUCKETS = tuple(range(1, 17)) + (24, 32, 48)
CHUNK = 4
SLOTS = math.ceil(MAX_NEW / CHUNK) * CHUNK + 1


class CudaRunner:
    def __init__(self, net: MangaOcrNet, fit: str = "aspect", buckets=BUCKETS):
        self.net, self.dev = net, net.dev
        self.pre = GpuPrep(self.dev, fit)
        self.stage = torch.empty(0, dtype=torch.uint8).pin_memory()
        self.pool = torch.cuda.graph_pool_handle()
        self.buckets = tuple(sorted(buckets))
        self.g = {b: self._capture(b) for b in self.buckets}

    def _graph(self, fn):
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                fn()
        torch.cuda.current_stream(self.dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            fn()
        torch.cuda.synchronize(self.dev)
        return g

    def _capture(self, B):
        net, dev = self.net, self.dev
        sk, sv = net.new_cache(B, SLOTS)
        buf = {"P": torch.zeros(B, 196, 768, device=dev), "V": torch.ones(B, 196, dtype=torch.bool, device=dev),
               "pe": torch.zeros(B, 196, 768, device=dev, dtype=net.edt),
               "ckv": [(torch.zeros(B, 12, 197, 64, device=dev, dtype=net.ddt), torch.zeros(B, 12, 197, 64, device=dev, dtype=net.ddt))
                       for _ in net.dec],
               "xm": torch.ones(B, 1, 1, 197, dtype=torch.bool, device=dev), "sk": sk, "sv": sv,
               "tok": torch.full((B,), START, dtype=torch.long, device=dev), "pos": torch.zeros(1, dtype=torch.long, device=dev),
               "slot": torch.arange(SLOTS, device=dev), "ids": torch.zeros(B, CHUNK, dtype=torch.long, device=dev)}

        def enc():
            mem, mask = net.encode(buf["P"], buf["V"], buf["pe"])
            for (k, v), (k2, v2) in zip(buf["ckv"], net.cross_kv(mem)):
                k.copy_(k2); v.copy_(v2)
            buf["xm"].copy_(mask[:, None, None, :])
            buf["tok"].fill_(START); buf["pos"].zero_()

        def dec():
            for j in range(CHUNK):
                live = (buf["slot"] <= buf["pos"])[None, None, None, :]
                nxt = net.step(buf["tok"], buf["pos"], buf["sk"], buf["sv"], buf["ckv"], buf["xm"], live)
                buf["ids"][:, j].copy_(nxt); buf["tok"].copy_(nxt); buf["pos"].add_(1)

        with torch.inference_mode():
            return self._graph(enc), self._graph(dec), buf

    def _upload(self, arrs):
        need = sum(a.size for a in arrs)
        if self.stage.numel() < need:
            self.stage = torch.empty(int(need * 1.5), dtype=torch.uint8).pin_memory()
        out, off = [], 0
        for a in arrs:
            n = a.size
            self.stage[off:off + n].numpy()[:] = a.reshape(-1)
            out.append(self.stage[off:off + n].to(self.dev, non_blocking=True).view(a.shape))
            off += n
        return out

    @torch.inference_mode()
    def ids(self, arrs: list[np.ndarray]) -> list[list[int]]:
        """uint8 [H, W, 3] RGB crops -> generated ids per crop"""
        top, out = self.buckets[-1], []
        for i in range(0, len(arrs), top):
            out += self._chunk(arrs[i:i + top])
        return out

    def _chunk(self, arrs):
        b = len(arrs)
        B = next(x for x in self.buckets if x >= b)
        ge, gd, buf = self.g[B]
        for j, x in enumerate(self._upload(arrs)):
            gh, gw = self.pre(x, buf["P"][j], buf["V"][j])
            buf["pe"][j].copy_(self.net.pe(gh, gw))
        if B > b:
            for k in ("P", "V", "pe"):
                buf[k][b:B].copy_(buf[k][b - 1:b].expand(B - b, *buf[k].shape[1:]))
        ge.replay()
        chunks = []
        for _ in range(SLOTS // CHUNK):
            gd.replay()
            chunks.append(buf["ids"][:b].clone())
            if bool((torch.cat(chunks, 1) == 3).any(1).all()):
                break
        return torch.cat(chunks, 1)[:, :MAX_NEW].tolist()
