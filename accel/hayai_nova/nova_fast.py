"""Faster inference for Hayai v2.5 Nova (JustANormalTinkerer/hayai-ocr-v2.5-nova, Apache-2.0; tested at revision 2803d0dc).
Same weights, same greedy decoding (max_new 128),
same Siglip2 NaFlex preprocessing (max_num_patches), same fp16 fused engine kernels as the shipped modeling_hayai._FastEngine --
only the orchestration changes:
  * preprocessing: CPU does only PIL->uint8 + the uint8 antialiased bilinear resize (the exact torchvision call the HF fast
    processor makes); normalise (x-127.5)/127.5 in fp32, patchify, fp16 cast happen on the GPU (bitwise equal to the CPU path).
  * batch 1 "front" (vision encoder + projector gather + projector + decoder prefill + first argmax) is one CUDA graph per
    (n_patches, m, need_mask) key, captured lazily (cached, no LRU eviction).
  * decode: C decode steps per CUDA graph replay (the engine's own static decode_step), EOS checked after every chunk instead
    of every 16 steps (C = 1 by default, the fastest setting measured); decode states cached per (batch, m) without the engine's
    4-entry LRU (which forced re-captures).
  * batch > 1: eager vision/prefill (the engine's packed path) + the chunked decode graphs; optional batch padding buckets.
Graphs are captured lazily, the first time an input shape key is seen (about 0.1 s each); they are never evicted, so GPU memory
grows with the number of distinct shapes (41 keys on the 7,789 crops of JMangaBench_Read v1.3).
The front graph body below re-implements the non-packed path of modeling_hayai._FastVision / _FastEngine with the same operations.
Usage:
    from nova_fast import NovaFast
    ocr = NovaFast("path/to/hayai-ocr-v2.5-nova", max_num_patches=384)
    ocr([pil_image])          # -> [text]
"""
import math, sys, time
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TVF
from transformers.models.siglip2.image_processing_siglip2 import get_image_size_for_max_num_patches

PATCH = 16


def resize_u8(img, P):
    """CPU: identical to Siglip2ImageProcessorFast up to (and including) the resize (uint8, antialiased bilinear)."""
    t = TVF.pil_to_tensor(img.convert("RGB"))
    h, w = get_image_size_for_max_num_patches(t.shape[1], t.shape[2], PATCH, P)
    return TVF.resize(t, [h, w], interpolation=TVF.InterpolationMode.BILINEAR, antialias=True)


class NovaFast:
    def __init__(self, model_dir, max_num_patches=384, chunk=1, max_new=128, front_graph=True, gpu_prep=True,
                 batch_buckets=None, device="cuda"):
        from transformers import AutoModel, PreTrainedTokenizerFast
        t0 = time.perf_counter()
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).to(device).eval().half()
        self.tok = PreTrainedTokenizerFast.from_pretrained(model_dir)
        self.mod = sys.modules[type(model).__module__]
        self.dev, self.P, self.C, self.max_new = torch.device(device), max_num_patches, chunk, max_new
        self.front_graph, self.gpu_prep, self.batch_buckets = front_graph, gpu_prep, batch_buckets
        tk = self.tok
        self.bos = tk.bos_token_id or 1; self.eos = tk.eos_token_id or 2; self.pad = tk.pad_token_id or self.eos
        with torch.inference_mode():
            self.eng = self.mod._FastEngine(model, torch.float16)
        self.model = model
        self.vis = self.eng.vision
        self.stream = torch.cuda.Stream(self.dev)          # warm-up, capture AND replay all on this stream
        self.pool = torch.cuda.graph_pool_handle()          # one shared private pool (replays are serialised)
        self.states, self.fronts = {}, {}
        self.mean = torch.tensor([0.5] * 3) * (1.0 / 0.00392156862745098)   # == HF fused rescale+normalize constants
        self.std = torch.tensor([0.5] * 3) * (1.0 / 0.00392156862745098)
        self.mean_d = self.mean.to(self.dev).view(-1, 1, 1); self.std_d = self.std.to(self.dev).view(-1, 1, 1)
        self.capture_log = []                                # (kind, key, seconds)
        self.load_sec = time.perf_counter() - t0

    # ------------------------------------------------------------------ preprocessing
    def resize_u8(self, img):
        return resize_u8(img, self.P)

    def patches_gpu(self, u8, out=None):
        """uint8 (3,h,w) CPU -> fp16 (h/16*w/16, 768) patches on the GPU (normalise in fp32 exactly as torchvision)."""
        x = u8.pin_memory().to(self.dev, non_blocking=True).to(torch.float32)
        x = x.sub_(self.mean_d).div_(self.std_d)
        c, h, w = x.shape; hp, wp = h // PATCH, w // PATCH
        x = x.reshape(c, hp, PATCH, wp, PATCH).permute(1, 3, 2, 4, 0).reshape(hp * wp, -1)
        if out is None:
            return x.half()
        out.copy_(x)
        return out

    def patches_cpu(self, u8):
        x = TVF.normalize(u8.to(torch.float32), self.mean.tolist(), self.std.tolist())
        c, h, w = x.shape; hp, wp = h // PATCH, w // PATCH
        return x.reshape(c, hp, PATCH, wp, PATCH).permute(1, 3, 2, 4, 0).reshape(hp * wp, -1)

    # ------------------------------------------------------------------ decode states / chunk graphs
    def _capture(self, fn, warm=2):
        s = self.stream
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(warm):
                fn()
        s.synchronize()
        # torch.cuda.graph's __enter__ runs gc.collect() + empty_cache() (~0.3 s with 7.8k PIL images alive) -> capture directly
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            g.capture_begin(self.pool, capture_error_mode="thread_local")
            try:
                fn()
            finally:
                g.capture_end()
        s.synchronize()
        return g

    def state(self, b, m):
        key = (b, m)
        st = self.states.get(key)
        if st is not None:
            return st
        t0 = time.perf_counter()
        E = self.eng
        L_total = m + self.max_new + 1
        with torch.inference_mode():
            st = E._get_state(b, L_total, self.max_new, self.eos, self.pad, False)     # buffers only, no engine graph
            E._states.clear()                                                          # we own it now
            fn = E.decode_step
            def steps(n):
                def f():
                    for _ in range(n):
                        fn(st)
                return f
            E._reset_dummy(st)
            st.g_one = self._capture(steps(1)); E._reset_dummy(st)
            st.g_chunk = st.g_one if self.C == 1 else self._capture(steps(self.C)); E._reset_dummy(st)
        st.nbytes = sum(k.numel() * k.element_size() * 2 for k, _ in st.kv)
        # batch > 1 states (KV ~1 GB at b=512) are evicted oldest-first beyond a 3 GB KV budget; batch-1 states are tiny and kept
        while b > 1 and sum(s.nbytes for kk, s in self.states.items() if kk[0] > 1) + st.nbytes > 3 * 2**30:
            old = next((kk for kk in self.states if kk[0] > 1), None)
            if old is None:
                break
            del self.states[old]
        self.states[key] = st
        self.capture_log.append(("decode", key, time.perf_counter() - t0))
        return st

    # ------------------------------------------------------------------ prefill bookkeeping shared by both paths
    def _start(self, st, x, vis_valid, need_mask, cos, sin, m):
        E = self.eng
        st.base_valid.fill_(True)
        st.base_valid[:, :m] = vis_valid
        st.out.fill_(self.pad)
        st.out[:, 0] = self.bos
        st.step.fill_(1)
        st.cidx.fill_(m + 1)
        st.need_mask = need_mask
        logits = E.prefill(x, vis_valid, need_mask, cos, sin, st, m)
        nxt = logits.argmax(-1)
        st.out[:, 1] = nxt
        st.unfinished.copy_((nxt != self.eos) & (nxt != self.pad))
        st.tok.copy_(nxt)

    def _gather_idx(self, shapes, base, r, bucket):
        """CPU replica of DSCProjector.gather_patches' index math (int64, exact). shapes: list[(h,w)], base: list[int]."""
        ss = torch.tensor(shapes, dtype=torch.long)
        h, w = ss[:, 0], ss[:, 1]
        ho, wo = (h + r - 1) // r, (w + r - 1) // r
        lens = ho * wo
        L = int(lens.max()); L = max(1, -(-L // bucket) * bucket)
        b = len(shapes)
        p = torch.arange(L)[None]
        woc = wo.clamp(min=1)[:, None]
        oy, ox = p // woc, p % woc
        off = torch.arange(r)
        sy = torch.minimum(oy[..., None, None] * r + off[:, None], (h - 1)[:, None, None, None])
        sx = torch.minimum(ox[..., None, None] * r + off[None, :], (w - 1)[:, None, None, None])
        idx = (sy.clamp(min=0) * w[:, None, None, None] + sx.clamp(min=0)).reshape(b, L * r * r) + torch.tensor(base)[:, None]
        return idx.reshape(-1), torch.stack([ho, wo], 1), lens, L

    # ------------------------------------------------------------------ batch-1 front graph
    def front(self, n, m, need_mask):
        key = (n, m, need_mask)
        fr = self.fronts.get(key)
        if fr is not None:
            return fr
        t0 = time.perf_counter()
        E, V, dev, D = self.eng, self.vis, self.dev, self.vis.D
        st = self.state(1, m)
        fr = SimpleNamespace(pv=torch.zeros(1, n, 768, dtype=torch.float16, device=dev),
                             pos=torch.zeros(n, D, dtype=torch.float16, device=dev),
                             idx=torch.zeros(m * E.r * E.r, dtype=torch.long, device=dev),
                             shapes=torch.ones(1, 2, dtype=torch.long, device=dev),
                             lens=torch.ones(1, dtype=torch.long, device=dev), st=st)
        fr.idx.zero_()
        bos_emb = F.embedding(torch.full((1, 1), self.bos, dtype=torch.long, device=dev), E.emb_w)
        cmf = self.mod.compute_batch_2d_mrope_freqs
        H, hd, eps = V.H, V.hd, V.eps

        def body():
            x = F.linear(fr.pv, V.patch_w, V.patch_b)                      # (1, n, D) == _FastVision non-packed path
            x[0, :n].add_(fr.pos)
            for L in V.layers:
                h = F.layer_norm(x, (D,), L.ln1_w, L.ln1_b, eps)
                qkv = F.linear(h, L.w_qkv, L.b_qkv)
                q, k, v = qkv.view(1, n, 3, H, hd).permute(2, 0, 3, 1, 4).unbind(0)
                a = F.scaled_dot_product_attention(q, k, v, attn_mask=None).transpose(1, 2).reshape(1, n, D)
                x.add_(F.linear(a, L.w_o, L.b_o))
                h = F.layer_norm(x, (D,), L.ln2_w, L.ln2_b, eps)
                x.add_(F.linear(V.act(F.linear(h, L.fc1_w, L.fc1_b)), L.fc2_w, L.fc2_b))
            x = F.layer_norm(x, (D,), V.post_w, V.post_b, eps)
            flat = x.reshape(-1, D)
            g = flat.index_select(0, fr.idx).view(1, m, E.r * E.r, D)
            vis_emb = E._project(g)
            xx = torch.cat([vis_emb, bos_emb], dim=1)
            cos, sin = cmf(fr.shapes, m, 1, d_head=2 * E.d_axis, device=dev)
            vis_valid = torch.arange(m, device=dev)[None] < fr.lens[:, None]
            self._start(st, xx, vis_valid, need_mask, cos, sin, m)

        with torch.inference_mode():
            fr.graph = self._capture(body) if self.front_graph else None
            fr.body = body
        self.fronts[key] = fr
        self.capture_log.append(("front", key, time.perf_counter() - t0))
        return fr

    # ------------------------------------------------------------------ decode loop
    def _decode(self, st):
        """Replay C-step chunks until all rows hit EOS/pad; never more than max_new-1 decode steps (= engine loop bound)."""
        left = self.max_new - 1
        while left > 0:
            if left >= self.C:
                st.g_chunk.replay(); left -= self.C
            else:
                st.g_one.replay(); left -= 1
                if left > 0:
                    continue
            if not bool(st.unfinished.any()):
                break
        return left

    def _texts(self, out_rows):
        res = []
        for seq in out_rows:
            clean = [t for t in seq[1:] if t not in (self.eos, self.pad)]
            res.append(self.tok.decode(clean, skip_special_tokens=True))
        return res

    @torch.inference_mode()
    def predict1(self, img, u8=None):
        E, dev = self.eng, self.dev
        if u8 is None:
            u8 = self.resize_u8(img)
        hp, wp = u8.shape[1] // PATCH, u8.shape[2] // PATCH
        n = hp * wp
        idx, new_shapes, lens, m = self._gather_idx([(hp, wp)], [0], E.r, 32)
        need_mask = bool((lens < m).any())
        fr = self.front(n, m, need_mask)
        with torch.cuda.stream(self.stream):
            if self.gpu_prep:
                self.patches_gpu(u8, out=fr.pv[0])
            else:
                fr.pv[0].copy_(self.patches_cpu(u8).half(), non_blocking=False)
            fr.pos.copy_(self.vis.pos_embed(hp, wp))
            fr.idx.copy_(idx, non_blocking=True)
            fr.shapes.copy_(new_shapes, non_blocking=True)
            fr.lens.copy_(lens, non_blocking=True)
            if fr.graph is not None:
                fr.graph.replay()
            else:
                fr.body()
            left = self._decode(fr.st)
            ncols = self.max_new + 1 - left
            out = fr.st.out[:, :ncols].cpu()
        return self._texts(out.tolist())[0]

    def predict_batch(self, imgs):
        return self.predict_u8([self.resize_u8(i) for i in imgs])

    @torch.inference_mode()
    def predict_u8(self, u8s):
        """b > 1: engine's eager vision (packed) + gather/project/prefill, then chunked decode graphs keyed by (b, m).
        u8s: list of resized uint8 (3,h,w) CPU tensors (output of resize_u8)."""
        E, dev = self.eng, self.dev
        b = len(u8s)
        if b == 1 and self.front_graph:
            return [self.predict1(None, u8s[0])]
        shapes = [(u.shape[1] // PATCH, u.shape[2] // PATCH) for u in u8s]
        hw = [h * w for h, w in shapes]; n = max(hw)
        bb = b
        if self.batch_buckets:
            bb = next((x for x in self.batch_buckets if x >= b), b)
        with torch.cuda.stream(self.stream):
            pv = torch.zeros(bb, n, 768, dtype=torch.float16, device=dev)
            for i, u in enumerate(u8s):
                if self.gpu_prep:
                    self.patches_gpu(u, out=pv[i, :hw[i]])
                else:
                    pv[i, :hw[i]].copy_(self.patches_cpu(u).half())
            sh = shapes + [shapes[0]] * (bb - b)                   # padded rows: duplicate shape of row 0, zero pixels
            ss_cpu = torch.tensor(sh, dtype=torch.long)
            flat, base = self.vis.forward(pv, ss_cpu)
            gathered, new_shapes, lens, lens_c, m = self.mod.DSCProjector.gather_patches(flat, base, ss_cpu, E.r, 32)
            need_mask = bool((lens_c < m).any())
            vis_emb = E._project(gathered)
            bos_emb = F.embedding(torch.full((bb, 1), self.bos, dtype=torch.long, device=dev), E.emb_w)
            x = torch.cat([vis_emb, bos_emb], dim=1)
            cos, sin = self.mod.compute_batch_2d_mrope_freqs(new_shapes, m, 1, d_head=2 * E.d_axis, device=dev)
            st = self.state(bb, m)
            vis_valid = torch.arange(m, device=dev)[None] < lens[:, None]
            self._start(st, x, vis_valid, need_mask, cos, sin, m)
            if bb > b:
                st.unfinished[b:] = False
            left = self._decode(st)
            out = st.out[:b, :self.max_new + 1 - left].cpu()
        return self._texts(out.tolist())

    def __call__(self, imgs):
        if len(imgs) == 1:
            return [self.predict1(imgs[0])]
        return self.predict_batch(imgs)
