"""Faster batch-1 inference for PaddleOCR-VL-For-Manga (jzhang533/PaddleOCR-VL-For-Manga, Apache-2.0). Same weights, fp16, sdpa,
greedy, max_new_tokens 512, same processor parameters (smart_resize min_pixels 147,384 / max_pixels 2,822,400, PIL bicubic,
rescale/normalize). Tested with transformers 4.55.4.

What changes is only the implementation:
  * preprocessing: the PIL bicubic resize re-implemented bit for bit on the GPU (gpu_resize.py), rescale/normalize via a 256-entry
    LUT built with the processor's own rescale()/normalize() (bit-identical per value), patchify on the GPU; prompt ids from a
    cached template (no chat template / tokenizer per crop); 3D rope positions computed in closed form.
  * vision tower + projector + LLM prefill: the original modules, called directly (no HF generate / prepare_inputs / mask
    utilities / host syncs).
  * decode_mode="exact" (default): static KV cache; each decode step runs as 19 captured CUDA-graph segments (embedding + rope,
    the parts of every layer before and after attention, final norm + argmax) with the attention itself run eagerly at the exact
    KV length through HF's own sdpa_attention_forward, so every kernel sees the same shapes as the baseline. Outputs are
    identical to the baseline on all 7,789 crops of JMangaBench_Read v1.3.
  * decode_mode="bucket" (chunk=4): the whole step is one CUDA graph per KV-length bucket of 128 with masked fp32 attention;
    faster, but not bit-identical (7,783/7,789 identical outputs on Read v1.3, same scores).
Usage:
    from fast_paddle import load
    ocr = load("path/to/PaddleOCR-VL-For-Manga")            # or load(path, decode_mode="bucket", chunk=4)
    ocr.ocr(pil_image)                                     # -> text
Exactness is checked against the baseline outputs, not assumed."""
import math, os, sys, time, collections
import numpy as np
import torch
from PIL import Image

EOS = 2


class _PrefillCache:
    """HF Cache stand-in for the prefill: stores K/V of the prompt into the static buffers, returns them unchanged (as DynamicCache does
    on its first update)."""
    def __init__(self, K, V): self.K, self.V = K, V
    def update(self, k, v, layer_idx, cache_kwargs=None):
        P = k.shape[2]
        self.K[layer_idx][:, :, :P].copy_(k); self.V[layer_idx][:, :, :P].copy_(v)
        return k, v


class _StepCache:
    """HF Cache stand-in for one decode step at KV length L (python int fixed per captured graph): write the new K/V at L-1, return [:L]."""
    def __init__(self, K, V): self.K, self.V, self.L = K, V, None
    def update(self, k, v, layer_idx, cache_kwargs=None):
        L = self.L
        self.K[layer_idx][:, :, L - 1:L].copy_(k); self.V[layer_idx][:, :, L - 1:L].copy_(v)
        return self.K[layer_idx][:, :, :L], self.V[layer_idx][:, :, :L]


class FastPaddle:
    def __init__(self, model, proc, max_new=512, chunk=1, full_prefill_logits=True, gpu_resize=True, decode_mode="exact", bucket=128,
                 pre_buckets_upto=1024, max_bucket_graphs=24, lean_prefill=True, verbose=True):
        t0 = time.perf_counter()
        self.m, self.proc, self.ip, self.tok = model, proc, proc.image_processor, proc.tokenizer
        self.max_new, self.chunk, self.full_prefill_logits = max_new, chunk, full_prefill_logits
        self.decode_mode, self.bucket, self.max_bucket_graphs, self.lean_prefill = decode_mode, bucket, max_bucket_graphs, lean_prefill
        cfg = model.config
        self.dev = next(model.parameters()).device
        self.dtype = model.lm_head.weight.dtype
        self.img_id = cfg.image_token_id
        self.smart_resize = sys.modules[type(self.ip).__module__].smart_resize
        ip = self.ip
        self.factor = ip.patch_size * ip.merge_size
        # LUT uint8 -> normalized value, built with the processor's own functions (float32), then cast like the model does (.type(fp16))
        arr = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)  # (1, 256, 3) HWC
        x = ip.rescale(arr, scale=ip.rescale_factor, input_data_format="channels_last")
        x = ip.normalize(x, mean=ip.image_mean, std=ip.image_std, input_data_format="channels_last")
        assert x.dtype == np.float32, x.dtype
        self.lut = torch.from_numpy(np.ascontiguousarray(x[0])).to(self.dev).to(self.dtype)  # (256, 3)
        self.ch = torch.arange(3, device=self.dev)
        self.gpu_resize = gpu_resize
        if gpu_resize:
            try:
                from gpu_resize import GpuBicubic
            except ImportError:
                from .gpu_resize import GpuBicubic
            assert int(self.ip.resample) == 3, self.ip.resample
            self.rs = GpuBicubic(self.dev)
        # prompt template: processor output for a tiny image -> prefix / suffix around the image-token run
        tiny = Image.new("RGB", (64, 64), (255, 255, 255))
        msgs = [[{"role": "user", "content": [{"type": "image", "image": tiny}, {"type": "text", "text": "OCR:"}]}]]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = proc(text=text, images=[[tiny]], return_tensors="pt", padding=True)["input_ids"][0].tolist()
        pos = [i for i, t in enumerate(ids) if t == self.img_id]
        assert pos == list(range(pos[0], pos[-1] + 1)), "image tokens not contiguous"
        self.prefix, self.suffix = ids[:pos[0]], ids[pos[-1] + 1:]
        # static KV cache sized for the largest possible prompt (max_pixels) + max_new
        max_img_tok = ip.max_pixels // (self.factor ** 2)
        self.l_max = len(self.prefix) + max_img_tok + len(self.suffix) + max_new + 1
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        self.K = [torch.zeros(1, nkv, self.l_max + self.bucket, hd, device=self.dev, dtype=self.dtype) for _ in range(cfg.num_hidden_layers)]
        self.V = [torch.zeros(1, nkv, self.l_max + self.bucket, hd, device=self.dev, dtype=self.dtype) for _ in range(cfg.num_hidden_layers)]
        self.pcache = _PrefillCache(self.K, self.V)
        # decode-step static buffers (graph inputs / outputs)
        H, nh = cfg.hidden_size, cfg.num_attention_heads
        z = lambda *sh, dt=self.dtype: torch.zeros(*sh, dtype=dt, device=self.dev)
        self.tok_in, self.pos_in = z(1, 1, dt=torch.long), z(3, 1, 1, dt=torch.long)
        self.kv_idx, self.step_ctr = z(1, dt=torch.long), z(1, dt=torch.long)
        self.out_buf = z(max_new + 8, dt=torch.long)
        self.h_buf, self.q_buf, self.a_buf = z(1, 1, H), z(1, nh, 1, hd), z(1, 1, nh * hd)
        self.cos_buf, self.sin_buf = z(3, 1, 1, hd), z(3, 1, 1, hd)
        mod = sys.modules[type(model.model.layers[0].self_attn).__module__]
        self.rope_fn = mod.apply_multimodal_rotary_pos_emb
        self.rotate_half = mod.rotate_half
        self.mrope = list(cfg.rope_scaling["mrope_section"])
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        self.sdpa = sdpa_attention_forward
        self.vis_cache = {}
        self.n_layers = cfg.num_hidden_layers
        # vision rope table: SigLIPRotaryEmbedding(seqlen) = outer(arange(seqlen), inv_freq) is called with a GPU scalar (pids.max()+1)
        # -> host sync. Rows are independent of seqlen, so a fixed 1024-row table indexed by the same pids gives identical values.
        enc = model.visual.vision_model.encoder
        if not getattr(enc.rotary_pos_emb, "_fixed_table", False):
            tbl = enc.rotary_pos_emb(1024)
            enc.rotary_pos_emb.forward = lambda seqlen, _t=tbl: _t
            enc.rotary_pos_emb._fixed_table = True
        min_img_tok = math.ceil(ip.min_pixels / self.factor ** 2)
        self.l_lo = len(self.prefix) + min_img_tok + len(self.suffix) + 1
        with torch.inference_mode():
            # warm up every segment eagerly on a side stream (cuBLAS handles / workspaces), then capture the 19 segment graphs once
            self.kv_idx.fill_(self.l_lo - 1)
            s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    for i in range(self.n_layers + 1):
                        self._seg(i)
                        if i < self.n_layers: self._attn(i, self.l_lo)
            torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
            self.pool = torch.cuda.graph_pool_handle()
            self.segs = []
            for i in range(self.n_layers + 1):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool):
                    self._seg(i)
                self.segs.append(g)
            self.bgraphs = collections.OrderedDict(); self.n_bucket_captures = 0
            if decode_mode == "bucket":
                self.ar = torch.arange(self.l_max + bucket, device=self.dev)
                s2 = torch.cuda.Stream(); s2.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s2):
                    for _ in range(2): self._bstep(bucket)
                torch.cuda.current_stream().wait_stream(s2); torch.cuda.synchronize()
                for Lb in range(bucket, pre_buckets_upto + 1, bucket):
                    self._bgraph(Lb)
        torch.cuda.synchronize()
        self.startup_s = time.perf_counter() - t0
        self.n_bucket_captures = 0
        if verbose:
            print(f"FastPaddle: mode {decode_mode}, {len(self.segs)} segment graphs, {len(self.bgraphs)} bucket graphs, l_max {self.l_max}, startup {self.startup_s:.1f}s, "
                  f"mem {torch.cuda.memory_allocated() / 2**20:.0f} MB", flush=True)

    # ---------------- decode step: 19 captured segments + eager exact-length attention ----------------
    def _pre(self, i, h):
        """Ernie4_5DecoderLayer i up to the attention call (same ops as HF): residual, input RMSNorm, q/k/v, 3D rope, KV write."""
        layer = self.m.model.layers[i]; at = layer.self_attn
        self.h_buf.copy_(h)
        x = layer.input_layernorm(h)
        shp = (1, 1, -1, at.head_dim)
        q = at.q_proj(x).view(shp).transpose(1, 2)
        k = at.k_proj(x).view(shp).transpose(1, 2)
        v = at.v_proj(x).view(shp).transpose(1, 2)
        q, k = self.rope_fn(q, k, self.cos_buf, self.sin_buf, at.rope_scaling["mrope_section"])
        self.K[i].index_copy_(2, self.kv_idx, k); self.V[i].index_copy_(2, self.kv_idx, v)
        self.q_buf.copy_(q)

    def _post(self, i):
        """rest of layer i after attention: o_proj, residual, post-attention RMSNorm, MLP, residual."""
        layer = self.m.model.layers[i]
        h = self.h_buf + layer.self_attn.o_proj(self.a_buf)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def _seg(self, i):
        mm = self.m.model
        if i == 0:
            h = mm.embed_tokens(self.tok_in)
            cos, sin = mm.rotary_emb(h, self.pos_in)
            self.cos_buf.copy_(cos); self.sin_buf.copy_(sin)
        else:
            h = self._post(i - 1)
        if i < self.n_layers:
            self._pre(i, h); return
        h = mm.norm(h)
        nxt = self.m.lm_head(h)[:, -1, :].to(copy=True, dtype=torch.float32).argmax(-1)  # (1,)
        self.tok_in.copy_(nxt.view(1, 1))
        self.pos_in.add_(1); self.kv_idx.add_(1)
        self.out_buf.index_copy_(0, self.step_ctr, nxt)
        self.step_ctr.add_(1)

    def _attn(self, i, L):
        """eager, exact length L, through HF's own sdpa_attention_forward (no mask -> enable_gqa, same kernels as the baseline)."""
        at = self.m.model.layers[i].self_attn
        o, _ = self.sdpa(at, self.q_buf, self.K[i][:, :, :L], self.V[i][:, :, :L], None, dropout=0.0, scaling=at.scaling)
        self.a_buf.copy_(o.reshape(1, 1, -1))

    def _mcat(self, cos, sin):
        """the per-layer mrope section select of apply_multimodal_rotary_pos_emb, done once (pure data movement, identical values)."""
        ms = self.mrope * 2
        c = torch.cat([m[i % 3] for i, m in enumerate(cos.split(ms, dim=-1))], dim=-1).unsqueeze(1)
        s = torch.cat([m[i % 3] for i, m in enumerate(sin.split(ms, dim=-1))], dim=-1).unsqueeze(1)
        return c, s

    def _bstep(self, Lb):
        """bucket mode: one whole decode step, attention over the first Lb cache slots with a mask (> kv_idx = -inf), fp32 like the
        math sdpa path but without repeat_interleave (GQA via a (kv_heads, group) view). Not bit-identical to the baseline."""
        mm = self.m.model
        h = mm.embed_tokens(self.tok_in)
        cos, sin = mm.rotary_emb(h, self.pos_in)
        cm, sm = self._mcat(cos, sin)
        rh = self.rotate_half
        mask = torch.zeros(Lb, device=self.dev, dtype=torch.float32).masked_fill(self.ar[:Lb] > self.kv_idx, float("-inf"))
        for i, layer in enumerate(mm.layers):
            at = layer.self_attn; hd = at.head_dim
            res = h
            x = layer.input_layernorm(h)
            q = at.q_proj(x).view(1, 1, -1, hd).transpose(1, 2)
            k = at.k_proj(x).view(1, 1, -1, hd).transpose(1, 2)
            v = at.v_proj(x).view(1, 1, -1, hd).transpose(1, 2)
            q = q * cm + rh(q) * sm; k = k * cm + rh(k) * sm
            self.K[i].index_copy_(2, self.kv_idx, k); self.V[i].index_copy_(2, self.kv_idx, v)
            sf = math.sqrt(at.scaling)
            nkv = k.shape[1]
            q32 = (q.float() * sf).view(1, nkv, -1, hd)                       # (1, kv, group, hd)
            k32 = self.K[i][:, :, :Lb].float()
            a = torch.matmul(q32, k32.transpose(-2, -1) * sf) + mask
            a = torch.softmax(a, dim=-1)
            o = torch.matmul(a, self.V[i][:, :, :Lb].float()).to(h.dtype)     # (1, kv, group, hd)
            o = o.reshape(1, -1, 1, hd).transpose(1, 2).reshape(1, 1, -1)
            h = res + at.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = mm.norm(h)
        nxt = self.m.lm_head(h)[:, -1, :].to(copy=True, dtype=torch.float32).argmax(-1)
        self.tok_in.copy_(nxt.view(1, 1))
        self.pos_in.add_(1); self.kv_idx.add_(1)
        self.out_buf.index_copy_(0, self.step_ctr, nxt)
        self.step_ctr.add_(1)

    def _bgraph(self, Lb):
        g = self.bgraphs.get(Lb)
        if g is not None:
            self.bgraphs.move_to_end(Lb); return g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool):
            self._bstep(Lb)
        self.bgraphs[Lb] = g; self.n_bucket_captures += 1
        if len(self.bgraphs) > self.max_bucket_graphs: self.bgraphs.popitem(last=False)
        return g

    def _decode_step(self, L):
        if self.decode_mode == "bucket":
            self._bgraph(-(-L // self.bucket) * self.bucket).replay(); return
        for i in range(self.n_layers):
            self.segs[i].replay(); self._attn(i, L)
        self.segs[self.n_layers].replay()

    # ---------------- preprocessing ----------------
    def _target(self, h, w):
        return self.smart_resize(h, w, factor=self.factor, min_pixels=self.ip.min_pixels, max_pixels=self.ip.max_pixels)

    def prep_cpu(self, im):
        """PIL RGB image -> uint8 HWC array. gpu_resize: the original pixels (resize happens on the GPU, bit-exact to PIL);
        otherwise resized here with the same smart_resize + PIL bicubic call as the HF processor."""
        if self.gpu_resize:
            return np.asarray(im)
        w, h = im.size
        rh, rw = self._target(h, w)
        return np.asarray(im.resize((rw, rh), resample=self.ip.resample))

    def _patches(self, arr):
        x = torch.from_numpy(arr).to(self.dev, non_blocking=False)
        if self.gpu_resize:
            rh, rw = self._target(arr.shape[0], arr.shape[1])
            x = self.rs(x, rh, rw)
        H, W, _ = x.shape; p = self.ip.patch_size; gh, gw = H // p, W // p
        x = self.lut[x.long(), self.ch]                                      # (H, W, 3) fp16
        x = x.permute(2, 0, 1).reshape(3, gh, p, gw, p).permute(1, 3, 0, 2, 4).reshape(gh * gw, 3, p, p)
        return x, gh, gw

    def _shape_consts(self, gh, gw):
        key = (gh, gw)
        c = self.vis_cache.get(key)
        if c is not None: return c
        ms = self.ip.merge_size; lh, lw = gh // ms, gw // ms; n_img = lh * lw
        a, b = len(self.prefix), len(self.suffix)
        ids = torch.tensor(self.prefix + [self.img_id] * n_img + self.suffix, dtype=torch.long, device=self.dev)[None]
        txt0 = np.arange(a)
        hh = np.repeat(np.arange(lh), lw); ww = np.tile(np.arange(lw), lh)
        img = np.stack([np.zeros(n_img, np.int64), hh, ww]) + a
        st = a + max(lh, lw)
        txt1 = np.arange(b) + st
        pos = np.concatenate([np.stack([txt0] * 3), img, np.stack([txt1] * 3)], axis=1)  # (3, P)
        P = pos.shape[1]; delta = int(pos.max()) + 1 - P
        N = gh * gw
        c = {"ids": ids, "pos": torch.from_numpy(pos)[:, None, :].to(self.dev), "delta": delta, "P": P, "a": a, "n_img": n_img,
             "vis_pos": torch.arange(N, device=self.dev), "sample_idx": torch.zeros(N, dtype=torch.long, device=self.dev),
             "cu": torch.tensor([0, N], dtype=torch.int32)}   # CPU: only used for the final split (no flash-attn) -> no host sync
        if len(self.vis_cache) > 512: self.vis_cache.pop(next(iter(self.vis_cache)))
        self.vis_cache[key] = c
        return c

    # ---------------- prefill ----------------
    def _prefill(self, pix, gh, gw, c):
        m = self.m; tm = getattr(self, "timing", None)
        if tm is not None: torch.cuda.synchronize(); tv = time.perf_counter()
        emb = m.model.embed_tokens(c["ids"])
        vo = m.visual(pixel_values=pix.unsqueeze(0), image_grid_thw=[(1, gh, gw)], position_ids=c["vis_pos"], vision_return_embed_list=True,
                      interpolate_pos_encoding=True, sample_indices=c["sample_idx"], cu_seqlens=c["cu"], return_pooler_output=False,
                      use_rope=True, window_size=-1)
        feats = m.mlp_AR(vo.last_hidden_state, [(1, gh, gw)])
        feats = torch.cat(feats, dim=0).to(emb.dtype)
        if tm is not None: torch.cuda.synchronize(); tm["vision_ms"] = 1000 * (time.perf_counter() - tv)
        a = c["a"]; emb[0, a:a + c["n_img"]] = feats
        pos = c["pos"]
        cos, sin = m.model.rotary_emb(emb, pos)
        h = emb
        if not self.lean_prefill:
            for layer in m.model.layers:
                h = layer(h, attention_mask=None, position_ids=pos, past_key_value=self.pcache, position_embeddings=(cos, sin))
        else:
            cm, sm = self._mcat(cos, sin); rh = self.rotate_half; P = h.shape[1]
            for i, layer in enumerate(m.model.layers):      # same aten ops as Ernie4_5DecoderLayer + HF sdpa_attention_forward
                at = layer.self_attn; shp = (1, P, -1, at.head_dim)
                res = h
                x = layer.input_layernorm(h)
                q = at.q_proj(x).view(shp).transpose(1, 2)
                k = at.k_proj(x).view(shp).transpose(1, 2)
                v = at.v_proj(x).view(shp).transpose(1, 2)
                q = (q * cm) + (rh(q) * sm); k = (k * cm) + (rh(k) * sm)
                self.K[i][:, :, :P].copy_(k); self.V[i][:, :, :P].copy_(v)
                o, _ = self.sdpa(at, q, k, v, None, dropout=0.0, scaling=at.scaling)
                h = res + at.o_proj(o.reshape(1, P, -1).contiguous())
                h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = m.model.norm(h)
        logits = m.lm_head(h)[:, -1, :] if self.full_prefill_logits else m.lm_head(h[:, -1:, :])[:, -1, :]
        return logits.to(copy=True, dtype=torch.float32).argmax(-1)  # (1,)

    # ---------------- full crop ----------------
    @torch.inference_mode()
    def ocr_ids(self, arr):
        pix, gh, gw = self._patches(arr)
        c = self._shape_consts(gh, gw)
        P, delta = c["P"], c["delta"]
        tok0 = self._prefill(pix, gh, gw, c)
        self.tok_in.copy_(tok0.view(1, 1))
        self.pos_in.fill_(P + delta)          # rope position of the token at KV index P (= cache_position + rope_deltas)
        self.kv_idx.fill_(P)
        self.step_ctr.zero_()
        out = [int(tok0.item())]
        if out[0] == EOS: return out
        L = P + 1
        while len(out) < self.max_new:
            k = min(self.chunk, self.max_new - len(out))
            for _ in range(k):
                self._decode_step(L); L += 1
            new = self.out_buf[len(out) - 1:len(out) - 1 + k].tolist()
            for t in new:
                out.append(t)
                if t == EOS: return out
        return out

    def ocr(self, im):
        return self.tok.decode(self.ocr_ids(self.prep_cpu(im)), skip_special_tokens=True)


def load(model_dir, verbose=True, **kw):
    """fp16, sdpa (as the baseline on GPUs without flash-attn); batch 1 needs no change to the vision attention."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda().eval()
    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=True)
    return FastPaddle(model, proc, verbose=verbose, **kw)
