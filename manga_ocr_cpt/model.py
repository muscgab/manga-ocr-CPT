"""Network: the mangaOCR architecture (ViT-B/16 encoder, 2-layer post-LN BERT decoder with cross-attention, output layer tied to
the input embedding), written in plain PyTorch on the weights in model.safetensors. Differences from kha-white/manga-ocr-base:
6,822 output tokens (5 special + our 6,817 characters). Input (config.json "fit"): "square" (manga-ocr-base-CPT) = the original
224 x 224 stretch, 14 x 14 patches, the position table as is; "aspect" (manga-ocr-base-CPT-flex) = aspect kept, at most 196
patches, positions = the 14 x 14 learned table bicubically resized to the patch grid, padded patches masked out of attention.

Precision: on CUDA the encoder runs its matrix products in fp16 with LayerNorm and the residual stream in fp32 (= the fp16
autocast used in training and evaluation); the decoder runs in fp32 (decoder="fp32", default, as evaluated) or with fp16 matrix
products (decoder="fp16"). On CPU everything is fp32.
Token ids: 0 pad, 1 unk, 2 start, 3 end, 4 mask, 5 + i = character i."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

D, HEADS, EPS = 768, 12, 1e-12
START, END, NSPEC = 2, 3, 5
MAX_NEW = 65                     # at most 64 characters + end


class MangaOcrNet:
    def __init__(self, w: dict, device, half_encoder: bool, half_decoder: bool):
        self.dev = torch.device(device)
        e16 = torch.float16 if half_encoder else torch.float32
        d16 = torch.float16 if half_decoder else torch.float32
        self.edt, self.ddt = e16, d16
        g = lambda k, dt=torch.float32: w[k].to(self.dev, dt)
        lin = lambda p, dt: (g(p + ".weight", dt), g(p + ".bias", dt))
        ln = lambda p: (g(p + ".weight"), g(p + ".bias"))
        self.patch = lin("enc.patch", e16)
        self.cls, self.pos = g("enc.cls"), g("enc.pos")                  # [768], [197, 768]
        self.enc = [{"ln1": ln(f"enc.{i}.ln1"), "q": lin(f"enc.{i}.q", e16), "k": lin(f"enc.{i}.k", e16), "v": lin(f"enc.{i}.v", e16),
                     "o": lin(f"enc.{i}.o", e16), "ln2": ln(f"enc.{i}.ln2"), "f1": lin(f"enc.{i}.f1", e16), "f2": lin(f"enc.{i}.f2", e16)}
                    for i in range(12)]
        self.enc_ln = ln("enc.ln")
        self.word, self.posw, self.tt0, self.ln_e = g("dec.word"), g("dec.pos"), g("dec.type0"), ln("dec.ln")
        self.dec = [{n: lin(f"dec.{i}.{n}", d16) for n in ("q", "k", "v", "o", "cq", "ck", "cv", "co", "f1", "f2")} |
                    {n: ln(f"dec.{i}.{n}") for n in ("ln1", "ln2", "ln3")} for i in range(2)]
        self.head_t, self.ln_h = lin("dec.head", d16), ln("dec.head_ln")
        self.out = (g("dec.word", d16), g("dec.out_bias", d16))           # tied to the input embedding
        self._pe = {}

    @classmethod
    def load(cls, model_dir, device="cuda", half_decoder=False) -> "MangaOcrNet":
        from safetensors.torch import load_file
        w = load_file(str(Path(model_dir) / "model.safetensors"))
        cuda = torch.device(device).type == "cuda"
        return cls(w, device, half_encoder=cuda, half_decoder=half_decoder and cuda)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _lin(x: Tensor, wb, dt) -> Tensor:
        return F.linear(x.to(dt), wb[0], wb[1])

    @staticmethod
    def _ln(x: Tensor, wb) -> Tensor:
        return F.layer_norm(x.float(), (D,), wb[0], wb[1], EPS)

    @staticmethod
    def _heads(x: Tensor, B: int) -> Tensor:
        return x.view(B, -1, HEADS, D // HEADS).transpose(1, 2)

    def pe(self, gh: int, gw: int) -> Tensor:
        """position rows for a gh x gw grid, [196, 768] (zeros on padded patches), cached"""
        key = (gh, gw)
        if key not in self._pe:
            table = self.pos[1:].reshape(14, 14, D).permute(2, 0, 1)[None]
            t = table if key == (14, 14) else F.interpolate(table, size=key, mode="bicubic", align_corners=False)
            out = torch.zeros(196, D, device=self.dev, dtype=self.edt)
            out[: gh * gw] = t[0].flatten(1).T.to(self.edt)
            self._pe[key] = out
        return self._pe[key]

    # ------------------------------------------------------------------ encoder
    def encode(self, P: Tensor, V: Tensor, pe: Tensor):
        """P [B, 196, 768] float32, V [B, 196] bool, pe [B, 196, 768] -> memory [B, 197, 768] fp32, mask [B, 197] bool"""
        dt = self.edt
        B = P.shape[0]
        x = self._lin(P, self.patch, dt) + pe
        h = torch.cat([(self.cls + self.pos[0]).expand(B, 1, D), x.float()], 1)
        mask = torch.cat([torch.ones_like(V[:, :1]), V], 1)
        am = mask[:, None, None, :]
        for L in self.enc:
            y = self._ln(h, L["ln1"])
            q, k, v = (self._heads(self._lin(y, L[n], dt), B) for n in "qkv")
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B, -1, D)
            h = h + self._lin(y, L["o"], dt)
            h = h + self._lin(F.gelu(self._lin(self._ln(h, L["ln2"]), L["f1"], dt)), L["f2"], dt)
        return self._ln(h, self.enc_ln), mask

    def cross_kv(self, mem: Tensor):
        B = mem.shape[0]
        return [(self._heads(self._lin(mem, L["ck"], self.ddt), B), self._heads(self._lin(mem, L["cv"], self.ddt), B)) for L in self.dec]

    # ------------------------------------------------------------------ one greedy decoder step
    def step(self, tok: Tensor, pos: Tensor, sk, sv, ckv, xm: Tensor, live: Tensor) -> Tensor:
        """tok [B] ids being fed, pos [1] their index; sk/sv per layer [B, 12, S, 64] caches (written at pos); ckv cross K/V;
        xm [B, 1, 1, 197] memory mask; live [1, 1, 1, S] cache slots <= pos -> next ids [B]"""
        dt = self.ddt
        B = tok.shape[0]
        x = self._ln(self.word[tok][:, None] + self.posw[pos][None] + self.tt0, self.ln_e)
        for i, L in enumerate(self.dec):
            q, k, v = (self._heads(self._lin(x, L[n], dt), B) for n in "qkv")
            sk[i].index_copy_(2, pos, k); sv[i].index_copy_(2, pos, v)
            y = F.scaled_dot_product_attention(q, sk[i], sv[i], attn_mask=live)
            x = self._ln(self._lin(y.transpose(1, 2).reshape(B, 1, D), L["o"], dt) + x, L["ln1"])
            y = F.scaled_dot_product_attention(self._heads(self._lin(x, L["cq"], dt), B), ckv[i][0], ckv[i][1], attn_mask=xm)
            x = self._ln(self._lin(y.transpose(1, 2).reshape(B, 1, D), L["co"], dt) + x, L["ln2"])
            x = self._ln(self._lin(F.gelu(self._lin(x, L["f1"], dt)), L["f2"], dt) + x, L["ln3"])
        z = self._ln(F.gelu(self._lin(x, self.head_t, dt)), self.ln_h)
        return self._lin(z[:, 0], self.out, dt).argmax(-1)

    def new_cache(self, B: int, slots: int):
        z = lambda: torch.zeros(B, HEADS, slots, D // HEADS, device=self.dev, dtype=self.ddt)
        return [z() for _ in self.dec], [z() for _ in self.dec]

    @torch.inference_mode()
    def greedy(self, P: Tensor, V: Tensor, pe: Tensor) -> Tensor:
        """plain (eager) greedy decoding -> ids [B, <=65]; used on CPU and other non-CUDA devices"""
        mem, mask = self.encode(P, V, pe)
        ckv = self.cross_kv(mem)
        B = P.shape[0]
        sk, sv = self.new_cache(B, MAX_NEW + 1)
        slot = torch.arange(MAX_NEW + 1, device=self.dev)
        tok = torch.full((B,), START, dtype=torch.long, device=self.dev)
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        out = []
        for t in range(MAX_NEW):
            pos = torch.tensor([t], device=self.dev)
            tok = self.step(tok, pos, sk, sv, ckv, mask[:, None, None, :], (slot <= pos)[None, None, None, :])
            out.append(tok)
            done |= tok == END
            if bool(done.all()):
                break
        return torch.stack(out, 1)
