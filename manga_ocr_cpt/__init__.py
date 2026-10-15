"""manga-ocr-CPT: kha-white/manga-ocr-base after continued pre-training on our data (Japanese manga text-box OCR, 111M parameters).
Two models, same code: manga-ocr-base-CPT-flex (aspect ratio kept, recommended) and manga-ocr-base-CPT (the original 224 x 224
input). On an NVIDIA GPU the fast path runs preprocessing on the GPU and replays greedy decoding from CUDA graphs; on CPU, Apple
MPS or any other torch device the plain PyTorch path runs. Both paths give the same outputs.

    from manga_ocr_cpt import MangaOcrCPT
    ocr = MangaOcrCPT("flex")            # or "224", or a local directory holding model.safetensors + config.json
    ocr("crop.png")                      # one crop -> str
    ocr([crop1, crop2, ...])             # several crops (e.g. one page) -> list[str], run as one batch

Crops may be file paths, PIL images or uint8 HWC RGB arrays."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .preprocess import preprocess

__version__ = "1.0.0"
__all__ = ["MangaOcrCPT", "MODELS"]
MODELS = {"flex": "muscgab/manga-ocr-base-CPT-flex", "224": "muscgab/manga-ocr-base-CPT"}
PLAIN_BATCH = 16


def _rgb(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        a = x if x.ndim == 3 else np.stack([x] * 3, -1)
        return np.ascontiguousarray(a[..., :3], dtype=np.uint8)
    if isinstance(x, Image.Image):
        return np.asarray(x.convert("RGB"))
    with Image.open(x) as im:
        im.load()
        return np.asarray(im.convert("RGB"))


def resolve(model: str) -> Path:
    """local directory -> itself; "flex" / "224" / a Hugging Face repo id -> downloaded snapshot (needs huggingface_hub)"""
    p = Path(os.path.expanduser(model))
    if (p / "model.safetensors").exists():
        return p
    repo = MODELS.get(model, model)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise FileNotFoundError(f"{model!r} is not a local model directory; install huggingface_hub to download {repo}, "
                                "or download it yourself (Hugging Face / ModelScope) and pass the directory") from e
    return Path(snapshot_download(repo, allow_patterns=["model.safetensors", "config.json", "nar_vocabulary_6817.json"]))


class MangaOcrCPT:
    def __init__(self, model: str = "flex", device: str | None = None, backend: str = "auto", decoder: str = "fp32"):
        """model: "flex", "224", a Hugging Face repo id or a local directory. backend: "auto" (CUDA graphs on an NVIDIA GPU,
        plain PyTorch elsewhere), "cuda" or "plain". decoder: "fp32" (default, reproduces the evaluated outputs) or "fp16"
        (CUDA only; about 1 ms per crop faster)."""
        import torch
        from .model import MangaOcrNet
        if decoder not in ("fp32", "fp16"):
            raise ValueError(f"decoder must be 'fp32' or 'fp16', got {decoder!r}")
        if backend not in ("auto", "cuda", "plain"):
            raise ValueError(f"backend must be 'auto', 'cuda' or 'plain', got {backend!r}")
        self.model_dir = resolve(model)
        cfg = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.fit = cfg["input"]["fit"]
        self.chars = json.loads((self.model_dir / "nar_vocabulary_6817.json").read_text(encoding="utf-8"))["characters"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if backend == "cuda" and self.device.type != "cuda":
            raise ValueError(f"backend='cuda' needs a CUDA device, got {self.device}")
        self.net = MangaOcrNet.load(self.model_dir, self.device, half_decoder=decoder == "fp16")
        self.runner = None
        if self.device.type == "cuda" and backend != "plain":
            from .cuda import CudaRunner
            self.runner = CudaRunner(self.net, self.fit)
        self._torch = torch

    def _text(self, ids) -> str:
        ids = list(ids)
        ids = ids[: ids.index(3)] if 3 in ids else ids
        return "".join(self.chars[t - 5] for t in ids if t >= 5)

    def ids(self, arrs: list[np.ndarray]) -> list[list[int]]:
        if self.runner is not None:
            return self.runner.ids(arrs)
        torch, out = self._torch, []
        for i in range(0, len(arrs), PLAIN_BATCH):
            items = [preprocess(Image.fromarray(a), self.fit) for a in arrs[i:i + PLAIN_BATCH]]
            P = torch.from_numpy(np.stack([it[0] for it in items])).to(self.device)
            V = torch.from_numpy(np.stack([it[1] for it in items])).to(self.device)
            pe = torch.stack([self.net.pe(it[2], it[3]) for it in items])
            out += self.net.greedy(P, V, pe).tolist()
        return out

    def __call__(self, images):
        single = not isinstance(images, (list, tuple))
        arrs = [_rgb(x) for x in ([images] if single else images)]
        texts = [self._text(r) for r in self.ids(arrs)]
        return texts[0] if single else texts
