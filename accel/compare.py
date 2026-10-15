"""Baseline vs accelerated inference on a folder of text-box crops, batch 1: identical outputs and per-crop latency.

    python compare.py nova   --model-dir hayai-ocr-v2.5-nova --processor-dir siglip2-base-patch16-naflex --images crops/ [--n 100]
    python compare.py paddle --model-dir PaddleOCR-VL-For-Manga --images crops/ [--n 100] [--bucket]

Baselines are the models' own Hugging Face paths, as used for JMangaBench_Read v1.3: Hayai v2.5 Nova p384 in fp16 with
model.generate(..., max_new_tokens=128); PaddleOCR-VL-For-Manga in fp16 with sdpa, prompt "OCR:", greedy, max_new_tokens 512.
Latency includes preprocessing and is measured with CUDA synchronisation; the first --warmup crops are not timed."""
import argparse, os, statistics, sys, time
from pathlib import Path
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent


def nova(a):
    sys.path.insert(0, str(HERE / "hayai_nova"))
    from transformers import AutoModel, AutoProcessor, PreTrainedTokenizerFast
    from nova_fast import NovaFast
    model = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True).cuda().eval().half()
    tok = PreTrainedTokenizerFast.from_pretrained(a.model_dir)
    proc = AutoProcessor.from_pretrained(a.processor_dir)

    @torch.inference_mode()
    def base(im):
        x = proc(images=[im.convert("RGB")], max_num_patches=384, return_tensors="pt").to("cuda")
        x["pixel_values"] = x["pixel_values"].half()
        return model.generate(pixel_values=x["pixel_values"], pixel_attention_mask=x["pixel_attention_mask"],
                              spatial_shapes=x["spatial_shapes"], tokenizer=tok, max_new_tokens=128, repetition_penalty=1.0)[0]
    fast = NovaFast(a.model_dir, max_num_patches=384)
    return base, lambda im: fast([im.convert("RGB")])[0]


def paddle(a):
    sys.path.insert(0, str(HERE / "paddleocr_vl_manga"))
    from transformers import AutoModelForCausalLM, AutoProcessor
    import fast_paddle
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, trust_remote_code=True, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda().eval()
    proc = AutoProcessor.from_pretrained(a.model_dir, trust_remote_code=True, use_fast=True)

    @torch.inference_mode()
    def base(im):
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": "OCR:"}]}]]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        x = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in proc(text=text, images=[[im]], return_tensors="pt").items()}
        g = model.generate(**x, max_new_tokens=512, do_sample=False, use_cache=True)
        return proc.batch_decode(g[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    kw = {"decode_mode": "bucket", "chunk": 4} if a.bucket else {}
    fast = fast_paddle.FastPaddle(model, proc, **kw)
    return base, lambda im: fast.ocr(im.convert("RGB"))


def timed(fn, imgs, warmup):
    out, lat = [], []
    for i, im in enumerate(imgs):
        torch.cuda.synchronize(); t = time.perf_counter()
        out.append(fn(im)); torch.cuda.synchronize()
        if i >= warmup:
            lat.append(1000 * (time.perf_counter() - t))
    return out, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["nova", "paddle"])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--processor-dir", default="google/siglip2-base-patch16-naflex", help="nova only")
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--bucket", action="store_true", help="paddle only: bucket decode mode (faster, not bit-identical)")
    a = ap.parse_args()
    files = sorted(p for p in Path(a.images).iterdir()
                   if not p.name.startswith(".") and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[: a.n]
    imgs = []
    for f in files:
        with Image.open(f) as im:
            imgs.append(im.convert("RGB"))
    base, fast = (nova if a.model == "nova" else paddle)(a)
    tb, lb = timed(base, imgs, a.warmup)
    tf, lf = timed(fast, imgs, a.warmup)
    same = sum(x == y for x, y in zip(tb, tf))
    q = lambda v: f"mean {statistics.mean(v):.1f} ms, p50 {statistics.median(v):.1f} ms"
    print(f"{len(imgs)} crops: identical outputs {same}/{len(imgs)}")
    print(f"baseline    {q(lb)}")
    print(f"accelerated {q(lf)}")
    for f, x, y in zip(files, tb, tf):
        if x != y:
            print(f"  differs: {f.name}: {x!r} -> {y!r}")


if __name__ == "__main__":
    main()
