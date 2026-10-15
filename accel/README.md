# accel: faster inference for two other manga OCR models

[English](README.md) | [中文](README_zh.md)

Evidence for finding 5 of the [technical report](../TECHNICAL_REPORT.md#43-two-other-models-implementation-alone-speeds-them-up-too): autoregressive manga OCR models are slow mainly because of their implementation. These scripts run two existing models with **their original weights, greedy decoding, generation limits, preprocessing parameters and fp16 precision unchanged**; only the inference flow is rewritten (CUDA graphs, static KV cache, GPU preprocessing), without torch.compile.

| Model | Original (batch 1, mean / p50) | Accelerated | Outputs identical to the original |
|---|---|---|---|
| [Hayai v2.5 Nova](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova) p384 | 54.29 / 52.43 ms | 21.22 / 18.29 ms | 7,789/7,789 |
| [PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga), exact | 365.0 / 308.4 ms | 141.7 / 120.2 ms | 7,789/7,789 |
| PaddleOCR-VL-For-Manga, bucket | 365.0 / 308.4 ms | 101.5 / 93.4 ms | 7,783/7,789 (same scores) |

RTX 2080 Ti, fp16, all 7,789 boxes of [JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read), preprocessing included, the first 20 boxes not timed; the originals were re-measured on the same GPU. Details, page-level numbers and costs are in §4.3 of the report.

## Files

| File | Contents |
|---|---|
| `hayai_nova/nova_fast.py` | `NovaFast`: GPU preprocessing; one CUDA graph for vision encoder + projector + prefill per input shape; decoding replayed step by step from CUDA graphs with an end-token check after every step |
| `paddleocr_vl_manga/fast_paddle.py` | `FastPaddle`: no HF `generate`; static KV cache; each decoding step as 19 CUDA-graph segments with attention at the exact length (`exact`), or one graph per KV-length bucket of 128 (`bucket`) |
| `paddleocr_vl_manga/gpu_resize.py` | Pillow's 8-bit bicubic resize reproduced bit for bit on the GPU |
| `compare.py` | Runs the original and the accelerated path on a folder of crops: identical outputs and per-crop latency |

## Usage

Download the original models first (Hugging Face or a mirror). Hayai Nova was tested at revision `2803d0dc` with transformers 4.57; PaddleOCR-VL-For-Manga with transformers 4.55.4. Both need an NVIDIA GPU.

```bash
python compare.py nova   --model-dir hayai-ocr-v2.5-nova --images crops/ --n 100
python compare.py paddle --model-dir PaddleOCR-VL-For-Manga --images crops/ --n 100            # add --bucket for bucket mode
```

```python
import sys; sys.path.insert(0, "accel/hayai_nova")
from nova_fast import NovaFast
ocr = NovaFast("hayai-ocr-v2.5-nova", max_num_patches=384)
ocr([pil_image])                                       # -> [text]

sys.path.insert(0, "accel/paddleocr_vl_manga")
from fast_paddle import load
ocr = load("PaddleOCR-VL-For-Manga")                   # load(..., decode_mode="bucket", chunk=4) for bucket mode
ocr.ocr(pil_image)                                     # -> text
```

## Caveats

- **Nova captures a graph the first time an input shape appears** (about 0.1 s; 41 shapes on Read v1.3, 31 boxes 35–40 ms slower because of it). Graphs are kept, so GPU memory grows with the number of distinct shapes (whole-page inference 925 → 1,560 MB). Gains shrink for long outputs (more than 65 characters: 171 → 148 ms).
- **Paddle `bucket` mode is not bit-identical** (6 of 7,789 outputs differ, mostly in how ellipses are written). `exact` mode is.
- Tested on one GPU model (RTX 2080 Ti) only. Batch 1 is the main target; batched inference was not optimised beyond what is described in the report.
- These are research scripts, not packages. They are not affiliated with or endorsed by the authors of the two models.

## License

Apache License 2.0, as the rest of this repository. Both upstream models are released under Apache License 2.0; the scripts call the models' own code (`trust_remote_code`) and re-implement parts of it with the same operations. See `../NOTICE.md`.
