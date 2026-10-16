# manga-ocr-CPT

[English](README.md) | [中文](README_zh.md)

This is an experimental repository. We kept the architecture of [mangaOCR](https://github.com/kha-white/manga-ocr) (`kha-white/manga-ocr-base`) unchanged, replaced only its vocabulary, and continued pre-training it (CPT) on new training data.

The main content of this repository is the experimental findings; see the [technical report](TECHNICAL_REPORT.md) for the full account. The repository also provides two models and their inference code.

| Model | Input | Weights |
|---|---|---|
| **manga-ocr-base-CPT-flex** (recommended) | Aspect ratio kept, at most 196 patches | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-base-CPT-flex) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-base-CPT-flex) |
| manga-ocr-base-CPT | Stretched to 224 × 224, exactly as the original mangaOCR | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-base-CPT) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-base-CPT) |

The two models share the same architecture, parameter count (111M) and training settings; only the input differs. Below they are called CPT-flex and CPT-224.

## Main findings

1. **Training data is the main factor.** With the original architecture, parameter count and 224 × 224 input, and only the training data and vocabulary replaced, EM on [JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read) rises from 81.29% to 91.00% (both with greedy decoding, $p = 4\times10^{-110}$).
2. **A 2-layer autoregressive decoder is enough.** With the same data, CPT-flex ties our 54M non-autoregressive model manga-ocr-nar-small (EM 91.42% vs 91.09%, $p = 0.24$).
3. **Keeping the aspect ratio helps only on extreme shapes.** The overall gain is just 0.42 EM points and not significant. But on boxes with aspect ratio ≥ 8 it is 85.40% vs 76.64%, and on horizontal boxes 87.02% vs 81.78%; both are significant.
4. **The original default decoding hurts on manga.** The original manga-ocr package uses `no_repeat_ngram_size=3`. On the 301 boxes whose label contains a repeated 3-character fragment, EM is 21.59% with the default settings and 68.77% with greedy decoding.
5. **The original is slow because of its implementation.** With the same network architecture, per-box time at batch 1 drops from 84.9 ms (original package) to 11.0 ms. Changing only the implementation, with identical outputs, also makes two other models 2.6× faster: Hayai v2.5 Nova 54.3 → 21.2 ms, PaddleOCR-VL-For-Manga 365 → 142 ms. The code is in [`accel/`](accel/).

See the [technical report](TECHNICAL_REPORT.md) for the data, tests and limitations behind each finding.

## Accuracy

All 7,789 boxes of JMangaBench_Read v1.3, RIN v1 normalisation, batch 1, RTX 2080 Ti. See the benchmark's documentation for the model settings and scoring.

| Model | Params | Score ↑ | CER % ↓ | EM % ↑ | EM printed | EM non-printed |
|---|---|---|---|---|---|---|
| **CPT-flex** | 111M | **97.45** | **1.53** | **91.42** | 94.16 | **83.00** |
| CPT-224 | 111M | 97.43 | 1.77 | 91.00 | 93.82 | 82.32 |
| mangaOCR (original, package defaults) | 111M | 90.23 | 4.84 | 78.25 | 88.04 | 48.17 |
| mangaOCR (original, greedy) | 111M | 89.82 | 6.16 | 81.29 | 91.87 | 48.80 |
| manga-ocr-nar-small (our non-autoregressive model) | 54M | 97.26 | 1.73 | 91.09 | **94.44** | 80.81 |
| Hayai v2.5 Nova p384 | 157M | 96.62 | 2.40 | 88.00 | 91.70 | 76.62 |
| PaddleOCR-VL-For-Manga | 959M | 94.98 | 2.03 | 87.52 | 94.78 | 65.22 |

- The original is shown with two decoding settings: the package defaults (4-beam search with `no_repeat_ngram_size=3`) and greedy decoding. Both run in fp16 on the GPU.
- Paired per box: 521 boxes are read correctly only by CPT-flex and 254 only by Hayai v2.5 Nova p384 ($p = 5\times10^{-22}$).
- CPT-flex is listed on the JMangaBench_Read leaderboard (since 2026-10-16); CPT-224 is not.

## Speed

RTX 2080 Ti, Read v1.3 speed protocol, preprocessing included. For page latency a page is one batch; a page has 8.7 boxes on average.

| Inference path | Per crop (batch 1) | Page p50 | Peak throughput |
|---|---|---|---|
| CPT-flex, CUDA path | 11.0 ms | 43.0 ms | 260 crops/s (batch 32) |
| CPT-flex, plain path | 27.6 ms | 71.4 ms | 242 crops/s (batch 80) |
| CPT-224, CUDA path | 10.4 ms | 38.1 ms | 277 crops/s (batch 32) |
| CPT-224, plain path | 27.6 ms | 69.7 ms | 234 crops/s (batch 64) |
| mangaOCR (original manga-ocr package, fp16) | 84.9 ms | 242.4 ms | 42 crops/s (batch 56) |

- **CUDA path**: preprocessing runs on the GPU, and the encoder and decoding steps are replayed from CUDA graphs. Capturing the graphs at start-up takes about 1–3 s.
- **Plain path**: plain PyTorch; runs on CPU, Apple MPS and other devices.
- CPT-224 was timed on 2026-10-15. In the same session, the CPT-flex CUDA path measured 10.9 ms / 42.9 ms / 264 crops/s, within 2% of the table above.
- Both paths use the same weights. On all 7,789 boxes of Read v1.3, the batch-1 outputs of both paths are identical to the evaluated outputs.
- About half of the original package's slowness comes from beam search with no-repeat, and the other half from the overhead of the `transformers` generation loop (technical report §4.2).
- **GPU down-clocking**: after a few idle seconds the GPU clocks down and the next request is slower. On the CPT-flex CUDA path, the first crop after 20 s idle takes 28.1 ms, vs 9.5 ms when running continuously.

## Which model to use

- **For general use, choose CPT-flex.** It is clearly better on elongated and horizontal boxes and ties CPT-224 elsewhere.
- **CPT-224 is mainly a control.** Its input is exactly that of the original mangaOCR, which makes it the right model for comparing the effect of changing only the data.

## Installation and usage

```bash
pip install git+https://github.com/muscgab/manga-ocr-CPT
```

```python
from manga_ocr_cpt import MangaOcrCPT

ocr = MangaOcrCPT("flex")                # or "224"; downloads the weights from Hugging Face on first use (about 444 MB)
ocr("crop.png")                          # one crop -> str
ocr(["crop1.png", "crop2.png"])          # the crops of one page -> list[str], run as one batch
```

- **Input**: file paths, PIL images or uint8 RGB arrays. Inputs must be cropped text boxes; this package does no text detection.
- **Device**: with an NVIDIA GPU the CUDA path is used automatically; otherwise the plain path. Choose a device with `device="cpu"` or `"mps"`, and force the plain path with `backend="plain"`.
- **fp16 decoder**: `decoder="fp16"` runs the decoder's matrix products in fp16 (CUDA only), about 1 ms faster per crop. For CPT-flex it changes 2 outputs on Read v1.3 without changing the scores. The default `"fp32"` reproduces the evaluated outputs exactly.
- **Downloading from ModelScope**: download first, then pass the local directory:

```bash
modelscope download --model muscgab/manga-ocr-base-CPT-flex --local_dir manga-ocr-base-CPT-flex
```

```python
ocr = MangaOcrCPT("manga-ocr-base-CPT-flex")    # local directory
```

Command line:

```bash
manga-ocr-cpt crop1.png crop2.png               # one "path<TAB>text" line per crop; --model 224 switches model
```

## Differences from the original manga-ocr

- **Vocabulary**: replaced by 6,817 characters. Hangul and some simplified Chinese characters of the original vocabulary are removed. The weights therefore cannot be used in the original manga-ocr package.
- **Length**: at most 64 characters per box; longer boxes must be split first.
- **Decoding**: greedy only; no beam search and no repetition blocking.
- **Inference code**: re-implemented; does not depend on the original manga-ocr package or `transformers`.

## Limitations

- The benchmark JMangaBench_Read was built by the author of this repository and consists mainly of works published online. We did not evaluate on scanned manga such as Manga109-s; the original was trained on Manga109-s, so a comparison there would not be fair either.
- Parts of Read v1.2 and v1.3 were used during development. The released weights are from the last training step and were not selected on this basis.
- Each model was trained once.
- Long boxes of more than 32 characters remain clearly behind: both CPT models score 71%–73% EM there, PaddleOCR-VL-For-Manga 82.84%.
- Handwriting and stylised lettering produce clearly more errors than printed text.
- Aimed at dialogue, narration and small annotations; onomatopoeia is not a target.
- The training data and the synthetic renderer are not released.

## Files

| File | Contents |
|---|---|
| `manga_ocr_cpt/` | Inference code: network, CPU / GPU preprocessing, CUDA-graph runner, command line |
| `accel/` | Accelerated inference for Hayai v2.5 Nova and PaddleOCR-VL-For-Manga (with their original weights), and a script that compares them box by box with the originals |
| `TECHNICAL_REPORT.md` / `TECHNICAL_REPORT_zh.md` | Technical report (English / Chinese) |
| `samples/` | 8 example crops from JMangaBench_Read v1.3 |
| `NOTICE.md` | Provenance and acknowledgements |
| `LICENSE` | Apache License 2.0 (code) |

This repository contains no weights. The weights are in the two model repositories on Hugging Face and ModelScope.

## License

- **Code**: Apache License 2.0.
- **Model weights**: CC BY-NC-SA 4.0; see each model repository. The training data AnimeText-OCR is CC BY-NC-SA 4.0.
- The weights are initialised from kha-white/manga-ocr-base (Apache License 2.0). See `NOTICE.md` for provenance and acknowledgements.
