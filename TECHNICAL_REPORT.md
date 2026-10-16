# manga-ocr-CPT technical report

[English](TECHNICAL_REPORT.md) | [中文](TECHNICAL_REPORT_zh.md)

Date: 2026-10-15

We kept the architecture of [mangaOCR](https://github.com/kha-white/manga-ocr) (`kha-white/manga-ocr-base`) unchanged, replaced only its vocabulary, and continued pre-training it (CPT) on new training data.

We release two models. The only difference between them is the input:

| Name | Input | Purpose |
|---|---|---|
| `manga-ocr-base-CPT` | Stretched to 224 × 224, pixel-identical to the original `ViTImageProcessor` | Control with exactly the original mangaOCR input |
| `manga-ocr-base-CPT-flex` | Aspect ratio kept, at most 196 patches of 16 px, position embeddings interpolated to the patch grid | Recommended |

Below they are called **CPT-224** and **CPT-flex**.

## Summary

1. **Training data is the main factor.** With the original architecture, parameter count (111M) and 224 × 224 input, and only the training data and vocabulary replaced, EM on Read v1.3 rises from 81.29% to 91.00%. Both use greedy decoding; paired per box, 994 boxes are read correctly only by CPT-224 and 238 only by the original, $p = 4\times10^{-110}$. The vocabulary change can explain at most 0.17 percentage points (§3.1).
2. **With the same data, architecture matters little.** CPT-flex is a 111M autoregressive model with only 2 decoder layers. Our non-autoregressive manga-ocr-nar-small has 54M parameters and was trained on the same data with the same number of samples. EM is 91.42% vs 91.09%, not significantly different ($p = 0.24$).
3. **Keeping the aspect ratio helps only on extreme shapes.** CPT-flex beats CPT-224 by 0.42 EM points overall, which is not significant under the rule fixed before training ($p = 0.087$). It is significant only on boxes with aspect ratio ≥ 8 (85.40% vs 76.64%, $p = 0.0075$) and horizontal boxes (87.02% vs 81.78%, $p = 0.0014$).
4. **The original default decoding hurts on manga.** The original manga-ocr package uses `no_repeat_ngram_size=3`. On the 301 boxes whose label contains a repeated 3-character fragment, EM is 21.59% with the default settings and 68.77% with greedy decoding.
5. **The original is slow because of its implementation, not its network.** Per box at batch 1 on an RTX 2080 Ti: 84.9 ms for the original package, 27.6 ms for our plain PyTorch implementation, and 11.0 ms with CUDA graphs and GPU preprocessing. All three run the same architecture; the last two give identical outputs on all 7,789 boxes. The same approach applied to two other models keeps their outputs identical: Hayai v2.5 Nova goes from 54.3 ms to 21.2 ms, PaddleOCR-VL-For-Manga from 365 ms to 142 ms (§4.3).
6. **Cutting parameters in a small model buys little end-to-end latency.** The parameter counts differ by about 2×. Model-only time differs by about 1.9× between manga-ocr-nar-small and CPT-flex, but end to end it is 9.1 ms vs 11.0 ms: fixed overhead dilutes the gap (§4.4).
7. **Long boxes remain a shared weakness.** On the 169 boxes longer than 32 characters, both CPT models and manga-ocr-nar-small (all trained on texts of at most 64 characters) score 71%–74% EM. PaddleOCR-VL-For-Manga scores 82.84% on this subset, clearly higher.

## 1. Model

- **Architecture**: identical to the original. ViT-B/16 encoder (12 layers, width 768, initialised from DeiT-base) and a 2-layer BERT decoder (width 768, post-LN, GELU, absolute positions, cross-attention); autoregressive; output layer tied to the input embedding. 110,935,974 parameters (excluding the unused ViT pooler).
- **Vocabulary**: replaced by our 6,817 characters plus 5 special tokens. The 5,133 characters already in the original vocabulary keep their embeddings and output biases; new characters are initialised from $\mathcal{N}(0, 0.02^2)$. Hangul and some simplified Chinese characters of the original vocabulary are removed.
- **Decoding**: greedy, at most 65 tokens (64 characters plus the end token).

**Input**:

- CPT-224: greyscale, then stretched to 224 × 224, i.e. 14 × 14 = 196 patches. This is pixel-identical to the original `ViTImageProcessor` (checked on 52 boxes, maximum difference 0.0).
- CPT-flex: greyscale, then resized bilinearly with the aspect ratio kept into the largest grid $g_h \times g_w$ of at most 196 patches. The rest of the grid is filled with the median border colour, and padded patches are masked in encoder self-attention and decoder cross-attention. The 14 × 14 position table is bicubically interpolated to each image's grid. On a full 14 × 14 grid, this encoder path matches HF ViT to within $3.8\times10^{-6}$.

Both use a budget of 196 patches, i.e. the same number of pixels; they differ only in whether the crop is stretched.

## 2. Training

### 2.1 Data

Identical to manga-ocr-nar-small, 4,471,985 images:

| Source | Images | Share |
|---|---|---|
| Synthetic printed text (project renderer) | 1,795,869 | 40.2% |
| Synthetic handwriting (project renderer) | 1,670,000 | 37.3% |
| [AnimeText-OCR](https://huggingface.co/datasets/JustANormalTinkerer/animetext-ocr) (real crops, sequence confidence 0.90–0.98) | 1,006,116 | 22.5% |

- Labels are 1–64 characters long; only in-vocabulary characters are kept.
- One round of manual review removed 603 samples and corrected 1,398.
- Several distribution parameters of the synthetic data follow summary statistics of the JMangaBench_Read candidate pool (see §7).
- How the synthetic handwriting is made: [manga-ocr-nar/docs/synthetic_handwriting.md](https://github.com/muscgab/manga-ocr-nar/blob/main/docs/synthetic_handwriting.md).
- No image from the Read candidate pool or benchmark was used for training.
- The original mangaOCR model card lists Manga109-s as training data. Our data contains no Manga109-s.

### 2.2 Settings

Both models use identical settings except for the input.

| Item | Setting |
|---|---|
| Initialisation | `kha-white/manga-ocr-base` |
| Steps | 139,749 steps, global batch 256 (4 GPUs × 64), 8 epochs, about 35.78M samples |
| Optimiser | AdamW, $\beta = (0.9, 0.98)$; weight decay 0.01 on matrices only; gradient clipping 1.0 |
| Learning rate | Decoder (incl. embeddings) $5\times10^{-4}$, 2,000 warm-up steps; encoder $1\times10^{-4}$, held at 0 for the first 1,000 steps, then 2,000 warm-up steps; both cosine to 0 |
| Loss | Token cross-entropy, teacher forcing, no label smoothing |
| Precision | Encoder fp16 mixed precision, decoder fp32; GradScaler; DDP gradient communication compressed to fp16 |
| Hardware | 4 × RTX 2080 Ti. CPT-flex about 611 images/s, CPT-224 about 695 images/s. CPT-224 took about 14.4 hours |
| Released weights | Last step (fixed in advance; no checkpoint selection) |

**CPT-flex trained about 4.4% longer.** At step 126,200 a false alarm of the divergence guard stopped CPT-flex (see §6); it was resumed from the step-120,000 checkpoint with a new data seed. It therefore ran 6,200 more steps than CPT-224, about 1.59M samples.

### 2.3 Validation during training

Three validation shards, the first 2,000 samples of each; greedy EM (%). AT = AnimeText-OCR validation, OC = synthetic printed, HW = synthetic handwriting.

| Step | flex AT | 224 AT | flex OC | 224 OC | flex HW | 224 HW |
|---|---|---|---|---|---|---|
| 10,000 | 88.75 | 89.30 | 89.00 | 90.20 | 81.95 | 82.35 |
| 20,000 | 92.70 | 91.00 | 93.60 | 93.30 | 88.10 | 87.65 |
| 40,000 | 94.25 | 94.50 | 95.20 | 95.25 | 92.30 | 91.15 |
| 70,000 | 96.85 | 95.70 | 97.20 | 96.50 | 95.95 | 94.85 |
| 100,000 | 98.20 | 96.95 | 98.20 | 97.95 | 97.35 | 97.00 |
| 139,749 | 98.50 | 97.85 | 98.55 | 98.45 | 98.50 | 97.60 |

- CPT-224 sees exactly the input of the original pre-training, so it starts faster: it is higher on all three sets at 10k steps, and its loss at step 100 is 1.20 vs 1.68 for CPT-flex.
- From 60k steps on, CPT-flex leads on all three sets.

## 3. Accuracy

### 3.1 Evaluation protocol

- **Benchmark**: [JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read), 7,789 text boxes, RIN v1 normalisation.
- **Metrics**:
  - CER = total edit distance / total label characters;
  - EM = share of boxes matched exactly;
  - Score = equal-weight mean of $1-\mathrm{NED}$ over the printed and non-printed parts:

$$s_i = 1 - \frac{\mathrm{ED}(p_i, g_i)}{\max(|p_i|, |g_i|)}, \qquad \text{Score} = 100 \times \tfrac{1}{2}\left(\overline{s}_{\text{printed}} + \overline{s}_{\text{non-printed}}\right)$$

- **Paired test**: on the same boxes, let $b$ be the number read correctly only by A and $c$ only by B; exact two-sided McNemar test:

$$p = \min\!\left(1,\; 2\sum_{i=0}^{\min(b,c)} \binom{b+c}{i} 2^{-(b+c)}\right)$$

- **Run paths**:
  - Main CPT-flex results: GPU, encoder fp16, decoder fp32, batch 1.
  - Both CPT models are run this way (GPU, batch 1).
  - Both were also run on CPU in fp32 at batch 32, with box-by-box EM identical to GPU batch 1; the text is identical on 7,789/7,789 boxes for CPT-224 and 7,788/7,789 for CPT-flex. The 70k-checkpoint comparison (§3.3) uses the CPU path.
- **Two results for the original mangaOCR**:
  - The leaderboard row uses the original manga-ocr package defaults: 4-beam search, `no_repeat_ngram_size=3`, GPU fp16, batch 1.
  - The greedy row uses the original weights, also on GPU in fp16 at batch 1; only the decoding is changed to greedy.
- **Vocabulary coverage**: only 13 boxes in Read v1.3 (0.17%) contain characters outside the original vocabulary (ellipses count as covered under RIN). The vocabulary change itself can therefore directly explain at most 0.17 EM points.

### 3.2 Main results

All 7,789 boxes of Read v1.3. Score, CER and EM are over the full set; the last five columns are EM (%) on subsets, whose sizes are given below the table.

| Model | Params | Score | CER % | EM % | Printed | Non-printed | >32 chars | Horizontal | Aspect ≥ 8 |
|---|---|---|---|---|---|---|---|---|---|
| mangaOCR original (package default: beam 4 + no-repeat) | 111M | 90.23 | 4.84 | 78.25 | 88.04 | 48.17 | 48.52 | 46.24 | 52.55 |
| mangaOCR original (greedy) | 111M | 89.82 | 6.16 | 81.29 | 91.87 | 48.80 | 61.54 | 46.47 | 58.39 |
| **CPT-224** | 111M | 97.43 | 1.77 | 91.00 | 93.82 | 82.32 | 71.01 | 81.78 | 76.64 |
| **CPT-flex** | 111M | **97.45** | **1.53** | **91.42** | 94.16 | **83.00** | 73.37 | **87.02** | **85.40** |
| manga-ocr-nar-small (non-autoregressive) | 54M | 97.26 | 1.73 | 91.09 | 94.44 | 80.81 | 73.96 | 84.51 | 85.40 |
| Hayai v2.5 Nova p384 | 157M | 96.62 | 2.40 | 88.00 | 91.70 | 76.62 | 55.03 | 83.60 | 85.40 |
| PaddleOCR-VL-For-Manga | 959M | 94.98 | 2.03 | 87.52 | **94.78** | 65.22 | **82.84** | 71.53 | 84.67 |

Subset sizes: printed 5,877, non-printed 1,912, >32 characters 169, horizontal 439, aspect ≥ 8 137.

Except for the two CPT rows and the original-greedy row, all rows use the batch-1 predictions of the public Read v1.3 leaderboard, rescored with the same scoring script.

Paired per box (all 7,789 boxes):

| A vs B | Only A correct | Only B correct | $p$ |
|---|---|---|---|
| CPT-224 vs original (greedy) | 994 | 238 | $4\times10^{-110}$ |
| CPT-224 vs original (package default) | 1,209 | 216 | $2\times10^{-167}$ |
| CPT-flex vs original (package default) | 1,245 | 219 | $3\times10^{-174}$ |
| Original (greedy) vs original (package default) | 298 | 61 | $1\times10^{-38}$ |
| CPT-flex vs CPT-224 | 192 | 159 | 0.087 |
| CPT-flex vs manga-ocr-nar-small | 235 | 209 | 0.24 |
| CPT-224 vs manga-ocr-nar-small | 233 | 240 | 0.78 |
| CPT-flex vs Hayai v2.5 Nova p384 | 521 | 254 | $5\times10^{-22}$ |
| CPT-224 vs Hayai v2.5 Nova p384 | 521 | 287 | $2\times10^{-16}$ |
| CPT-flex vs PaddleOCR-VL-For-Manga | 622 | 318 | $2\times10^{-23}$ |
| CPT-224 vs PaddleOCR-VL-For-Manga | 615 | 344 | $2\times10^{-18}$ |

### 3.3 Input: CPT-flex vs CPT-224

Both use the GPU batch-1 results. The decision rule was fixed before training: paired per-box $p < 0.05$ on all boxes, with the subsets aspect ≥ 4, >32 characters and horizontal reported separately.

| Subset | Boxes | EM flex | EM 224 | Only flex correct | Only 224 correct | $p$ |
|---|---|---|---|---|---|---|
| All | 7,789 | 91.42 | 91.00 | 192 | 159 | 0.087 |
| Aspect < 2 | 1,965 | 90.43 | 89.67 | 57 | 42 | 0.16 |
| Aspect 2–4 | 4,279 | 92.19 | 91.73 | 97 | 77 | 0.15 |
| Aspect ≥ 4 | 1,545 | 90.55 | 90.68 | 38 | 40 | 0.91 |
| Aspect ≥ 8 | 137 | 85.40 | 76.64 | 15 | 3 | 0.0075 |
| 17–32 chars | 1,441 | 91.81 | 91.33 | 34 | 27 | 0.44 |
| >32 chars | 169 | 73.37 | 71.01 | 12 | 8 | 0.50 |
| Vertical | 7,158 | 91.79 | 91.69 | 147 | 140 | 0.72 |
| Horizontal | 439 | 87.02 | 81.78 | 36 | 13 | 0.0014 |
| Printed | 5,877 | 94.16 | 93.82 | 85 | 65 | 0.12 |
| Non-printed | 1,912 | 83.00 | 82.32 | 107 | 94 | 0.40 |

- Under the pre-set rule, the overall difference is not significant.
- Stretching costs clearly on two kinds of boxes. One is very elongated boxes: at aspect ≥ 8, stretching changes the glyphs' aspect ratio by a factor of 8 or more. The other is horizontal boxes (92% of Read is vertical). Our guess for why horizontal boxes suffer most is that stretched horizontal glyphs are furthest from most training samples; this is unverified.
- **The gap was larger mid-training.** At the 70k checkpoint, on the same CPU path, EM was 90.61% for CPT-flex and 89.47% for CPT-224 (295 boxes correct only for flex, 206 only for 224, $p = 8.1\times10^{-5}$); after full training the gap narrowed. Our guess is that keeping the aspect ratio mainly makes the model learn faster, and that most of the loss from stretching is learned back with enough training. One seed only; unverified.

### 3.4 The original decoding settings

The original manga-ocr package uses 4-beam search with `no_repeat_ngram_size=3` by default, i.e. a 3-character fragment may appear at most once in the output. Yet repeated fragments are not rare in manga dialogue: 301 boxes (3.9%) have a label containing a repeated 3-character fragment.

| Boxes | Count | Package default EM | Greedy EM |
|---|---|---|---|
| Label contains a repeated 3-character fragment (raw label text) | 301 | 21.59 | 68.77 |
| Others | 7,488 | 80.53 | 81.80 |

- Of the 298 boxes read correctly only by greedy, 143 have a label with a repeated 3-character fragment; of the 61 read correctly only by the default settings, 1 does.
- The price of greedy decoding is runaway output. Greedy produces 17 boxes (4,794 characters in total) whose prediction is longer than twice the label plus 10 characters; the default settings produce none. This is why greedy has the higher CER (6.16% vs 4.84%).
- Both CPT models decode greedily with at most 65 tokens and have no runaway boxes. On the 301 boxes with repeated fragments, EM is 78.07% for CPT-flex and 75.08% for CPT-224.

### 3.5 Long boxes

- Read v1.3 has 169 boxes longer than 32 characters; 8 are longer than 64, and the longest has 161.
- CPT and manga-ocr-nar-small were trained on texts of at most 64 characters, and all score 71%–74% EM on the >32 subset. Autoregressive decoding gives CPT no advantage on long boxes.
- PaddleOCR-VL-For-Manga uses high input resolution and has no 64-character limit. It scores 82.84% EM on this subset, above CPT-flex (24 boxes correct only for Paddle, 8 only for CPT-flex, $p = 0.007$).

## 4. Speed

All timings are on an RTX 2080 Ti under the Read v1.3 speed protocol: preprocessing included; when a page is one batch, a page has 8.7 boxes on average.

### 4.1 Three inference paths

| Inference path (original weights for the original row) | Per crop (batch 1) | Page p50 | Peak throughput |
|---|---|---|---|
| Original manga-ocr package (fp16, beam 4 + no-repeat) | 84.9 ms | 242.4 ms | 42 crops/s (batch 56) |
| CPT-flex, plain PyTorch (greedy, KV cache) | 27.6 ms | 71.4 ms | 242 crops/s (batch 80) |
| CPT-flex, CUDA graphs + GPU preprocessing | 11.0 ms (p50 10.09) | 43.0 ms | 260 crops/s (batch 32) |
| CPT-224, plain PyTorch (greedy, KV cache) | 27.6 ms | 69.7 ms | 234 crops/s (batch 64) |
| CPT-224, CUDA graphs + GPU preprocessing | 10.4 ms (p50 9.44) | 38.1 ms | 277 crops/s (batch 32) |
| Reference: manga-ocr-nar-small | 9.1 ms | 35.5 ms | 299 crops/s (batch 48) |

- For both models, the plain PyTorch and CUDA paths give batch-1 outputs identical to the evaluated outputs on all 7,789 boxes.
- CPT-224 was timed on 2026-10-15, on a different day from CPT-flex. In the same session, the CPT-flex CUDA path measured 10.9 ms / 42.9 ms / 264 crops/s, within 2% of the 10-14 numbers in the table. CPT-224 is slightly faster, possibly because its grid is fixed at 14 × 14 and needs no position-table interpolation or padding (a guess).

### 4.2 Why the original is slow

Every 15th box of Read v1.3, 520 boxes, batch 1, fp16:

| Setting | Time per box | Boxes with the same output as the default |
|---|---|---|
| Package default: beam 4 + no-repeat | preprocessing 3.6 + generation 83.1 ms | — |
| Beam 4, no no-repeat | 76.4 ms | 487 / 520 |
| Greedy (HF `generate`) | 57.0 ms (3.6 ms per token) | 447 / 520 |

- Profiling a single box: the default settings spend 27.9 ms in 6,261 GPU kernels; greedy spends 15.9 ms in 3,200 kernels. The rest is CPU launch and HF `generate` overhead.
- Of the drop from 84.9 ms to 27.6 ms, about half comes from beam search and no-repeat (about 26 ms) and about half from the HF generation loop and module overhead.
- Our accelerated implementation does three things:
  1. the encoder and the cross-attention K/V are captured as one CUDA graph;
  2. greedy decoding uses a static KV cache, with every 4 steps captured as one CUDA graph and a check after each chunk for whether all sequences have ended;
  3. preprocessing runs on the GPU.

  On the 520-box sample, the prototype (preprocessing still on CPU) splits a box into GPU encoder 2.8 ms, decoder 6.4 ms and CPU preprocessing 2.1 ms on average.

### 4.3 Two other models: implementation alone speeds them up too

To check that "slow because of the implementation" is not specific to mangaOCR, we did the same for two other manga OCR models: weights, greedy decoding, maximum generation length, preprocessing parameters and fp16 precision all unchanged; only the inference flow was rewritten, without torch.compile. The code is in [`accel/`](accel/).

Protocol: RTX 2080 Ti, fp16, all 7,789 boxes of Read v1.3, batch 1, preprocessing included, the first 20 boxes not timed. The originals were re-measured on the same GPU.

| Model | Original: mean / p50 | Accelerated: mean / p50 | Outputs identical to the original | Score / EM / CER |
|---|---|---|---|---|
| Hayai v2.5 Nova p384 | 54.29 / 52.43 ms | 21.22 / 18.29 ms (2.6×) | 7,789/7,789 | unchanged |
| PaddleOCR-VL-For-Manga (exact) | 365.0 / 308.4 ms | 141.7 / 120.2 ms (2.6×) | 7,789/7,789 | unchanged |
| PaddleOCR-VL-For-Manga (bucket) | 365.0 / 308.4 ms | 101.5 / 93.4 ms (3.6×) | 7,783/7,789 | 94.99 / 87.52% / 2.03% (original 94.98 / 87.52% / 2.03%) |

**Hayai v2.5 Nova.** The official implementation already uses CUDA graphs, a static KV cache and fused matrix products, but wastes time in three places:
1. **It checks for the end token only every 16 steps.** A box produces 6.3 tokens on average but runs 16.5 steps. Checking after every step brings decoding from 30.1 ms to 11.7 ms.
2. **Vision encoding and prefill are not captured as CUDA graphs.** They launch about 900 kernels. One graph for vision, projector and prefill brings this from 19.8 ms to 6.1 ms.
3. **Only 4 decoding states are cached.** Whole-page inference re-captures graphs whenever the batch size changes. Removing the limit brings page p50 from 254.2 ms to 70.8 ms and peak throughput from 141.5 to 220.9 crops/s.

In addition, moving preprocessing to the GPU brings it from 2.6 ms to 1.4 ms.

The remaining ~21 ms comes mainly from the architecture: the 12-layer decoder runs all 12 layers serially for every generated character (0.13 ms and about 50 kernels per layer per step, 1.8 ms per step), and the visual tokens, as a prefix, first pass through those 12 layers (prefill 2.8–3.4 ms).

Costs:
- A graph is captured the first time each input shape appears. On the 7,789 boxes, 31 boxes take 35–40 ms longer for this reason; 41 graphs are captured in total, 3.8 s overall.
- Higher memory: whole-page inference goes from 925 MB to 1,560 MB.
- The longer the output, the smaller the gain: boxes of more than 65 characters only go from 171.2 ms to 147.5 ms.

**PaddleOCR-VL-For-Manga.** The original has the HF `generate` problem: each box launches 15,000–24,000 kernels and the GPU is busy only 14–19% of the time.

- **exact mode (outputs unchanged):**
  - `generate` replaced by a static KV cache;
  - each decoding step split into 19 CUDA-graph segments, with attention still run at the exact length through the original sdpa call;
  - the PIL bicubic resize reproduced bit for bit on the GPU.
- **bucket mode:** the whole step is one graph per KV-length bucket of 128; 6 outputs change.

On the 7,769 timed boxes, no box is slower than the original in either mode.

Page p50 in bucket mode is 774.1 ms, against the official 1,024.4 ms for the original (not re-measured on the same GPU). The remaining time is mainly in the vision tower (about 31–36 ms, already GPU-bound) and the LLM prefill (about 25 ms).

**Summary:** all three models can be made 2.6–7.7× faster by changing only the implementation, with outputs unchanged (or almost unchanged). How much depends on where the time goes: models dominated by the generation loop and kernel launches (mangaOCR, Nova) gain a lot; a model with a heavy vision tower (Paddle) gains less.

### 4.4 Parameters and latency

- Model only: manga-ocr-nar-small about 4.9 ms (measured earlier with a different script); CPT-flex about 9.2 ms (encoder 2.8 plus decoder 6.4 from the previous section).
- End to end: 9.1 ms vs 11.0 ms. End to end minus model only is about 4.2 ms for manga-ocr-nar-small and about 1.8 ms for CPT-flex; the two use different preprocessing implementations.
- At batch 1 the GPU is idle most of the time: manga-ocr-nar-small runs at about 9.4% GPU utilisation. Latency is set mainly by the number of serial kernels and fixed overhead (preprocessing, copies, launches).
- In the 50M–111M range, model-only time differs by 1.9× but end-to-end time by only about 1.2×. We did not test larger models.

## 5. Conclusions

1. For manga text-box OCR, **training data (including its match to the target domain) matters more than architecture**. The same 2022 architecture moves from the bottom of the leaderboard to the top tier just by changing its data.
2. **A 2-layer autoregressive decoder is enough**: it ties a 4-layer non-autoregressive decoder. Decoder depth is not the bottleneck for this task (on boxes of ≤ 64 characters).
3. **Keeping the aspect ratio is a worthwhile small change**: it avoids clear losses on elongated and horizontal boxes and makes training converge faster (a guess). The gain in the overall average is not significant.
4. **Do not use n-gram no-repeat.** It conflicts directly with repeated text in manga. CPT uses greedy decoding with a 65-token limit and shows no runaway output on Read.
5. **Autoregressive OCR is slow mainly because of decoding settings and the implementation of the generation loop.** Dropping beam search and using CUDA graphs with a static KV cache cuts per-box latency from 84.9 ms (original package) to 11.0 ms; the two paths that change only the implementation (27.6 ms vs 11.0 ms) give identical outputs. Changing only the implementation also makes Hayai v2.5 Nova and PaddleOCR-VL-For-Manga 2.6× faster with identical outputs; what remains is set by their architecture (Nova's 12-layer decoder, Paddle's vision tower).
6. **Long boxes need their own solution.** Models whose training texts are capped at 64 characters all fall behind on long boxes.

## 6. Training notes

**The first attempt diverged.**
- Settings: global batch 128, decoder also in fp16 mixed precision, learning rates as in §2.2.
- What happened: from step 14.7k the loss crept up; the gradient norm rose from about 3 to 282 at step 15.8k, and the loss scale fell from 4,096 to 16; after step 16k the loss rose above 4 and training collapsed.
- At step 20k, the largest attention-query activation in decoder layer 1 was 1,219 (the network-wide maximum was 406 at step 10k). We suspect fp16 attention overflow combined with a high learning rate; the cause is unverified.
- The final runs restarted from the original weights with two changes at once: decoder in fp32 and global batch 256. All 139,749 steps were then stable. Because both changes were made together, their separate effects cannot be told apart.

**False alarm of the divergence guard.**
- The original rule declared divergence when the 500-step mean loss exceeded twice the best value so far.
- This ratio fails when the loss is near 0: at step 126,200 CPT-flex had a 500-step mean of 0.0056 and a best of 0.0026, and was stopped although the gradient norm was normal.
- Fix: an absolute margin was added; the mean must also exceed the best by 0.05 to count as divergence.

**fp16 safeguards.** Throughout training we monitored the following and stopped on any anomaly:
- whether the GradScaler scale hit its floor;
- per-layer maximum activations;
- loss spikes;
- the gradient norm.

## 7. Limitations

- **Benchmark**: JMangaBench_Read was built by the author of this report and consists mainly of works published online. We did not evaluate CPT on scanned manga such as Manga109-s; the original was trained on Manga109-s, so a comparison there would not be fair either. The contributions of "data" and "domain match" in conclusion 1 therefore cannot be separated.
- **Development use**: during development, Read v1.3 was used to check the 70k and 90k checkpoints and for the ablation analysis in this report. No checkpoint was selected on this basis; the released weights are from the last step. The training data design drew on Read v1.2 and the Sol handwriting development set; see Read's [Development use](https://github.com/muscgab/JMangaBench_Read#development-use) section.
- **Seeds**: each model was trained once; seed-to-seed variance is not estimated.
- **Training length**: CPT-flex trained about 4.4% longer than CPT-224 (§2.2).
- **Untested settings**: CPT with beam search (without no-repeat) was not tested.
- **Onomatopoeia**: not a target. On the onomatopoeia set shipped with Read (380 boxes), EM is 80.53% for CPT-flex and 81.58% for CPT-224, for reference only.

## Appendix: reproducibility

- All training settings are listed in §2.2. The training script depends on an unreleased data pipeline and is not part of this repository; the two models differ only in their input (§1).
- Evaluation: Read v1.3 `eval/score.py` (RIN v1); paired tests are computed with the formula in §3.1.
- The training data and the synthetic renderer are not released.
