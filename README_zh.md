# manga-ocr-CPT

[English](README.md) | [中文](README_zh.md)

这是一个试验性仓库。我们没有改动 [mangaOCR](https://github.com/kha-white/manga-ocr)（`kha-white/manga-ocr-base`）的结构，只换了词表，在新的训练数据上做了继续预训练（continued pre-training，CPT）。

这个仓库最主要的内容是实验结论，完整内容见[技术报告](TECHNICAL_REPORT_zh.md)。仓库同时提供两个模型和它们的推理代码。

| 模型 | 输入 | 权重 |
|---|---|---|
| **manga-ocr-base-CPT-flex**（推荐） | 保持长宽比，最多 196 个 patch | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-base-CPT-flex) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-base-CPT-flex) |
| manga-ocr-base-CPT | 拉伸到 224 × 224，与原版 mangaOCR 完全相同 | [Hugging Face](https://huggingface.co/muscgab/manga-ocr-base-CPT) · [ModelScope](https://modelscope.cn/models/muscgab/manga-ocr-base-CPT) |

两个模型的结构、参数量（1.11 亿）和训练设置都相同，只有输入方式不同。下文简称 CPT-flex 和 CPT-224。

## 主要发现

1. **训练数据是主要因素。** 结构、参数量和 224 × 224 输入都与原版相同，只换训练数据和词表，[JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read) 的 EM 从 81.29% 升到 91.00%（两者都用贪心解码，$p = 4\times10^{-110}$）。
2. **2 层的自回归解码器就够用。** 在相同数据下，CPT-flex 与我们 5,400 万参数的非自回归模型 manga-ocr-nar-small 持平（EM 91.42% 对 91.09%，$p = 0.24$）。
3. **保持长宽比只在极端形状上有用。** 总体只多 0.42 个 EM 百分点，差异不显著。但在长宽比 ≥ 8 的框上是 85.40% 对 76.64%，在横排框上是 87.02% 对 81.78%，这两处都显著。
4. **原版的默认解码设置在漫画上有害。** 原版 manga-ocr 包使用 `no_repeat_ngram_size=3`。标签中含有重复三字片段的 301 框上，默认设置的 EM 是 21.59%，贪心解码是 68.77%。
5. **原版慢，是实现的问题。** 网络结构相同，batch 1 时每框耗时从原版包的 84.9 ms 降到 11.0 ms。只换实现、输出逐字不变，另外两个模型也快了 2.6 倍：Hayai v2.5 Nova 54.3 → 21.2 ms，PaddleOCR-VL-For-Manga 365 → 142 ms。代码在 [`accel/`](accel/)。

每一条的数据、检验方法和局限见[技术报告](TECHNICAL_REPORT_zh.md)。

## 准确率

JMangaBench_Read v1.3 全部 7,789 框，RIN v1 归一化，batch 1，RTX 2080 Ti。各模型的定义与评分方法见该评测集的说明。

| 模型 | 参数 | 综合分 ↑ | CER % ↓ | EM % ↑ | EM 印刷 | EM 非印刷 |
|---|---|---|---|---|---|---|
| **CPT-flex** | 111M | **97.45** | **1.53** | **91.42** | 94.16 | **83.00** |
| CPT-224 | 111M | 97.43 | 1.77 | 91.00 | 93.82 | 82.32 |
| mangaOCR（原版，包默认设置） | 111M | 90.23 | 4.84 | 78.25 | 88.04 | 48.17 |
| mangaOCR（原版，贪心） | 111M | 89.82 | 6.16 | 81.29 | 91.87 | 48.80 |
| manga-ocr-nar-small（我们的非自回归模型） | 54M | 97.26 | 1.73 | 91.09 | **94.44** | 80.81 |
| Hayai v2.5 Nova p384 | 157M | 96.62 | 2.40 | 88.00 | 91.70 | 76.62 |
| PaddleOCR-VL-For-Manga | 959M | 94.98 | 2.03 | 87.52 | 94.78 | 65.22 |

- 原版用了两种解码方式：包默认设置是 4 路束搜索加 `no_repeat_ngram_size=3`；另一种是贪心解码。两者都是 GPU fp16。
- 逐框配对：CPT-flex 只有自己读对的 521 框，Hayai v2.5 Nova p384 只有自己读对的 254 框（$p = 5\times10^{-22}$）。
- 两个 CPT 模型的分数没有写进 JMangaBench_Read 的成绩表。

## 速度

RTX 2080 Ti，按 Read v1.3 的测速方法，包含预处理。整页时一页作为一批，平均每页 8.7 框。

| 推理路径 | 逐张（batch 1） | 整页 p50 | 最大吞吐 |
|---|---|---|---|
| CPT-flex，CUDA 路径 | 11.0 ms | 43.0 ms | 260 张/秒（batch 32） |
| CPT-flex，普通路径 | 27.6 ms | 71.4 ms | 242 张/秒（batch 80） |
| CPT-224，CUDA 路径 | 10.4 ms | 38.1 ms | 277 张/秒（batch 32） |
| CPT-224，普通路径 | 27.6 ms | 69.7 ms | 234 张/秒（batch 64） |
| mangaOCR（原版 manga-ocr 包，fp16） | 84.9 ms | 242.4 ms | 42 张/秒（batch 56） |

- **CUDA 路径**：预处理放在 GPU 上，编码器和解码步用 CUDA 图回放。启动时录图约需 1–3 秒。
- **普通路径**：纯 PyTorch，可在 CPU、Apple MPS 和其他设备上运行。
- CPT-224 的速度测于 2026-10-15。同一次测量中，CPT-flex 的 CUDA 路径为 10.9 ms / 42.9 ms / 264 张/秒，与上表相差不到 2%。
- 两条路径用同一份权重。在 Read v1.3 全部 7,789 框上，两条路径的 batch 1 输出与评测输出完全相同。
- 原版包慢，约一半来自束搜索和禁止重复，另一半来自 `transformers` 生成循环的开销（技术报告 §4.2）。
- **GPU 降频**：显卡空闲几秒后会降频，下一次请求会变慢。CPT-flex 的 CUDA 路径，空闲 20 秒后第一框要 28.1 ms，连续处理时是 9.5 ms。

## 怎么选

- **一般用途选 CPT-flex。** 它在细长框和横排框上明显更好，其余情况与 CPT-224 持平。
- **CPT-224 主要作对照用。** 它的输入与原版 mangaOCR 完全相同，适合比较“只换数据”的效果。

## 安装与使用

```bash
pip install git+https://github.com/muscgab/manga-ocr-CPT
```

```python
from manga_ocr_cpt import MangaOcrCPT

ocr = MangaOcrCPT("flex")                # 或 "224"；首次使用时从 Hugging Face 下载权重（约 444 MB）
ocr("crop.png")                          # 一张裁图 -> str
ocr(["crop1.png", "crop2.png"])          # 一页的裁图 -> list[str]，作为一批推理
```

- **输入**：文件路径、PIL 图像或 uint8 RGB 数组。输入必须是裁好的文字框；本包不做文字检测。
- **设备**：有 NVIDIA 显卡时自动走 CUDA 路径，否则走普通路径。可以用 `device="cpu"` 或 `"mps"` 指定设备，用 `backend="plain"` 强制走普通路径。
- **fp16 解码器**：`decoder="fp16"` 让解码器的矩阵乘法用 fp16，只在 CUDA 上有效，每框约快 1 ms。CPT-flex 在 Read v1.3 上有 2 框输出改变，分数不变。默认的 `"fp32"` 与评测输出完全一致。
- **从 ModelScope 下载**：先下载，再传入本地目录：

```bash
modelscope download --model muscgab/manga-ocr-base-CPT-flex --local_dir manga-ocr-base-CPT-flex
```

```python
ocr = MangaOcrCPT("manga-ocr-base-CPT-flex")    # 本地目录
```

命令行：

```bash
manga-ocr-cpt crop1.png crop2.png               # 每行输出“路径<TAB>文字”；--model 224 换模型
```

## 与原版 manga-ocr 的区别

- **词表**：换成 6,817 字。原词表中的韩文字母和部分简体字已移除。因此权重不能放进原版 manga-ocr 包使用。
- **长度**：每框最多 64 字，更长的框需要先切开。
- **解码**：只用贪心解码，不用束搜索，也不禁止重复。
- **推理代码**：重新实现，不依赖原版 manga-ocr 包或 `transformers`。

## 局限

- 评测集 JMangaBench_Read 由本仓库作者制作，网络发布的作品占主体。我们没有在 Manga109-s 这类扫描漫画上评测；原版训练过 Manga109-s，在它上面比较也不公平。
- 开发中用过 Read v1.2 和 v1.3 的部分框；发布的权重是训练最后一步的权重，没有据此挑选。
- 每个模型只训练了一次。
- 超过 32 字的长框仍明显落后：两个 CPT 模型的 EM 在 71%–73%，PaddleOCR-VL-For-Manga 是 82.84%。
- 手写和美术字的错误明显多于印刷体。
- 面向对白、旁白和小字注释，拟声词不是目标。
- 训练数据和合成渲染器不公开。

## 文件

| 文件 | 内容 |
|---|---|
| `manga_ocr_cpt/` | 推理代码：网络、CPU / GPU 预处理、CUDA 图运行器、命令行 |
| `accel/` | Hayai v2.5 Nova 与 PaddleOCR-VL-For-Manga 的加速推理代码（使用各自的原版权重），以及与原版逐框对比的脚本 |
| `TECHNICAL_REPORT_zh.md` / `TECHNICAL_REPORT.md` | 技术报告（中文 / 英文） |
| `samples/` | 取自 JMangaBench_Read v1.3 的 8 张示例裁图 |
| `NOTICE.md` | 来源与致谢 |
| `LICENSE` | Apache License 2.0（代码） |

本仓库不含权重。权重在 Hugging Face 和 ModelScope 的两个模型仓库中。

## 许可

- **代码**：Apache License 2.0。
- **模型权重**：CC BY-NC-SA 4.0，见各模型仓库。训练数据 AnimeText-OCR 为 CC BY-NC-SA 4.0。
- 权重由 kha-white/manga-ocr-base（Apache License 2.0）初始化。来源与致谢见 `NOTICE.md`。
