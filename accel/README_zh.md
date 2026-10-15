# accel：另外两个漫画 OCR 模型的加速推理

[English](README.md) | [中文](README_zh.md)

这是[技术报告](../TECHNICAL_REPORT_zh.md)第 5 条发现的论据：自回归漫画 OCR 模型慢，主要是实现的问题。这里的脚本运行两个现有模型，**原版权重、贪心解码、生成长度上限、预处理参数和 fp16 精度都不变**，只重写推理流程（CUDA 图、静态 KV 缓存、GPU 预处理），不使用 torch.compile。

| 模型 | 原版（batch 1，均值 / p50） | 加速版 | 与原版输出逐字相同 |
|---|---|---|---|
| [Hayai v2.5 Nova](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.5-nova) p384 | 54.29 / 52.43 ms | 21.22 / 18.29 ms | 7,789/7,789 |
| [PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga)，exact | 365.0 / 308.4 ms | 141.7 / 120.2 ms | 7,789/7,789 |
| PaddleOCR-VL-For-Manga，bucket | 365.0 / 308.4 ms | 101.5 / 93.4 ms | 7,783/7,789（分数不变） |

RTX 2080 Ti，fp16，[JMangaBench_Read v1.3](https://github.com/muscgab/JMangaBench_Read) 全部 7,789 框，含预处理，前 20 框热身不计时；原版在同一张卡上重测。细节、整页数字和代价见报告 §4.3。

## 文件

| 文件 | 内容 |
|---|---|
| `hayai_nova/nova_fast.py` | `NovaFast`：GPU 预处理；每种输入形状把视觉编码、投影和 prefill 录成一张 CUDA 图；解码逐步回放 CUDA 图，每步检查是否结束 |
| `paddleocr_vl_manga/fast_paddle.py` | `FastPaddle`：不用 HF `generate`；静态 KV 缓存；每个解码步拆成 19 段 CUDA 图、注意力按精确长度计算（`exact`），或按 KV 长度每 128 分一档、整步一张图（`bucket`） |
| `paddleocr_vl_manga/gpu_resize.py` | 在 GPU 上逐位复现 Pillow 的 8 位双三次缩放 |
| `compare.py` | 在一个裁图文件夹上分别跑原版和加速版，报告输出相同的框数和每框延迟 |

## 使用

先下载原版模型（Hugging Face 或镜像）。Hayai Nova 测试时用的是 `2803d0dc` 版本、transformers 4.57；PaddleOCR-VL-For-Manga 用的是 transformers 4.55.4。两者都需要 NVIDIA 显卡。

```bash
python compare.py nova   --model-dir hayai-ocr-v2.5-nova --images crops/ --n 100
python compare.py paddle --model-dir PaddleOCR-VL-For-Manga --images crops/ --n 100            # 加 --bucket 用 bucket 模式
```

```python
import sys; sys.path.insert(0, "accel/hayai_nova")
from nova_fast import NovaFast
ocr = NovaFast("hayai-ocr-v2.5-nova", max_num_patches=384)
ocr([pil_image])                                       # -> [text]

sys.path.insert(0, "accel/paddleocr_vl_manga")
from fast_paddle import load
ocr = load("PaddleOCR-VL-For-Manga")                   # bucket 模式：load(..., decode_mode="bucket", chunk=4)
ocr.ocr(pil_image)                                     # -> text
```

## 注意

- **Nova 每遇到一种新的输入形状，要临时录一张图**（约 0.1 秒；Read v1.3 上共 41 种，其中 31 框因此多花 35–40 ms）。录过的图会一直保留，所以显存随形状种类增加（整页推理从 925 MB 增到 1,560 MB）。输出越长，提速越小（65 字以上：171 → 148 ms）。
- **Paddle 的 `bucket` 模式不逐位一致**（7,789 框中 6 框输出不同，多为省略号写法）。`exact` 模式逐位一致。
- 只在一种显卡（RTX 2080 Ti）上测过。主要针对 batch 1；批量推理只做了报告里写到的那些优化。
- 这些是研究用脚本，不是安装包。与两个模型的作者没有关联，也未经其认可。

## 许可

与本仓库其余部分相同，采用 Apache License 2.0。两个上游模型都以 Apache License 2.0 发布；脚本调用模型自带的代码（`trust_remote_code`），并用相同的运算重新实现了其中一部分。见 `../NOTICE.md`。
