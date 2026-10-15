"""manga-ocr-cpt [--model flex|224|DIR] [--device DEV] [--backend auto|cuda|plain] [--decoder fp32|fp16] crop.png ...
-> one "path<TAB>text" line per crop; all crops given in one call are run as one batch."""
from __future__ import annotations

import argparse
import sys

from . import MangaOcrCPT


def main(argv=None):
    ap = argparse.ArgumentParser(prog="manga-ocr-cpt")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="flex", help='"flex" (default), "224", a Hugging Face repo id or a local directory')
    ap.add_argument("--device", default=None, help="torch device, e.g. cuda, mps or cpu (default: cuda if available, else cpu)")
    ap.add_argument("--backend", default="auto", choices=("auto", "cuda", "plain"))
    ap.add_argument("--decoder", default="fp32", choices=("fp32", "fp16"))
    a = ap.parse_args(argv)
    ocr = MangaOcrCPT(a.model, a.device, a.backend, a.decoder)
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    for path, text in zip(a.images, ocr(list(a.images))):
        out.write(f"{path}\t{text}\n")


if __name__ == "__main__":
    main()
