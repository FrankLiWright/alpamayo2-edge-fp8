# SPDX-License-Identifier: Apache-2.0
"""Reproduce the FP8 weight-only Alpamayo 2 Edge checkpoint from the BF16 release.

    python quantize_fp8.py --source /path/to/nvidia-Alpamayo2-Edge --target ./Alpamayo2-Edge-FP8

The source directory must be a complete ``nvidia/Alpamayo2-Edge`` snapshot
(config, tokenizer/processor assets and the safetensors shards).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).parent))
from fp8_quant import SCHEME, quantize_model_inplace  # noqa: E402

COPIED = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "generation_config.json",
    "LICENSE",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="nvidia/Alpamayo2-Edge snapshot")
    ap.add_argument("--target", required=True, type=Path, help="output directory")
    ap.add_argument("--min-params", type=int, default=0, help="skip linears below this size")
    args = ap.parse_args()

    src, dst = args.source.expanduser().resolve(), args.target.expanduser().resolve()
    if not (src / "config.json").is_file():
        ap.error(f"source is not a checkpoint: {src}")

    from alpamayo2_edge.checkpoint import load_model

    print(f"source: {src}\ntarget: {dst}", flush=True)
    t = time.time()
    model, _ = load_model(str(src), device="cpu", dtype="bfloat16")
    print(f"loaded BF16 on CPU in {time.time() - t:.1f}s", flush=True)

    before = sum(p.numel() for p in model.parameters())
    t = time.time()
    swapped = quantize_model_inplace(model, min_params=args.min_params)
    print(f"swapped {len(swapped)} modules to FP8 in {time.time() - t:.1f}s", flush=True)

    state = model.state_dict()
    n_fp8 = sum(v.numel() for v in state.values() if v.dtype == torch.float8_e4m3fn)
    bytes_total = sum(v.numel() * v.element_size() for v in state.values())
    print(
        f"{len(state)} tensors | fp8 {n_fp8/1e6:.1f}M elems | {bytes_total/2**30:.2f} GiB "
        f"(was {before*2/2**30:.2f} GiB in BF16)",
        flush=True,
    )

    dst.mkdir(parents=True, exist_ok=True)
    t = time.time()
    save_file(state, str(dst / "quant.safetensors"), metadata={"format": "pt", "quant": SCHEME})
    print(f"saved quant.safetensors in {time.time() - t:.1f}s", flush=True)

    for name in COPIED:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)

    (dst / "quant_config.json").write_text(
        json.dumps(
            {
                "quantization_scheme": SCHEME,
                "quantized_dtype": "float8_e4m3fn",
                "scale_dtype": "float32",
                "granularity": "per_output_channel_linear_per_input_row_embedding",
                "e4m3_max": 448.0,
                "source_checkpoint": "nvidia/Alpamayo2-Edge",
                "quantized_modules": swapped,
                "n_quantized_modules": len(swapped),
                "n_parameters_before": before,
                "compute_dtype": "bfloat16",
                "note": (
                    "Weight-only FP8. Forward dequantizes to the activation dtype, so the "
                    "official alpamayo2_edge model code runs unmodified via load_fp8.py."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    total = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    print(f"checkpoint on disk: {total/2**30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
