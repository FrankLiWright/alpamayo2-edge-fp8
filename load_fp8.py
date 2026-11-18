# SPDX-License-Identifier: Apache-2.0
"""Load a FP8 weight-only Alpamayo 2 Edge checkpoint produced by quantize_fp8.py."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from fp8_quant import SCHEME, swap_to_fp8


def _classes_defining_initialize_weights():
    from transformers import PreTrainedModel

    seen: list[type] = []
    stack = [PreTrainedModel]
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return [cls for cls in seen if "initialize_weights" in cls.__dict__]


def _build_skeleton(config, fast: bool = True):
    """Construct the official model, optionally skipping the random weight init.

    ``initialize_weights`` randomly fills 2.65B parameters and dominates load
    time. Every parameter is overwritten by the checkpoint anyway, so
    suppressing it is safe. The non-persistent derived buffers (rotary
    ``inv_freq``, Fourier tables, action-space scalars) are computed in each
    module's ``__init__``, not by ``initialize_weights``, so eager construction
    keeps them correct -- which is why this does not build on meta device.
    """
    from alpamayo2_edge.models.alpamayo2_edge import Alpamayo2Edge

    if not fast:
        return Alpamayo2Edge(config)

    patched = [(cls, cls.initialize_weights) for cls in _classes_defining_initialize_weights()]
    if not patched:
        return Alpamayo2Edge(config)
    for cls, _ in patched:
        cls.initialize_weights = lambda self: None  # type: ignore[method-assign]
    try:
        model = Alpamayo2Edge(config)
    finally:
        for cls, fn in patched:
            cls.initialize_weights = fn  # type: ignore[method-assign]
    return model


def _assert_derived_buffers_ok(model) -> list[str]:
    """Every non-persistent buffer must hold finite values (they come from config)."""
    bad = []
    for name, buf in model.named_buffers():
        if buf is None or buf.numel() == 0 or buf.device.type == "meta":
            continue
        if not torch.isfinite(buf.float()).all():
            bad.append(name)
    return bad


def load_fp8_model(
    quant_dir: str | Path,
    device: str = "cuda",
    dtype: str = "bfloat16",
    attn_implementation: str = "sdpa",
    fast_build: bool = True,
):
    """Build the official model skeleton, swap in FP8 modules and load the weights.

    Returns ``(model, info)`` where ``info`` reports what was swapped and loaded.
    """
    quant_dir = Path(quant_dir).expanduser().resolve()
    quant_cfg_path = quant_dir / "quant_config.json"
    if not quant_cfg_path.is_file():
        raise FileNotFoundError(f"Not an FP8 checkpoint (missing quant_config.json): {quant_dir}")
    quant_cfg = json.loads(quant_cfg_path.read_text(encoding="utf-8"))
    if quant_cfg.get("quantization_scheme") != SCHEME:
        raise ValueError(f"Unsupported quantization scheme: {quant_cfg.get('quantization_scheme')!r}")

    from alpamayo2_edge.config import Alpamayo2EdgeConfig, normalize_checkpoint_config

    raw = json.loads((quant_dir / "config.json").read_text(encoding="utf-8"))
    config = Alpamayo2EdgeConfig(**normalize_checkpoint_config(raw))
    config._name_or_path = str(quant_dir)
    config.set_dtype(dtype)
    config._attn_implementation = attn_implementation

    t0 = time.time()
    model = _build_skeleton(config, fast=fast_build)
    t_skel = time.time() - t0

    t0 = time.time()
    swapped = swap_to_fp8(model, structural=True, compute_dtype=getattr(torch, dtype))
    t_swap = time.time() - t0

    t0 = time.time()
    state = load_file(str(quant_dir / "quant.safetensors"))
    t_read = time.time() - t0

    t0 = time.time()
    model.load_state_dict(state, strict=True, assign=True)
    del state
    t_load = time.time() - t0

    bad = _assert_derived_buffers_ok(model)
    if bad:
        raise RuntimeError(f"derived buffers are non-finite after load: {bad}")

    model.eval()
    t_to = 0.0
    if device:
        t0 = time.time()
        model.to(device)
        t_to = time.time() - t0

    n_q = sum(v.numel() for v in model.state_dict().values() if v.dtype == torch.float8_e4m3fn)
    info = {
        "quantization_scheme": quant_cfg["quantization_scheme"],
        "swapped_modules": len(swapped),
        "expected_swapped_modules": quant_cfg.get("n_quantized_modules"),
        "fp8_elements": n_q,
        "fast_build": fast_build,
        "timing_s": {
            "skeleton": round(t_skel, 2),
            "swap": round(t_swap, 2),
            "read_safetensors": round(t_read, 2),
            "load_state_dict": round(t_load, 2),
            "to_device": round(t_to, 2),
        },
    }
    if len(swapped) != quant_cfg.get("n_quantized_modules"):
        raise RuntimeError(
            f"Swapped {len(swapped)} modules, checkpoint says "
            f"{quant_cfg.get('n_quantized_modules')}"
        )
    return model, info
