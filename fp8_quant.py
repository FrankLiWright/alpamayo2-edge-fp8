# SPDX-License-Identifier: Apache-2.0
"""FP8 (e4m3) weight-only quantization for Alpamayo 2 Edge.

Weights are stored as ``float8_e4m3fn`` with one FP32 scale per output channel
(per input row for embeddings). Norms, biases and hand-written parameters stay
in their checkpoint dtype. ``nn.Linear`` / ``nn.Embedding`` are swapped for
drop-in modules whose ``forward`` dequantizes to the activation dtype, so the
official model code runs unmodified.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

E4M3_MAX = 448.0
SCHEME = "fp8_e4m3_weight_only_per_channel"


def _quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[out, in]`` (or ``[n, d]``) along the last dim, per leading row."""
    w = weight.detach().to(torch.float32)
    scale = w.abs().amax(dim=-1, keepdim=True) / E4M3_MAX
    scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
    q = torch.clamp(w / scale, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


class FP8Linear(nn.Module):
    """``nn.Linear`` with FP8 weights and per-output-channel scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "qweight", torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn)
        )
        self.register_buffer("weight_scale", torch.ones(out_features, 1, dtype=torch.float32))
        self.register_buffer(
            "bias", torch.zeros(out_features, dtype=compute_dtype) if bias else None
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.dequant_weight()

    def dequant_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.compute_dtype
        return self.qweight.to(dtype) * self.weight_scale.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequant_weight(x.dtype), self.bias)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "FP8Linear":
        q, scale = _quantize_rows(lin.weight)
        mod = cls(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            compute_dtype=lin.weight.dtype,
        )
        mod.qweight = q
        mod.weight_scale = scale
        if lin.bias is not None:
            mod.bias = lin.bias.detach().to(lin.weight.dtype)
        return mod

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, quant={SCHEME}"
        )


class FP8Embedding(nn.Module):
    """``nn.Embedding`` with FP8 weights; only the looked-up rows are dequantized."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        compute_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compute_dtype = compute_dtype
        self.register_buffer(
            "qweight", torch.zeros(num_embeddings, embedding_dim, dtype=torch.float8_e4m3fn)
        )
        self.register_buffer("weight_scale", torch.ones(num_embeddings, 1, dtype=torch.float32))

    @property
    def weight(self) -> torch.Tensor:
        return self.qweight.to(self.compute_dtype) * self.weight_scale.to(self.compute_dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        rows = self.qweight[input].to(self.compute_dtype)
        return rows * self.weight_scale[input].to(self.compute_dtype)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding) -> "FP8Embedding":
        q, scale = _quantize_rows(emb.weight)
        mod = cls(emb.num_embeddings, emb.embedding_dim, compute_dtype=emb.weight.dtype)
        mod.qweight = q
        mod.weight_scale = scale
        return mod

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"quant={SCHEME}"
        )


def swap_to_fp8(
    model: nn.Module,
    min_params: int = 0,
    structural: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> list[str]:
    """Replace ``nn.Linear`` / ``nn.Embedding`` in place. Returns swapped names.

    ``structural=True`` creates empty modules from the shape only, without
    reading the existing weights. Use it for meta-device skeletons whose
    weights are then filled by ``load_state_dict(..., assign=True)``.
    """
    swapped: list[str] = []

    def visit(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent.named_children()):
            full = f"{prefix}{name}"
            n = child.weight.numel() if hasattr(child, "weight") else 0
            if isinstance(child, nn.Linear) and n >= min_params:
                new = (
                    FP8Linear(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=compute_dtype,
                    )
                    if structural
                    else FP8Linear.from_linear(child)
                )
                setattr(parent, name, new)
                swapped.append(full)
            elif isinstance(child, nn.Embedding) and n >= min_params:
                new = (
                    FP8Embedding(
                        child.num_embeddings, child.embedding_dim, compute_dtype=compute_dtype
                    )
                    if structural
                    else FP8Embedding.from_embedding(child)
                )
                setattr(parent, name, new)
                swapped.append(full)
            else:
                visit(child, f"{full}.")

    visit(model, "")
    return swapped


def quantize_model_inplace(model: nn.Module, min_params: int = 0) -> list[str]:
    """Swap modules to FP8 and free the replaced BF16 tensors."""
    swapped = swap_to_fp8(model, min_params)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return swapped
