# alpamayo2-edge-fp8

**FP8 weight-only quantization of [NVIDIA Alpamayo 2 Edge](https://huggingface.co/nvidia/Alpamayo2-Edge)** — a 2 B-parameter in-vehicle reasoning vision-language-action model — so it fits on **8 GB consumer GPUs**.

**Weights (2.49 GiB):** [`FrankLiWright/Alpamayo2-Edge-FP8`](https://huggingface.co/FrankLiWright/Alpamayo2-Edge-FP8) — that repo also carries the full model card.

| | |
|---|---|
| **Before** | 5.30 GB on disk, ~5.0 GiB of VRAM for weights alone |
| **After** | 2.49 GiB on disk, ~2.5 GiB of VRAM for weights |

> Community derivative. **Not** NVIDIA's official FP8 recipe
> (`NVlabs/alpamayo-recipes` → TensorRT Edge-LLM → DRIVE AGX Thor). See
> [How this differs from the official FP8 path](#how-this-differs-from-the-official-fp8-path).

## What it does

Weight-only FP8 (`e4m3fn`) storage with **one FP32 scale per output channel** (per input row for embeddings). Norms, biases and hand-written parameters stay in BF16.

**Compute is unchanged**: weights dequantize to the activation dtype at `forward`. So it runs on **any BF16-capable CUDA GPU** — FP8 tensor cores are *not* required, and there is no FP8 speed benefit. The point is purely that the weights stop eating your VRAM.

508 modules are quantized: 506 `nn.Linear` + 2 `nn.Embedding`. Embeddings dequantize only the looked-up rows.

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.17.0 safetensors einops hydra-core
pip install "alpamayo2-edge @ git+https://github.com/NVlabs/alpamayo2-edge"

hf download FrankLiWright/Alpamayo2-Edge-FP8 --local-dir ./Alpamayo2-Edge-FP8
cd Alpamayo2-Edge-FP8
python example.py --model . --traj-sets 6
```

`load_fp8.py` builds the official `Alpamayo2Edge` skeleton, swaps `nn.Linear` / `nn.Embedding` for FP8 modules, and fills them from `quant.safetensors`. **The upstream model code is not modified.**

```python
import sys, torch
sys.path.insert(0, "./Alpamayo2-Edge-FP8")
from load_fp8 import load_fp8_model
from alpamayo2_edge.helper import prepare_model_inputs, to_device

model, info = load_fp8_model("./Alpamayo2-Edge-FP8", device="cuda")
print(info["swapped_modules"], "FP8 modules")          # 508

inputs = to_device(prepare_model_inputs(sample, model.config, model.tokenizer), device="cuda")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    xyz, rotations, _, extra = model.sample_trajectories_from_data(
        inputs, num_traj_sets=1, num_traj_samples=6, max_generation_length=256,
        diffusion_kwargs={"inference_step": 10,
                          "generator": torch.Generator(device="cuda").manual_seed(43)},
        return_extra=True,
    )
print(xyz.shape)   # [1, 6, 1, 64, 3] -> 64 poses at 0.1 s intervals
```

`example.py` builds a synthetic 22-camera-frame capture so you can run without the gated dataset. For real clips use the PhysicalAI-AV loader from [`NVlabs/alpamayo2-edge`](https://github.com/NVlabs/alpamayo2-edge) (the dataset has its own access terms).

## Requirements

* Python 3.12
* `torch` with CUDA (tested `2.8.0+cu128`)
* `transformers==5.17.0` — the first release with the `cosmos3_edge` backbone
* [`alpamayo2-edge`](https://github.com/NVlabs/alpamayo2-edge) (Apache-2.0)

## Measured results

RTX 5060 Laptop GPU (8 GB, `sm_120`), Windows, driver 621.24, CUDA 12.8,
`torch 2.8.0+cu128`, `transformers 5.17.0`, Python 3.12.15. Warm timings on the
public 22-frame Edge profile (3 261 prompt tokens).

| | BF16 release | **FP8 weight-only** |
|---|---|---|
| Checkpoint on disk | 5.30 GB | **2.49 GiB** |
| Weights in VRAM | ~5.0 GiB | **~2.5 GiB** |
| 1 trajectory — time | 2.43 s | 3.00 s |
| 1 trajectory — peak VRAM | 7.01 GiB | **4.53 GiB** |
| 6 trajectories — time | 2 842 s | **6.10 s** |
| 6 trajectories — peak VRAM | 8.55 GiB | **6.07 GiB** |
| Model load | 5 s | 26 s |

* **1 trajectory is ~20 % slower** than BF16 — dequantize-on-forward is not free.
* **6 trajectories is ~465× faster**, but that is *not* a quantization speedup. BF16 peaks at 8.55 GiB on an 8 GiB card and spills into system memory through WDDM. FP8 just fits. On a 24 GB card BF16 is the faster of the two.

## How this differs from the official FP8 path

| | This | `recipes/alpamayo2edge_edgellm` (official) |
|---|---|---|
| Weights | FP8 e4m3, per-channel scales | FP8 |
| Activations / K/V cache | BF16 | FP8 |
| Runtime | PyTorch + `transformers` | TensorRT Edge-LLM |
| Target hardware | any BF16 CUDA GPU | NVIDIA DRIVE AGX Thor |
| Calibration | none (weight-only) | per-clip, host GPU ≥ 16 GB |
| Latency | 2.4–3.0 s on a laptop GPU | ~200 ms per request on Thor |

Deploying on Thor and need real-time → use the official recipe. Want to run it on a desktop/laptop GPU in PyTorch → use this.

## Reproducing the quantization

No calibration data needed — weights are produced from the public BF16 release:

```bash
hf download nvidia/Alpamayo2-Edge --local-dir ./Alpamayo2-Edge
python quantize_fp8.py --source ./Alpamayo2-Edge --target ./Alpamayo2-Edge-FP8
```

## Limitations — read before deploying

**Not evaluated here:** trajectory accuracy on real driving data. Upstream reports minADE₆ at 5 s of **0.52 m** on 1 392 PhysicalAI-AV samples — that is for **BF16** and has **not** been re-measured for this checkpoint (the evaluation dataset requires accepting its own access terms). There is no closed-loop (AlpaSim) evaluation either. **If you use this beyond research, run those evaluations yourself.**

**What was checked:** it loads, produces finite 64-point trajectories, and on one synthetic 22-frame input with a fixed seed its output is close to BF16 — mean per-waypoint displacement **0.32 m** and **0.78 m** in two runs, over a trajectory spanning ~35 m. Treat this as a smoke test, not a benchmark: the model samples its chain-of-thought stochastically, so a small logit perturbation can change the sampled reasoning and with it the trajectory. Part of that displacement is sampling divergence, not quantization error.

**Other caveats**

* Weight-only — activations and K/V cache stay BF16, so no FP8 GEMM kernels are involved.
* `load_fp8.py` suppresses the random weight init during skeleton construction (every parameter is overwritten by the checkpoint). Non-persistent derived buffers (rotary `inv_freq`, Fourier tables, action-space scalars) come from each module's `__init__` and are checked to be finite after loading.
* Pinned to `transformers==5.17.0`; newer versions may restructure the `cosmos3_edge` backbone.

## License and attribution

The weights are derived from [nvidia/Alpamayo2-Edge](https://huggingface.co/nvidia/Alpamayo2-Edge), developed by **NVIDIA Corporation**, and are distributed under **[OpenMDW-1.1](https://openmdw.ai/license/1-1/)**. They live in the [Hugging Face repo](https://huggingface.co/FrankLiWright/Alpamayo2-Edge-FP8), which also carries the OpenMDW-1.1 agreement and the upstream notices, as that licence requires.

New material — the quantization scheme, the quantized weights, and the code in this repository — is **Apache License 2.0**. See [`NOTICE`](NOTICE) for full attribution.

This repository is not affiliated with, endorsed by, or supported by NVIDIA. For upstream questions use the [Alpamayo NV Developer Forum](https://forums.developer.nvidia.com/c/autonomous-vehicles/alpamayo/766) and [NVlabs/alpamayo2-edge issues](https://github.com/NVlabs/alpamayo2-edge/issues).
