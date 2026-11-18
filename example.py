# SPDX-License-Identifier: Apache-2.0
"""Minimal FP8 Alpamayo 2 Edge run on a synthetic 22-camera-frame capture.

    python example.py --model .            # this directory
    python example.py --model . --traj-sets 6

The synthetic input follows the public Edge input profile (six cameras, the
documented per-camera frame counts and pixel budgets, 20 ego poses, a
navigation instruction). It proves the checkpoint loads and runs; it is not a
driving-quality evaluation. For real inputs use the PhysicalAI-AV loader from
https://github.com/NVlabs/alpamayo2-edge (the dataset has its own access terms).
"""

from __future__ import annotations

import argparse
import time

import torch


def build_sample(t0_us: int = 4_000_000) -> dict:
    """Synthetic 22-frame capture matching the documented Edge input profile."""
    from alpamayo2_edge.sample import CAMERA_FRAME_COUNTS

    offsets_s = {1: [-2, -1, -0.5, -0.1, 0], 6: [-2, -1, -0.5, -0.1, 0]}
    for cam in (0, 2, 3, 5):
        offsets_s[cam] = [-1, -0.1, 0]

    h, w = 1080, 1920  # the public example's raw capture resolution
    frames, cams, tss = [], [], []
    for cam in sorted(CAMERA_FRAME_COUNTS):
        for dt in offsets_s[cam]:
            frames.append(torch.zeros(1, 1, 3, h, w, dtype=torch.uint8))
            cams.append(cam)
            tss.append(int(t0_us + dt * 1_000_000))

    n = 20
    xyz = torch.zeros(1, n, 3)
    rot = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, n, 1, 1)
    for i in range(n):
        t = -1.9 + 0.1 * i
        xyz[0, i] = torch.tensor([t * 8.0, 0.05 * (t * 8.0) ** 2, 0.0])
        yaw = 0.05 * (t * 8.0)
        c, s = torch.cos(torch.tensor(yaw)), torch.sin(torch.tensor(yaw))
        rot[0, i] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    return {
        "image_frames": torch.cat(frames, dim=0).contiguous(),
        "camera_indices": torch.tensor(cams, dtype=torch.int64),
        "absolute_timestamps": torch.tensor(tss, dtype=torch.int64),
        "ego_history_xyz": xyz,
        "ego_history_rot": rot,
        "nav_text": ["Turn right in 30m"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=".", help="FP8 checkpoint directory")
    ap.add_argument("--traj-sets", type=int, default=1)
    ap.add_argument("--traj-samples", type=int, default=1)
    ap.add_argument("--diffusion-steps", type=int, default=5)
    ap.add_argument("--max-generation-length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from load_fp8 import load_fp8_model

    t = time.time()
    model, info = load_fp8_model(args.model, device=args.device, dtype="bfloat16")
    print(f"loaded in {time.time() - t:.1f}s | {info['swapped_modules']} FP8 modules", flush=True)
    if args.device.startswith("cuda"):
        print(f"weight VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    sample = build_sample()
    inputs = to_device(
        prepare_model_inputs(sample, model.config, model.tokenizer), device=args.device
    )
    print(f"prompt tokens: {inputs['tokenized_data']['input_ids'].shape[-1]}", flush=True)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        xyz, rotations, _, extra = model.sample_trajectories_from_data(
            inputs,
            num_traj_sets=args.traj_sets,
            num_traj_samples=args.traj_samples,
            max_generation_length=args.max_generation_length,
            diffusion_kwargs={
                "inference_step": args.diffusion_steps,
                "generator": torch.Generator(device=args.device).manual_seed(43),
            },
            return_extra=True,
        )
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"inference {time.time() - t:.1f}s", flush=True)
    if args.device.startswith("cuda"):
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    print(f"pred_xyz {tuple(xyz.shape)} finite={bool(torch.isfinite(xyz).all())}", flush=True)
    print(f"pred_rot {tuple(rotations.shape)} finite={bool(torch.isfinite(rotations).all())}", flush=True)
    for key in ("decoded_text", "cot", "text", "reasoning"):
        if isinstance(extra, dict) and extra.get(key):
            print(f"{key}: {extra[key]}", flush=True)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
