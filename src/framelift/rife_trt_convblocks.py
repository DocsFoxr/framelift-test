"""Conservative TensorRT acceleration of RIFE's convolution-only submodules.

RIFE's FP32 flow warps, FP16 frame blending, and feature extraction remain
in PyTorch. This avoids compiling mixed-precision grid sampling in the same
graph as the convolution kernels and makes per-block numerical checks possible.

Enabled via FRAMELIFT_RIFE_TRT_STRATEGY=convblocks.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional

log = logging.getLogger(__name__)


def _indices(raw: str) -> tuple[int, ...]:
    try:
        blocks = tuple(int(x.strip()) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError("FRAMELIFT_RIFE_TRT_BLOCKS must contain indices 0..4") from exc
    if not blocks or len(set(blocks)) != len(blocks) or any(i not in range(5) for i in blocks):
        raise ValueError("FRAMELIFT_RIFE_TRT_BLOCKS must be a unique list of indices 0..4")
    return blocks


def _compare(reference: torch.Tensor, candidate: torch.Tensor, name: str) -> float:
    if candidate.shape != reference.shape:
        raise RuntimeError(f"{name}: TensorRT output shape mismatch")
    if not bool(torch.isfinite(candidate).all()):
        raise RuntimeError(f"{name}: TensorRT produced non-finite values")
    delta = (candidate.float() - reference.float()).abs()
    mean = float(delta.mean().item())
    max_error = float(delta.max().item())
    over_5pct = float((delta > 0.05).float().mean().item())
    log.warning(
        "RIFE_TRT_COMPARE %s | mean=%.6f | max=%.6f | over_0.05=%.6f",
        name, mean, max_error, over_5pct,
    )
    if mean > 0.015:
        raise RuntimeError(
            f"{name}: TensorRT numerical validation failed (mean={mean:.6f}, "
            f"max={max_error:.6f}, fraction>0.05={over_5pct:.6f})"
        )
    return mean


def compile_convolution_blocks(
    step: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    size: tuple[int, int],
    torch_tensorrt,
    cache_dir: Path,
    gpu_id: int,
    optimization_level: int,
) -> torch.nn.Module:
    """Compile selected _ResConv stacks; do not compile any RIFE warping operators.

    Validate original and TensorRT stacks independently, then run the entire
    frame interpolator through both paths with a smooth moving image.
    If validation fails, the original PyTorch modules are restored.
    """
    from .tensorrt_backend import _engine_lock

    selected = _indices(os.getenv("FRAMELIFT_RIFE_TRT_BLOCKS", "4"))
    width, height = size
    started = time.perf_counter()
    originals = {}
    replacements = {}
    cache_dir = cache_dir / "convblocks"

    with torch.cuda.device(device), _engine_lock(cache_dir), torch.inference_mode():
        # Low-frequency synthetic images better resemble video than white
        # noise; no artificial differences are waived for the final check.
        seed = torch.Generator(device=device).manual_seed(426)
        small = torch.rand(
            (1, 3, max(8, height // 16), max(8, width // 16)),
            generator=seed, dtype=dtype, device=device,
        )
        img0 = functional.interpolate(
            small, size=(height, width), mode="bilinear", align_corners=False
        )
        img1 = torch.roll(img0, shifts=(3, 7), dims=(-2, -1))
        f0 = step.network.encode(img0)
        f1 = step.network.encode(img1)
        t = torch.tensor([0.5], device=device, dtype=dtype).view(1, 1, 1, 1)
        inputs = (img0, img1, f0, f1, t)
        original_result = step(*inputs).detach()

        for index in selected:
            block = getattr(step.network, f"block{index}")
            module = block.convblock
            originals[index] = module
            scale = step.network.scale_list[index]
            # _IFBlock.conv0 has two 3x3, stride-2 layers. The residual
            # stack sees ceil(ceil(input/scale)/4) in each spatial axis.
            inner_h = math.ceil(math.ceil(height / scale) / 4)
            inner_w = math.ceil(math.ceil(width / scale) / 4)
            channels = module[0].conv.in_channels
            example = torch.randn(
                (1, channels, inner_h, inner_w),
                generator=seed, dtype=dtype, device=device,
            ) * 0.1
            baseline = module(example).detach()

            log.warning(
                "RIFE_TRT_BLOCK_BUILD GPU%d | block%d | shape=(1,%d,%d,%d)",
                gpu_id, index, channels, inner_h, inner_w,
            )
            compiled = torch_tensorrt.compile(
                module,
                ir="dynamo",
                arg_inputs=(example,),
                use_explicit_typing=True,
                device=torch_tensorrt.Device(gpu_id=gpu_id),
                min_block_size=3,
                require_full_compilation=False,
                immutable_weights=False,
                cache_built_engines=True,
                reuse_cached_engines=True,
                engine_cache_dir=str(cache_dir),
                optimization_level=optimization_level,
            )
            actual_engines = sum(
                "tensorrt" in type(submodule).__name__.lower()
                or "trtengine" in type(submodule).__name__.lower()
                for submodule in compiled.modules()
            )
            if actual_engines == 0:
                raise RuntimeError(f"RIFE block{index}: no TensorRT engine was produced")
            result = compiled(example)
            _compare(baseline, result, f"GPU{gpu_id}:block{index}")
            replacements[index] = compiled

        try:
            for index, compiled in replacements.items():
                getattr(step.network, f"block{index}").convblock = compiled
            full_result = step(*inputs)
            mean_error = _compare(
                original_result, full_result, f"GPU{gpu_id}:complete_RIFE"
            )
        except Exception:
            # Never leave a partially modified reference model in use.
            for index, module in originals.items():
                getattr(step.network, f"block{index}").convblock = module
            raise

        torch.cuda.synchronize(device)
        log.warning(
            "RIFE_TRT_ACTIVE GPU%d | strategy=convblocks | blocks=%s | "
            "compile_and_validation=%.1fs | mean_absolute_error=%.6f",
            gpu_id, ",".join(map(str, selected)),
            time.perf_counter() - started, mean_error,
        )
    return step
