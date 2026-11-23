"""Opt-in Torch-TensorRT compilation for the CUDA RIFE inference step.

This module is imported ONLY when FRAMELIFT_RIFE_TRT=1. The regular PyTorch
path remains the default, and failures can be made fatal via
FRAMELIFT_RIFE_TRT_REQUIRED=1 for honest A/B benchmarks.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

log = logging.getLogger(__name__)


@contextmanager
def _engine_lock(cache_dir: Path):
    """Serialize builds for workers on one GPU so they share the engine cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".build.lock"
    with lock_path.open("a+b") as lock_file:
        if os.name == "posix":
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def try_compile_rife_step(
    step: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    size: tuple[int, int],
) -> torch.nn.Module:
    """Compile only RIFE's interpolation step, keeping feature extraction in PyTorch.

    Inputs have fixed 1-frame shapes. Compilation overhead is intentionally
    recorded, and the compiled engine is checked for real TensorRT blocks and
    numerical consistency before activation.
    """
    if device.type != "cuda" or dtype is not torch.float16:
        raise RuntimeError("RIFE TensorRT currently requires CUDA and FP16")

    import torch_tensorrt

    gpu_id = device.index if device.index is not None else torch.cuda.current_device()
    width, height = size
    cache_root = Path(os.getenv(
        "FRAMELIFT_RIFE_TRT_CACHE",
        str(Path.home() / ".cache" / "framelift" / "tensorrt"),
    )).expanduser()
    # One lock per GPU; engines can still be shared between processes on that GPU.
    cache_dir = cache_root / f"gpu{gpu_id}"
    optimization = int(os.getenv("FRAMELIFT_RIFE_TRT_OPT_LEVEL", "3"))
    if not 0 <= optimization <= 5:
        raise ValueError("FRAMELIFT_RIFE_TRT_OPT_LEVEL must be between 0 and 5")

    started = time.perf_counter()
    with torch.cuda.device(device), _engine_lock(cache_dir), torch.inference_mode():
        # Representative valid inputs. Feature dimensions match _Head's output.
        img0 = torch.rand((1, 3, height, width), dtype=dtype, device=device)
        img1 = torch.rand((1, 3, height, width), dtype=dtype, device=device)
        f0 = step.network.encode(img0)
        f1 = step.network.encode(img1)
        t = torch.tensor([0.5], dtype=dtype, device=device).reshape(1, 1, 1, 1)
        examples = (img0, img1, f0, f1, t)

        log.warning(
            "RIFE TensorRT build/cache lookup starting on GPU %d (%dx%d, FP16); "
            "the first compilation can take minutes.",
            gpu_id, width, height,
        )
        compiled = torch_tensorrt.compile(
            step,
            ir="dynamo",
            arg_inputs=examples,
            # Torch-TensorRT 2.11 defaults to strong/explicit typing. Passing
            # enabled_precisions alongside it raises AssertionError. The RIFE
            # network and the example tensors are already FP16; _warp uses
            # intentional FP32 math for sampling coordinates.
            use_explicit_typing=True,
            device=torch_tensorrt.Device(gpu_id=gpu_id),
            min_block_size=5,
            require_full_compilation=False,
            cache_built_engines=True,
            reuse_cached_engines=True,
            engine_cache_dir=str(cache_dir),
            optimization_level=optimization,
        )
        engine_count = sum(
            "tensorrtmodule" in type(submodule).__name__.lower()
            or "trtenginemodule" in type(submodule).__name__.lower()
            for submodule in compiled.modules()
        )
        if engine_count == 0:
            raise RuntimeError(
                "Torch-TensorRT compiled no actual TensorRT engine blocks; "
                "cannot label this run as TensorRT."
            )

        # Compare representative full-sized tensors; use mean error to tolerate
        # reasonable FP16 kernel differences while refusing clearly bad outputs.
        reference = step(*examples)
        candidate = compiled(*examples)
        if candidate.shape != reference.shape:
            raise RuntimeError("TensorRT output shape is different from PyTorch")
        delta = (candidate.float() - reference.float()).abs()
        mean_error = float(delta.mean().item())
        if not bool(torch.isfinite(candidate).all()) or mean_error > 0.015:
            raise RuntimeError(
                f"TensorRT numerical check failed (mean absolute error {mean_error:.5f})"
            )

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        log.warning(
            "RIFE_TRT_ACTIVE GPU %d | engines=%d | prepare_and_check=%.1fs "
            "| mean_absolute_error=%.6f",
            gpu_id, engine_count, elapsed, mean_error,
        )
    return compiled
