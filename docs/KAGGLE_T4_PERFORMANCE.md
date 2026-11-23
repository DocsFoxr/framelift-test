# Kaggle T4 ×2: optional inference optimizations

This branch adds three **opt-in** paths without changing FrameLift's defaults.

- `FRAMELIFT_RIFE_CUDA_RESIZE=1`: when `FRAMELIFT_RIFE_RENDER_SCALE < 1`, bicubic-upsample RIFE predictions on CUDA before copying them back to host. The full-resolution AI-enhanced source frames are untouched. RIFE predictions can differ slightly from the OpenCV resize.
- `FRAMELIFT_RIFE_BATCH=2` (or 3): compute multiple timestamps between the same two source frames in one RIFE network invocation. This duplicates tensors logically with `expand`; RIFE's warps and activations still consume memory. **Not guaranteed faster on T4**; compare `1`, `2`, and `3`. The default is 1. A CUDA out-of-memory exception triggers serial fallback; other exceptions propagate.
- `FRAMELIFT_CUDA_CHANNELS_LAST=1`: opt-in NHWC-like memory layout for the super-resolution model and its input tensor. This may help convolutions; benchmarking determines whether it helps a particular GPU.

Already present on main: `FRAMELIFT_CUDA_FAST_RESIZE=1` (bicubic 4x-to-2x resize on GPU) and `FRAMELIFT_RIFE_RENDER_SCALE=0.5` (run RIFE on 1080p for 4K output).

Recommended **quality/speed starting point** for Kaggle T4, with 1080p→4K60:

```bash
export FRAMELIFT_CUDA_FAST_RESIZE=1
export FRAMELIFT_RIFE_RENDER_SCALE=0.5
export FRAMELIFT_RIFE_CUDA_RESIZE=1
export FRAMELIFT_RIFE_BATCH=1
export FRAMELIFT_CUDA_CHANNELS_LAST=0
```

Use independent output directories and the same input for each benchmark; compare total processing time with `RIFE_BATCH=1,2,3` and `CUDA_CHANNELS_LAST=0,1`. Do not assume batching increases throughput; large GPU kernels and grid sampling can be bandwidth-bound. Keep the original 4-worker `[0,0,1,1]` benchmark as baseline. For accurate timings, run without `--debug-timings`; for diagnosis, enable it separately.

## Known caveats

This is **not a TensorRT implementation** and does **not implement GPU-resident NVENC**. Both require separate validated inference/export and buffer lifetime work. In this branch, full-resolution frames still travel through CPU/FFmpeg. No T4 performance measurements have been made by this change, so no speedup is promised.

`RIFE_RENDER_SCALE=0.5` is a perceptual trade-off: motion predictions are inferred at reduced resolution and upscaled; verify fine detail and fast motion against scale 1.0.

## Tests

`pytest -q tests/test_interpolation.py -m "not slow"`

The grouped timestamp tests compare serial vs grouped frame ordering and exact time arguments. GPU numeric consistency and throughput should be tested on Kaggle with the actual RIFE weights.
