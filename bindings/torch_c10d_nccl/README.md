# torch_c10d_nccl: PyTorch ProcessGroupNCCL for Windows

Native NVIDIA NCCL distributed communication backend (`ProcessGroupNCCL`) for PyTorch on Windows.

## Why this exists
Official PyTorch Windows binaries currently omit `ProcessGroupNCCL` via `if(NOT WIN32)` in CMake, preventing standard multi-GPU distributed training with NCCL on Windows machines.

`torch_c10d_nccl` compiles PyTorch's native `ProcessGroupNCCL` out-of-tree against Windows-compatible NCCL (`nccl.dll` / `nccl.lib`) and bundles `nccl.dll` directly inside the wheel.

## Quick Start
```bash
pip install torch_c10d_nccl
```

In your training script:
```python
import torch
import torch_c10d_nccl
import torch.distributed as dist

# Native NCCL is now available!
print("NCCL Available:", dist.is_nccl_available())  # True

dist.init_process_group(backend="nccl")
```

## Features
- **100% Native C++ Execution**: Direct CUDA stream execution, CUDA event caching, and watchdog thread monitoring.
- **Self-Contained**: Automatically bundles `nccl.dll` and registers it with the Windows DLL loader.
- **Drop-in Compatibility**: Monkeypatches `torch._C._distributed_c10d.ProcessGroupNCCL` and registers with `dist.Backend` for standard PyTorch distributed semantics.
