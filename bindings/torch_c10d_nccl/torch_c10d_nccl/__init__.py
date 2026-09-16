# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
torch_c10d_nccl: Out-of-tree PyTorch ProcessGroupNCCL extension for Windows.

Enables native NVIDIA NCCL distributed communications in PyTorch on Windows
without requiring a full rebuild of PyTorch itself.
"""

import os
import sys

# On Windows, register package directory with DLL loader so bundled nccl.dll is found
if sys.platform == "win32":
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(pkg_dir)
        except Exception:
            pass
    os.environ["PATH"] = pkg_dir + os.pathsep + os.environ.get("PATH", "")

import torch
import torch.distributed as dist
import torch._C._distributed_c10d as c10d
from . import _C

ProcessGroupNCCL = _C.ProcessGroupNCCL

# Inject ProcessGroupNCCL into torch distributed namespaces
c10d.ProcessGroupNCCL = _C.ProcessGroupNCCL
dist.distributed_c10d.ProcessGroupNCCL = _C.ProcessGroupNCCL
dist.ProcessGroupNCCL = _C.ProcessGroupNCCL

# Mark NCCL as available in PyTorch
dist.distributed_c10d._NCCL_AVAILABLE = True
dist.is_nccl_available = lambda: True

# Update backend registries
if not hasattr(dist.Backend, "NCCL"):
    setattr(dist.Backend, "NCCL", "nccl")
if "nccl" not in dist.Backend.backend_list:
    dist.Backend.backend_list.append("nccl")

dist.Backend.default_device_backend_map["cuda"] = "nccl"
dist.Backend.backend_capability["nccl"] = ["cuda"]
dist.Backend.backend_type_map["nccl"] = dist.ProcessGroup.BackendType.NCCL

# Register backend creation hook as plugin fallback
def _create_nccl_process_group(store, rank, size, timeout):
    return _C.ProcessGroupNCCL(store, rank, size, timeout)

try:
    dist.Backend.register_backend(
        "nccl",
        _create_nccl_process_group,
        extended_api=False,
        devices=["cuda"],
    )
except Exception:
    pass

__version__ = "2.11.0"
__all__ = ["ProcessGroupNCCL"]
