import os
import sys
import shutil

if sys.platform == "win32":
    os.environ["DISTUTILS_USE_SDK"] = "1"
    # Ensure MSVC's link.exe takes precedence over Git's /usr/bin/link.exe
    vc_tools_dir = os.environ.get("VCToolsInstallDir", "")
    host_arch = os.environ.get("VSCMD_ARG_HOST_ARCH", "x64")
    tgt_arch = os.environ.get("VSCMD_ARG_TGT_ARCH", "x64")
    if vc_tools_dir:
        msvc_bin = os.path.join(vc_tools_dir, "bin", f"Host{host_arch}", tgt_arch)
        if os.path.exists(os.path.join(msvc_bin, "link.exe")):
            os.environ["PATH"] = msvc_bin + os.pathsep + os.environ.get("PATH", "")
            print(f"Prepended MSVC linker to PATH: {msvc_bin}")

from setuptools import setup, find_packages

import torch
from torch.utils import cpp_extension

this_dir = os.path.dirname(os.path.abspath(__file__))


# 1. Locate NCCL
nccl_dir = os.environ.get("NCCL_DIR", "")
if not nccl_dir:
    for candidate in [
        os.path.abspath(os.path.join(this_dir, "..", "..", "install_nccl")),
        os.path.abspath(os.path.join(this_dir, "..", "..", "build")),
    ]:
        if os.path.exists(candidate):
            nccl_dir = candidate
            break

if nccl_dir:
    if sys.platform == "win32" and nccl_dir.startswith("/") and len(nccl_dir) > 2 and nccl_dir[2] == "/":
        nccl_dir = nccl_dir[1] + ":" + nccl_dir[2:]
    nccl_dir = os.path.abspath(nccl_dir)

nccl_include_dirs = []
nccl_library_dirs = []

if nccl_dir:
    # Check standard install prefix layout
    cand_inc = os.path.join(nccl_dir, "include")
    cand_lib = os.path.join(nccl_dir, "lib")
    cand_bin = os.path.join(nccl_dir, "bin")

    if os.path.exists(cand_inc):
        nccl_include_dirs.append(cand_inc)
    if os.path.exists(cand_lib):
        nccl_library_dirs.append(cand_lib)
    if os.path.exists(cand_bin):
        nccl_library_dirs.append(cand_bin)

    # If nccl.dll is present, bundle it into the wheel package
    for d in [cand_bin, cand_lib, nccl_dir]:
        cand_dll = os.path.join(d, "nccl.dll")
        if os.path.exists(cand_dll):
            dest_dir = os.path.join(this_dir, "torch_c10d_nccl")
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(cand_dll, os.path.join(dest_dir, "nccl.dll"))
            print(f"Bundling {cand_dll} into torch_c10d_nccl package")
            break

# Also include nccl repository include dirs as fallback
repo_src_include = os.path.abspath(os.path.join(this_dir, "..", "..", "src", "include"))
if os.path.exists(repo_src_include):
    nccl_include_dirs.append(repo_src_include)

# Generate local nccl.h if not found in include dirs
local_nccl_h = os.path.join(this_dir, "include", "nccl.h")
if not os.path.exists(local_nccl_h):
    nccl_h_in = os.path.abspath(os.path.join(this_dir, "..", "..", "src", "nccl.h.in"))
    if os.path.exists(nccl_h_in):
        with open(nccl_h_in, "r", encoding="utf-8") as f:
            content = f.read()
        replacements = {
            "${nccl:Major}": "2",
            "${nccl:Minor}": "32",
            "${nccl:Patch}": "2",
            "${nccl:Suffix}": "",
            "${nccl:Version}": "23202",
        }
        for k, v in replacements.items():
            content = content.replace(k, v)
        with open(local_nccl_h, "w", encoding="utf-8") as f:
            f.write(content)
        print("Generated bindings/torch_c10d_nccl/include/nccl.h from src/nccl.h.in")

# 2. Locate CUDA
cuda_home = cpp_extension.CUDA_HOME or os.environ.get("CUDA_PATH", "")
cuda_include_dirs = []
cuda_library_dirs = []
if cuda_home and os.path.exists(cuda_home):
    cand_inc = os.path.join(cuda_home, "include")
    cand_lib = os.path.join(cuda_home, "lib", "x64")
    if os.path.exists(cand_inc):
        cuda_include_dirs.append(cand_inc)
    if os.path.exists(cand_lib):
        cuda_library_dirs.append(cand_lib)

# 3. Torch paths
torch_dir = os.path.dirname(torch.__file__)
torch_lib_dir = os.path.join(torch_dir, "lib")

local_include = os.path.join(this_dir, "include")
sources = [
    os.path.join("src", "binding.cpp"),
    os.path.join("src", "ProcessGroupNCCL.cpp"),
    os.path.join("src", "NCCLUtils.cpp"),
    os.path.join("src", "nccl.cpp"),
    os.path.join("src", "CUDAEventCache.cpp"),
]

include_dirs = [
    local_include,
] + nccl_include_dirs + cuda_include_dirs

library_dirs = [
    torch_lib_dir,
] + nccl_library_dirs + cuda_library_dirs

extra_compile_args = {
    "cxx": [
        "/std:c++20",
        "/EHsc",
        "/MD",
        "/O2",
        "/FS",
        "/utf-8",
        "/DNOMINMAX",
        "/DWIN32_LEAN_AND_MEAN",
        "/DUSE_C10D_NCCL=1",
        "/DNCCL_HAS_CONFIG=1",
        "/DNCCL_HAS_COMM_SPLIT=1",
        "/DNCCL_HAS_COMM_NONBLOCKING=1",
        "/DNCCL_HAS_INIT_RANK_SCALABLE=1",
        "/DNCCL_HAS_COMM_REGISTER=1",
        "/DNCCL_HAS_COMM_WINDOW_REGISTER=1",
        "/DNCCL_HAS_MEM_ALLOC=1",
        "/DNCCL_HAS_REMOTE_ERROR=1",
        "/DNCCL_HAS_COMM_SHRINK=1",
        "/wd4273",
        "/wd4267",
        "/wd4244",
        "/wd4251",
        "/wd4275",
        "/wd4005",
    ]
}



libraries = [
    "torch_cpu",
    "torch_cuda",
    "c10",
    "c10_cuda",
    "torch_python",
    "cuda",
    "cudart",
    "nccl",
    "fmt",
]

ext_modules = [
    cpp_extension.CppExtension(
        name="torch_c10d_nccl._C",
        sources=sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )
]

setup(
    name="torch_c10d_nccl",
    version="2.11.0",
    description="Native ProcessGroupNCCL distributed communication backend for PyTorch on Windows",
    packages=find_packages(),
    package_data={"torch_c10d_nccl": ["*.dll"]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    zip_safe=False,
)
