import os
import subprocess
import sys
from pathlib import Path

import pybind11
from pybind11.setup_helpers import build_ext
from setuptools import Extension, setup

ROOT = Path(__file__).resolve().parent

# Environment variables
THUNDERKITTENS_ROOT = os.environ.get("THUNDERKITTENS_ROOT") or str(ROOT.parent / "ThunderKittens")
MEGAKERNELS_ROOT = os.environ.get("MEGAKERNELS_ROOT") or str(ROOT.parent)
PYTHON_VERSION = os.environ.get("PYTHON_VERSION", f"{sys.version_info.major}.{sys.version_info.minor}")

# Target GPU (default to HOPPER)
TARGET = os.environ.get("TARGET_GPU", "HOPPER")  # or BLACKWELL

# Source file
SRC = "src/linear_training_mk_demo.cu"

# Get Python include directory
def get_python_include():
    try:
        python_include = (
            subprocess.check_output(
                ["python", "-c", "import sysconfig; print(sysconfig.get_path('include'))"]
            )
            .decode()
            .strip()
        )
        return python_include
    except subprocess.CalledProcessError:
        return ""


def get_torch_includes_and_libs():
    try:
        import torch
        from torch.utils.cpp_extension import include_paths, library_paths

        return include_paths(), library_paths()
    except Exception:
        return [], []

# Base NVCC flags
nvcc_flags = [
    "-DNDEBUG",
    "-Xcompiler=-fPIE",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "--use_fast_math",
    "-forward-unknown-to-host-compiler",
    "-O3",
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    "-std=c++20",
    "-x",
    "cu",
    "-lrt",
    "-lpthread",
    "-ldl",
    "-lcuda",
    "-lcudadevrt",
    "-lcudart_static",
    "-lcublas",
    "-lineinfo",
    "-shared",
    "-fPIC",
    f"-lpython{PYTHON_VERSION}",
]

# Include directories
torch_includes, torch_library_dirs = get_torch_includes_and_libs()
if not torch_includes:
    raise RuntimeError("PyTorch must be installed to build linear_training_mk_demo.")

include_dirs = [
    f"{THUNDERKITTENS_ROOT}/include",
    f"{MEGAKERNELS_ROOT}/include",
    pybind11.get_include(),
    get_python_include(),
    *torch_includes,
]
include_dirs = [path for path in include_dirs if path]

# Get python config flags
def get_python_config_flags():
    try:
        ldflags = subprocess.check_output(["python3-config", "--ldflags"]).decode().strip().split()
        return ldflags
    except subprocess.CalledProcessError:
        return []

# Add python config flags
nvcc_flags.extend(get_python_config_flags())

# Torch libs
for lib_dir in torch_library_dirs:
    if lib_dir:
        nvcc_flags.extend(["-L", lib_dir, f"-Wl,-rpath,{lib_dir}"])
nvcc_flags.extend(["-lc10", "-ltorch", "-ltorch_python"])

# Conditional setup based on target GPU
if TARGET == "HOPPER":
    nvcc_flags.extend(["-DKITTENS_HOPPER", "-arch=sm_90a"])
elif TARGET == "BLACKWELL":
    nvcc_flags.extend(["-DKITTENS_HOPPER", "-DKITTENS_BLACKWELL", "-arch=sm_100a"])
else:
    raise ValueError(f"Invalid target: {TARGET}")

# Get python extension suffix
def get_extension_suffix():
    try:
        suffix = subprocess.check_output(["python3-config", "--extension-suffix"]).decode().strip()
        return suffix
    except subprocess.CalledProcessError:
        return ".so"

# Custom build extension class to use nvcc
class CudaExtension(Extension):
    def __init__(self, name, sources, **kwargs):
        super().__init__(name, sources, **kwargs)

class CudaBuildExt(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CudaExtension):
            self.build_cuda_extension(ext)
        else:
            super().build_extension(ext)
    
    def build_cuda_extension(self, ext):
        nvcc = os.environ.get('NVCC', 'nvcc')
        
        # Get the output file path
        ext_path = self.get_ext_fullpath(ext.name)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        
        # Build the nvcc command
        cmd = [nvcc] + ext.sources + nvcc_flags + ['-o', ext_path]
        
        # Add include directories
        for include_dir in include_dirs:
            cmd.extend(['-I', include_dir])
        
        print(f"Building CUDA extension with command: {' '.join(cmd)}")
        
        # Execute the command
        subprocess.check_call(cmd)

# Define the extension
ext_modules = [
    CudaExtension(
        'linear_training_mk_demo',
        sources=[SRC],
    )
]

setup(
    name='linear_training_mk_demo',
    ext_modules=ext_modules,
    cmdclass={'build_ext': CudaBuildExt},
    zip_safe=False,
    python_requires=">=3.6",
)
