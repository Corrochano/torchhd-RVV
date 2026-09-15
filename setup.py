"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import os
import platform
import sys

from torch.utils.cpp_extension import BuildExtension, CppExtension
from setuptools import find_packages, setup

# Read the version without importing any dependencies
version = {}
with open("torchhd/version.py") as f:
    exec(f.read(), version)

machine = platform.machine().lower()
force_rvv_build = os.environ.get("TORCHHD_BUILD_RVV", "").lower() in {"1", "true", "yes", "on"}
should_build_rvv = force_rvv_build or machine in {"riscv64", "riscv32", "riscv"}

extra_compile_args = ["-O3"]
if should_build_rvv and sys.platform != "win32":
    extra_compile_args.extend(["-march=rv64gcv"])

ext_modules = []
if should_build_rvv:
    ext_modules = [
        CppExtension(
            name="torchhd._C",
            sources=[
                "torchhd/rvv_extension.cpp",
                "torchhd/riscv/vectorialKernels.cpp",
            ],
            extra_compile_args={"cxx": extra_compile_args},
            include_dirs=["torchhd/riscv"],
        )
    ]

setup(
    name="torchHDrvv",
    version=version["__version__"],
    cmdclass={"build_ext": BuildExtension},
    description="Torchhd is a Python library for Hyperdimensional Computing and Vector Symbolic Architectures",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hyperdimensional-computing/torchhd",
    license="MIT",
    install_requires=[
        "torch>=1.9.0",
        "scipy",
        "pandas",
        "numpy",
        "requests",
        "tqdm",
        "openpyxl",
    ],
    packages=find_packages(exclude=["docs", "torchhd.tests", "examples"]),
    ext_modules=ext_modules,
    python_requires=">=3.8, <4",
    project_urls={
        "Source": "https://github.com/hyperdimensional-computing/torchhd",
        "Documentation": "https://torchhd.readthedocs.io",
    },
)
