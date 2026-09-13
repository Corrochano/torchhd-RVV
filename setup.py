"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import sys
from torch.utils.cpp_extension import CppExtension, BuildExtension
from setuptools import setup, find_packages

# Read the version without importing any dependencies
version = {}
with open("torchhd/version.py") as f:
    exec(f.read(), version)

extra_compile_args = ['-O3']
if sys.platform != 'win32':
    extra_compile_args.extend(['-march=rv64gcv'])

setup(
    name="torch-hd",  # use torch-hd on PyPi to install torchhd, torchhd is too similar according to PyPi
    version=version["__version__"],
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
    ext_modules=[
        CppExtension(
            name="torchhd._C",
            sources=["torchhd/rvv_extension.cpp"],
            extra_compile_args={'cxx': extra_compile_args}
        )
    ],
    python_requires=">=3.8, <4",
    project_urls={
        "Source": "https://github.com/hyperdimensional-computing/torchhd",
        "Documentation": "https://torchhd.readthedocs.io",
    },
)
