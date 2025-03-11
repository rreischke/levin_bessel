import sys
import os
from setuptools import setup
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension

import distutils.sysconfig

__version__ = "1.1.0"


if (sys.platform[:6] == "darwin"
        and (distutils.sysconfig.get_config_var("CC") == "clang"
                or os.environ.get("CC", "") == "clang")):
    compiler_args = ["-Xpreprocessor"]
    linker_args = ["-mlinker-version=305", "-Xpreprocessor"]
else:
    compiler_args = []
    linker_args = []

compiler_args += ["-fopenmp","-O3", "-ffast-math", "-fassociative-math", "-pedantic"]
linker_args += ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "pylevin",
        ["src/pylevin.cpp", "src/pybind11_interface.cpp"],
        cxx_std=20,
        include_dirs=["src"],
        libraries=["m", "gsl", "gslcblas"],
        extra_compile_args=compiler_args,
        extra_link_args=linker_args
        ),
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name="pylevin",
    version=__version__,
    author="Robert Reischke",
    author_email="reischke@posteo.net",
    url="https://github.com/rreischke/levin_bessel",
    description="Implementing the Levin method to calculate integrals over product of up to three Bessel functions",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license_files = ('LICENSE.txt'),
    ext_modules=ext_modules,
    zip_safe=False,
    headers=['src/pylevin.h'],
)
