# setup.py
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("cpp_functions.pyx"),
    include_dirs=[numpy.get_include()]
)