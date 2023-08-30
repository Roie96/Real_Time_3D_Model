from setuptools import setup, Extension
from Cython.Build import cythonize

setup(ext_modules=cythonize([Extension('dismap', ['dismap.pyx'], extra_compile_args=["-O3", '-march=native', '-mtune=native', '-fsigned-char'])], annoate=True, langauge_level="3", compiler_directives={"boundscheck":False, "wraparound": False, "cdivision": True}))