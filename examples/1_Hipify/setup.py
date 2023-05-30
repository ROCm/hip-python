from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
  ext_modules = cythonize([
      Extension("ccuda_stream", ["ccuda_stream.pyx"],
                #libraries=["calg"]
                )
      ])
)