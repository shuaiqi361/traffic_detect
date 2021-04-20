from setuptools import setup, Extension

# Compile *cmm.cpp* into a shared library
setup(
    #...
    ext_modules=[Extension('cmm', ['cmm.cpp', 'cmm_truncate_sides.cpp']), ],
)
