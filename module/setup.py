from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

extension = Extension("find_triangular",
                      sources=["find_triangular.pyx"],
                      include_dirs=[numpy.get_include()],
                      language="c++")

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize(extension,
                          compiler_directives={'language_level': "3"}
                          )
),
