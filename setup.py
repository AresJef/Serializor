import os, numpy as np, platform
from setuptools import setup, Extension
from Cython.Build import cythonize

__package__ = "serializor"


# Create Extension
def extension(filename: str, include_np: bool, *extra_compile_args: str) -> Extension:
    # Extra arguments
    extra_args = list(extra_compile_args) if extra_compile_args else None
    # Name
    name: str = "%s.%s" % (__package__, filename.split(".")[0])
    source: str = os.path.join("src", __package__, filename)
    # Create extension
    if include_np:
        return Extension(
            name,
            sources=[source],
            extra_compile_args=extra_args,
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    else:
        return Extension(name, sources=[source], extra_compile_args=extra_args)


# Build Extensions
if platform.system() == "Windows":
    extensions = [
        extension("crypto.py", True),
        extension("deserialize.py", True),
        extension("prefix.py", True),
        extension("serialize.py", True),
        extension("typeref.py", False),
    ]
else:
    extensions = [
        extension("crypto.py", True, "-Wno-unreachable-code"),
        extension("deserialize.py", True, "-Wno-unreachable-code"),
        extension("prefix.py", True, "-Wno-unreachable-code"),
        extension("serialize.py", True, "-Wno-unreachable-code"),
        extension("typeref.py", False, "-Wno-unreachable-code"),
    ]


# Build
setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
        annotate=True,
    ),
)
