# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython

# Python imports
import numpy as np

# Basic Types
STR: cython.char = ord("s")
BOOL: cython.char = ord("o")
FLOAT: cython.char = ord("f")
INT: cython.char = ord("i")
NONE: cython.char = ord("n")

# Date&Time Types
DATE: cython.char = ord("d")
TIME: cython.char = ord("t")
DATETIME: cython.char = ord("z")
TIMEDELTA: cython.char = ord("l")
STRUCT_TIME: cython.char = ord("r")

# Numeric Types
DECIMAL: cython.char = ord("e")
COMPLEX: cython.char = ord("c")

# Bytes Types
BYTES: cython.char = ord("b")
BYTEARRAY: cython.char = ord("y")

# Sequence Types
LIST: cython.char = ord("L")
TUPLE: cython.char = ord("T")
SET: cython.char = ord("E")
FROZENSET: cython.char = ord("R")
RANGE: cython.char = ord("g")

# Mapping Types
DICT: cython.char = ord("D")

# NumPy Types
# fmt: off
DATETIME64: cython.char = ord("M")
TIMEDELTA64: cython.char = ord("m")
NDARRAY: cython.char = ord("N")
NDARRAY_OBJECT: cython.char = ord(np.array(None, dtype="O").dtype.kind)  # 'O'
NDARRAY_INT: cython.char = ord(np.array(1, dtype=np.int64).dtype.kind)  # 'i'
NDARRAY_UINT: cython.char = ord(np.array(1, dtype=np.uint64).dtype.kind)  # 'u'
NDARRAY_FLOAT: cython.char = ord(np.array(0.1, dtype=np.float64).dtype.kind)  # 'f'
NDARRAY_BOOL: cython.char = ord(np.array(True, dtype=bool).dtype.kind)  # 'b'
NDARRAY_DT64: cython.char = ord(np.array(1, dtype="datetime64[ns]").dtype.kind)  # 'M'
NDARRAY_TD64: cython.char = ord(np.array(1, dtype="timedelta64[ns]").dtype.kind)  # 'm'
NDARRAY_COMPLEX: cython.char = ord(np.array(1 + 1j, dtype=np.complex128).dtype.kind)  # 'c'
NDARRAY_BYTES: cython.char = ord(np.array(b"1", dtype="S").dtype.kind)  # 'S'
NDARRAY_UNICODE: cython.char = ord(np.array("1", dtype="U").dtype.kind)  # 'U'
# fmt: on

# Pandas Types
SERIES: cython.char = ord("S")
DATAFRAME: cython.char = ord("F")
PD_TIMESTAMP: cython.char = ord("p")
PD_TIMEDELTA: cython.char = ord("a")
DATETIMEINDEX: cython.char = ord("Z")
TIMEDELTAINDEX: cython.char = ord("X")


### Test for duplicates ###
_duplicates_test: cython.bint = False
if not _duplicates_test:

    def _test_duplicates(category: str, ids: list[str]) -> None:
        seen: set[str] = set()
        for i in ids:
            assert i not in seen, f"Duplicated %s: %r." % (category, i)
            seen.add(i)

    _object_ids = [
        # Basic Types
        STR,
        BOOL,
        FLOAT,
        INT,
        NONE,
        # Date&Time Types
        DATE,
        TIME,
        DATETIME,
        TIMEDELTA,
        STRUCT_TIME,
        # Numeric Types
        DECIMAL,
        COMPLEX,
        # Bytes Types
        BYTES,
        BYTEARRAY,
        # Sequence Types
        LIST,
        TUPLE,
        SET,
        FROZENSET,
        RANGE,
        # Mapping Types
        DICT,
        # Numpy Types
        DATETIME64,
        TIMEDELTA64,
        NDARRAY,
        # Pandas Types
        SERIES,
        DATAFRAME,
        PD_TIMESTAMP,
        PD_TIMEDELTA,
        DATETIMEINDEX,
        TIMEDELTAINDEX,
    ]
    _test_duplicates("object identifier", _object_ids)
    _ndarray_dtype_ids = [
        NDARRAY_OBJECT,
        NDARRAY_INT,
        NDARRAY_UINT,
        NDARRAY_FLOAT,
        NDARRAY_BOOL,
        NDARRAY_DT64,
        NDARRAY_TD64,
        NDARRAY_COMPLEX,
        NDARRAY_BYTES,
        NDARRAY_UNICODE,
    ]
    _test_duplicates("ndarray dtype identifier", _ndarray_dtype_ids)

    del _test_duplicates, _object_ids, _ndarray_dtype_ids

    _duplicates_test = True
