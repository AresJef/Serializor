# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_READ_CHAR as read_char  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FromOrdinal as str_fr_ucs4  # type: ignore

np.import_array()
np.import_umath()

# Python imports
import numpy as np

# Basic Types -----------------------------------------------------------------------
STR: str = "s"
BOOL: str = "o"
FLOAT: str = "f"
INT: str = "i"
NONE: str = "n"
# . identifier
STR_ID: cython.Py_UCS4 = read_char(STR, 0)
BOOL_ID: cython.Py_UCS4 = read_char(BOOL, 0)
FLOAT_ID: cython.Py_UCS4 = read_char(FLOAT, 0)
INT_ID: cython.Py_UCS4 = read_char(INT, 0)
NONE_ID: cython.Py_UCS4 = read_char(NONE, 0)
# . variant
BOOL_TRUE: str = BOOL + "1"  # 'o1'
BOOL_FALSE: str = BOOL + "0"  # 'o0'

# Date&Time Types -------------------------------------------------------------------
DATE: str = "d"
TIME: str = "t"
DATETIME: str = "z"
TIMEDELTA: str = "l"
# . identifier
DATE_ID: cython.Py_UCS4 = read_char(DATE, 0)
TIME_ID: cython.Py_UCS4 = read_char(TIME, 0)
DATETIME_ID: cython.Py_UCS4 = read_char(DATETIME, 0)
TIMEDELTA_ID: cython.Py_UCS4 = read_char(TIMEDELTA, 0)

# Numeric Types ---------------------------------------------------------------------
DECIMAL: str = "e"
COMPLEX: str = "c"
# . identifier
DECIMAL_ID: cython.Py_UCS4 = read_char(DECIMAL, 0)
COMPLEX_ID: cython.Py_UCS4 = read_char(COMPLEX, 0)

# Bytes Types -----------------------------------------------------------------------
BYTES: str = "b"
# . identifier
BYTES_ID: cython.Py_UCS4 = read_char(BYTES, 0)

# Mapping ---------------------------------------------------------------------------
DICT: str = "D"
# . identifier
DICT_ID: cython.Py_UCS4 = read_char(DICT, 0)

# Sequence Types --------------------------------------------------------------------
LIST: str = "L"
TUPLE: str = "T"
SET: str = "E"
# . identifier
LIST_ID: cython.Py_UCS4 = read_char(LIST, 0)
TUPLE_ID: cython.Py_UCS4 = read_char(TUPLE, 0)
SET_ID: cython.Py_UCS4 = read_char(SET, 0)

# Numpy Types -----------------------------------------------------------------------
DATETIME64: str = "M"
TIMEDELTA64: str = "m"
# . identifier
DATETIME64_ID: cython.Py_UCS4 = read_char(DATETIME64, 0)
TIMEDELTA64_ID: cython.Py_UCS4 = read_char(TIMEDELTA64, 0)
# . variant
DATETIME64_Y: str = DATETIME64 + "Y"  # 'MY'
DATETIME64_M: str = DATETIME64 + "M"  # 'MM'
DATETIME64_W: str = DATETIME64 + "W"  # 'MW'
DATETIME64_D: str = DATETIME64 + "D"  # 'MD'
DATETIME64_B: str = DATETIME64 + "B"  # 'MB'
DATETIME64_H: str = DATETIME64 + "h"  # 'Mh'
DATETIME64_MI: str = DATETIME64 + "m"  # 'Mm'
DATETIME64_S: str = DATETIME64 + "s"  # 'Ms'
DATETIME64_MS: str = DATETIME64 + "ms"  # 'Mms'
DATETIME64_US: str = DATETIME64 + "us"  # 'Mus'
DATETIME64_NS: str = DATETIME64 + "ns"  # 'Mns'
DATETIME64_PS: str = DATETIME64 + "ps"  # 'Mps'
DATETIME64_FS: str = DATETIME64 + "fs"  # 'Mfs'
DATETIME64_AS: str = DATETIME64 + "as"  # 'Mas'
TIMEDELTA64_Y: str = TIMEDELTA64 + "Y"  # 'mY'
TIMEDELTA64_M: str = TIMEDELTA64 + "M"  # 'mM'
TIMEDELTA64_W: str = TIMEDELTA64 + "W"  # 'mW'
TIMEDELTA64_D: str = TIMEDELTA64 + "D"  # 'mD'
TIMEDELTA64_B: str = TIMEDELTA64 + "B"  # 'mB'
TIMEDELTA64_H: str = TIMEDELTA64 + "h"  # 'mh'
TIMEDELTA64_MI: str = TIMEDELTA64 + "m"  # 'mm'
TIMEDELTA64_S: str = TIMEDELTA64 + "s"  # 'ms'
TIMEDELTA64_MS: str = TIMEDELTA64 + "ms"  # 'mms'
TIMEDELTA64_US: str = TIMEDELTA64 + "us"  # 'mus'
TIMEDELTA64_NS: str = TIMEDELTA64 + "ns"  # 'mns'
TIMEDELTA64_PS: str = TIMEDELTA64 + "ps"  # 'mps'
TIMEDELTA64_FS: str = TIMEDELTA64 + "fs"  # 'mfs'
TIMEDELTA64_AS: str = TIMEDELTA64 + "as"  # 'mas'

# Numpy.ndarray ---------------------------------------------------------------------
NDARRAY: str = "N"
# fmt: off
arr: np.ndarray = np.array(None, dtype=object)
NDARRAY_DTYPE_OBJECT: str = str_fr_ucs4(arr.descr.kind)  # 'O'
arr: np.ndarray = np.array(1.1, dtype=np.float64)
NDARRAY_DTYPE_FLOAT: str = str_fr_ucs4(arr.descr.kind)  # 'f'
arr: np.ndarray = np.array(1, dtype=np.int64)
NDARRAY_DTYPE_INT: str = str_fr_ucs4(arr.descr.kind)  # 'i'
arr: np.ndarray = np.array(1, dtype=np.uint64)
NDARRAY_DTYPE_UINT: str = str_fr_ucs4(arr.descr.kind)  # 'u'
arr: np.ndarray = np.array(1, dtype=np.bool_)
NDARRAY_DTYPE_BOOL: str = str_fr_ucs4(arr.descr.kind)  # 'b'
arr: np.ndarray = np.array(1, dtype="datetime64[ns]")
NDARRAY_DTYPE_DT64: str = str_fr_ucs4(arr.descr.kind)  # 'M'
arr: np.ndarray = np.array(1, dtype="timedelta64[ns]")
NDARRAY_DTYPE_TD64: str = str_fr_ucs4(arr.descr.kind)  # 'm'
arr: np.ndarray = np.array(1 + 1j, dtype=np.complex128)
NDARRAY_DTYPE_COMPLEX: str = str_fr_ucs4(arr.descr.kind)  # 'c'
arr: np.ndarray = np.array(b"", dtype="S")
NDARRAY_DTYPE_BYTES: str = str_fr_ucs4(arr.descr.kind)  # 'S'
arr: np.ndarray = np.array("", dtype="U")
NDARRAY_DTYPE_UNICODE: str = str_fr_ucs4(arr.descr.kind)  # 'U'
# fmt: on
# . identifier
NDARRAY_ID: cython.Py_UCS4 = read_char(NDARRAY, 0)
NDARRAY_DTYPE_OBJECT_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_OBJECT, 0)
NDARRAY_DTYPE_FLOAT_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_FLOAT, 0)
NDARRAY_DTYPE_INT_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_INT, 0)
NDARRAY_DTYPE_UINT_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_UINT, 0)
NDARRAY_DTYPE_BOOL_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_BOOL, 0)
NDARRAY_DTYPE_DT64_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_DT64, 0)
NDARRAY_DTYPE_TD64_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_TD64, 0)
NDARRAY_DTYPE_COMPLEX_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_COMPLEX, 0)
NDARRAY_DTYPE_BYTES_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_BYTES, 0)
NDARRAY_DTYPE_UNICODE_ID: cython.Py_UCS4 = read_char(NDARRAY_DTYPE_UNICODE, 0)
# . variant dtype
NDARRAY_DTYPE_DT64_Y: str = NDARRAY_DTYPE_DT64 + "Y"  # 'MY'
NDARRAY_DTYPE_DT64_M: str = NDARRAY_DTYPE_DT64 + "M"  # 'MM'
NDARRAY_DTYPE_DT64_W: str = NDARRAY_DTYPE_DT64 + "W"  # 'MW'
NDARRAY_DTYPE_DT64_D: str = NDARRAY_DTYPE_DT64 + "D"  # 'MD'
NDARRAY_DTYPE_DT64_B: str = NDARRAY_DTYPE_DT64 + "B"  # 'MB'
NDARRAY_DTYPE_DT64_H: str = NDARRAY_DTYPE_DT64 + "h"  # 'Mh'
NDARRAY_DTYPE_DT64_MI: str = NDARRAY_DTYPE_DT64 + "m"  # 'Mm'
NDARRAY_DTYPE_DT64_S: str = NDARRAY_DTYPE_DT64 + "s"  # 'Ms'
NDARRAY_DTYPE_DT64_MS: str = NDARRAY_DTYPE_DT64 + "ms"  # 'Mms'
NDARRAY_DTYPE_DT64_US: str = NDARRAY_DTYPE_DT64 + "us"  # 'Mus'
NDARRAY_DTYPE_DT64_NS: str = NDARRAY_DTYPE_DT64 + "ns"  # 'Mns'
NDARRAY_DTYPE_DT64_PS: str = NDARRAY_DTYPE_DT64 + "ps"  # 'Mps'
NDARRAY_DTYPE_DT64_FS: str = NDARRAY_DTYPE_DT64 + "fs"  # 'Mfs'
NDARRAY_DTYPE_DT64_AS: str = NDARRAY_DTYPE_DT64 + "as"  # 'Mas'
NDARRAY_DTYPE_TD64_Y: str = NDARRAY_DTYPE_TD64 + "Y"  # 'mY'
NDARRAY_DTYPE_TD64_M: str = NDARRAY_DTYPE_TD64 + "M"  # 'mM'
NDARRAY_DTYPE_TD64_W: str = NDARRAY_DTYPE_TD64 + "W"  # 'mW'
NDARRAY_DTYPE_TD64_D: str = NDARRAY_DTYPE_TD64 + "D"  # 'mD'
NDARRAY_DTYPE_TD64_B: str = NDARRAY_DTYPE_TD64 + "B"  # 'mB'
NDARRAY_DTYPE_TD64_H: str = NDARRAY_DTYPE_TD64 + "h"  # 'mh'
NDARRAY_DTYPE_TD64_MI: str = NDARRAY_DTYPE_TD64 + "m"  # 'mm'
NDARRAY_DTYPE_TD64_S: str = NDARRAY_DTYPE_TD64 + "s"  # 'ms'
NDARRAY_DTYPE_TD64_MS: str = NDARRAY_DTYPE_TD64 + "ms"  # 'mms'
NDARRAY_DTYPE_TD64_US: str = NDARRAY_DTYPE_TD64 + "us"  # 'mus'
NDARRAY_DTYPE_TD64_NS: str = NDARRAY_DTYPE_TD64 + "ns"  # 'mns'
NDARRAY_DTYPE_TD64_PS: str = NDARRAY_DTYPE_TD64 + "ps"  # 'mps'
NDARRAY_DTYPE_TD64_FS: str = NDARRAY_DTYPE_TD64 + "fs"  # 'mfs'
NDARRAY_DTYPE_TD64_AS: str = NDARRAY_DTYPE_TD64 + "as"  # 'mas'
# variant ndarray
NDARRAY_OBJECT: str = NDARRAY + NDARRAY_DTYPE_OBJECT  # 'NO'
NDARRAY_FLOAT: str = NDARRAY + NDARRAY_DTYPE_FLOAT  # 'Nf'
NDARRAY_INT: str = NDARRAY + NDARRAY_DTYPE_INT  # 'Ni'
NDARRAY_UINT: str = NDARRAY + NDARRAY_DTYPE_UINT  # 'Nu'
NDARRAY_BOOL: str = NDARRAY + NDARRAY_DTYPE_BOOL  # 'Nb'
NDARRAY_COMPLEX: str = NDARRAY + NDARRAY_DTYPE_COMPLEX  # 'Nc'
NDARRAY_BYTES: str = NDARRAY + NDARRAY_DTYPE_BYTES  # 'NS'
NDARRAY_UNICODE: str = NDARRAY + NDARRAY_DTYPE_UNICODE  # 'NU'

# Pandas Types ----------------------------------------------------------------------
DATETIMEINDEX: str = "Z"
TIMEDELTAINDEX: str = "X"
# . identifier
DATETIMEINDEX_ID: cython.Py_UCS4 = read_char(DATETIMEINDEX, 0)
TIMEDELTAINDEX_ID: cython.Py_UCS4 = read_char(TIMEDELTAINDEX, 0)

# Pandas.Series ---------------------------------------------------------------------
SERIES: str = "I"
# . identifier
SERIES_ID: cython.Py_UCS4 = read_char(SERIES, 0)
# . variant
SERIES_OBJECT: str = SERIES + NDARRAY_DTYPE_OBJECT  # 'IO'
SERIES_FLOAT: str = SERIES + NDARRAY_DTYPE_FLOAT  # 'If'
SERIES_INT: str = SERIES + NDARRAY_DTYPE_INT  # 'Ii'
SERIES_UINT: str = SERIES + NDARRAY_DTYPE_UINT  # 'Iu'
SERIES_BOOL: str = SERIES + NDARRAY_DTYPE_BOOL  # 'Ib'
SERIES_COMPLEX: str = SERIES + NDARRAY_DTYPE_COMPLEX  # 'Ic'
SERIES_BYTES: str = SERIES + NDARRAY_DTYPE_BYTES  # 'IS'
SERIES_UNICODE: str = SERIES + NDARRAY_DTYPE_UNICODE  # 'IU'

# Pandas.DataFrame ------------------------------------------------------------------
DATAFRAME: str = "F"
# . identifier
DATAFRAME_ID: cython.Py_UCS4 = read_char(DATAFRAME, 0)
# . variant
DATAFRAME_COL_OBJECT: str = NDARRAY_DTYPE_OBJECT  # 'O'
DATAFRAME_COL_FLOAT: str = NDARRAY_DTYPE_FLOAT  # 'f'
DATAFRAME_COL_INT: str = NDARRAY_DTYPE_INT  # 'i'
DATAFRAME_COL_UINT: str = NDARRAY_DTYPE_UINT  # 'u'
DATAFRAME_COL_BOOL: str = NDARRAY_DTYPE_BOOL  # 'b'
DATAFRAME_COL_COMPLEX: str = NDARRAY_DTYPE_COMPLEX  # 'c'
DATAFRAME_COL_BYTES: str = NDARRAY_DTYPE_BYTES  # 'S'
DATAFRAME_COL_UNICODE: str = NDARRAY_DTYPE_UNICODE  # 'U'
