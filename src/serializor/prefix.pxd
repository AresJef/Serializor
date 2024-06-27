# cython: language_level=3

cimport numpy as np

# Basic Types
cdef:
    str STR, BOOL, FLOAT, INT, BYTES, NONE
    # . identifier
    Py_UCS4 STR_ID, BOOL_ID, FLOAT_ID, INT_ID, BYTES_ID, NONE_ID
    # . variant
    str BOOL_TRUE, BOOL_FALSE

# Date&Time Types
cdef:
    str DATE, TIME, DATETIME, TIMEDELTA
    # . identifier
    Py_UCS4 DATE_ID, TIME_ID, DATETIME_ID, TIMEDELTA_ID

# Numeric Types
cdef:
    str DECIMAL, COMPLEX
    # . identifier
    Py_UCS4 DECIMAL_ID, COMPLEX_ID

# Bytes Types
cdef:
    str BYTES
    # . identifier
    Py_UCS4 BYTES_ID

# Mapping Types
cdef:
    str DICT
    # . identifier
    Py_UCS4 DICT_ID

# Sequence Types
cdef:
    str LIST, TUPLE, SET
    # . identifier
    Py_UCS4 LIST_ID, TUPLE_ID, SET_ID

# Numpy Types
cdef:
    str DATETIME64, TIMEDELTA64
    # . identifier
    Py_UCS4 DATETIME64_ID, TIMEDELTA64_ID
    # . variant
    str DATETIME64_Y, DATETIME64_M, DATETIME64_W, DATETIME64_D 
    str DATETIME64_B, DATETIME64_H, DATETIME64_MI, DATETIME64_S
    str DATETIME64_MS, DATETIME64_US, DATETIME64_NS
    str DATETIME64_PS, DATETIME64_FS, DATETIME64_AS
    str TIMEDELTA64_Y, TIMEDELTA64_M, TIMEDELTA64_W, TIMEDELTA64_D
    str TIMEDELTA64_B, TIMEDELTA64_H, TIMEDELTA64_MI, TIMEDELTA64_S
    str TIMEDELTA64_MS, TIMEDELTA64_US, TIMEDELTA64_NS
    str TIMEDELTA64_PS, TIMEDELTA64_FS, TIMEDELTA64_AS

# Numpy.ndarray
cdef:
    np.ndarray _arr
    str NDARRAY
    str NDARRAY_DTYPE_OBJECT, NDARRAY_DTYPE_FLOAT
    str NDARRAY_DTYPE_INT, NDARRAY_DTYPE_UINT, NDARRAY_DTYPE_BOOL
    str NDARRAY_DTYPE_DT64, NDARRAY_DTYPE_TD64, NDARRAY_DTYPE_COMPLEX
    str NDARRAY_DTYPE_BYTES, NDARRAY_DTYPE_UNICODE
    # . identifier
    Py_UCS4 NDARRAY_ID
    Py_UCS4 NDARRAY_DTYPE_OBJECT_ID, NDARRAY_DTYPE_FLOAT_ID
    Py_UCS4 NDARRAY_DTYPE_INT_ID, NDARRAY_DTYPE_UINT_ID, NDARRAY_DTYPE_BOOL_ID
    Py_UCS4 NDARRAY_DTYPE_DT64_ID, NDARRAY_DTYPE_TD64_ID, NDARRAY_DTYPE_COMPLEX_ID
    Py_UCS4 NDARRAY_DTYPE_BYTES_ID, NDARRAY_DTYPE_UNICODE_ID
    # . variant dtype
    str NDARRAY_DTYPE_DT64_Y, NDARRAY_DTYPE_DT64_M, NDARRAY_DTYPE_DT64_W, NDARRAY_DTYPE_DT64_D
    str NDARRAY_DTYPE_DT64_B, NDARRAY_DTYPE_DT64_H, NDARRAY_DTYPE_DT64_MI, NDARRAY_DTYPE_DT64_S
    str NDARRAY_DTYPE_DT64_MS, NDARRAY_DTYPE_DT64_US, NDARRAY_DTYPE_DT64_NS
    str NDARRAY_DTYPE_DT64_PS, NDARRAY_DTYPE_DT64_FS, NDARRAY_DTYPE_DT64_AS
    str NDARRAY_DTYPE_TD64_Y, NDARRAY_DTYPE_TD64_M, NDARRAY_DTYPE_TD64_W, NDARRAY_DTYPE_TD64_D
    str NDARRAY_DTYPE_TD64_B, NDARRAY_DTYPE_TD64_H, NDARRAY_DTYPE_TD64_MI, NDARRAY_DTYPE_TD64_S
    str NDARRAY_DTYPE_TD64_MS, NDARRAY_DTYPE_TD64_US, NDARRAY_DTYPE_TD64_NS
    str NDARRAY_DTYPE_TD64_PS, NDARRAY_DTYPE_TD64_FS, NDARRAY_DTYPE_TD64_AS
    # . variant
    str NDARRAY_OBJECT, NDARRAY_FLOAT, NDARRAY_INT, NDARRAY_UINT
    str NDARRAY_BOOL, NDARRAY_COMPLEX, NDARRAY_BYTES, NDARRAY_UNICODE

# Pandas Types
cdef:
    str DATETIMEINDEX, TIMEDELTAINDEX
    # . identifier
    Py_UCS4 DATETIMEINDEX_ID, TIMEDELTAINDEX_ID

# Pandas.Series
cdef:
    str SERIES
    # . identifier
    Py_UCS4 SERIES_ID
    # . variant
    str SERIES_OBJECT, SERIES_FLOAT, SERIES_INT, SERIES_UINT
    str SERIES_BOOL, SERIES_COMPLEX, SERIES_BYTES, SERIES_UNICODE

# Pandas.DataFrame
cdef:
    str DATAFRAME
    # . identifier
    Py_UCS4 DATAFRAME_ID
    # . variant
    str DATAFRAME_COL_OBJECT, DATAFRAME_COL_FLOAT, DATAFRAME_COL_INT, DATAFRAME_COL_UINT
    str DATAFRAME_COL_BOOL, DATAFRAME_COL_COMPLEX, DATAFRAME_COL_BYTES, DATAFRAME_COL_UNICODE
