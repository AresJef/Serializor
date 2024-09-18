# cython: language_level=3

from cpython.bytes cimport (
    PyBytes_Size as bytes_len,
    PyBytes_AsString as bytes_to_chars,
)
from cpython.unicode cimport (
    PyUnicode_Decode,
    PyUnicode_DecodeUTF8,
    PyUnicode_DecodeASCII,
    PyUnicode_AsEncodedString,
    PyUnicode_ReadChar as str_read,
    PyUnicode_GET_LENGTH as str_len,
    PyUnicode_Substring as str_substr,
)
cimport numpy as np
from numpy cimport (
    NPY_ORDER,
    PyArray_Flatten,
    PyArray_GETPTR1,
    PyArray_GETPTR2,
    PyArray_GETPTR3,
    PyArray_GETPTR4,
    PyArray_GETITEM,
    PyArray_SETITEM,
)

# Constants
cdef:
    set AVAILABLE_TIMEZONES

# Utils: encode / decode
cdef inline bytes encode_str(object obj, char* encoding):
    """Encode string to bytes using the 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""
    return PyUnicode_AsEncodedString(obj, encoding, b"surrogateescape")

cdef inline str decode_bytes(object data, char* encoding):
    """Decode bytes to string using the 'encoding' with
    "surrogateescape" error handling `<'str'>`."""
    return PyUnicode_Decode(bytes_to_chars(data), bytes_len(data), encoding, b"surrogateescape")

cdef inline str decode_bytes_utf8(object data):
    """Decode bytes to string using 'utf-8' encoding with
    'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeUTF8(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

cdef inline str decode_bytes_ascii(object data):
    """Decode bytes to string using 'ascii' encoding with
    'surrogateescape' error handling `<'str'>`."""
    return PyUnicode_DecodeASCII(bytes_to_chars(data), bytes_len(data), b"surrogateescape")

# NumPy: ndarray get item
cdef inline object arr_getitem_1d(np.ndarray arr, np.npy_intp i):
    """Get item from 1-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    return PyArray_GETITEM(arr, itemptr)

cdef inline object arr_getitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from 2-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    return PyArray_GETITEM(arr, itemptr)

cdef inline object arr_getitem_3d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k):
    """Get item from 3-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR3(arr, i, j, k)
    return PyArray_GETITEM(arr, itemptr)

cdef inline object arr_getitem_4d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, np.npy_intp l):
    """Get item from 4-dimensional numpy ndarray as `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR4(arr, i, j, k, l)
    return PyArray_GETITEM(arr, itemptr)

# NumPy: ndarray set item
cdef inline bint arr_setitem_1d(np.ndarray arr, object item, np.npy_intp i) except -1:
    """Set item for 1-dimensional numpy ndarray."""
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline bint arr_setitem_2d(np.ndarray arr, object item, np.npy_intp i, np.npy_intp j) except -1:
    """Set item for 2-dimensional numpy ndarray."""
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline bint arr_setitem_3d(np.ndarray arr, object item, np.npy_intp i, np.npy_intp j, np.npy_intp k) except -1:
    """Set item for 3-dimensional numpy ndarray."""
    cdef void* itemptr = <void*>PyArray_GETPTR3(arr, i, j, k)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline bint arr_setitem_4d(np.ndarray arr, object item, np.npy_intp i, np.npy_intp j, np.npy_intp k, np.npy_intp l) except -1:
    """Set item for 4-dimensional numpy ndarray."""
    cdef void* itemptr = <void*>PyArray_GETPTR4(arr, i, j, k, l)
    return PyArray_SETITEM(arr, itemptr, item)

# NumPy: ndarray flatten
cdef inline np.ndarray arr_flatten(np.ndarray arr):
    """Flatten multi-dimensional numpy ndarray into 
    1-dimension with 'C-ORDER' `<'np.ndarray'>`."""
    return PyArray_Flatten(arr, NPY_ORDER.NPY_CORDER)

# NumPy: nptime unit
cdef inline str map_nptime_unit_int2str(int unit):
    """Map ndarray[datetime64/timedelta64] unit from integer
    to the corresponding string representation `<'str'>`."""
    # Common units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return "ns"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return "us"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return "ms"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return "s"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return "m"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return "h"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return "D"

    # Uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return "Y"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return "M"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return "W"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return "ps"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return "fs"
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return "as"
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return "B"

    # Unsupported unit
    raise ValueError("unsupported numpy time unit: %d." % unit)

cdef inline int map_nptime_unit_str2int(str unit):
    """Map ndarray[datetime64/timedelta64] unit from string
    representation to the corresponding integer `<'int'>`."""
    cdef:
        Py_ssize_t size = str_len(unit)
        Py_UCS4 ch

    # Unit: 'ns', 'us', 'ms', 'ps', 'fs', 'as'
    if size == 2:
        # Common units
        ch = str_read(unit, 0)
        if ch == "n":
            return np.NPY_DATETIMEUNIT.NPY_FR_ns
        if ch == "u":
            return np.NPY_DATETIMEUNIT.NPY_FR_us
        if ch == "m":
            return np.NPY_DATETIMEUNIT.NPY_FR_ms
        # Uncommon units
        if ch == "p":
            return np.NPY_DATETIMEUNIT.NPY_FR_ps
        if ch == "f":
            return np.NPY_DATETIMEUNIT.NPY_FR_fs
        if ch == "a":
            return np.NPY_DATETIMEUNIT.NPY_FR_as

    # Unit: 's', 'm', 'h', 'D', 'Y', 'M', 'W', 'B'
    elif size == 1:
        # Common units
        ch = str_read(unit, 0)
        if ch == "s":
            return np.NPY_DATETIMEUNIT.NPY_FR_s
        if ch == "m":
            return np.NPY_DATETIMEUNIT.NPY_FR_m
        if ch == "h":
            return np.NPY_DATETIMEUNIT.NPY_FR_h
        if ch == "D":
            return np.NPY_DATETIMEUNIT.NPY_FR_D
        # Uncommon units
        if ch == "Y":
            return np.NPY_DATETIMEUNIT.NPY_FR_Y
        if ch == "M":
            return np.NPY_DATETIMEUNIT.NPY_FR_M
        if ch == "W":
            return np.NPY_DATETIMEUNIT.NPY_FR_W
        # if ch == "B":
        #     return np.NPY_DATETIMEUNIT.NPY_FR_B

    # Unsupported unit
    raise ValueError("unsupported numpy time unit: %s." % unit)

cdef inline int parse_arr_nptime_unit(np.ndarray arr):
    """Parse numpy datetime64/timedelta64 unit from the
    given 'arr', returns the unit in `<'int'>`."""
    cdef:
        str dtype_str = arr.dtype.str
        Py_ssize_t size = str_len(dtype_str)

    if size < 6:
        raise ValueError("unable to parse ndarray time unit from '%s'." % dtype_str)
    dtype_str = str_substr(dtype_str, 4, size - 1)
    try:
        return map_nptime_unit_str2int(dtype_str)
    except ValueError as err:
        raise ValueError("unable to parse ndarray time unit from '%s'." % dtype_str) from err
