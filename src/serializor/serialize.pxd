# cython: language_level=3

from cpython.bytes cimport PyBytes_AsString, PyBytes_GET_SIZE
from cpython.bytearray cimport PyByteArray_AsString, PyByteArray_GET_SIZE
from cpython.unicode cimport PyUnicode_DecodeUTF8, PyUnicode_AsEncodedString
cimport numpy as np
from numpy cimport PyArray_GETITEM, PyArray_SETITEM
from numpy cimport PyArray_GETPTR1, PyArray_GETPTR2, PyArray_GETPTR3, PyArray_GETPTR4

# Constants
cdef:
    # . functions
    object FN_ORJSON_DUMPS

# Utils
# . encode
cdef inline object encode_str(object obj):
    """Encode string to bytes using 'utf-8' encoding 
    with 'surrogateescape' error handling `<'bytes'>`."""
    return PyUnicode_AsEncodedString(obj, "utf-8", "surrogateescape")

# . decode
cdef inline str decode_bytes(object obj):
    """Decode bytes to string using 'utf-8' encoding 
    with 'surrogateescape' error handling `<'str'>`."""
    cdef char* s = PyBytes_AsString(obj)
    cdef Py_ssize_t size = PyBytes_GET_SIZE(obj)
    return PyUnicode_DecodeUTF8(s, size, "surrogateescape")

cdef inline str decode_bytearray(object obj):
    """Decode bytearray to string using 'utf-8' encoding 
    with 'surrogateescape' error handling `<'str'>`."""
    cdef char* s = PyByteArray_AsString(obj)
    cdef Py_ssize_t size = PyByteArray_GET_SIZE(obj)
    return PyUnicode_DecodeUTF8(s, size, "surrogateescape")

# . get ndarray item
cdef inline object ndarray_getitem_1d(np.ndarray arr, np.npy_intp i):
    """Get item from 1-dimensional numpy ndarray `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline object ndarray_getitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    """Get item from 2-dimensional numpy ndarray `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline object ndarray_getitem_3d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k):
    """Get item from 3-dimensional numpy ndarray `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR3(arr, i, j, k)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline object ndarray_getitem_4d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, np.npy_intp l):
    """Get item from 4-dimensional numpy ndarray `<'object'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR4(arr, i, j, k, l)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

# . set ndarray item
cdef inline int ndarray_setitem_1d(np.ndarray arr, np.npy_intp i, object item):
    """Set item for 1-dimensional numpy ndarray `<'int'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline int ndarray_setitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j, object item):
    """Set item for 2-dimensional numpy ndarray `<'int'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline int ndarray_setitem_3d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, object item):
    """Set item for 3-dimensional numpy ndarray `<'int'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR3(arr, i, j, k)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline int ndarray_setitem_4d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, np.npy_intp l, object item):
    """Set item for 4-dimensional numpy ndarray `<'int'>`."""
    cdef void* itemptr = <void*>PyArray_GETPTR4(arr, i, j, k, l)
    return PyArray_SETITEM(arr, itemptr, item)

# Serialize
cpdef str serialize(object obj)
