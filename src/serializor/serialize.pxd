# cython: language_level=3

from cpython.bytes cimport PyBytes_AsString, PyBytes_GET_SIZE
from cpython.bytearray cimport PyByteArray_AsString, PyByteArray_GET_SIZE
from cpython.unicode cimport PyUnicode_DecodeUTF8, PyUnicode_AsEncodedString
cimport numpy as np
from numpy cimport PyArray_GETITEM, PyArray_SETITEM
from numpy cimport PyArray_GETPTR1, PyArray_GETPTR2, PyArray_GETPTR3, PyArray_GETPTR4

# Constants
cdef:
    # . characters
    char* CHARS_UTF8 = "utf-8"
    char* CHARS_SRGE = "surrogateescape"
    # . functions
    object FN_ORJSON_DUMPS

# Bytes
# . encode
cdef inline object bytes_encode_utf8(object obj):
    return PyUnicode_AsEncodedString(obj, CHARS_UTF8, CHARS_SRGE)

# . decode
cdef inline str bytes_decode_utf8(object obj):
    cdef char* s = PyBytes_AsString(obj)
    cdef Py_ssize_t size = PyBytes_GET_SIZE(obj)
    return PyUnicode_DecodeUTF8(s, size, CHARS_SRGE)

cdef inline str bytearray_decode_utf8(object obj):
    cdef char* s = PyByteArray_AsString(obj)
    cdef Py_ssize_t size = PyByteArray_GET_SIZE(obj)
    return PyUnicode_DecodeUTF8(s, size, CHARS_SRGE)

# Numpy ndarray
# . get item
cdef inline object ndarray_getitem_1d(np.ndarray arr, np.npy_intp i):
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline object ndarray_getitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j):
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline object ndarray_getitem_3d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k):
    cdef void* itemptr = <void*>PyArray_GETPTR3(arr, i, j, k)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

cdef inline object ndarray_getitem_4d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, np.npy_intp l):
    cdef void* itemptr = <void*>PyArray_GETPTR4(arr, i, j, k, l)
    cdef object item = PyArray_GETITEM(arr, itemptr)
    return item

# . set item
cdef inline int ndarray_setitem_1d(np.ndarray arr, np.npy_intp i, object item):
    cdef void* itemptr = <void*>PyArray_GETPTR1(arr, i)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline int ndarray_setitem_2d(np.ndarray arr, np.npy_intp i, np.npy_intp j, object item):
    cdef void* itemptr = <void*>PyArray_GETPTR2(arr, i, j)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline int ndarray_setitem_3d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, object item):
    cdef void* itemptr = <void*>PyArray_GETPTR3(arr, i, j, k)
    return PyArray_SETITEM(arr, itemptr, item)

cdef inline int ndarray_setitem_4d(np.ndarray arr, np.npy_intp i, np.npy_intp j, np.npy_intp k, np.npy_intp l, object item):
    cdef void* itemptr = <void*>PyArray_GETPTR4(arr, i, j, k, l)
    return PyArray_SETITEM(arr, itemptr, item)

# Serialize
cdef str capi_serialize(object obj)