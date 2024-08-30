# cython: language_level=3
from libc.stdlib cimport strtoll, strtold
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.unicode cimport PyUnicode_DecodeUTF8, PyUnicode_DecodeASCII
from serializor.binary.ser cimport (  # type: ignore
    UINT8_ENCODE_VALUE,
    UINT16_ENCODE_VALUE,
    UINT32_ENCODE_VALUE,
    UINT64_ENCODE_VALUE,
)

# Utils: Unpack unsigned integers
cdef inline unsigned char unpack_uint8(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 8-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    return <unsigned char> data[pos]

cdef inline unsigned short unpack_uint16(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 16-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned short p0 = <unsigned char> data[pos]
        unsigned short p1 = <unsigned char> data[pos + 1]
        unsigned short res = p0 | (p1 << 8)
    return res

cdef inline unsigned int unpack_uint24(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 24-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned int p0 = <unsigned char> data[pos]
        unsigned int p1 = <unsigned char> data[pos + 1]
        unsigned int p2 = <unsigned char> data[pos + 2]
        unsigned int res = p0 | (p1 << 8) | (p2 << 16)
    return res

cdef inline unsigned int unpack_uint32(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 32-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned int p0 = <unsigned char> data[pos]
        unsigned int p1 = <unsigned char> data[pos + 1]
        unsigned int p2 = <unsigned char> data[pos + 2]
        unsigned int p3 = <unsigned char> data[pos + 3]
        unsigned int res = p0 | (p1 << 8) | (p2 << 16) | (p3 << 24)
    return res

cdef inline unsigned long long unpack_uint64(char* data, Py_ssize_t pos):
    """Unpack (read) `UNSIGNED` 64-bit integer from 'data'
    at the given 'pos' in little-endian order `<'int'>`.
    """
    cdef:
        unsigned long long p0 = <unsigned char> data[pos]
        unsigned long long p1 = <unsigned char> data[pos + 1]
        unsigned long long p2 = <unsigned char> data[pos + 2]
        unsigned long long p3 = <unsigned char> data[pos + 3]
        unsigned long long p4 = <unsigned char> data[pos + 4]
        unsigned long long p5 = <unsigned char> data[pos + 5]
        unsigned long long p6 = <unsigned char> data[pos + 6]
        unsigned long long p7 = <unsigned char> data[pos + 7]
        unsigned long long res = (
            p0 | (p1 << 8) | (p2 << 16) | (p3 << 24) | (p4 << 32)
            | (p5 << 40) | (p6 << 48) | (p7 << 56) )
    return res

# Utils: Unpack signed integer
cdef inline signed char unpack_int8(char* data, Py_ssize_t pos):
    """Read (unpack) `SIGNED` 8-bit integer from 'data'
    at givent 'pos' in little-endian order `<'int'>`."""
    return <signed char> data[pos]

cdef inline short unpack_int16(char* data, Py_ssize_t pos):
    """Read (unpack) `SIGNED` 16-bit integer from 'data'
    at givent 'pos' in little-endian order `<'int'>`."""
    return <short> unpack_uint16(data, pos)

cdef inline int unpack_int24(char* data, Py_ssize_t pos):
    """Read (unpack) `SIGNED` 24-bit integer from 'data'
    at givent 'pos' in little-endian order `<'int'>`."""
    cdef int res = <int> unpack_uint24(data, pos)
    return res if res < 0x800000 else res - 0x1000000

cdef inline int unpack_int32(char* data, Py_ssize_t pos):
    """Read (unpack) `SIGNED` 32-bit integer from 'data'
    at givent 'pos' in little-endian order `<'int'>`."""
    return <int> unpack_uint32(data, pos)

cdef inline long long unpack_int64(char* data, Py_ssize_t pos):
    """Read (unpack) `SIGNED` 64-bit integer from 'data'
    at givent 'pos' in little-endian order `<'int'>`."""
    return <long long> unpack_uint64(data, pos)

# Encoded integer
cdef inline unsigned long long dec_encoded_int(char* data, Py_ssize_t pos[1]):
    """Decode encoded integer from 'data' at the given 'pos' pointer `<'int'>`.
    Note:
    - The 'pos' argument accepts the position pointer instead of the position value.
    - After decoding, the 'pos' pointer will be updated to the next position.
    """
    cdef:
        Py_ssize_t idx = pos[0]
        unsigned char code = <unsigned char> data[idx]

    if code <= UINT8_ENCODE_VALUE:
        pos[0] = idx + 1
        return code
    elif code == UINT16_ENCODE_VALUE:
        pos[0] = idx + 3
        return unpack_uint16(data, idx + 1)
    elif code == UINT32_ENCODE_VALUE:
        pos[0] = idx + 5
        return unpack_uint32(data, idx + 1)
    elif code == UINT64_ENCODE_VALUE:
        pos[0] = idx + 9
        return unpack_uint64(data, idx + 1)
    else:
        raise ValueError("Invalid encoded integer value: %d." % code)

# Slice & convert
cdef inline char* slice_to_chars(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice 'data' from 'start' with the given 'size' to `<'char*'>`."""
    cdef Py_ssize_t end = start + size
    if end < start:
        raise ValueError("Invalid slice size: [start]%d -> [end]%d." % (start, end))
    return data[start:end]

cdef inline bytes slice_to_bytes(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice 'data' from 'start' with the given 'size' to `<'bytes'>`."""
    return PyBytes_FromStringAndSize(data + start, size)

cdef inline str slice_to_str_utf8(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice 'data' from 'start' with the given 'size' to `<'str'>`.

    With 'UTF-8' encoding & 'surrogateescape' error handler.
    """
    return PyUnicode_DecodeUTF8(data + start, size, "surrogateescape")

cdef inline str slice_to_str_ascii(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice 'data' from 'start' with the given 'size' to `<'str'>`.

    With 'ascii' encoding & 'surrogateescape' error handler.
    """
    return PyUnicode_DecodeASCII(data + start, size, "surrogateescape")

cdef inline long long slice_to_int(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice 'data' from 'start' with the given 'size' to `<'int'>`."""
    return strtoll(slice_to_chars(data, start, size), NULL, 10)

cdef inline long double slice_to_float(char* data, Py_ssize_t start, Py_ssize_t size):
    """Slice 'data' from 'start' with the given 'size' to `<'float'>`."""
    return strtold(slice_to_chars(data, start, size), NULL)

# Deserialize
cpdef object deserialize(bytes data)
