# cython: language_level=3
from libc.limits cimport USHRT_MAX, UINT_MAX
from cpython.bytes cimport PyBytes_FromStringAndSize

# Constants
cdef:
    # . orjson options
    object OPT_SERIALIZE_NUMPY
    # . encoded integer
    unsigned char UINT8_ENCODE_VALUE
    unsigned char UINT16_ENCODE_VALUE
    unsigned char UINT32_ENCODE_VALUE
    unsigned char UINT64_ENCODE_VALUE
    # . native types identifier
    bytes DATETIME_ID
    bytes DATE_ID
    bytes TIME_ID
    bytes TIMEDELTA_ID
    bytes STRUCT_TIME_ID
    bytes COMPLEX_ID
    bytes LIST_ID
    bytes TUPLE_ID
    bytes SET_ID
    bytes FROZENSET_ID
    bytes RANGE_ID
    bytes DICT_ID
    # . numpy identifider & dtype
    bytes DT64_ID
    bytes TD64_ID
    bytes NDARRAY_ID
    bytes NDARRAY_OBJECT_DT
    bytes NDARRAY_INT_DT
    bytes NDARRAY_UINT_DT
    bytes NDARRAY_FLOAT_DT
    bytes NDARRAY_BOOL_DT
    bytes NDARRAY_DT64_DT
    bytes NDARRAY_TD64_DT
    bytes NDARRAY_COMPLEX_DT
    bytes NDARRAY_BYTES_DT
    bytes NDARRAY_UNICODE_DT
    bytes NDARRAY_OBJECT_IDDT
    bytes NDARRAY_INT_IDDT
    bytes NDARRAY_UINT_IDDT
    bytes NDARRAY_FLOAT_IDDT
    bytes NDARRAY_BOOL_IDDT
    bytes NDARRAY_DT64_IDDT
    bytes NDARRAY_TD64_IDDT
    bytes NDARRAY_COMPLEX_IDDT
    bytes NDARRAY_BYTES_IDDT
    bytes NDARRAY_UNICODE_IDDT
    # . pandas identifider
    bytes PD_TIMESTAMP_ID
    bytes PD_TIMEDELTA_ID
    bytes SERIES_ID
    bytes DATETIMEINDEX_ID
    bytes TIMEDELTAINDEX_ID
    bytes DATAFRAME_ID
    # . value
    bytes TRUE_VALUE
    bytes FALSE_VALUE
    bytes NONE_VALUE

# Utils: Pack unsigned integers
cdef inline bytes pack_uint8(unsigned int value):
    """Pack `UNSIGNED` 8-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<B", value)
    """
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return PyBytes_FromStringAndSize(buffer, 1)

cdef inline bytes pack_uint16(unsigned int value):
    """Pack `UNSIGNED` 16-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<H", value)
    """
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 2)

cdef inline bytes pack_uint24(unsigned int value):
    """Pack `UNSIGNED` 24-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<I", value)[:3]
    """
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 3)

cdef inline bytes pack_uint32(unsigned long long value):
    """Pack `UNSIGNED` 32-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<I", value)
    """
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_uint64(unsigned long long value):
    """Pack `UNSIGNED` 64-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<Q", value)
    """
    cdef char buffer[8]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    buffer[4] = (value >> 32) & 0xFF
    buffer[5] = (value >> 40) & 0xFF
    buffer[6] = (value >> 48) & 0xFF
    buffer[7] = (value >> 56) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 8)

# Utils: Pack signed integer
cdef inline bytes pack_int8(int value):
    """Pack `SIGNED` 8-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<b", value)
    """
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return PyBytes_FromStringAndSize(buffer, 1)

cdef inline bytes pack_int16(int value):
    """Pack `SIGNED` 16-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<h", value)
    """
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 2)

cdef inline bytes pack_int24(int value):
    """Pack `SIGNED` 24-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<i", value)[:3]
    """
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 3)

cdef inline bytes pack_int32(long long value):
    """Pack `SIGNED` 32-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<i", value)
    """
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 4)

cdef inline bytes pack_int64(long long value):
    """Pack `SIGNED` 64-bit integer in little-endian order to `<'bytes'>`.
    
    Equivalent to:
    >>> struct.pack("<q", value)
    """
    cdef char buffer[8]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    buffer[4] = (value >> 32) & 0xFF
    buffer[5] = (value >> 40) & 0xFF
    buffer[6] = (value >> 48) & 0xFF
    buffer[7] = (value >> 56) & 0xFF
    return PyBytes_FromStringAndSize(buffer, 8)

# Encoded integer
cdef inline bytes gen_encoded_int(unsigned long long i):
    """Generate encoded integer `<'bytes'>`."""
    cdef char buffer[9]
    if i <= UINT8_ENCODE_VALUE:
        buffer[0] = i & 0xFF
        return PyBytes_FromStringAndSize(buffer, 1)
    elif i <= USHRT_MAX:
        buffer[0] = UINT16_ENCODE_VALUE
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 3)
    elif i <= UINT_MAX:
        buffer[0] = UINT32_ENCODE_VALUE
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        buffer[4] = (i >> 24) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 5)
    else:
        buffer[0] = UINT64_ENCODE_VALUE
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        buffer[4] = (i >> 24) & 0xFF
        buffer[5] = (i >> 32) & 0xFF
        buffer[6] = (i >> 40) & 0xFF
        buffer[7] = (i >> 48) & 0xFF
        buffer[8] = (i >> 56) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 9)

# Header
cdef inline bytes gen_header(unsigned char identifier, unsigned long long i):
    """Generate header '[identifier]+[encoded integer]' `<'bytes'>`."""
    cdef char buffer[10]
    if i <= UINT8_ENCODE_VALUE:
        buffer[0] = identifier
        buffer[1] = i & 0xFF
        return PyBytes_FromStringAndSize(buffer, 2)
    elif i <= USHRT_MAX:
        buffer[0] = identifier
        buffer[1] = UINT16_ENCODE_VALUE
        buffer[2] = i & 0xFF
        buffer[3] = (i >> 8) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 4)
    elif i <= UINT_MAX:
        buffer[0] = identifier
        buffer[1] = UINT32_ENCODE_VALUE
        buffer[2] = i & 0xFF
        buffer[3] = (i >> 8) & 0xFF
        buffer[4] = (i >> 16) & 0xFF
        buffer[5] = (i >> 24) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 6)
    else:
        buffer[0] = identifier
        buffer[1] = UINT64_ENCODE_VALUE
        buffer[2] = i & 0xFF
        buffer[3] = (i >> 8) & 0xFF
        buffer[4] = (i >> 16) & 0xFF
        buffer[5] = (i >> 24) & 0xFF
        buffer[6] = (i >> 32) & 0xFF
        buffer[7] = (i >> 40) & 0xFF
        buffer[8] = (i >> 48) & 0xFF
        buffer[9] = (i >> 56) & 0xFF
        return PyBytes_FromStringAndSize(buffer, 10)

# Serialize
cpdef bytes serialize(object obj)
