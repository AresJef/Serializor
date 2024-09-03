# cython: language_level=3
from libc.limits cimport USHRT_MAX, UINT_MAX

# Constants
cdef:
    # . orjson options
    object OPT_SERIALIZE_NUMPY
    # . encoded integer
    Py_UCS4 UINT8_ENCODE_VALUE
    Py_UCS4 UINT16_ENCODE_VALUE
    Py_UCS4 UINT32_ENCODE_VALUE
    Py_UCS4 UINT64_ENCODE_VALUE
    # . native types identifier
    str DATETIME_ID
    str DATE_ID
    str TIME_ID
    str TIMEDELTA_ID
    str STRUCT_TIME_ID
    str COMPLEX_ID
    str LIST_ID
    str TUPLE_ID
    str SET_ID
    str FROZENSET_ID
    str RANGE_ID
    str DICT_ID
    # . numpy identifider & dtype
    str DT64_ID
    str TD64_ID
    str NDARRAY_ID
    str NDARRAY_OBJECT_DT
    str NDARRAY_INT_DT
    str NDARRAY_UINT_DT
    str NDARRAY_FLOAT_DT
    str NDARRAY_BOOL_DT
    str NDARRAY_DT64_DT
    str NDARRAY_TD64_DT
    str NDARRAY_COMPLEX_DT
    str NDARRAY_BYTES_DT
    str NDARRAY_UNICODE_DT
    str NDARRAY_OBJECT_IDDT
    str NDARRAY_INT_IDDT
    str NDARRAY_UINT_IDDT
    str NDARRAY_FLOAT_IDDT
    str NDARRAY_BOOL_IDDT
    str NDARRAY_DT64_IDDT
    str NDARRAY_TD64_IDDT
    str NDARRAY_COMPLEX_IDDT
    str NDARRAY_BYTES_IDDT
    str NDARRAY_UNICODE_IDDT
    # . pandas identifider
    str PD_TIMESTAMP_ID
    str PD_TIMEDELTA_ID
    str SERIES_ID
    str DATETIMEINDEX_ID
    str TIMEDELTAINDEX_ID
    str DATAFRAME_ID
    # . value
    str TRUE_VALUE
    str FALSE_VALUE
    str NONE_VALUE

# Utils: Pack unsigned integers
cdef inline str pack_uint8(unsigned int value):
    """Pack `UNSIGNED` 8-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<B", value).decode("latin1")
    """
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return buffer[0:1].decode("latin1")

cdef inline str pack_uint16(unsigned int value):
    """Pack `UNSIGNED` 16-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<H", value).decode("latin1")
    """
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return buffer[0:2].decode("latin1")

cdef inline str pack_uint24(unsigned int value):
    """Pack `UNSIGNED` 24-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> (struct.pack("<I", value)[:3]).decode("latin1")
    """
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return buffer[0:3].decode("latin1")

cdef inline str pack_uint32(unsigned long long value):
    """Pack `UNSIGNED` 32-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<I", value).decode("latin1")
    """
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return buffer[0:4].decode("latin1")

cdef inline str pack_uint64(unsigned long long value):
    """Pack `UNSIGNED` 64-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<Q", value).decode("latin1")
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
    return buffer[0:8].decode("latin1")

# Utils: Pack signed integer
cdef inline str pack_int8(int value):
    """Pack `SIGNED` 8-bit integer in little-endian order to `<'str'>`.
    
    Equivalent to:
    >>> struct.pack("<b", value).encode("latin1")
    """
    cdef char buffer[1]
    buffer[0] = value & 0xFF
    return buffer[0:1].decode("latin1")

cdef inline str pack_int16(int value):
    """Pack `SIGNED` 16-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<h", value).decode("latin1")
    """
    cdef char buffer[2]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    return buffer[0:2].decode("latin1")

cdef inline str pack_int24(int value):
    """Pack `SIGNED` 24-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> (struct.pack("<i", value)[:3]).decode("latin1")
    """
    cdef char buffer[3]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    return buffer[0:3].decode("latin1")

cdef inline str pack_int32(long long value):
    """Pack `SIGNED` 32-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<i", value).decode("latin1")
    """
    cdef char buffer[4]
    buffer[0] = value & 0xFF
    buffer[1] = (value >> 8) & 0xFF
    buffer[2] = (value >> 16) & 0xFF
    buffer[3] = (value >> 24) & 0xFF
    return buffer[0:4].decode("latin1")

cdef inline str pack_int64(long long value):
    """Pack `SIGNED` 64-bit integer in little-endian order to `<'str'>`.

    Equivalent to:
    >>> struct.pack("<q", value).decode("latin1")
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
    return buffer[0:8].decode("latin1")

# Encoded integer
cdef inline str gen_encoded_int(unsigned long long i):
    """Generate encoded integer `<'str'>`."""
    cdef char buffer[9]
    if i <= UINT8_ENCODE_VALUE:
        buffer[0] = i & 0xFF
        return buffer[0:1].decode("latin1")
    elif i <= USHRT_MAX:
        buffer[0] = UINT16_ENCODE_VALUE
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        return buffer[0:3].decode("latin1")
    elif i <= UINT_MAX:
        buffer[0] = UINT32_ENCODE_VALUE
        buffer[1] = i & 0xFF
        buffer[2] = (i >> 8) & 0xFF
        buffer[3] = (i >> 16) & 0xFF
        buffer[4] = (i >> 24) & 0xFF
        return buffer[:5].decode("latin1")
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
        return buffer[0:9].decode("latin1")

# Header
cdef inline str gen_header(unsigned char identifier, unsigned long long i):
    """Generate header '[identifier]+[encoded integer]' `<'str'>`."""
    cdef char buffer[10]
    if i <= UINT8_ENCODE_VALUE:
        buffer[0] = identifier
        buffer[1] = i & 0xFF
        return buffer[0:2].decode("latin1")
    elif i <= USHRT_MAX:
        buffer[0] = identifier
        buffer[1] = UINT16_ENCODE_VALUE
        buffer[2] = i & 0xFF
        buffer[3] = (i >> 8) & 0xFF
        return buffer[0:4].decode("latin1")
    elif i <= UINT_MAX:
        buffer[0] = identifier
        buffer[1] = UINT32_ENCODE_VALUE
        buffer[2] = i & 0xFF
        buffer[3] = (i >> 8) & 0xFF
        buffer[4] = (i >> 16) & 0xFF
        buffer[5] = (i >> 24) & 0xFF
        return buffer[:6].decode("latin1")
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
        return buffer[0:10].decode("latin1")

# Serialize
cpdef str serialize(object obj)
