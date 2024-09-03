# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_Size as bytes_len  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_AsString as bytes_to_chars  # type: ignore
from cython.cimports.cpython.list import PyList_AsTuple as list_to_tuple  # type: ignore
from cython.cimports.cpython.complex import PyComplex_FromDoubles as gen_complex  # type: ignore
from cython.cimports.serializor import identifier, typeref, utils  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import numpy as np, datetime
from zoneinfo import ZoneInfo
from orjson import loads as _loads
from serializor import identifier, typeref, utils, errors

__all__ = ["deserialize"]


# Orjson loads --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _orjson_loads(data: object) -> object:
    """(cfunc) Deserialize JSON string to python `<'object'>`.

    Based on [orjson](https://github.com/ijl/orjson) `'loads()'` function.
    """
    return _loads(data)


# Basic Types ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_str(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> str:
    """(cfunc) Deserialize 'data' at the given position as `<'str'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    idx: cython.Py_ssize_t = pos[0]
    pos[0] = idx + size  # set new position
    return slice_to_str_utf8(data, idx, size)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _des_int(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'int'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    idx: cython.Py_ssize_t = pos[0]
    pos[0] = idx + size  # set new position
    # Must in long long range: use c-func 'strtoll()'
    if size < 19:
        return slice_to_int(data, idx, size)  # type: ignore
    # Possible out of range: use built-in 'int()'
    return int(slice_to_bytes(data, idx, size))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _des_float(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'float'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    idx: cython.Py_ssize_t = pos[0]
    pos[0] = idx + size  # set new position
    return slice_to_float(data, idx, size)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _des_bool(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'bool'>`.

    Position updates to the next byte after deserialization.
    """
    val: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
    pos[0] += 2  # skip identifier & boolean value
    return True if val == 1 else False


@cython.cfunc
@cython.inline(True)
def _des_none(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'None'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    return None


# Date&Time Types -----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_datetime(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'datetime.datetime'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0]
    flag: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    yy: cython.int = unpack_uint16(data, idx + 2)  # type: ignore
    mm: cython.int = unpack_uint8(data, idx + 4)  # type: ignore
    dd: cython.int = unpack_uint8(data, idx + 5)  # type: ignore
    hh: cython.int = unpack_uint8(data, idx + 6)  # type: ignore
    mi: cython.int = unpack_uint8(data, idx + 7)  # type: ignore
    ss: cython.int = unpack_uint8(data, idx + 8)  # type: ignore
    us: cython.int = unpack_uint32(data, idx + 9)  # type: ignore
    # Without timezone
    if flag == 0:
        pos[0] += 13  # set new position
        return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, None, 0)
    # Timezone: approach 1 (tz_name)
    elif flag == 1:
        pos[0] += 13  # set new position
        size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        idx = pos[0]
        tz_name: str = slice_to_str_ascii(data, idx, size)  # type: ignore
        pos[0] += size  # set new position
        return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, ZoneInfo(tz_name), 0)
    # Timezone: approach 2 (tz_offset)
    else:
        offset = datetime.timedelta_new(
            unpack_int32(data, idx + 13),  # type: ignore
            unpack_int32(data, idx + 17),  # type: ignore
            unpack_int32(data, idx + 21),  # type: ignore
        )
        pos[0] += 25  # set new position
        tzinfo = datetime.timezone_new(offset, None)
        return datetime.datetime_new(yy, mm, dd, hh, mi, ss, us, tzinfo, 0)


@cython.cfunc
@cython.inline(True)
def _des_date(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'datetime.date'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    year: cython.int = unpack_uint16(data, idx)  # type: ignore
    month: cython.int = unpack_uint8(data, idx + 2)  # type: ignore
    day: cython.int = unpack_uint8(data, idx + 3)  # type: ignore
    pos[0] += 5  # set new position
    return datetime.date_new(year, month, day)


@cython.cfunc
@cython.inline(True)
def _des_time(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'datetime.time'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0]
    flag: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    hh: cython.int = unpack_uint8(data, idx + 2)  # type: ignore
    mi: cython.int = unpack_uint8(data, idx + 3)  # type: ignore
    ss: cython.int = unpack_uint8(data, idx + 4)  # type: ignore
    us: cython.int = unpack_uint32(data, idx + 5)  # type: ignore
    # Without timezone
    if flag == 0:
        pos[0] += 9
        return datetime.time_new(hh, mi, ss, us, None, 0)
    # Timezone: approach 1 (tz_name)
    elif flag == 1:
        pos[0] += 9  # set new position
        size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        idx = pos[0]
        tz_name: str = slice_to_str_ascii(data, idx, size)  # type: ignore
        pos[0] += size  # set new position
        return datetime.time_new(hh, mi, ss, us, ZoneInfo(tz_name), 0)
    # Timezone: approach 2 (tz_offset)
    else:
        offset = datetime.timedelta_new(
            unpack_int32(data, idx + 9),  # type: ignore
            unpack_int32(data, idx + 13),  # type: ignore
            unpack_int32(data, idx + 17),  # type: ignore
        )
        pos[0] += 21  # set new position
        tzinfo = datetime.timezone_new(offset, None)
        return datetime.time_new(hh, mi, ss, us, tzinfo, 0)


@cython.cfunc
@cython.inline(True)
def _des_timedelta(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'datetime.timedelta'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    days: cython.int = unpack_int32(data, idx)  # type: ignore
    seconds: cython.int = unpack_int32(data, idx + 4)  # type: ignore
    microseconds: cython.int = unpack_int32(data, idx + 8)  # type: ignore
    pos[0] += 13  # set new position
    return datetime.timedelta_new(days, seconds, microseconds)


@cython.cfunc
@cython.inline(True)
def _des_struct_time(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'time.struct_time'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    return typeref.STRUCT_TIME(_des_list(data, pos))


# Numeric Types -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_decimal(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'decimal.Decimal'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    idx: cython.Py_ssize_t = pos[0]
    pos[0] = idx + size  # set new position
    return typeref.DECIMAL(slice_to_str_ascii(data, idx, size))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _des_complex(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'complex'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    # Complex: real
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    real: cython.longdouble = slice_to_float(data, pos[0], size)  # type: ignore
    pos[0] += size  # set new position
    # Complex: imag
    size = dec_encoded_int(data, pos)  # type: ignore
    imag: cython.longdouble = slice_to_float(data, pos[0], size)  # type: ignore
    pos[0] += size  # set new position
    # Generate complex
    return gen_complex(real, imag)


# Bytes Types ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_bytes(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> bytes:
    """(cfunc) Deserialize 'data' at the given position as `<'bytes'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    idx: cython.Py_ssize_t = pos[0]
    pos[0] = idx + size  # set new position
    return slice_to_bytes(data, idx, size)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _des_bytearray(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'bytearray'>`.

    Position updates to the next byte after deserialization.
    """
    return bytearray(_des_bytes(data, pos))


# Sequence Types ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_list(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> list:
    """(cfunc) Deserialize 'data' at the given position as `<'list'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    return [_des_data(data, pos) for _ in range(size)]


@cython.cfunc
@cython.inline(True)
def _des_tuple(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'tuple'>`.

    Position updates to the next byte after deserialization.
    """
    return list_to_tuple(_des_list(data, pos))


@cython.cfunc
@cython.inline(True)
def _des_set(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'set'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    return {_des_data(data, pos) for _ in range(size)}


@cython.cfunc
@cython.inline(True)
def _des_frozenset(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'frozenset'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    return frozenset([_des_data(data, pos) for _ in range(size)])


@cython.cfunc
@cython.inline(True)
def _des_range(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'range'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    return range(
        _des_int(data, pos),  # start
        _des_int(data, pos),  # stop
        _des_int(data, pos),  # step
    )


# Mapping Types -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_dict(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'dict'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
    return {_des_data(data, pos): _des_data(data, pos) for _ in range(size)}


# NumPy Types ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_datetime64(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.datetime64'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    unit: cython.uchar = unpack_uint8(data, idx)  # type: ignore
    val: cython.longlong = unpack_int64(data, idx + 1)  # type: ignore
    pos[0] += 10  # set new position
    return typeref.DATETIME64(val, utils.map_nptime_unit_int2str(unit))


@cython.cfunc
@cython.inline(True)
def _des_timedelta64(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.timedelta64'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    unit: cython.uchar = unpack_uint8(data, idx)  # type: ignore
    val: cython.longlong = unpack_int64(data, idx + 1)  # type: ignore
    pos[0] += 10  # set new position
    return typeref.TIMEDELTA64(val, utils.map_nptime_unit_int2str(unit))


@cython.cfunc
@cython.inline(True)
def _des_ndarray(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    Position updates to the next byte after deserialization.
    """
    pos[0] += 1  # skip identifier
    # Get ndarray dtype
    arr_dtype: cython.uchar = data[pos[0]]

    # Deserialize ndarray
    # . ndarray[object]
    if arr_dtype == identifier.NDARRAY_OBJECT:
        return _des_ndarray_object(data, pos)
    # . ndarray[int]
    if arr_dtype == identifier.NDARRAY_INT:
        return _des_ndarray_numeric(data, pos)
    # . ndarray[uint]
    if arr_dtype == identifier.NDARRAY_UINT:
        return _des_ndarray_numeric(data, pos)
    # . ndarray[float]
    if arr_dtype == identifier.NDARRAY_FLOAT:
        return _des_ndarray_numeric(data, pos)
    # . ndarray[bool]
    if arr_dtype == identifier.NDARRAY_BOOL:
        return _des_ndarray_numeric(data, pos)
    # . ndarray[datetime64]
    if arr_dtype == identifier.NDARRAY_DT64:
        return _des_ndarray_nptime(data, pos, True)
    # . ndarray[timedelta64]
    if arr_dtype == identifier.NDARRAY_TD64:
        return _des_ndarray_nptime(data, pos, False)
    # . ndarray[complex]
    if arr_dtype == identifier.NDARRAY_COMPLEX:
        return _des_ndarray_complex(data, pos)
    # . ndarray[bytes]
    if arr_dtype == identifier.NDARRAY_BYTES:
        return _des_ndarray_bytes(data, pos)
    # . ndarray[str]
    if arr_dtype == identifier.NDARRAY_UNICODE:
        return _des_ndarray_unicode(data, pos)
    # . invalid dtype
    raise ValueError("unsupported <'numpy.ndarray'> dtype '%s'." % chr(arr_dtype))


@cython.cfunc
@cython.inline(True)
def _des_ndarray_object(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    #### This function is only for ndarray dtype: `"O" (object)`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip array dtype
    npy_dtype = unpack_uint8(data, idx)  # type: ignore
    ndim: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    pos[0] += 3  # set new position
    arr: np.ndarray
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i], npy_dtype, 0)
        if s_i == 0:
            return arr  # exit: empty
        # . deserialize values
        for i in range(s_i):
            utils.arr_setitem_1d(arr, _des_data(data, pos), i)
        return arr

    # 2-dimensions
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j], npy_dtype, 0)
        if s_j == 0:
            return arr  # exit: empty
        # . deserialize values
        for i in range(s_i):
            for j in range(s_j):
                utils.arr_setitem_2d(arr, _des_data(data, pos), i, j)
        return arr

    # 3-dimensions
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j, s_k], npy_dtype, 0)
        if s_k == 0:
            return arr  # exit: empty
        # . deserialize values
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    utils.arr_setitem_3d(arr, _des_data(data, pos), i, j, k)
        return arr

    # 4-dimensions
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_l: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j, s_k, s_l], npy_dtype, 0)
        if s_l == 0:
            return arr  # exit: empty
        # . deserialize values
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        utils.arr_setitem_4d(arr, _des_data(data, pos), i, j, k, l)
        return arr

    # Invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _des_ndarray_numeric(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    #### This function is for ndarray dtype: `"i" (int)`, `"u" (uint)`, `"f" (float)` and `"b" (bool)`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip array dtype
    npy_dtype = unpack_uint8(data, idx)  # type: ignore
    ndim: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    pos[0] += 3  # set new position
    arr: np.ndarray
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i], npy_dtype, 0)
        if s_i == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            utils.arr_setitem_1d(arr, next(values), i)
        return arr

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j], npy_dtype, 0)
        if s_j == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                utils.arr_setitem_2d(arr, next(values), i, j)
        return arr

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j, s_k], npy_dtype, 0)
        if s_k == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    utils.arr_setitem_3d(arr, next(values), i, j, k)
        return arr

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_l: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j, s_k, s_l], npy_dtype, 0)
        if s_l == 0:
            return arr
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        utils.arr_setitem_4d(arr, next(values), i, j, k, l)
        return arr

    # Invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _des_ndarray_nptime(
    data: cython.pchar,
    pos: cython.Py_ssize_t[1],
    is_dt64: cython.bint,
) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    #### This function is for ndarray dtype: `"M" (datetime64)`, `"m" (timedelta64)`.

    Position updates to the next byte after deserialization.
    """
    ndim: cython.uchar = unpack_uint8(data, pos[0] + 2)  # type: ignore
    pos[0] += 3  # set new position
    arr: np.ndarray
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        unit: str = utils.map_nptime_unit_int2str(unpack_uint8(data, pos[0]))  # type: ignore
        pos[0] += 1  # skip unit
        arr = np.empty(
            s_i,
            "datetime64[%s]" % unit if is_dt64 else "timedelta64[%s]" % unit,
        )
        if s_i == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            utils.arr_setitem_1d(arr, next(values), i)
        return arr

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        unit: str = utils.map_nptime_unit_int2str(unpack_uint8(data, pos[0]))  # type: ignore
        pos[0] += 1  # skip unit
        arr = np.empty(
            [s_i, s_j],
            "datetime64[%s]" % unit if is_dt64 else "timedelta64[%s]" % unit,
        )
        if s_j == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                utils.arr_setitem_2d(arr, next(values), i, j)
        return arr

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        unit: str = utils.map_nptime_unit_int2str(unpack_uint8(data, pos[0]))  # type: ignore
        pos[0] += 1  # skip unit
        arr = np.empty(
            [s_i, s_j, s_k],
            "datetime64[%s]" % unit if is_dt64 else "timedelta64[%s]" % unit,
        )
        if s_k == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    utils.arr_setitem_3d(arr, next(values), i, j, k)
        return arr

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_l: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        unit: str = utils.map_nptime_unit_int2str(unpack_uint8(data, pos[0]))  # type: ignore
        pos[0] += 1  # skip unit
        arr = np.empty(
            [s_i, s_j, s_k, s_l],
            "datetime64[%s]" % unit if is_dt64 else "timedelta64[%s]" % unit,
        )
        if s_l == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        utils.arr_setitem_4d(arr, next(values), i, j, k, l)
        return arr

    # Invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _des_ndarray_complex(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    #### This function is for only ndarray dtype: `"c" (complex)`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip array dtype
    npy_dtype = unpack_uint8(data, idx)  # type: ignore
    ndim: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    pos[0] += 3  # set new position
    arr: np.ndarray
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i], npy_dtype, 0)
        if s_i == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            utils.arr_setitem_1d(arr, gen_complex(next(values), next(values)), i)
        return arr

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j], npy_dtype, 0)
        if s_j == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                utils.arr_setitem_2d(arr, gen_complex(next(values), next(values)), i, j)
        return arr

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j, s_k], npy_dtype, 0)
        if s_k == 0:
            return arr  # exit: empty
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    utils.arr_setitem_3d(arr, gen_complex(next(values), next(values)), i, j, k)  # type: ignore
        return arr

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_l: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        arr = np.PyArray_EMPTY(ndim, [s_i, s_j, s_k, s_l], npy_dtype, 0)  # type: ignore
        if s_l == 0:
            return arr
        # . deserialize values
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        utils.arr_setitem_4d(arr, gen_complex(next(values), next(values)), i, j, k, l)  # type: ignore
        return arr

    # Invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _des_ndarray_bytes(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    #### This function is only for ndarray dtype: `"S" (String)`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip array dtype
    npy_dtype = unpack_uint8(data, idx)  # type: ignore
    ndim: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    pos[0] += 3  # set new position
    arr: np.ndarray
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_i == 0:
            return np.PyArray_EMPTY(ndim, [s_i], npy_dtype, 0)  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty(s_i, "S%d" % ch_size)
        for i in range(s_i):
            utils.arr_setitem_1d(arr, _des_bytes(data, pos), i)
        return arr

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_j == 0:
            return np.PyArray_EMPTY(ndim, [s_i, s_j], npy_dtype, 0)  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty([s_i, s_j], "S%d" % ch_size)
        for i in range(s_i):
            for j in range(s_j):
                utils.arr_setitem_2d(arr, _des_bytes(data, pos), i, j)
        return arr

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_k == 0:
            return np.PyArray_EMPTY(ndim, [s_i, s_j, s_k], npy_dtype, 0)  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty([s_i, s_j, s_k], "S%d" % ch_size)
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    utils.arr_setitem_3d(arr, _des_bytes(data, pos), i, j, k)
        return arr

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_l: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_l == 0:
            return np.PyArray_EMPTY(
                ndim, [s_i, s_j, s_k, s_l], npy_dtype, 0
            )  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty([s_i, s_j, s_k, s_l], "S%d" % ch_size)
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        utils.arr_setitem_4d(arr, _des_bytes(data, pos), i, j, k, l)
        return arr

    # Invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _des_ndarray_unicode(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'np.ndarray'>`.

    #### This function is only for ndarray dtype: `"U" (Unicode)`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip array dtype
    npy_dtype = unpack_uint8(data, idx)  # type: ignore
    ndim: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    pos[0] += 3  # set new position
    arr: np.ndarray
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_i == 0:
            return np.PyArray_EMPTY(ndim, [s_i], npy_dtype, 0)  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty(s_i, "U%d" % ch_size)
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            utils.arr_setitem_1d(arr, next(values), i)
        return arr

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_j == 0:
            return np.PyArray_EMPTY(ndim, [s_i, s_j], npy_dtype, 0)  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty([s_i, s_j], "U%d" % ch_size)
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                utils.arr_setitem_2d(arr, next(values), i, j)
        return arr

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_k == 0:
            return np.PyArray_EMPTY(ndim, [s_i, s_j, s_k], npy_dtype, 0)  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty([s_i, s_j, s_k], "U%d" % ch_size)
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    utils.arr_setitem_3d(arr, next(values), i, j, k)
        return arr

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_j: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_k: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        s_l: cython.Py_ssize_t = dec_encoded_int(data, pos)  # type: ignore
        if s_l == 0:
            return np.PyArray_EMPTY(
                ndim, [s_i, s_j, s_k, s_l], npy_dtype, 0
            )  # exit: empty
        # . deserialize values
        ch_size: cython.int = dec_encoded_int(data, pos)  # type: ignore
        arr = np.empty([s_i, s_j, s_k, s_l], "U%d" % ch_size)
        values_len: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        values_json: bytes = slice_to_bytes(data, pos[0], values_len)  # type: ignore
        pos[0] += values_len  # set new position
        values = iter(_orjson_loads(values_json))
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        utils.arr_setitem_4d(arr, next(values), i, j, k, l)
        return arr

    # Invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


# Pandas Types --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_pd_timestamp(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'pd.Timestamp'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0]
    flag: cython.uchar = unpack_uint8(data, idx + 1)  # type: ignore
    val: cython.longlong = unpack_int64(data, idx + 2)  # type: ignore
    # Without timezone
    if flag == 0:
        pos[0] += 10  # set new position
        return typeref.PD_TIMESTAMP(val)
    # Timezone: approach 1 (tz_name)
    elif flag == 1:
        pos[0] += 10  # set new position
        size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        idx = pos[0]
        tz_name: str = slice_to_str_ascii(data, idx, size)  # type: ignore
        pos[0] += size  # set new position
        return typeref.PD_TIMESTAMP(val, tz=tz_name)
    # Timezone: approach 2 (tz_offset)
    else:
        offset = datetime.timedelta_new(
            unpack_int32(data, idx + 10),  # type: ignore
            unpack_int32(data, idx + 14),  # type: ignore
            unpack_int32(data, idx + 18),  # type: ignore
        )
        pos[0] += 22  # set new position
        tzinfo = datetime.timezone_new(offset, None)
        return typeref.PD_TIMESTAMP(val, tzinfo=tzinfo)


@cython.cfunc
@cython.inline(True)
def _des_pd_timedelta(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'pd.Timedelta'>`.

    Position updates to the next byte after deserialization.
    """
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    val: cython.longlong = unpack_int64(data, idx)  # type: ignore
    pos[0] += 9  # set new position
    return typeref.PD_TIMEDELTA(val)


@cython.cfunc
@cython.inline(True)
def _des_series(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'pd.Series'>`.

    Position updates to the next byte after deserialization.
    """
    # Deserialize name
    has_name: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
    pos[0] += 2  # skip identifier & has_name
    name = None if has_name == 0 else _des_data(data, pos)

    # Deserialize values
    arr_dtype: cython.uchar = data[pos[0]]
    # . ndarray[object]
    if arr_dtype == identifier.NDARRAY_OBJECT:
        arr = _des_ndarray_object(data, pos)
    # . ndarray[int]
    elif arr_dtype == identifier.NDARRAY_INT:
        arr = _des_ndarray_numeric(data, pos)
    # . ndarray[uint]
    elif arr_dtype == identifier.NDARRAY_UINT:
        arr = _des_ndarray_numeric(data, pos)
    # . ndarray[float]
    elif arr_dtype == identifier.NDARRAY_FLOAT:
        arr = _des_ndarray_numeric(data, pos)
    # . ndarray[bool]
    elif arr_dtype == identifier.NDARRAY_BOOL:
        arr = _des_ndarray_numeric(data, pos)
    # . ndarray[datetime64]
    elif arr_dtype == identifier.NDARRAY_DT64:
        flag: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
        pos[0] += 2  # skip arr dtype & flag
        # . Without timezone
        if flag == 0:
            arr = _des_ndarray_nptime(data, pos, True)
        # . Timezone: approach 1 (tz_name)
        elif flag == 1:
            size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
            tz_name: str = slice_to_str_ascii(data, pos[0], size)  # type: ignore
            pos[0] += size  # set new position
            arr = _des_ndarray_nptime(data, pos, True)
            series = typeref.SERIES(arr, name=name, copy=False)
            return series.dt.tz_localize("UTC").dt.tz_convert(tz_name)  # exit
        # . Timezone: approach 2 (tz_offset)
        else:
            idx: cython.Py_ssize_t = pos[0]
            offset = datetime.timedelta_new(
                unpack_int32(data, idx),  # type: ignore
                unpack_int32(data, idx + 4),  # type: ignore
                unpack_int32(data, idx + 8),  # type: ignore
            )
            pos[0] += 12  # set new position
            tzinfo = datetime.timezone_new(offset, None)
            arr = _des_ndarray_nptime(data, pos, True)
            series = typeref.SERIES(arr, name=name, copy=False)
            return series.dt.tz_localize("UTC").dt.tz_convert(tzinfo)  # exit
    # . ndarray[timedelta64]
    elif arr_dtype == identifier.NDARRAY_TD64:
        arr = _des_ndarray_nptime(data, pos, False)
    # . ndarray[complex]
    elif arr_dtype == identifier.NDARRAY_COMPLEX:
        arr = _des_ndarray_complex(data, pos)
    # . ndarray[bytes]
    elif arr_dtype == identifier.NDARRAY_BYTES:
        arr = _des_ndarray_bytes(data, pos)
    # . ndarray[str]
    elif arr_dtype == identifier.NDARRAY_UNICODE:
        arr = _des_ndarray_unicode(data, pos)
    # . invalid dtype
    else:
        raise ValueError("unsupported <'pandas.Series'> dtype '%s'." % chr(arr_dtype))

    # Generate Series
    return typeref.SERIES(arr, name=name, copy=False)


@cython.cfunc
@cython.inline(True)
def _des_datetime_index(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'pd.DatetimeIndex'>`.

    Position updates to the next byte after deserialization.
    """
    # Deserialize name
    has_name: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
    pos[0] += 2  # skip identifier & has_name
    name = None if has_name == 0 else _des_data(data, pos)

    # Deserialize values
    flag: cython.uchar = unpack_uint8(data, pos[0])  # type: ignore
    pos[0] += 1  # skip arr dtype & flag
    # . Without timezone
    if flag == 0:
        arr = _des_ndarray_nptime(data, pos, True)
        return typeref.DATETIMEINDEX(arr, name=name, copy=False)
    # . Timezone: approach 1 (tz_name)
    elif flag == 1:
        size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
        tz_name: str = slice_to_str_ascii(data, pos[0], size)  # type: ignore
        pos[0] += size  # set new position
        arr = _des_ndarray_nptime(data, pos, True)
        dindex = typeref.DATETIMEINDEX(arr, name=name, copy=False)
        return dindex.tz_localize("UTC").tz_convert(tz_name)
    # . Timezone: approach 2 (tz_offset)
    else:
        idx: cython.Py_ssize_t = pos[0]
        offset = datetime.timedelta_new(
            unpack_int32(data, idx),  # type: ignore
            unpack_int32(data, idx + 4),  # type: ignore
            unpack_int32(data, idx + 8),  # type: ignore
        )
        pos[0] += 12
        tzinfo = datetime.timezone_new(offset, None)
        arr = _des_ndarray_nptime(data, pos, True)
        dindex = typeref.DATETIMEINDEX(arr, name=name, copy=False)
        return dindex.tz_localize("UTC").tz_convert(tzinfo)


@cython.cfunc
@cython.inline(True)
def _des_timedelta_index(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'pd.TimedeltaIndex'>`.

    Position updates to the next byte after deserialization.
    """
    # Deserialize name
    has_name: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
    pos[0] += 2  # skip identifier & has_name
    name = None if has_name == 0 else _des_data(data, pos)

    # Deserialize values
    arr = _des_ndarray_nptime(data, pos, False)
    return typeref.TIMEDELTAINDEX(arr, name=name, copy=False)


@cython.cfunc
@cython.inline(True)
def _des_dataframe(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position as `<'pd.DataFrame'>`.

    Position updates to the next byte after deserialization.
    """
    # Deserialize columns name
    has_cols: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
    pos[0] += 2  # skip identifier & has_cols
    # . empty w/o columns
    if has_cols == 0:
        return typeref.DATAFRAME()  # exit
    cols: list = _des_list(data, pos)
    has_row: cython.uchar = unpack_uint8(data, pos[0])  # type: ignore
    pos[0] += 1  # skip has_row
    # . empty w/o rows
    if has_row == 0:
        return typeref.DATAFRAME(columns=cols)  # exit

    # Deserialize columns values
    values: dict = {}
    for col in cols:
        arr_dtype: cython.uchar = data[pos[0]]
        # . ndarray[object]
        if arr_dtype == identifier.NDARRAY_OBJECT:
            values[col] = _des_ndarray_object(data, pos)
        # . ndarray[int]
        elif arr_dtype == identifier.NDARRAY_INT:
            values[col] = _des_ndarray_numeric(data, pos)
        # . ndarray[uint]
        elif arr_dtype == identifier.NDARRAY_UINT:
            values[col] = _des_ndarray_numeric(data, pos)
        # . ndarray[float]
        elif arr_dtype == identifier.NDARRAY_FLOAT:
            values[col] = _des_ndarray_numeric(data, pos)
        # . ndarray[bool]
        elif arr_dtype == identifier.NDARRAY_BOOL:
            values[col] = _des_ndarray_numeric(data, pos)
        # . ndarray[datetime64]
        elif arr_dtype == identifier.NDARRAY_DT64:
            flag: cython.uchar = unpack_uint8(data, pos[0] + 1)  # type: ignore
            pos[0] += 2  # skip arr dtype & flag
            # . Without timezone
            if flag == 0:
                values[col] = _des_ndarray_nptime(data, pos, True)
            # . Timezone: approach 1 (tz_name)
            elif flag == 1:
                size: cython.ulonglong = dec_encoded_int(data, pos)  # type: ignore
                tz_name: str = slice_to_str_ascii(data, pos[0], size)  # type: ignore
                pos[0] += size  # set new position
                arr = _des_ndarray_nptime(data, pos, True)
                dindex = typeref.DATETIMEINDEX(arr, copy=False)
                values[col] = dindex.tz_localize("UTC").tz_convert(tz_name)
            # . Timezone: approach 2 (tz_offset)
            else:
                idx: cython.Py_ssize_t = pos[0]
                offset = datetime.timedelta_new(
                    unpack_int32(data, idx),  # type: ignore
                    unpack_int32(data, idx + 4),  # type: ignore
                    unpack_int32(data, idx + 8),  # type: ignore
                )
                pos[0] += 12  # set new position
                tzinfo = datetime.timezone_new(offset, None)
                arr = _des_ndarray_nptime(data, pos, True)
                dindex = typeref.DATETIMEINDEX(arr, copy=False)
                values[col] = dindex.tz_localize("UTC").tz_convert(tzinfo)
        # . ndarray[timedelta64]
        elif arr_dtype == identifier.NDARRAY_TD64:
            values[col] = _des_ndarray_nptime(data, pos, False)
        # . ndarray[complex]
        elif arr_dtype == identifier.NDARRAY_COMPLEX:
            values[col] = _des_ndarray_complex(data, pos)
        # . ndarray[bytes]
        elif arr_dtype == identifier.NDARRAY_BYTES:
            values[col] = _des_ndarray_bytes(data, pos)
        # . ndarray[str]
        elif arr_dtype == identifier.NDARRAY_UNICODE:
            values[col] = _des_ndarray_unicode(data, pos)
        # . invalid dtype
        else:
            raise ValueError(
                "unsupported <'pandas.DataFrame'> column's dtype '%s'." % chr(arr_dtype)
            )

    # Generate DataFrame
    return typeref.DATAFRAME(values, columns=cols, copy=False)


# Deserialize ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _des_data(data: cython.pchar, pos: cython.Py_ssize_t[1]) -> object:
    """(cfunc) Deserialize 'data' at the given position back to an `<'object'>`.

    Position updates to the next byte after deserialization.
    """
    dtype_id: cython.uchar = data[pos[0]]

    # Basic Types
    # <'str'>
    if dtype_id == identifier.STR:
        return _des_str(data, pos)
    # <'int'>
    if dtype_id == identifier.INT:
        return _des_int(data, pos)
    # <'float'>
    if dtype_id == identifier.FLOAT:
        return _des_float(data, pos)
    # <'bool'>
    if dtype_id == identifier.BOOL:
        return _des_bool(data, pos)
    # <'NoneType'>
    if dtype_id == identifier.NONE:
        return _des_none(data, pos)

    # Date&Time Types
    # <'datetime.datetime'>
    if dtype_id == identifier.DATETIME:
        return _des_datetime(data, pos)
    # <'datetime.date'>
    if dtype_id == identifier.DATE:
        return _des_date(data, pos)
    # <'datetime.time'>
    if dtype_id == identifier.TIME:
        return _des_time(data, pos)
    # <'datetime.timedelta'>
    if dtype_id == identifier.TIMEDELTA:
        return _des_timedelta(data, pos)
    # <'time.struct_time'>
    if dtype_id == identifier.STRUCT_TIME:
        return _des_struct_time(data, pos)
    # <'pandas.Timestamp'>
    if dtype_id == identifier.PD_TIMESTAMP:
        return _des_pd_timestamp(data, pos)
    # <'pandas.Timedelta'>
    if dtype_id == identifier.PD_TIMEDELTA:
        return _des_pd_timedelta(data, pos)
    # <'numpy.datetime64'>
    if dtype_id == identifier.DATETIME64:
        return _des_datetime64(data, pos)
    # <'numpy.timedelta64'>
    if dtype_id == identifier.TIMEDELTA64:
        return _des_timedelta64(data, pos)

    # Numeric Types
    # <'decimal.Decimal'>
    if dtype_id == identifier.DECIMAL:
        return _des_decimal(data, pos)
    # <'complex'>
    if dtype_id == identifier.COMPLEX:
        return _des_complex(data, pos)

    # Bytes Types
    # <'bytes'>
    if dtype_id == identifier.BYTES:
        return _des_bytes(data, pos)
    # <'bytearray'>
    if dtype_id == identifier.BYTEARRAY:
        return _des_bytearray(data, pos)

    # Sequence Types
    # <'list'>
    if dtype_id == identifier.LIST:
        return _des_list(data, pos)
    # <'tuple'>
    if dtype_id == identifier.TUPLE:
        return _des_tuple(data, pos)
    # <'set'>
    if dtype_id == identifier.SET:
        return _des_set(data, pos)
    # <'fronzenset'>
    if dtype_id == identifier.FROZENSET:
        return _des_frozenset(data, pos)
    # <'range'>
    if dtype_id == identifier.RANGE:
        return _des_range(data, pos)

    # Mapping Types
    # <'dict'>
    if dtype_id == identifier.DICT:
        return _des_dict(data, pos)

    # NumPy Types
    # <'numpy.ndarray'>
    if dtype_id == identifier.NDARRAY:
        return _des_ndarray(data, pos)

    # Pandas Types
    # <'pandas.Series'>
    if dtype_id == identifier.SERIES:
        return _des_series(data, pos)
    # <'pandas.DataFrame'>
    if dtype_id == identifier.DATAFRAME:
        return _des_dataframe(data, pos)
    # <'pandas.DatetimeIndex'>
    if dtype_id == identifier.DATETIMEINDEX:
        return _des_datetime_index(data, pos)
    # <'pandas.TimedeltaIndex'>
    if dtype_id == identifier.TIMEDELTAINDEX:
        return _des_timedelta_index(data, pos)

    # Invalid 'data'
    raise ValueError("unknown 'data' identifer: '%s'." % chr(dtype_id))


@cython.ccall
def deserialize(data: bytes) -> object:
    """Deserialize the data (bytes) back to an `<'object'>`."""
    try:
        eof: cython.Py_ssize_t = bytes_len(data)
        if eof == 0:
            raise ValueError("'data' is empty.")
        pos: cython.Py_ssize_t[1] = [0]
        obj = _des_data(bytes_to_chars(data), pos)
        if pos[0] != eof:
            raise ValueError("improper [pos] %d / [eof] %d." % (pos[0], eof))
        return obj
    except Exception as err:
        raise errors.DeserializeError(
            "<'Serializor'>\nFailed to deserialize from: "
            "%s\n%r\nError: %s" % (type(data), data, err)
        ) from err


########## The following functions are for testing purpose only ##########
def _test_utils() -> None:
    _test_unpack_int8()
    _test_unpack_uint8()
    _test_unpack_int16()
    _test_unpack_uint16()
    _test_unpack_int24()
    _test_unpack_uint24()
    _test_unpack_int32()
    _test_unpack_uint32()
    _test_unpack_int64()
    _test_unpack_uint64()
    _test_dec_encoded_int()
    _test_slice_n_convert()


def _test_unpack_int8() -> None:
    import struct

    for val in range(-128, 128):
        b = struct.pack("<b", val)
        i = unpack_int8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack int8".ljust(80))

    del struct


def _test_unpack_uint8() -> None:
    import struct

    for val in range(0, 256):
        b = struct.pack("<B", val)
        i = unpack_uint8(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack uint8".ljust(80))

    del struct


def _test_unpack_int16() -> None:
    import struct

    for val in range(-32768, 32768):
        b = struct.pack("<h", val)
        i = unpack_int16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack int16".ljust(80))

    del struct


def _test_unpack_uint16() -> None:
    import struct

    for val in range(0, 65536):
        b = struct.pack("<H", val)
        i = unpack_uint16(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack uint16".ljust(80))

    del struct


def _test_unpack_int24() -> None:
    import struct

    for val in (-8388608, -8388607, 0, 8388606, 8388607):
        b = struct.pack("<i", val)[:3]
        i = unpack_int24(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack int24".ljust(80))

    del struct


def _test_unpack_uint24() -> None:
    import struct

    for val in (0, 1, 16777213, 16777214, 16777215):
        b = struct.pack("<I", val)[:3]
        i = unpack_uint24(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack uint24".ljust(80))

    del struct


def _test_unpack_int32() -> None:
    import struct

    for val in (-2147483648, -2147483647, 0, 2147483646, 2147483647):
        b = struct.pack("<i", val)
        i = unpack_int32(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack int32".ljust(80))

    del struct


def _test_unpack_uint32() -> None:
    import struct

    for val in (0, 1, 4294967293, 4294967294, 4294967295):
        b = struct.pack("<I", val)
        i = unpack_uint32(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack uint32".ljust(80))

    del struct


def _test_unpack_int64() -> None:
    import struct

    for val in (
        -9223372036854775808,
        -9223372036854775807,
        0,
        9223372036854775806,
        9223372036854775807,
    ):
        b = struct.pack("<q", val)
        i = unpack_int64(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack int64".ljust(80))

    del struct


def _test_unpack_uint64() -> None:
    import struct

    for val in (
        0,
        1,
        18446744073709551613,
        18446744073709551614,
        18446744073709551615,
    ):
        b = struct.pack("<Q", val)
        i = unpack_uint64(b, 0)  # type: ignore
        assert val == i, f"val {val} vs i {i}"
        print("- pass: %s" % val, end="\r")
    print("Pass Unpack uint64".ljust(80))

    del struct


def _test_dec_encoded_int() -> None:
    for enc, num in [
        (b"\xfb", 251),
        (b"\xfc\xfc\x00", 252),
        (b"\xfc\xff\xff", 65535),
        (b"\xfd\x00\x00\x01\x00", 65536),
        (b"\xfd\xff\xff\xff\xff", 4294967295),
        (b"\xfe\x00\x00\x00\x00\x01\x00\x00\x00", 4294967296),
        (b"\xfe\x00\x00\x00\x00\x02\x00\x00\x00", 8589934592),
    ]:
        i = dec_encoded_int(enc, [0])  # type: ignore
        assert num == i, f"num {num} vs i {i} - encode: {enc}"
    print("Pass Dec Encoded Integer".ljust(80))


def _test_slice_n_convert() -> None:
    val: bytes = "hello 1234 3.14 english 中文".encode("utf-8")
    chs: cython.pchar = val
    # slice to chars
    n1 = chs[0:5]
    x1: bytes = slice_to_chars(chs, 0, 5)  # type: ignore
    assert n1 == x1, f"n {n1} vs x {x1}"
    # slice to bytes
    n2 = val[0:5]
    x2 = slice_to_bytes(chs, 0, 5)  # type: ignore
    assert n2 == x2, f"n {n2} vs x {x2}"
    # slice to int
    n3 = 1234
    x3 = slice_to_int(chs, 6, 4)  # type: ignore
    assert n3 == x3, f"n {n3} vs x {x3}"
    # slice to float
    n4 = 3.14
    x4 = slice_to_float(chs, 11, 4)  # type: ignore
    assert n4 == x4, f"n {n4} vs x {x4}"
    # slice to str ascii
    n5 = "english"
    x5 = slice_to_str_ascii(chs, 16, 7)  # type: ignore
    assert n5 == x5, f"n {repr(n5)} vs x {repr(x5)}"
    # slice to str utf-8
    n6 = "中文"
    x6 = slice_to_str_utf8(chs, 24, len(n6.encode("utf8")))  # type: ignore
    assert n6 == x6, f"n {repr(n6)} vs x {repr(x6)}"
    print("Pass Slice & Convert".ljust(80))
