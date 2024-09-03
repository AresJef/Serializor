# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.libc.math import isnormal  # type: ignore
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.set import PySet_GET_SIZE as set_len  # type: ignore
from cython.cimports.cpython.set import PySet_Contains as set_contains  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_Size as bytes_len  # type: ignore
from cython.cimports.cpython.list import PyList_GET_SIZE as list_len  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GET_SIZE as tuple_len  # type: ignore
from cython.cimports.cpython.dict import PyDict_Size as dict_len  # type: ignore
from cython.cimports.cpython.complex import PyComplex_RealAsDouble as complex_getreal  # type: ignore
from cython.cimports.cpython.complex import PyComplex_ImagAsDouble as complex_getimag  # type: ignore
from cython.cimports.serializor import identifier, typeref, utils  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Iterable
import numpy as np, datetime
from time import struct_time
from orjson import dumps as _dumps, OPT_SERIALIZE_NUMPY
from pandas import (
    Timestamp,
    Timedelta,
    Series,
    DatetimeIndex,
    TimedeltaIndex,
    DataFrame,
)
from serializor import identifier, typeref, utils, errors

__all__ = ["serialize"]


# Constants -----------------------------------------------------------------------------------
# . encoded integer
UINT8_ENCODE_VALUE: cython.uchar = 251
UINT16_ENCODE_VALUE: cython.uchar = 252
UINT32_ENCODE_VALUE: cython.uchar = 253
UINT64_ENCODE_VALUE: cython.uchar = 254
# . native types identifier
DATETIME_ID: bytes = pack_uint8(identifier.DATETIME)  # type: ignore
DATE_ID: bytes = pack_uint8(identifier.DATE)  # type: ignore
TIME_ID: bytes = pack_uint8(identifier.TIME)  # type: ignore
TIMEDELTA_ID: bytes = pack_uint8(identifier.TIMEDELTA)  # type: ignore
STRUCT_TIME_ID: bytes = pack_uint8(identifier.STRUCT_TIME)  # type: ignore
COMPLEX_ID: bytes = pack_uint8(identifier.COMPLEX)  # type: ignore
LIST_ID: bytes = pack_uint8(identifier.LIST)  # type: ignore
TUPLE_ID: bytes = pack_uint8(identifier.TUPLE)  # type: ignore
SET_ID: bytes = pack_uint8(identifier.SET)  # type: ignore
FROZENSET_ID: bytes = pack_uint8(identifier.FROZENSET)  # type: ignore
RANGE_ID: bytes = pack_uint8(identifier.RANGE)  # type: ignore
DICT_ID: bytes = pack_uint8(identifier.DICT)  # type: ignore
# . numpy identifider & dtype
DT64_ID: bytes = pack_uint8(identifier.DATETIME64)  # type: ignore
TD64_ID: bytes = pack_uint8(identifier.TIMEDELTA64)  # type: ignore
NDARRAY_ID: bytes = pack_uint8(identifier.NDARRAY)  # type: ignore
NDARRAY_OBJECT_DT: bytes = pack_uint8(identifier.NDARRAY_OBJECT)  # type: ignore
NDARRAY_INT_DT: bytes = pack_uint8(identifier.NDARRAY_INT)  # type: ignore
NDARRAY_UINT_DT: bytes = pack_uint8(identifier.NDARRAY_UINT)  # type: ignore
NDARRAY_FLOAT_DT: bytes = pack_uint8(identifier.NDARRAY_FLOAT)  # type: ignore
NDARRAY_BOOL_DT: bytes = pack_uint8(identifier.NDARRAY_BOOL)  # type: ignore
NDARRAY_DT64_DT: bytes = pack_uint8(identifier.NDARRAY_DT64)  # type: ignore
NDARRAY_TD64_DT: bytes = pack_uint8(identifier.NDARRAY_TD64)  # type: ignore
NDARRAY_COMPLEX_DT: bytes = pack_uint8(identifier.NDARRAY_COMPLEX)  # type: ignore
NDARRAY_BYTES_DT: bytes = pack_uint8(identifier.NDARRAY_BYTES)  # type: ignore
NDARRAY_UNICODE_DT: bytes = pack_uint8(identifier.NDARRAY_UNICODE)  # type: ignore
NDARRAY_OBJECT_IDDT: bytes = NDARRAY_ID + NDARRAY_OBJECT_DT
NDARRAY_INT_IDDT: bytes = NDARRAY_ID + NDARRAY_INT_DT
NDARRAY_UINT_IDDT: bytes = NDARRAY_ID + NDARRAY_UINT_DT
NDARRAY_FLOAT_IDDT: bytes = NDARRAY_ID + NDARRAY_FLOAT_DT
NDARRAY_BOOL_IDDT: bytes = NDARRAY_ID + NDARRAY_BOOL_DT
NDARRAY_DT64_IDDT: bytes = NDARRAY_ID + NDARRAY_DT64_DT
NDARRAY_TD64_IDDT: bytes = NDARRAY_ID + NDARRAY_TD64_DT
NDARRAY_COMPLEX_IDDT: bytes = NDARRAY_ID + NDARRAY_COMPLEX_DT
NDARRAY_BYTES_IDDT: bytes = NDARRAY_ID + NDARRAY_BYTES_DT
NDARRAY_UNICODE_IDDT: bytes = NDARRAY_ID + NDARRAY_UNICODE_DT
# . pandas identifider
PD_TIMESTAMP_ID: bytes = pack_uint8(identifier.PD_TIMESTAMP)  # type: ignore
PD_TIMEDELTA_ID: bytes = pack_uint8(identifier.PD_TIMEDELTA)  # type: ignore
SERIES_ID: bytes = pack_uint8(identifier.SERIES)  # type: ignore
DATETIMEINDEX_ID: bytes = pack_uint8(identifier.DATETIMEINDEX)  # type: ignore
TIMEDELTAINDEX_ID: bytes = pack_uint8(identifier.TIMEDELTAINDEX)  # type: ignore
DATAFRAME_ID: bytes = pack_uint8(identifier.DATAFRAME)  # type: ignore
# . value
TRUE_VALUE: bytes = gen_header(identifier.BOOL, 1)  # type: ignore
FALSE_VALUE: bytes = gen_header(identifier.BOOL, 0)  # type: ignore
NONE_VALUE: bytes = pack_uint8(identifier.NONE)  # type: ignore


# Orjson dumps --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _orjson_dumps(obj: object) -> bytes:
    """(cfunc) Serialize python object to JSON string `<'bytes'>`.

    Based on [orjson](https://github.com/ijl/orjson) `'dumps()'` function.
    """
    return _dumps(obj)


@cython.cfunc
@cython.inline(True)
def _orjson_dumps_numpy(obj: object) -> bytes:
    """(cfunc) Serialize numpy.ndarray to JSON string `<'bytes'>`.

    Based on [orjson](https://github.com/ijl/orjson) `'dumps()'` function.
    """
    return _dumps(obj, option=OPT_SERIALIZE_NUMPY)


# Basic Types ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_str(obj: object) -> bytes:
    """(cfunc) Serialize 'str' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    val: bytes = utils.encode_str(obj, b"utf-8")
    return gen_header(identifier.STR, bytes_len(val)) + val  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_int(obj: object) -> bytes:
    """(cfunc) Serialize 'int' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    val: bytes = utils.encode_str(str(obj), b"ascii")
    return gen_header(identifier.INT, bytes_len(val)) + val  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_float(obj: object) -> bytes:
    """(cfunc) Serialize 'float' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    # For normal float numbers, orjson performs
    # faster than Python built-in `str()` function.
    if isnormal(obj):
        val: bytes = _orjson_dumps(obj)
        return gen_header(identifier.FLOAT, bytes_len(val)) + val  # type: ignore
    # For other float objects, fallback to Python
    # built-in `str()` approach.
    return _ser_float64(obj)


@cython.cfunc
@cython.inline(True)
def _ser_bool(obj: object) -> bytes:
    """(cfunc) Serialize 'bool' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][value]'
    """
    return TRUE_VALUE if bool(obj) else FALSE_VALUE


@cython.cfunc
@cython.inline(True)
def _ser_none(_: object) -> bytes:
    """(cfunc) Serialize 'None' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)]'
    """
    return NONE_VALUE


# Date&Time Types -----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_datetime(obj: object) -> bytes:
    """(cfunc) Serialize 'datetime.datetime' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][FLAG(1B)][YY(2B)][MM(1B)][DD(1B)][hh(1B)][mm(1B)][ss(1B)][us(4B)][TIMEZONE]'

    [FLAG(1B)]: Different approaches to serialize datatime,
                also determines what [TIMEZONE] part is.

    - 0: Without timezone, [TIMEZONE] part is `OMITTED`.

    - 1: Timezone name appraoch:
    >>> b'...[us(4B)][tzNameSize(ENC)][tzName]'

    - 2: Timezone offset approach:
    >>> b'...[us(4B)][offsetDays(4B)][offsetSeconds(4B)][offsetUS(4B)]'
    """
    # With timezone
    if (tz := datetime.datetime_tzinfo(obj)) is not None:
        # . approach 1: timezone name
        if set_contains(utils.AVAILABLE_TIMEZONES, tz_name := str(tz)):
            name = utils.encode_str(tz_name, b"ascii")
            return b"".join(
                [
                    DATETIME_ID,  # identifier
                    b"\x01",  # approach 1
                    pack_uint16(datetime.datetime_year(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_month(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_day(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_hour(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_minute(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_second(obj)),  # type: ignore
                    pack_uint32(datetime.datetime_microsecond(obj)),  # type: ignore
                    gen_encoded_int(bytes_len(name)),  # type: ignore
                    name,
                ]
            )
        # . approach 2: timezone offset
        if (offset := tz.utcoffset(obj)) is not None:
            return b"".join(
                [
                    DATETIME_ID,  # identifier
                    b"\x02",  # approach 2
                    pack_uint16(datetime.datetime_year(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_month(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_day(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_hour(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_minute(obj)),  # type: ignore
                    pack_uint8(datetime.datetime_second(obj)),  # type: ignore
                    pack_uint32(datetime.datetime_microsecond(obj)),  # type: ignore
                    pack_int32(datetime.timedelta_days(offset)),  # type: ignore
                    pack_int32(datetime.timedelta_seconds(offset)),  # type: ignore
                    pack_int32(datetime.timedelta_microseconds(offset)),  # type: ignore
                ]
            )
    # Without timzone / Fallback
    return b"".join(
        [
            DATETIME_ID,  # identifier
            b"\x00",  # approach 0
            pack_uint16(datetime.datetime_year(obj)),  # type: ignore
            pack_uint8(datetime.datetime_month(obj)),  # type: ignore
            pack_uint8(datetime.datetime_day(obj)),  # type: ignore
            pack_uint8(datetime.datetime_hour(obj)),  # type: ignore
            pack_uint8(datetime.datetime_minute(obj)),  # type: ignore
            pack_uint8(datetime.datetime_second(obj)),  # type: ignore
            pack_uint32(datetime.datetime_microsecond(obj)),  # type: ignore
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_date(obj: object) -> bytes:
    """(cfunc) Serialize 'datetime.date' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][YY(2B)][MM(1B)][DD(1B)]'
    """
    return b"".join(
        [
            DATE_ID,  # identifier
            pack_uint16(datetime.date_year(obj)),  # type: ignore
            pack_uint8(datetime.date_month(obj)),  # type: ignore
            pack_uint8(datetime.date_day(obj)),  # type: ignore
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_time(obj: object) -> bytes:
    """(cfunc) Serialize 'datetime.time' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][FLAG(1B)][hh(1B)][mm(1B)][ss(1B)][us(4B)][TIMEZONE]'

    [FLAG(1B)]: Different approaches to serialize datetime.time,
                also determines what [TIMEZONE] part is.

    - 0: Without timezone, [TIMEZONE] part is `OMITTED`.

    - 1: Timezone name appraoch:
    >>> b'...[us(4B)][tzNameSize(ENC)][tzName]'

    - 2: Timezone offset approach:
    >>> b'...[us(4B)][offsetDays(4B)][offsetSeconds(4B)][offsetUS(4B)]'
    """
    # With timezone
    if (tz := datetime.time_tzinfo(obj)) is not None:
        # . approach 1: timezone name
        if set_contains(utils.AVAILABLE_TIMEZONES, tz_name := str(tz)):
            name = utils.encode_str(tz_name, b"ascii")
            return b"".join(
                [
                    TIME_ID,  # identifier
                    b"\x01",  # approach 1
                    pack_uint8(datetime.time_hour(obj)),  # type: ignore
                    pack_uint8(datetime.time_minute(obj)),  # type: ignore
                    pack_uint8(datetime.time_second(obj)),  # type: ignore
                    pack_uint32(datetime.time_microsecond(obj)),  # type: ignore
                    gen_encoded_int(bytes_len(name)),  # type: ignore
                    name,
                ]
            )
        # . approach 2: timezone offset
        if (offset := tz.utcoffset(None)) is not None:
            return b"".join(
                [
                    TIME_ID,  # identifier
                    b"\x02",  # approach 2
                    pack_uint8(datetime.time_hour(obj)),  # type: ignore
                    pack_uint8(datetime.time_minute(obj)),  # type: ignore
                    pack_uint8(datetime.time_second(obj)),  # type: ignore
                    pack_uint32(datetime.time_microsecond(obj)),  # type: ignore
                    pack_int32(datetime.timedelta_days(offset)),  # type: ignore
                    pack_int32(datetime.timedelta_seconds(offset)),  # type: ignore
                    pack_int32(datetime.timedelta_microseconds(offset)),  # type: ignore
                ]
            )
    # Without timzone / Fallback
    return b"".join(
        [
            TIME_ID,  # identifier
            b"\x00",  # approach 0
            pack_uint8(datetime.time_hour(obj)),  # type: ignore
            pack_uint8(datetime.time_minute(obj)),  # type: ignore
            pack_uint8(datetime.time_second(obj)),  # type: ignore
            pack_uint32(datetime.time_microsecond(obj)),  # type: ignore
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_timedelta(obj: object) -> bytes:
    """(cfunc) Serialize 'datetime.timedelta' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][days(4B)][seconds(4B)][microseconds(4B)]'
    """
    return b"".join(
        [
            TIMEDELTA_ID,  # identifier
            pack_int32(datetime.timedelta_days(obj)),  # type: ignore
            pack_int32(datetime.timedelta_seconds(obj)),  # type: ignore
            pack_int32(datetime.timedelta_microseconds(obj)),  # type: ignore
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_struct_time(obj: struct_time) -> bytes:
    """(cfunc) Serialize 'time.struct_time' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][LIST(SER)]'
    """
    items = list(obj)
    if (tm_zone := obj.tm_zone) is not None:
        items.append(tm_zone)
    if (tm_gmtoff := obj.tm_gmtoff) is not None:
        items.append(tm_gmtoff)
    return STRUCT_TIME_ID + _ser_list(items)


# Numeric Types -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_decimal(obj: object) -> bytes:
    """(cfunc) Serialize 'decimal.Decimal' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    val: bytes = utils.encode_str(str(obj), b"ascii")
    return gen_header(identifier.DECIMAL, bytes_len(val)) + val  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_complex(obj: object) -> bytes:
    """(cfunc) Serialize 'complex' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][real_size(ENC)][real][imag_size(ENC)][imag]'
    """
    real_val: bytes = _orjson_dumps(complex_getreal(obj))
    real_size: bytes = gen_encoded_int(bytes_len(real_val))  # type: ignore
    imag_val: bytes = _orjson_dumps(complex_getimag(obj))
    imag_size: bytes = gen_encoded_int(bytes_len(imag_val))  # type: ignore
    return b"".join([COMPLEX_ID, real_size, real_val, imag_size, imag_val])


# Bytes Types ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_bytes(obj: object) -> bytes:
    """(cfunc) Serialize 'bytes' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    return gen_header(identifier.BYTES, bytes_len(obj)) + obj  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_bytearray(obj: object) -> bytes:
    """(cfunc) Serialize 'bytearray' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    val = bytes(obj)
    return gen_header(identifier.BYTEARRAY, bytes_len(val)) + val  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_memoryview(obj: memoryview) -> bytes:
    """(cfunc) Serialize 'memoryview' object to `<'bytes'>`.

    This function converts 'memoryview' object to 'bytes',
    then uses '_ser_bytes()' function to serialize it.
    """
    return _ser_bytes(obj.tobytes())


# Sequence Types ------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_list(obj: list) -> bytes:
    """(cfunc) Serialize 'list' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][listSize(ENC)][i1(SER)]...[iN(SER)]'
    """
    # Get sequence size
    size: cython.Py_ssize_t = list_len(obj)
    if size == 0:
        return gen_header(identifier.LIST, 0)  # type: ignore
    # Serialize items
    items = [LIST_ID, gen_encoded_int(size)]  # type: ignore
    for i in obj:
        items.append(_ser_common(i))
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_tuple(obj: tuple) -> bytes:
    """(cfunc) Serialize 'tuple' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][tupleSize(ENC)][i1(SER)]...[iN(SER)]'
    """
    # Get sequence size
    size: cython.Py_ssize_t = tuple_len(obj)
    if size == 0:
        return gen_header(identifier.TUPLE, 0)  # type: ignore
    # Serialize items
    items = [TUPLE_ID, gen_encoded_int(size)]  # type: ignore
    for i in obj:
        items.append(_ser_common(i))
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_set(obj: set) -> bytes:
    """(cfunc) Serialize 'set' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][setSize(ENC)][i1(SER)]...[iN(SER)]'
    """
    # Get sequence size
    size: cython.Py_ssize_t = set_len(obj)
    if size == 0:
        return gen_header(identifier.SET, 0)  # type: ignore
    # Serialize items
    items = [SET_ID, gen_encoded_int(size)]  # type: ignore
    for i in obj:
        items.append(_ser_common(i))
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_frozenset(obj: frozenset) -> bytes:
    """(cfunc) Serialize 'frozenset' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][frozensetSize(ENC)][i1(SER)]...[iN(SER)]'
    """
    # Get sequence size
    size: cython.Py_ssize_t = set_len(obj)
    if size == 0:
        return gen_header(identifier.FROZENSET, 0)  # type: ignore
    # Serialize items
    items = [FROZENSET_ID, gen_encoded_int(size)]  # type: ignore
    for i in obj:
        items.append(_ser_common(i))
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_range(obj: object) -> bytes:
    """(cfunc) Serialize 'range' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][start(SER)][stop(SER)][step(SER)]'
    """
    return b"".join(
        [
            RANGE_ID,  # identifier
            _ser_int(obj.start),  # start
            _ser_int(obj.stop),  # stop
            _ser_int(obj.step),  # step
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_sequence(obj: Iterable) -> bytes:
    """(cfunc) Serialize 'sequence' object to `<'bytes'>`.

    This function is used to serialize uncommon sequence types,
    such as: 'collections.deque', 'dict_keys', 'dict_values',
    'numpy.record', etc. The result will be serialized as a
    'list' object.

    Composed of:
    >>> b'[id(1B)][seqSize(ENC)][i1(SER)]...[iN(SER)]'
    """
    # Get sequence size
    size: cython.Py_ssize_t = len(obj)
    if size == 0:
        return gen_header(identifier.LIST, 0)  # type: ignore
    # Serialize items
    items = [LIST_ID, gen_encoded_int(size)]  # type: ignore
    for i in obj:
        items.append(_ser_common(i))
    return b"".join(items)


# Mapping Types -------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_dict(obj: dict) -> bytes:
    """(cfunc) Serialize 'dict' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][dictSize(ENC)][k1(SER)][v1(SER)]...[kN(SER)][vN(SER)]'
    """
    # Get dict size
    size: cython.Py_ssize_t = dict_len(obj)
    if size == 0:
        return gen_header(identifier.DICT, 0)  # type: ignore
    # Serialize dict keys & values
    items = [DICT_ID, gen_encoded_int(size)]  # type: ignore
    for key, val in obj.items():
        items.append(_ser_common(key))
        items.append(_ser_common(val))
    return b"".join(items)


# NumPy Types ---------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_float64(obj: object) -> bytes:
    """(cfunc) Serialize 'np.float[16/32/64]' or other float related object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][size(ENC)][value]'
    """
    val: bytes = utils.encode_str(str(obj), b"ascii")
    return gen_header(identifier.FLOAT, bytes_len(val)) + val  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_datetime64(obj: object) -> bytes:
    """(cfunc) Serialize 'np.datetime64' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][nptime_unit(1B)][value(8B)]'
    """
    return b"".join(
        [
            DT64_ID,  # identifier
            pack_uint8(np.get_datetime64_unit(obj)),  # type: ignore
            pack_int64(np.get_datetime64_value(obj)),  # type: ignore
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_timedelta64(obj: object) -> bytes:
    """(cfunc) Serialize 'np.timedelta64' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][nptime_unit(1B)][value(8B)]'
    """
    return b"".join(
        [
            TD64_ID,  # identifier
            pack_uint8(np.get_datetime64_unit(obj)),  # type: ignore
            pack_int64(np.get_timedelta64_value(obj)),  # type: ignore
        ]
    )


@cython.cfunc
@cython.inline(True)
def _ser_complex64(obj: object) -> bytes:
    """(cfunc) Serialize 'np.complex[64]' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][real_size(ENC)][real][imag_size(ENC)][imag]'
    """
    real_val: bytes = _orjson_dumps(float(obj.real))
    real_size: bytes = gen_encoded_int(bytes_len(real_val))  # type: ignore
    imag_val: bytes = _orjson_dumps(float(obj.imag))
    imag_size: bytes = gen_encoded_int(bytes_len(imag_val))  # type: ignore
    return b"".join([COMPLEX_ID, real_size, real_val, imag_size, imag_val])


@cython.cfunc
@cython.inline(True)
def _ser_complex128(obj: object) -> bytes:
    """(cfunc) Serialize 'np.complex[128]' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][real_size(ENC)][real][imag_size(ENC)][imag]'
    """
    return _ser_complex(obj)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray(obj: np.ndarray) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray' object to `<'bytes'>`.

    #### Supports array from 1-4 dimensions.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # 2-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Determined by [arrDtype(1B)], and different
      array dtype can be serialized in different ways. For more
      information, please refer to '_ser_ndarray_*()' functions.
    """
    # Get ndarray kind
    npy_kind: cython.char = obj.descr.kind

    # . ndarray[object]
    if npy_kind == identifier.NDARRAY_OBJECT:
        return _ser_ndarray_object(obj, NDARRAY_OBJECT_IDDT)
    # . ndarray[int]
    if npy_kind == identifier.NDARRAY_INT:
        return _ser_ndarray_numeric(obj, NDARRAY_INT_IDDT, obj.descr.type_num)
    # . ndarray[uint]
    if npy_kind == identifier.NDARRAY_UINT:
        return _ser_ndarray_numeric(obj, NDARRAY_UINT_IDDT, obj.descr.type_num)
    # . ndarray[float]
    if npy_kind == identifier.NDARRAY_FLOAT:
        return _ser_ndarray_numeric(obj, NDARRAY_FLOAT_IDDT, obj.descr.type_num)
    # . ndarray[bool]
    if npy_kind == identifier.NDARRAY_BOOL:
        return _ser_ndarray_numeric(
            np.PyArray_Cast(obj, np.NPY_TYPES.NPY_INT64),  # cast to int64
            NDARRAY_BOOL_IDDT,
            obj.descr.type_num,
        )
    # . ndarray[datetime64]
    if npy_kind == identifier.NDARRAY_DT64:
        return _ser_ndarray_nptime(obj, NDARRAY_DT64_IDDT)
    # . ndarray[timedelta64]
    if npy_kind == identifier.NDARRAY_TD64:
        return _ser_ndarray_nptime(obj, NDARRAY_TD64_IDDT)
    # . ndarray[complex]
    if npy_kind == identifier.NDARRAY_COMPLEX:
        return _ser_ndarray_complex(obj, NDARRAY_COMPLEX_IDDT)
    # . ndarray[bytes]
    if npy_kind == identifier.NDARRAY_BYTES:
        return _ser_ndarray_bytes(obj, NDARRAY_BYTES_IDDT)
    # . ndarray[str]
    if npy_kind == identifier.NDARRAY_UNICODE:
        return _ser_ndarray_unicode(obj, NDARRAY_UNICODE_IDDT)
    # . invalid dtype
    raise TypeError("unsupported <'numpy.ndarray'> dtype '%s'." % obj.dtype)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray_object(arr: np.ndarray, arr_dtype: bytes) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray[object]' object to `<'bytes'>`.

    #### This function is only for ndarray dtype: `"O" (object)`.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # 2-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Serialize elements individually (multi-dimensional array is flatten in 'C-ORDER').
    >>> b'...[SHAPE][i1(SER)]...[iN(SER)]'
    """
    npy_dtype: cython.int = arr.descr.type_num
    ndim: cython.int = arr.ndim
    shape = arr.shape
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x01",  # 1-dimension
            gen_encoded_int(s_i),  # type: ignore
        ]
        # . serialize values
        if s_i != 0:
            for i in range(s_i):
                items.append(_ser_common(utils.arr_getitem_1d(arr, i)))
        return b"".join(items)

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x02",  # 2-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
        ]
        # . serialize values
        if s_j != 0:
            for i in range(s_i):
                for j in range(s_j):
                    items.append(_ser_common(utils.arr_getitem_2d(arr, i, j)))
        return b"".join(items)

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x03",  # 3-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
        ]
        # . serialize values
        if s_k != 0:
            for i in range(s_i):
                for j in range(s_j):
                    for k in range(s_k):
                        items.append(_ser_common(utils.arr_getitem_3d(arr, i, j, k)))
        return b"".join(items)

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        s_l: cython.Py_ssize_t = shape[3]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x04",  # 4-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
            gen_encoded_int(s_l),  # type: ignore
        ]
        # . serialize values
        if s_l != 0:
            for i in range(s_i):
                for j in range(s_j):
                    for k in range(s_k):
                        for l in range(s_l):
                            items.append(_ser_common(utils.arr_getitem_4d(arr, i, j, k, l)))  # type: ignore
        return b"".join(items)

    # invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray_numeric(
    arr: np.ndarray,
    arr_dtype: bytes,
    npy_dtype: cython.int,
) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray[numeric]' object to `<'bytes'>`.

    #### This function is for ndarray dtype: `"i" (int)`, `"u" (uint)` and `"f" (float)`.
    #### Array of dtype `"b" (bool)` should be cast to `np.int64` before serialization.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # n2-dim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Serialize elements to JSON (multi-dimensional array is flatten in 'C-ORDER').
    >>> b'...[SHAPE][values_json_size(ENC)][values_json]'
    """
    ndim: cython.int = arr.ndim
    shape = arr.shape
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x01",  # 1-dimension
            gen_encoded_int(s_i),  # type: ignore
        ]
        # . serialize values
        if s_i != 0:
            values_json = _orjson_dumps_numpy(arr)
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 2-dimension
    if ndim == 2:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x02",  # 2-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
        ]
        # . serialize values
        if s_j != 0:
            values_json = _orjson_dumps_numpy(utils.arr_flatten(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 3-dimension
    if ndim == 3:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x03",  # 3-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
        ]
        # . serialize values
        if s_k != 0:
            values_json = _orjson_dumps_numpy(utils.arr_flatten(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 4-dimension
    if ndim == 4:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        s_l: cython.Py_ssize_t = shape[3]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x04",  # 4-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
            gen_encoded_int(s_l),  # type: ignore
        ]
        # . serialize values
        if s_l != 0:
            values_json = _orjson_dumps_numpy(utils.arr_flatten(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray_nptime(arr: np.ndarray, arr_dtype: bytes) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray[nptime]' object to `<'bytes'>`.

    #### This function is for ndarray dtype: `"M" (datetime64)`, `"m" (timedelta64)`.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # 2-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Cast elements into int64 and then serialize
      to JSON (multi-dimensional array is flatten in 'C-ORDER').
    >>> b'...[SHAPE][npyTimeUnit(1B)][values_json_size(ENC)][values_json]'
    """
    npy_dtype: cython.int = arr.descr.type_num
    ndim: cython.int = arr.ndim
    shape = arr.shape
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x01",  # 1-dimension
            gen_encoded_int(s_i),  # type: ignore
        ]
        # . serialize values
        if s_i == 0:  # empty
            items.append(pack_uint8(utils.parse_arr_nptime_unit(arr)))  # type: ignore
        else:
            items.append(pack_uint8(np.get_datetime64_unit(arr[0])))  # type: ignore
            arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)  # cast to int64
            values_json = _orjson_dumps_numpy(arr)
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 2-dimension
    if ndim == 2:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x02",  # 2-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
        ]
        # . serialize values
        if s_j == 0:  # empty
            items.append(pack_uint8(utils.parse_arr_nptime_unit(arr)))  # type: ignore
        else:
            items.append(pack_uint8(np.get_datetime64_unit(arr[0, 0])))  # type: ignore
            arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)  # cast to int64
            values_json = _orjson_dumps_numpy(utils.arr_flatten(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 3-dimension
    if ndim == 3:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x03",  # 3-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
        ]
        # . serialize values
        if s_k == 0:  # empty
            items.append(pack_uint8(utils.parse_arr_nptime_unit(arr)))  # type: ignore
        else:
            items.append(pack_uint8(np.get_datetime64_unit(arr[0, 0, 0])))  # type: ignore
            arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)  # cast to int64
            values_json = _orjson_dumps_numpy(utils.arr_flatten(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 4-dimension
    if ndim == 4:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        s_l: cython.Py_ssize_t = shape[3]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x04",  # 4-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
            gen_encoded_int(s_l),  # type: ignore
        ]
        # . serialize values
        if s_l == 0:  # empty
            items.append(pack_uint8(utils.parse_arr_nptime_unit(arr)))  # type: ignore
        else:
            items.append(pack_uint8(np.get_datetime64_unit(arr[0, 0, 0, 0])))  # type: ignore
            arr = np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64)  # cast to int64
            values_json = _orjson_dumps_numpy(utils.arr_flatten(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray_complex(arr: np.ndarray, arr_dtype: bytes) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray[complex]' object to `<'bytes'>`.

    #### This function is for only ndarray dtype: `"c" (complex)`.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # 2-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Get real & imag part from each elements and then serialize
      to JSON (multi-dimensional array is flatten in 'C-ORDER').
    >>> b'...[SHAPE][values_json_size(ENC)][values_json]'
    """
    npy_dtype: cython.int = arr.descr.type_num
    ndim: cython.int = arr.ndim
    shape = arr.shape
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x01",  # 1-dimension
            gen_encoded_int(s_i),  # type: ignore
        ]
        # . serialize values
        if s_i != 0:
            values: list = []
            for i in range(s_i):
                item = utils.arr_getitem_1d(arr, i)
                values.append(complex_getreal(item))
                values.append(complex_getimag(item))
            values_json = _orjson_dumps(values)
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 2-dimension
    if ndim == 2:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x02",  # 2-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
        ]
        # . serialize values
        if s_j != 0:
            values: list = []
            for i in range(s_i):
                for j in range(s_j):
                    item = utils.arr_getitem_2d(arr, i, j)
                    values.append(complex_getreal(item))
                    values.append(complex_getimag(item))
            values_json = _orjson_dumps(values)
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 3-dimension
    if ndim == 3:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x03",  # 3-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
        ]
        # . serialize values
        if s_k != 0:
            values: list = []
            for i in range(s_i):
                for j in range(s_j):
                    for k in range(s_k):
                        item = utils.arr_getitem_3d(arr, i, j, k)
                        values.append(complex_getreal(item))
                        values.append(complex_getimag(item))
            values_json = _orjson_dumps(values)
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 4-dimension
    if ndim == 4:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        s_l: cython.Py_ssize_t = shape[3]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x04",  # 4-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
            gen_encoded_int(s_l),  # type: ignore
        ]
        # . serialize values
        if s_l != 0:
            values: list = []
            for i in range(s_i):
                for j in range(s_j):
                    for k in range(s_k):
                        for l in range(s_l):
                            item = utils.arr_getitem_4d(arr, i, j, k, l)
                            values.append(complex_getreal(item))
                            values.append(complex_getimag(item))
            values_json = _orjson_dumps(values)
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray_bytes(arr: np.ndarray, arr_dtype: bytes) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray[object]' object to `<'bytes'>`.

    #### This function is only for ndarray dtype: `"S" (String)`.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # 2-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Serialize elements individually (multi-dimensional array is flatten in 'C-ORDER').
    >>> b'...[SHAPE][i1(SER)]...[iN(SER)]'
    """
    npy_dtype: cython.int = arr.descr.type_num
    ndim: cython.int = arr.ndim
    shape = arr.shape
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x01",  # 1-dimension
            gen_encoded_int(s_i),  # type: ignore
        ]
        # . serialize values
        if s_i != 0:
            ch_size: cython.int = arr.descr.itemsize
            items.append(gen_encoded_int(ch_size))  # type: ignore
            for i in range(s_i):
                items.append(_ser_bytes(utils.arr_getitem_1d(arr, i)))
        return b"".join(items)

    # 2-dimension
    if ndim == 2:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x02",  # 2-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
        ]
        # . serialize values
        if s_j != 0:
            ch_size: cython.int = arr.descr.itemsize
            items.append(gen_encoded_int(ch_size))  # type: ignore
            for i in range(s_i):
                for j in range(s_j):
                    items.append(_ser_bytes(utils.arr_getitem_2d(arr, i, j)))
        return b"".join(items)

    # 3-dimension
    if ndim == 3:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x03",  # 3-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
        ]
        # . serialize values
        if s_k != 0:
            ch_size: cython.int = arr.descr.itemsize
            items.append(gen_encoded_int(ch_size))  # type: ignore
            for i in range(s_i):
                for j in range(s_j):
                    for k in range(s_k):
                        items.append(_ser_bytes(utils.arr_getitem_3d(arr, i, j, k)))
        return b"".join(items)

    # 4-dimension
    if ndim == 4:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        s_l: cython.Py_ssize_t = shape[3]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x04",  # 4-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
            gen_encoded_int(s_l),  # type: ignore
        ]
        # . serialize values
        if s_l != 0:
            ch_size: cython.int = arr.descr.itemsize
            items.append(gen_encoded_int(ch_size))  # type: ignore
            for i in range(s_i):
                for j in range(s_j):
                    for k in range(s_k):
                        for l in range(s_l):
                            items.append(_ser_bytes(utils.arr_getitem_4d(arr, i, j, k, l)))  # type: ignore
        return b"".join(items)

    # invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


@cython.cfunc
@cython.inline(True)
def _ser_ndarray_unicode(arr: np.ndarray, arr_dtype: bytes) -> bytes:
    """(cfunc) Serialize 'numpy.ndarray[unicode]' object to `<'bytes'>`.

    #### This function is only for ndarray dtype: `"U" (Unicode)`.

    Composed of:
    >>> b'[id(1B)][arrDtype(1B)][npyDtype(1B)][ndim(1B)][SHAPE][VALUES]'

    - [SHAPE]: Determined by [ndim(1B)].
    >>> b'...[ndim(1B)][s_i(ENC)][VALUES]'  # 1-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][VALUES]'  # n2-dim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][VALUES]'  # 3-ndim
    >>> b'...[ndim(1B)][s_i(ENC)][s_j(ENC)][s_k(ENC)][s_l(ENC)][VALUES]'  # 4-ndim

    - [VALUES]: Serialize elements to JSON (multi-dimensional array is flatten in 'C-ORDER').
    >>> b'...[SHAPE][values_json_size(ENC)][values_json]'
    """
    npy_dtype: cython.int = arr.descr.type_num
    ndim: cython.int = arr.ndim
    shape = arr.shape
    # 1-dimension
    if ndim == 1:
        # . array shape
        s_i: cython.Py_ssize_t = shape[0]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x01",  # 1-dimension
            gen_encoded_int(s_i),  # type: ignore
        ]
        # . serialize values
        if s_i != 0:
            ch_size: cython.int = int(arr.descr.itemsize / 4)
            items.append(gen_encoded_int(ch_size))  # type: ignore
            values_json = _orjson_dumps(np.PyArray_ToList(arr))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 2-dimension
    if ndim == 2:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x02",  # 2-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
        ]
        # . serialize values
        if s_j != 0:
            ch_size: cython.int = int(arr.descr.itemsize / 4)
            items.append(gen_encoded_int(ch_size))  # type: ignore
            values_json = _orjson_dumps(np.PyArray_ToList(utils.arr_flatten(arr)))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 3-dimension
    if ndim == 3:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x03",  # 3-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
        ]
        # . serialize values
        if s_k != 0:
            ch_size: cython.int = int(arr.descr.itemsize / 4)
            items.append(gen_encoded_int(ch_size))  # type: ignore
            values_json = _orjson_dumps(np.PyArray_ToList(utils.arr_flatten(arr)))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # 4-dimension
    if ndim == 4:
        s_i: cython.Py_ssize_t = shape[0]
        s_j: cython.Py_ssize_t = shape[1]
        s_k: cython.Py_ssize_t = shape[2]
        s_l: cython.Py_ssize_t = shape[3]
        items = [
            arr_dtype,  # identifier & dtype
            pack_uint8(npy_dtype),  # type: ignore
            b"\x04",  # 4-dimension
            gen_encoded_int(s_i),  # type: ignore
            gen_encoded_int(s_j),  # type: ignore
            gen_encoded_int(s_k),  # type: ignore
            gen_encoded_int(s_l),  # type: ignore
        ]
        # . serialize values
        if s_l != 0:
            ch_size: cython.int = int(arr.descr.itemsize / 4)
            items.append(gen_encoded_int(ch_size))  # type: ignore
            values_json = _orjson_dumps(np.PyArray_ToList(utils.arr_flatten(arr)))
            items.append(gen_encoded_int(bytes_len(values_json)))  # type: ignore
            items.append(values_json)
        return b"".join(items)

    # invalid
    raise ValueError("unsupported <'numpy.ndarray'> dimension [%d]." % ndim)


# Pandas Types --------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_pd_timestamp(obj: Timestamp) -> bytes:
    """(cfunc) Serialize 'pandas.Timestamp' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][FLAG(1B)][VALUE(8B)][TIMEZONE]'

    [FLAG(1B)]: Different approaches to serialize Timestamp,
                also determines what [TIMEZONE] part is.

    - 0: Without timezone, [TIMEZONE] part is `OMITTED`.

    - 1: Timezone name appraoch:
    >>> b'...[VALUE(8B)][tzNameSize(ENC)][tzName]'

    - 2: Timezone offset approach:
    >>> b'...[VALUE(4B)][offsetDays(4B)][offsetSeconds(4B)][offsetUS(4B)]'
    """
    # Get timestamp value
    try:
        val: cython.longlong = obj.value
    except Exception as err:
        raise TypeError("expects <'pandas.Timestamp'>, got %s." % type(obj)) from err
    # With timezone
    if (tz := datetime.datetime_tzinfo(obj)) is not None:
        # . approach 1: timezone name
        if set_contains(utils.AVAILABLE_TIMEZONES, tz_name := str(tz)):
            name = utils.encode_str(tz_name, b"ascii")
            return b"".join(
                [
                    PD_TIMESTAMP_ID,  # identifier
                    b"\x01",  # approach 1
                    pack_int64(val),  # type: ignore
                    gen_encoded_int(bytes_len(name)),  # type: ignore
                    name,
                ]
            )
        # . approach 2: timezone offset
        if (offset := tz.utcoffset(obj)) is not None:
            return b"".join(
                [
                    PD_TIMESTAMP_ID,  # identifier
                    b"\x02",  # approach 2
                    pack_int64(val),  # type: ignore
                    pack_int32(datetime.timedelta_days(offset)),  # type: ignore
                    pack_int32(datetime.timedelta_seconds(offset)),  # type: ignore
                    pack_int32(datetime.timedelta_microseconds(offset)),  # type: ignore
                ]
            )
    # Without timezone / Fallback
    return b"".join([PD_TIMESTAMP_ID, b"\x00", pack_int64(val)])  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_pd_timedelta(obj: Timedelta) -> bytes:
    """(cfunc) Serialize 'pandas.Timedelta' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][VALUE(8B)]'
    """
    try:
        val: cython.longlong = obj.value
    except Exception as err:
        raise TypeError("expects <'pandas.Timedelta'>, got %s." % type(obj)) from err
    return PD_TIMEDELTA_ID + pack_int64(val)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _ser_series(obj: Series) -> bytes:
    """(cfunc) Serialize 'pandas.Series' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][nameFlag(1B)][NAME][VALUES]'

    - [NAME]: Determined by [nameFlag(1B)].
        - If Series name is None, [NAME] part is `OMITTED`.
        - If Series name exists, [NAME] part is `SERIALIZED` by '_ser_common()'.

    - [VALUES]: Please refer to '_ser_ndarray_*()' functions.

    - [VALUES]: For dtype of 'M' (datetime64), [VALUES] part expends to the following:
        - Without timezone:
        >>> b'...[NAME][arrDtype(1B)][FLAG(1B)][VALUES]'

        - Timezone approach 1 - tz_name:
        >>> b'...[NAME][arrDtype(1B)][FLAG(1B)][tzNameSize(ENC)][tzName][VALUES]'

        - Timezone approach 2 - tz_offset:
        >>> b'...[NAME][arrDtype(1B)][FLAG(1B)][offsetDays(4B)][offsetSeconds(4B)][offsetUS(4B)][VALUES]'
    """
    # Access values and name
    try:
        arr: np.ndarray = obj.values
        name: object = obj.name
    except Exception as err:
        raise TypeError("expects <'pandas.Series'>, got %s." % type(obj)) from err

    # Serialize name
    if name is None:  # no name
        items = [SERIES_ID, b"\x00"]
    else:  # has name
        try:
            name_value = _ser_common(name)
        except Exception as err:
            raise ValueError("invalid <'pandas.Series'> name %r." % name) from err
        items = [SERIES_ID, b"\x01", name_value]

    # Serialize values
    npy_kind: cython.char = arr.descr.kind
    # . ndarray[object]
    if npy_kind == identifier.NDARRAY_OBJECT:
        items.append(_ser_ndarray_object(arr, NDARRAY_OBJECT_DT))
    # . ndarray[int]
    elif npy_kind == identifier.NDARRAY_INT:
        items.append(_ser_ndarray_numeric(arr, NDARRAY_INT_DT, arr.descr.type_num))
    # . ndarray[uint]
    elif npy_kind == identifier.NDARRAY_UINT:
        items.append(_ser_ndarray_numeric(arr, NDARRAY_UINT_DT, arr.descr.type_num))
    # . ndarray[float]
    elif npy_kind == identifier.NDARRAY_FLOAT:
        items.append(_ser_ndarray_numeric(arr, NDARRAY_FLOAT_DT, arr.descr.type_num))
    # . ndarray[bool]
    elif npy_kind == identifier.NDARRAY_BOOL:
        items.append(
            _ser_ndarray_numeric(
                np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64),  # cast to int64
                NDARRAY_BOOL_DT,
                arr.descr.type_num,
            )
        )
    # . ndarray[datetime64]
    elif npy_kind == identifier.NDARRAY_DT64:
        items.append(NDARRAY_DT64_DT)
        values = _ser_ndarray_nptime(arr, NDARRAY_DT64_DT)
        # . With timezone
        if (tz := obj.dt.tz) is not None:
            # . approach 1: timezone name
            if set_contains(utils.AVAILABLE_TIMEZONES, tz_name := str(tz)):
                name = utils.encode_str(tz_name, b"ascii")
                items.append(b"\x01")  # approach 1
                items.append(gen_encoded_int(bytes_len(name)))  # type: ignore
                items.append(name)
                items.append(values)
                return b"".join(items)  # exit
            # . approach 2: timezone offset
            if (offset := tz.utcoffset(None)) is not None:
                items.append(b"\x02")  # approach 2
                items.append(pack_int32(datetime.timedelta_days(offset)))  # type: ignore
                items.append(pack_int32(datetime.timedelta_seconds(offset)))  # type: ignore
                items.append(pack_int32(datetime.timedelta_microseconds(offset)))  # type: ignore
                items.append(values)
                return b"".join(items)  # exit
        # . Without timezone / Fallback
        items.append(b"\x00")  # approach 0
        items.append(values)
    # . ndarray[timedelta64]
    elif npy_kind == identifier.NDARRAY_TD64:
        items.append(_ser_ndarray_nptime(arr, NDARRAY_TD64_DT))
    # . ndarray[complex]
    elif npy_kind == identifier.NDARRAY_COMPLEX:
        items.append(_ser_ndarray_complex(arr, NDARRAY_COMPLEX_DT))
    # . ndarray[bytes]
    elif npy_kind == identifier.NDARRAY_BYTES:
        items.append(_ser_ndarray_bytes(arr, NDARRAY_BYTES_DT))
    # . ndarray[str]
    elif npy_kind == identifier.NDARRAY_UNICODE:
        items.append(_ser_ndarray_unicode(arr, NDARRAY_UNICODE_DT))
    # . invalid dtype
    else:
        raise TypeError("unsupported <'pandas.Series'> dtype '%s'." % obj.dtype)

    # Compose
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_datetime_index(obj: DatetimeIndex) -> bytes:
    """(cfunc) Serialize 'pandas.DatetimeIndex' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][nameFlag(1B)][NAME][VALUES]'

    - [NAME]: Determined by [nameFlag(1B)].
        - If DatetimeIndex name is None, [NAME] part is `OMITTED`.
        - If DatetimeIndex name exists, [NAME] part is `SERIALIZED` by '_ser_common()'.

    - [VALUES] part expends to the following:
        - Without timezone:
        >>> b'...[NAME][arrDtype(1B)][FLAG(1B)][VALUES]'

        - Timezone approach 1 - tz_name:
        >>> b'...[NAME][arrDtype(1B)][FLAG(1B)][tzNameSize(ENC)][tzName][VALUES]'

        - Timezone approach 2 - tz_offset:
        >>> b'...[NAME][arrDtype(1B)][FLAG(1B)][offsetDays(4B)][offsetSeconds(4B)][offsetUS(4B)][VALUES]'
    """
    # Access values and name
    try:
        arr: np.ndarray = obj.values
        name: object = obj.name
    except Exception as err:
        raise TypeError(
            "expects <'pandas.DatetimeIndex'>, got %s." % type(obj)
        ) from err

    # Serialize name
    if name is None:  # no name
        items = [DATETIMEINDEX_ID, b"\x00"]
    else:  # has name
        try:
            name_value = _ser_common(name)
        except Exception as err:
            raise ValueError(
                "invalid <'pandas.DatetimeIndex'> name %r." % name
            ) from err
        items = [DATETIMEINDEX_ID, b"\x01", name_value]

    # Serialize values
    values = _ser_ndarray_nptime(arr, NDARRAY_DT64_DT)
    # . With timezone
    if (tz := obj.tz) is not None:
        # . approach 1: timezone name
        if set_contains(utils.AVAILABLE_TIMEZONES, tz_name := str(tz)):
            name = utils.encode_str(tz_name, b"ascii")
            items.append(b"\x01")
            items.append(gen_encoded_int(bytes_len(name)))  # type: ignore
            items.append(name)
            items.append(values)
            return b"".join(items)  # exit
        # . approach 2: timezone offset
        if (offset := tz.utcoffset(None)) is not None:
            items.append(b"\x02")
            items.append(pack_int32(datetime.timedelta_days(offset)))  # type: ignore
            items.append(pack_int32(datetime.timedelta_seconds(offset)))  # type: ignore
            items.append(pack_int32(datetime.timedelta_microseconds(offset)))  # type: ignore
            items.append(values)
            return b"".join(items)  # exit
    # . Without timezone / Fallback
    items.append(b"\x00")
    items.append(values)
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_timedelta_index(obj: TimedeltaIndex) -> bytes:
    """(cfunc) Serialize 'pandas.TimedeltaIndex' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][nameFlag(1B)][NAME][VALUES]'

    - [NAME]: Determined by [nameFlag(1B)].
        - If TimedeltaIndex name is None, [NAME] part is `OMITTED`.
        - If TimedeltaIndex name exists, [NAME] part is `SERIALIZED` by '_ser_common()'.

    - [VALUES]: Please refer to '_ser_ndarray_nptime()' function.
    """
    # Access values and name
    try:
        arr: np.ndarray = obj.values
        name: object = obj.name
    except Exception as err:
        raise TypeError(
            "expects <'pandas.TimedeltaIndex'>, got %s." % type(obj)
        ) from err

    # Serialize
    values = _ser_ndarray_nptime(arr, NDARRAY_TD64_DT)
    if name is None:  # no name
        items = [TIMEDELTAINDEX_ID, b"\x00", values]
    else:  # has name
        try:
            name_value = _ser_common(name)
        except Exception as err:
            raise ValueError(
                "invalid <'pandas.TimedeltaIndex'> name %r." % name
            ) from err
        items = [TIMEDELTAINDEX_ID, b"\x01", name_value, values]
    return b"".join(items)


@cython.cfunc
@cython.inline(True)
def _ser_dataframe(obj: DataFrame) -> bytes:
    """(cfunc) Serialize 'pandas.DataFrame' object to `<'bytes'>`.

    Composed of:
    >>> b'[id(1B)][colsFlag(1B)][COLUMNS][rowsFlag(1B)][VALUES]'

    - [COLUMNS]: Determined by [colsFlag(1B)].
        - If DataFrame is empty and has no columns, everything after
          [colsFlag(1B)] is `OMITTED`.
        - Other case, [COLUMNS] is `SERIALIZED` by '_ser_list()'.

    - [VALUES]: Determined by [rowsFlag(1B)].
        - If DataFrame is empty without rows, everything after [rowsFlag(1B)] is `OMITTED`.
        - Other case for [VALUES] please refer to '_ser_series()' function.
    """
    # Serialize columns name
    cols: list = list(obj.columns)
    cols_num: cython.Py_ssize_t = list_len(cols)
    # . empty w/o columns
    if cols_num == 0:
        return gen_header(identifier.DATAFRAME, 0)  # type: ignore
    try:
        cols_value = _ser_list(cols)
    except Exception as err:
        raise ValueError("invalid <'pandas.DataFrame'> columns:\n%s" % cols) from err
    rows_num: cython.Py_ssize_t = len(obj.index)
    # . empty w/o rows
    if rows_num == 0:
        return b"".join([DATAFRAME_ID, b"\x01", cols_value, b"\x00"])

    # Serialize columns values
    items = [DATAFRAME_ID, b"\x01", cols_value, b"\x01"]
    for _, col in obj.items():
        try:
            arr: np.ndarray = col.values
        except Exception as err:
            raise ValueError(
                "invalid <'pandas.DataFrame'>, cannot access "
                "column's underlying 'ndarray.values':\n%s" % col
            ) from err
        npy_kind: cython.char = arr.descr.kind
        # . ndarray[object]
        if npy_kind == identifier.NDARRAY_OBJECT:
            items.append(_ser_ndarray_object(arr, NDARRAY_OBJECT_DT))
        # . ndarray[int]
        elif npy_kind == identifier.NDARRAY_INT:
            items.append(_ser_ndarray_numeric(arr, NDARRAY_INT_DT, arr.descr.type_num))
        # . ndarray[uint]
        elif npy_kind == identifier.NDARRAY_UINT:
            items.append(_ser_ndarray_numeric(arr, NDARRAY_UINT_DT, arr.descr.type_num))
        # . ndarray[float]
        elif npy_kind == identifier.NDARRAY_FLOAT:
            items.append(
                _ser_ndarray_numeric(arr, NDARRAY_FLOAT_DT, arr.descr.type_num)
            )
        # . ndarray[bool]
        elif npy_kind == identifier.NDARRAY_BOOL:
            items.append(
                _ser_ndarray_numeric(
                    np.PyArray_Cast(arr, np.NPY_TYPES.NPY_INT64),  # cast to int64
                    NDARRAY_BOOL_DT,
                    arr.descr.type_num,
                )
            )
        # . ndarray[datetime64]
        elif npy_kind == identifier.NDARRAY_DT64:
            items.append(NDARRAY_DT64_DT)
            values = _ser_ndarray_nptime(arr, NDARRAY_DT64_DT)
            # . With timezone
            if (tz := col.dt.tz) is not None:
                # . approach 1: timezone name
                if set_contains(utils.AVAILABLE_TIMEZONES, tz_name := str(tz)):
                    name = utils.encode_str(tz_name, b"ascii")
                    items.append(b"\x01")  # approach 1
                    items.append(gen_encoded_int(bytes_len(name)))  # type: ignore
                    items.append(name)
                    items.append(values)
                # . approach 2: timezone offset
                elif (offset := tz.utcoffset(None)) is not None:
                    items.append(b"\x02")  # approach 2
                    items.append(pack_int32(datetime.timedelta_days(offset)))  # type: ignore
                    items.append(pack_int32(datetime.timedelta_seconds(offset)))  # type: ignore
                    items.append(pack_int32(datetime.timedelta_microseconds(offset)))  # type: ignore
                    items.append(values)
                # . Fallback
                else:
                    items.append(b"\x00")  # approach 0
                    items.append(values)
            else:
                # . Without timezone
                items.append(b"\x00")  # approach 0
                items.append(values)
        # . ndarray[timedelta64]
        elif npy_kind == identifier.NDARRAY_TD64:
            items.append(_ser_ndarray_nptime(arr, NDARRAY_TD64_DT))
        # . ndarray[complex]
        elif npy_kind == identifier.NDARRAY_COMPLEX:
            items.append(_ser_ndarray_complex(arr, NDARRAY_COMPLEX_DT))
        # . ndarray[bytes]
        elif npy_kind == identifier.NDARRAY_BYTES:
            items.append(_ser_ndarray_bytes(arr, NDARRAY_BYTES_DT))
        # . ndarray[str]
        elif npy_kind == identifier.NDARRAY_UNICODE:
            items.append(_ser_ndarray_unicode(arr, NDARRAY_UNICODE_DT))
        # . invalid dtype
        else:
            raise TypeError(
                "unsupported <'pandas.DataFrame'> column dtype '%s'." % col.dtype
            )

    # Compose
    return b"".join(items)


# Serialize -----------------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _ser_common(obj: object) -> bytes:
    """(cfunc) Serialize common object to `<'bytes'>`."""
    # Get data type
    obj_dtype = type(obj)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if obj_dtype is str:
        return _ser_str(obj)
    # . <'int'>
    if obj_dtype is int:
        return _ser_int(obj)
    # . <'float'>
    if obj_dtype is float:
        return _ser_float(obj)
    # . <'bool'>
    if obj_dtype is bool:
        return _ser_bool(obj)
    # . <None>
    if obj_dtype is typeref.NONE:
        return _ser_none(obj)

    # Date&Time Types
    # . <'datetime.datetime'>
    if obj_dtype is datetime.datetime:
        return _ser_datetime(obj)
    # . <'datetime.date'>
    if obj_dtype is datetime.date:
        return _ser_date(obj)
    # . <'datetime.time'>
    if obj_dtype is datetime.time:
        return _ser_time(obj)
    # . <'datetime.timedelta'>
    if obj_dtype is datetime.timedelta:
        return _ser_timedelta(obj)

    # Bytes Types
    # . <'bytes'>
    if obj_dtype is bytes:
        return _ser_bytes(obj)

    # Sequence Types
    # . <'list'>
    if obj_dtype is list:
        return _ser_list(obj)
    # . <'tuple'>
    if obj_dtype is tuple:
        return _ser_tuple(obj)

    # Mapping Types
    # . <'dict'>
    if obj_dtype is dict:
        return _ser_dict(obj)

    ##### Uncommon Types #####
    return _ser_uncommon(obj, obj_dtype)


@cython.cfunc
@cython.inline(True)
def _ser_uncommon(obj: object, obj_dtype: type) -> bytes:
    """(cfunc) Serialize uncommon object to `<'bytes'>`."""
    ##### Uncommon Types #####
    # Basic Types
    # . <'numpy.float_'>
    if (
        obj_dtype is typeref.FLOAT64
        or obj_dtype is typeref.FLOAT32
        or obj_dtype is typeref.FLOAT16
    ):
        return _ser_float64(obj)
    # . <'numpy.int_'>
    if (
        obj_dtype is typeref.INT64
        or obj_dtype is typeref.INT32
        or obj_dtype is typeref.INT16
        or obj_dtype is typeref.INT8
    ):
        return _ser_int(obj)
    # . <'numpy.uint'>
    if (
        obj_dtype is typeref.UINT64
        or obj_dtype is typeref.UINT32
        or obj_dtype is typeref.UINT16
        or obj_dtype is typeref.UINT8
    ):
        return _ser_int(obj)
    # . <'numpy.bool_'>
    if obj_dtype is typeref.BOOL_:
        return _ser_bool(obj)

    # Date&Time Types
    # . <'numpy.datetime64'>
    if obj_dtype is typeref.DATETIME64:
        return _ser_datetime64(obj)
    # . <'numpy.timedelta64'>
    if obj_dtype is typeref.TIMEDELTA64:
        return _ser_timedelta64(obj)
    # . <'pandas.Timestamp'>
    if obj_dtype is typeref.PD_TIMESTAMP:
        return _ser_pd_timestamp(obj)
    # . <'pandas.Timedelta'>`
    if obj_dtype is typeref.PD_TIMEDELTA:
        return _ser_pd_timedelta(obj)
    # . <'time.struct_time'>`
    if obj_dtype is typeref.STRUCT_TIME:
        return _ser_struct_time(obj)

    # Numeric Types
    # . <'decimal.Decimal'>
    if obj_dtype is typeref.DECIMAL:
        return _ser_decimal(obj)
    # . <'complex'>
    if obj_dtype is complex:
        return _ser_complex(obj)
    # . <'numpy.complex128'>
    if obj_dtype is typeref.COMPLEX128:
        return _ser_complex128(obj)
    # . <'numpy.complex64'>
    if obj_dtype is typeref.COMPLEX64:
        return _ser_complex64(obj)

    # Bytes Types
    # . <'bytearray'>
    if obj_dtype is bytearray:
        return _ser_bytearray(obj)
    # . <'memoryview'>
    if obj_dtype is memoryview:
        return _ser_memoryview(obj)
    # . <'numpy.bytes_'>
    if obj_dtype is typeref.BYTES_:
        return _ser_bytes(obj)

    # String Types:
    # . <'numpy.str_'>
    if obj_dtype is typeref.STR_:
        return _ser_str(obj)

    # Sequence Types
    # . <'set'>
    if obj_dtype is set:
        return _ser_set(obj)
    # . <'frozenset'>
    if obj_dtype is frozenset:
        return _ser_frozenset(obj)
    # . <'range'>
    if obj_dtype is range:
        return _ser_range(obj)
    # . <'dict_keys'> & <'dict_values'> & <'collections.deque'>
    if (
        obj_dtype is typeref.DICT_KEYS
        or obj_dtype is typeref.DICT_VALUES
        or obj_dtype is typeref.DEQUE
    ):
        return _ser_sequence(obj)

    # Mapping Types
    # . <'dict_items'>
    if obj_dtype is typeref.DICT_ITEMS:
        return _ser_dict(dict(obj))

    # NumPy Types
    # . <'numpy.ndarray'>
    if obj_dtype is np.ndarray:
        return _ser_ndarray(obj)
    # . <'numpy.record'>
    if obj_dtype is typeref.RECORD:
        return _ser_sequence(obj)

    # Pandas Types
    # . <'pandas.Series'>
    if obj_dtype is typeref.SERIES:
        return _ser_series(obj)
    # . <'pandas.DataFrame'>
    if obj_dtype is typeref.DATAFRAME:
        return _ser_dataframe(obj)
    # . <'pandas.DatetimeIndex'>
    if obj_dtype is typeref.DATETIMEINDEX:
        return _ser_datetime_index(obj)
    # . <'pandas.TimedeltaIndex'>
    if obj_dtype is typeref.TIMEDELTAINDEX:
        return _ser_timedelta_index(obj)

    # Cytimes Types
    if typeref.CYTIMES_AVAILABLE:
        # . <'cytimes.pydt'>
        if obj_dtype is typeref.PYDT:
            return _ser_datetime(obj.dt)
        # . <'cytimes.pddt'>
        if obj_dtype is typeref.PDDT:
            return _ser_series(obj.dt)

    ##### Subclass Types #####
    return _ser_subclass(obj, obj_dtype)


@cython.cfunc
@cython.inline(True)
def _ser_subclass(obj: object, obj_dtype: type) -> bytes:
    """(cfunc) Serialize subclass object to `<'bytes'>`."""
    ##### Subclass Types #####
    # Basic Types
    # . subclass of <'str'>
    if isinstance(obj, str):
        return _ser_str(str(obj))
    # . subclass of <'int'>
    if isinstance(obj, int):
        return _ser_int(int(obj))
    # . subclass of <'float'>
    if isinstance(obj, float):
        return _ser_float(float(obj))
    # . subclass of <'bool'>
    if isinstance(obj, bool):
        return _ser_bool(bool(obj))

    # Date&Time Types
    # . subclass of <'datetime.datetime'>
    if isinstance(obj, datetime.datetime):
        return _ser_datetime(obj)
    # . subclass of <'datetime.date'>
    if isinstance(obj, datetime.date):
        return _ser_date(obj)
    # . subclass of <'datetime.time'>
    if isinstance(obj, datetime.time):
        return _ser_time(obj)
    # . subclass of <'datetime.timedelta'>
    if isinstance(obj, datetime.timedelta):
        return _ser_timedelta(obj)

    # Sequence Types
    # . subclass of <'list'>
    if isinstance(obj, list):
        return _ser_list(list(obj))
    # . subclass of <'tuple'>
    if isinstance(obj, tuple):
        return _ser_tuple(tuple(obj))
    # . subclass of <'set'>
    if isinstance(obj, set):
        return _ser_set(set(obj))
    # . subclass of <'frozenset'>
    if isinstance(obj, frozenset):
        return _ser_frozenset(frozenset(obj))

    # Mapping Types
    # . subclass of <'dict'>
    if isinstance(obj, dict):
        return _ser_dict(dict(obj))

    # Invalid Data Type
    raise TypeError("unsupported 'data' type %s." % obj_dtype)


@cython.ccall
def serialize(obj: object) -> bytes:
    """Serialize an object to `<'bytes'>`.

    Supports:
    - Python natives:
        - Ser <'str'> -> Des <'str'>
        - Ser <'int'> -> Des <'int'>
        - Ser <'float'> -> Des <'float'>
        - Ser <'bool'> -> Des <'bool'>
        - Ser <'datetime.datetime'> -> Des <'datetime.datetime'> `[Supports Timezone]`
        - Ser <'datetime.date'> -> Des <'datetime.date'>
        - Ser <'datetime.time'> -> Des <'datetime.time'> `[Supports Timezone]`
        - Ser <'datetime.timedelta'> -> Des <'datetime.timedelta'>
        - Ser <'time.struct_time'> -> Des <'time.struct_time'>
        - Ser <'decimal.Decimal'> -> Des <'decimal.Decimal'>
        - Ser <'complex'> -> Des <'comples'>
        - Ser <'bytes'> -> Des <'bytes'>
        - Ser <'bytearray'> -> Des <'bytearray'>
        - Ser <'memoryview'> -> Des <'bytes'>
        - Ser <'list'> -> Des <'list'>
        - Ser <'tuple'> -> Des <'tuple'>
        - Ser <'set'> -> Des <'set'>
        - Ser <'frozenset'> -> Des <'frozenset'>
        - Ser <'range'> -> Des <'range'>
        - Ser <'deque'> -> Des <'list'>
        - Ser <'dict'> -> Des <'dict'>
    - NumPy objects:
        - Ser <'np.str\_'> -> Des <'str'>
        - Ser <'np.int\*'> -> Des <'int'>
        - Ser <'np.uint\*'> -> Des <'int'>
        - Ser <'np.float\*'> -> Des <'float'>
        - Ser <'np.bool\_'> -> Des <'bool'>
        - Ser <'np.datetime64'> -> Des <'np.datetime64'>
        - Ser <'np.timedelta64'> -> Des <'np.timedelta64'>
        - Ser <'np.complex\*'> -> Des <'complex'>
        - Ser <'np.bytes\_'> -> Des <'bytes'>
        - Ser <'np.ndarray'> -> Des <'np.ndarray'> `[1-4 dimemsional]`
    - Pandas objects:
        - Ser <'pd.Timestamp'> -> Des <'pd.Timestamp'> `[Supports Timezone]`
        - Ser <'pd.Timedelta'> -> Des <'pd.Timedelta'>
        - Ser <'pd.Series'> -> Des <'pd.Series'>
        - Ser <'pd.DatetimeIndex'> -> Des <'pd.DatetimeIndex'> `[Supports Timezone]`
        - Ser <'pd.TimedeltaIndex'> -> Des <'pd.TimedeltaIndex'>
        - Ser <'pd.DataFrame'> -> Des <'pd.DataFrame'>
    """
    try:
        return _ser_common(obj)
    except Exception as err:
        raise errors.SerializeError(
            "<'Serializor'>\nFailed to serialize to 'bytes': "
            "%s\n%r\nError: %s" % (type(obj), obj, err)
        ) from err


########## The following functions are for testing purpose only ##########
def _test_utils() -> None:
    _test_pack_int8()
    _test_pack_uint8()
    _test_pack_int16()
    _test_pack_uint16()
    _test_pack_int24()
    _test_pack_uint24()
    _test_pack_int32()
    _test_pack_uint32()
    _test_pack_int64()
    _test_pack_uint64()
    _test_gen_encoded_int()
    _test_gen_header()


def _test_pack_int8() -> None:
    import struct

    for val in range(-128, 128):
        s = struct.pack("<b", val)
        b = pack_int8(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack int8".ljust(80))

    del struct


def _test_pack_uint8() -> None:
    import struct

    for val in range(0, 256):
        s = struct.pack("<B", val)
        b = pack_uint8(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack uint8".ljust(80))

    del struct


def _test_pack_int16() -> None:
    import struct

    for val in range(-32768, 32768):
        s = struct.pack("<h", val)
        b = pack_int16(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack int16".ljust(80))

    del struct


def _test_pack_uint16() -> None:
    import struct

    for val in range(0, 65536):
        s = struct.pack("<H", val)
        b = pack_uint16(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack uint16".ljust(80))

    del struct


def _test_pack_int24() -> None:
    import struct

    for val in (-8388608, -8388607, 0, 8388606, 8388607):
        s = struct.pack("<i", val)[:3]
        b = pack_int24(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack int24".ljust(80))

    del struct


def _test_pack_uint24() -> None:
    import struct

    for val in (0, 1, 16777213, 16777214, 16777215):
        s = struct.pack("<I", val)[:3]
        b = pack_uint24(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack uint24".ljust(80))

    del struct


def _test_pack_int32() -> None:
    import struct

    for val in (-2147483648, -2147483647, 0, 2147483646, 2147483647):
        s = struct.pack("<i", val)
        b = pack_int32(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack int32".ljust(80))

    del struct


def _test_pack_uint32() -> None:
    import struct

    for val in (0, 1, 4294967293, 4294967294, 4294967295):
        s = struct.pack("<I", val)
        b = pack_uint32(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack uint32".ljust(80))

    del struct


def _test_pack_int64() -> None:
    import struct

    for val in (
        -9223372036854775808,
        -9223372036854775807,
        0,
        9223372036854775806,
        9223372036854775807,
    ):
        s = struct.pack("<q", val)
        b = pack_int64(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack int64".ljust(80))

    del struct


def _test_pack_uint64() -> None:
    import struct

    for val in (
        0,
        1,
        18446744073709551613,
        18446744073709551614,
        18446744073709551615,
    ):
        s = struct.pack("<Q", val)
        b = pack_uint64(val)  # type: ignore
        assert s == b, f"{s} | {b}"
        print("- pass: %s" % val, end="\r")
    print("Pass Pack uint64".ljust(80))

    del struct


def _test_gen_encoded_int() -> None:
    # Test gen_encoded_int
    num = 251
    v = gen_encoded_int(num)  # type: ignore
    n = pack_uint8(num)  # type: ignore
    assert v == n, f"{v} | {n} - num: {num}"

    for num in (252, 65_535):
        v = gen_encoded_int(num)  # type: ignore
        n = pack_uint8(UINT16_ENCODE_VALUE) + pack_uint16(num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"

    for num in (65_536, 4_294_967_295):
        v = gen_encoded_int(num)  # type: ignore
        n = pack_uint8(UINT32_ENCODE_VALUE) + pack_uint32(num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"

    for num in (4_294_967_296, 4_294_967_296 * 2):
        v = gen_encoded_int(num)  # type: ignore
        n = pack_uint8(UINT64_ENCODE_VALUE) + pack_uint64(num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"
    print("Pass Gen Encoded Integer".ljust(80))


def _test_gen_header() -> None:
    dtype: bytes = b"\x01"
    dchar: cython.uchar = 1
    # Test gen_header
    num = 251
    v = dtype + gen_encoded_int(num)  # type: ignore
    n = gen_header(dchar, num)  # type: ignore
    assert v == n, f"{v} | {n} - num: {num}"

    for num in (252, 65_535):
        v = dtype + gen_encoded_int(num)  # type: ignore
        n = gen_header(dchar, num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"

    for num in (65_536, 4_294_967_295):
        v = dtype + gen_encoded_int(num)  # type: ignore
        n = gen_header(dchar, num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"

    for num in (4_294_967_296, 4_294_967_296 * 2):
        v = dtype + gen_encoded_int(num)  # type: ignore
        n = gen_header(dchar, num)  # type: ignore
        assert v == n, f"{v} | {n} - num: {num}"
    print("Pass Gen Header".ljust(80))
