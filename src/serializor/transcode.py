# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.numpy import PyArray_ToList as np_tolist  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.list import PyList_Check as is_list  # type: ignore
from cython.cimports.cpython.list import PyList_New as new_list  # type: ignore
from cython.cimports.cpython.list import PyList_Size as len_list  # type: ignore
from cython.cimports.cpython.dict import PyDict_New as new_dict  # type: ignore
from cython.cimports.cpython.dict import PyDict_Check as is_dict  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
from typing import Type
from decimal import Decimal
import orjson, pandas as pd
import datetime, numpy as np
from cytimes import cydatetime as cydt
from _collections_abc import dict_values, dict_keys

__all__ = ["dumps", "loads", "SerializorError"]

# Constants --------------------------------------------------------------------------------------------------------------
# Unique key
UNIQUE_KEY = "$RECD$"

# Special keys
DECIMAL_KEY: str = "$DECI$"  # Decimal
BYTES_KEY: str = "$BYTE$"  # bytes
DATE_KEY: str = "$DATE$"  # datetime.date
DATETIME_NAIVE_KEY: str = "$DTNA$"  # datetime.datetime (naive)
DATETIME_AWARE_KEY: str = "$DTAW$"  # datetime.datetime (aware)
TIME_NAIVE_KEY: str = "$TMNA$"  # datetime.time (naive)
TIME_AWARE_KEY: str = "$TMAW$"  # datetime.time (aware)
TIMEDELTA_KEY: str = "$TMDL$"  # datetime.timedelta
NDARRAY_KEY: str = "$NDAR$"  # numpy.ndarray
PDSERIES_JSON_KEY: str = "$SRJS$"  # pandas.Series (Json type)
PDSERIES_OBJT_KEY: str = "$SROB$"  # pandas.Series (Object type)
PDSERIES_TSNA_KEY: str = "$SRTN$"  # pandas.Series (Timestamp naive)
PDSERIES_TSAW_KEY: str = "$SRTZ$"  # pandas.Series (Timestamp aware)
PDSERIES_TMDL_KEY: str = "$SRTD$"  # pandas.Series (Timedelta)
PDDATAFRAME_KEY: str = "$PDDF$"  # pandas.DataFrame


# Encode ---------------------------------------------------------------------------------------------------------------
# Base types
@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _through(obj: object) -> object:
    return obj


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_bool(obj: cython.bint) -> cython.bint:
    return True if obj else False


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_int(obj: cython.longlong) -> cython.longlong:
    return int(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_float(obj: object) -> object:
    return float(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_decimal(obj: object) -> list:
    return [UNIQUE_KEY, DECIMAL_KEY, str(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_bytes(obj: bytes) -> list:
    return [UNIQUE_KEY, BYTES_KEY, obj.decode("utf-8")]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_date(obj: object) -> list:
    return [UNIQUE_KEY, DATE_KEY, cydt.to_ordinal(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_datetime(obj: object) -> list:
    us: cython.longlong = cydt.dt_to_microseconds(obj)
    tzinfo: object = cydt.get_dt_tzinfo(obj)
    if tzinfo is None:
        return [UNIQUE_KEY, DATETIME_NAIVE_KEY, us]

    offset: cython.longlong = (
        cydt.delta_to_microseconds(tzinfo.utcoffset(obj)) // cydt.US_SECOND
    )
    tzname: object = tzinfo.tzname(obj)
    return [UNIQUE_KEY, DATETIME_AWARE_KEY, us, offset, tzname]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_time(obj: object) -> list:
    us: cython.longlong = cydt.time_to_microseconds(obj)
    tzinfo: object = cydt.get_time_tzinfo(obj)
    if tzinfo is None:
        return [UNIQUE_KEY, TIME_NAIVE_KEY, us]

    offset: cython.longlong = (
        cydt.delta_to_microseconds(tzinfo.utcoffset(None)) // cydt.US_SECOND
    )
    tzname: object = tzinfo.tzname(None)
    return [UNIQUE_KEY, TIME_AWARE_KEY, us, offset, tzname]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_timedelta(obj: object) -> list:
    return [UNIQUE_KEY, TIMEDELTA_KEY, cydt.delta_to_microseconds(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_datetime64(obj: object) -> list:
    return [UNIQUE_KEY, DATETIME_NAIVE_KEY, cydt.dt64_to_microseconds(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_timedelta64(obj: object) -> list:
    return [UNIQUE_KEY, TIMEDELTA_KEY, cydt.delta64_to_microseconds(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_sequence(obj: object) -> list:
    return list(obj)


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_ndarray(obj: object) -> list:
    return [UNIQUE_KEY, NDARRAY_KEY, np_tolist(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_series(obj: pd.Series) -> list:
    kind: str = obj.dtype.kind
    name: object = obj.name
    # Object dtype
    if kind == "O":
        return [
            UNIQUE_KEY,
            PDSERIES_OBJT_KEY,
            [_fallback_handler(i) for i in obj],
            name,
        ]
    # Timestamp dtype
    elif kind == "M":
        values: np.ndarray = cydt.arraydt64_to_arrayint_ns(obj.values)
        tzinfo: object = obj.dt.tz
        # Timezone naive
        if tzinfo is None:
            return [UNIQUE_KEY, PDSERIES_TSNA_KEY, np_tolist(values), name]
        # Timezone aware
        tzname: object = tzinfo.tzname(None)
        offset: cython.longlong = (
            cydt.delta_to_microseconds(tzinfo.utcoffset(None)) // cydt.US_SECOND
        )
        return [UNIQUE_KEY, PDSERIES_TSAW_KEY, np_tolist(values), name, offset, tzname]
    # Timedelta dtype
    elif kind == "m":
        values: np.ndarray = cydt.arraydelta64_to_arrayint_ns(obj.values)
        return [UNIQUE_KEY, PDSERIES_TMDL_KEY, np_tolist(values), name]
    # Json dtype
    else:
        return [UNIQUE_KEY, PDSERIES_JSON_KEY, np_tolist(obj.values), name]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_dataframe(obj: pd.DataFrame) -> list:
    # Iterate over columns
    data: list = new_list(0)
    kind: str
    for name, col in obj.items():
        kind = col.dtype.kind
        # Object dtype
        if kind == "O":
            code = [PDSERIES_OBJT_KEY, [_fallback_handler(i) for i in col], name]
        # Timestamp dtype
        elif kind == "M":
            values: np.ndarray = cydt.arraydt64_to_arrayint_ns(col.values)
            tzinfo: object = col.dt.tz
            # Timezone naive
            if tzinfo is None:
                code = [PDSERIES_TSNA_KEY, np_tolist(values), name]
            # Timezone aware
            else:
                tzname: object = tzinfo.tzname(None)
                offset: cython.longlong = (
                    cydt.delta_to_microseconds(tzinfo.utcoffset(None)) // cydt.US_SECOND
                )
                code = [PDSERIES_TSAW_KEY, np_tolist(values), name, offset, tzname]
        # Timedelta dtype
        elif kind == "m":
            values: np.ndarray = cydt.arraydelta64_to_arrayint_ns(col.values)
            code = [PDSERIES_TMDL_KEY, np_tolist(values), name]
        # Json dtype
        else:
            code = [PDSERIES_JSON_KEY, np_tolist(col.values), name]
        # Append
        data.append(code)

    # Return
    return [UNIQUE_KEY, PDDATAFRAME_KEY, data]


# Encoder table
ENCODERS: dict[Type, cython.cfunc] = {
    # Base types
    bool: _through,
    np.bool_: _encode_bool,
    int: _through,
    np.int_: _encode_int,
    np.int8: _encode_int,
    np.int16: _encode_int,
    np.int32: _encode_int,
    np.int64: _encode_int,
    np.uint: _encode_int,
    np.uint16: _encode_int,
    np.uint32: _encode_int,
    np.uint64: _encode_int,
    float: _through,
    np.float_: _encode_float,
    np.float16: _encode_float,
    np.float32: _encode_float,
    np.float64: _encode_float,
    Decimal: _encode_decimal,
    str: _through,
    bytes: _encode_bytes,
    datetime.date: _encode_date,
    datetime.datetime: _encode_datetime,
    pd.Timestamp: _encode_datetime,
    datetime.timedelta: _encode_timedelta,
    pd.Timedelta: _encode_timedelta,
    datetime.time: _encode_time,
    np.datetime64: _encode_datetime64,
    np.timedelta64: _encode_timedelta64,
    type(None): _through,
    # Complex types
    list: _through,
    tuple: _encode_sequence,
    set: _encode_sequence,
    frozenset: _encode_sequence,
    dict_keys: _encode_sequence,
    dict_values: _encode_sequence,
    dict: _through,
    np.record: _encode_sequence,
    np.ndarray: _encode_ndarray,
    pd.Series: _encode_series,
    pd.DataFrame: _encode_dataframe,
}


# Fallback
@cython.cfunc
@cython.inline(True)
def _fallback_handler(obj: object) -> object:
    try:
        return ENCODERS[type(obj)](obj)
    except Exception:
        raise TypeError


# Master encode
@cython.cfunc
@cython.inline(True)
def _encode(obj: object) -> bytes:
    try:
        return orjson.dumps(
            obj, default=_fallback_handler, option=orjson.OPT_PASSTHROUGH_DATETIME
        )
    except Exception as err:
        raise SerializorError("<Serializor> %s" % err)


@cython.ccall
def dumps(obj: object) -> bytes:
    return _encode(obj)


# Decode ---------------------------------------------------------------------------------------------------------------
# Handle types
@cython.cfunc
@cython.inline(True)
def _decode_handler(obj: object) -> object:
    if is_list(obj):
        return _decode_list(obj)
    elif is_dict(obj):
        return _decode_dict(obj)
    else:
        return obj


# Complex types
@cython.cfunc
@cython.inline(True)
def _decode_list(obj: list) -> object:
    # Special Key
    if len_list(obj) > 2 and obj[0] == UNIQUE_KEY:
        try:
            # Try access key
            key: str = obj[1]
            # . pandas.DataFrame
            if key == PDDATAFRAME_KEY:
                dic: dict = new_dict()
                col: list
                for col in obj[2]:
                    skey: str = col[0]
                    vals: list = col[1]
                    name: str = col[2]
                    if skey == PDSERIES_JSON_KEY:
                        dic[name] = vals
                    elif skey == PDSERIES_OBJT_KEY:
                        dic[name] = [_decode_handler(i) for i in vals]
                    elif skey == PDSERIES_TSNA_KEY:
                        dic[name] = pd.DatetimeIndex(vals)
                    elif skey == PDSERIES_TSAW_KEY:
                        tzinfo: object = cydt.gen_timezone(col[3], col[4])
                        dic[name] = pd.DatetimeIndex(vals, tz=tzinfo)
                    elif skey == PDSERIES_TMDL_KEY:
                        dic[name] = pd.TimedeltaIndex(vals)
                    else:
                        raise TypeError
                return pd.DataFrame(dic)
            # . pandas.Series (Json)
            if key == PDSERIES_JSON_KEY:
                return pd.Series(obj[2], name=obj[3])
            # . pandas.Series (Object)
            if key == PDSERIES_OBJT_KEY:
                return pd.Series([_decode_handler(i) for i in obj[2]], name=obj[3])
            # . pandas.Series (Timestamp naive)
            if key == PDSERIES_TSNA_KEY:
                return pd.Series(pd.DatetimeIndex(obj[2]), name=obj[3])
            # . pandas.Series (Timestamp aware)
            if key == PDSERIES_TSAW_KEY:
                tzinfo: object = cydt.gen_timezone(obj[4], obj[5])
                return pd.Series(pd.DatetimeIndex(obj[2], tz=tzinfo), name=obj[3])
            # . pandas.Series (Timedelta)
            if key == PDSERIES_TMDL_KEY:
                return pd.Series(pd.TimedeltaIndex(obj[2]), name=obj[3])
            # . datetime-niave
            if key == DATETIME_NAIVE_KEY:
                return cydt.dt_fr_microseconds(obj[2])
            # . datetime-aware
            if key == DATETIME_AWARE_KEY:
                tzinfo: object = cydt.gen_timezone(obj[3], obj[4])
                return cydt.dt_fr_microseconds(obj[2], tzinfo)
            # . date
            if key == DATE_KEY:
                return cydt.date_fr_ordinal(obj[2])
            # . time-naive
            if key == TIME_NAIVE_KEY:
                return cydt.time_fr_microseconds(obj[2])
            # . time-aware
            if key == TIME_AWARE_KEY:
                tzinfo: object = cydt.gen_timezone(obj[3], obj[4])
                return cydt.time_fr_microseconds(obj[2], tzinfo)
            # . timedelta
            if key == TIMEDELTA_KEY:
                return cydt.delta_fr_microseconds(obj[2])
            # . decimal
            if key == DECIMAL_KEY:
                return Decimal(obj[2])
            # . bytes
            if key == BYTES_KEY:
                return obj[2].encode("utf-8")
            # . ndarray
            if key == NDARRAY_KEY:
                return np.PyArray_FROM_O([_decode_handler(i) for i in obj[2]])

        except Exception:
            # Fallback to normal key
            pass

    # Normal Key
    return [_decode_handler(i) for i in obj]


@cython.cfunc
@cython.inline(True)
def _decode_dict(obj: dict) -> object:
    return {key: _decode_handler(val) for key, val in obj.items()}


# Master decode
@cython.cfunc
@cython.inline(True)
def _decode(obj: bytes) -> object:
    try:
        return _decode_handler(orjson.loads(obj))
    except Exception as err:
        raise SerializorError("<Serializor> %s" % err)


@cython.ccall
def loads(obj: bytes) -> object:
    return _decode(obj)


# Exceptions -----------------------------------------------------------------------------------------------------------
class SerializorError(ValueError):
    """The one and only exception this module will raise."""
