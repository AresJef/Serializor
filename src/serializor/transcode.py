# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.numpy import PyArray_ToList as np_tolist  # type: ignore
from cython.cimports.numpy import PyArray_FROM_O as np_from_list  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.list import PyList_Check as is_list  # type: ignore
from cython.cimports.cpython.list import PyList_Size as len_list  # type: ignore
from cython.cimports.cpython.dict import PyDict_Check as is_dict  # type: ignore
from cython.cimports.cpython.int import PyInt_Check as is_int  # type: ignore
from cython.cimports.cpython.string import PyString_Check as is_str  # type: ignore
from cython.cimports.cytimes import cydatetime as cydt  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
from typing import Type
from decimal import Decimal
import datetime, time
import orjson, pandas as pd, numpy as np
from cytimes import cydatetime as cydt
from _collections_abc import dict_values, dict_keys

__all__ = ["dumps", "loads", "SerializorError"]

# Constants --------------------------------------------------------------------------------------------------------------
# Unique key
UNIQUE_KEY = "$@SL#%"

# Special keys
DECIMAL_KEY: str = "$@DE#%"  # Decimal
BYTES_KEY: str = "$@BY#%"  # bytes
DATE_KEY: str = "$@DT#%"  # datetime.date
DATETIME_NAIVE_KEY: str = "$@DN#%"  # datetime.datetime (naive)
DATETIME_AWARE_KEY: str = "$@DA#%"  # datetime.datetime (aware)
TIME_NAIVE_KEY: str = "$@TN#%"  # datetime.time (naive)
TIME_AWARE_KEY: str = "$@TA#%"  # datetime.time (aware)
TIMEDELTA_KEY: str = "$@DL#%"  # datetime.timedelta
NDARRAY_KEY: str = "$@ND#%"  # numpy.ndarray
PDSERIES_JSON_KEY: str = "$@SJ#%"  # pandas.Series (Json type)
PDSERIES_OBJT_KEY: str = "$@SO#%"  # pandas.Series (Object type)
PDSERIES_TSNA_KEY: str = "$@SN#%"  # pandas.Series (Timestamp naive)
PDSERIES_TSAW_KEY: str = "$@ST#%"  # pandas.Series (Timestamp aware)
PDSERIES_TMDL_KEY: str = "$@SD#%"  # pandas.Series (Timedelta)
PDDATAFRAME_KEY: str = "$@DF#%"  # pandas.DataFrame


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
    return float(str(obj))


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
    fold: cython.int = cydt.get_dt_fold(obj)
    return [UNIQUE_KEY, DATETIME_AWARE_KEY, us, offset, tzname, fold]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_datetime64(obj: object) -> list:
    return [UNIQUE_KEY, DATETIME_NAIVE_KEY, cydt.dt64_to_microseconds(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_struct_time(val: object) -> list:
    dt: datetime.datetime = cydt.gen_dt(
        val.tm_year,
        val.tm_mon,
        val.tm_mday,
        val.tm_hour,
        val.tm_min,
        val.tm_sec,
        0,
        None,
        0,
    )
    return [UNIQUE_KEY, DATETIME_NAIVE_KEY, cydt.dt_to_microseconds(dt)]


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
    fold: cython.int = cydt.get_time_fold(obj)
    return [UNIQUE_KEY, TIME_AWARE_KEY, us, offset, tzname, fold]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_timedelta(obj: object) -> list:
    return [UNIQUE_KEY, TIMEDELTA_KEY, cydt.delta_to_microseconds(obj)]


@cython.cfunc
@cython.inline(True)
@cython.exceptval(check=False)
def _encode_timedelta64(obj: object) -> list:
    return [UNIQUE_KEY, TIMEDELTA_KEY, cydt.delta64_to_microseconds(obj)]


# Complex types
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
    data: list = []
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
    np.datetime64: _encode_datetime64,
    time.struct_time: _encode_struct_time,
    datetime.time: _encode_time,
    datetime.timedelta: _encode_timedelta,
    pd.Timedelta: _encode_timedelta,
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
def encode(obj: object) -> bytes:
    """cfunction for `dumps`."""
    return orjson.dumps(
        obj, default=_fallback_handler, option=orjson.OPT_PASSTHROUGH_DATETIME
    )


@cython.ccall
def dumps(obj: object) -> bytes:
    """Serielize an object to bytes.

    ### Supported data types includes:
    - boolean: `bool` & `numpy.bool_` -> deserialized to `bool`
    - integer: `int` & `numpy.int` & `numpy.uint` -> deserialized to `int`
    - float: `float` & `numpy.float_` -> deserialized to `float`
    - decimal: `decimal.Decimal` -> deserialized to `decimal.Decimal`
    - string: `str` -> deserialized to `str`
    - bytes: `bytes` -> deserialized to `bytes`
    - date: `datetime.date` -> deserialized to `datetime.date`
    - time: `datetime.time` -> deserialized to `datetime.time`
    - datetime: `datetime.datetime` & `pandas.Timestamp` -> deserialize to `datetime.datetime`
    - datetime64*: `numpy.datetime64` & `time.struct_time` -> deserialize to `datetime.datetime`
    - timedelta: `datetime.timedelta` & `pandas.Timedelta` -> deserialize to `datetime.timedelta`
    - timedelta64: `numpy.timedelta64` -> deserialize to `datetime.timedelta`
    - None: `None` & `numpy.nan` -> deserialized to `None`
    - list: `list` of above supported data types -> deserialized to `list`
    - tuple: `tuple` of above supported data types -> deserialized to `list`
    - set: `set` of above supported data types -> deserialized to `list`
    - frozenset: `frozenset` of above supported data types -> deserialized to `list`
    - dict: `dict` of above supported data types -> deserialized to `dict`
    - numpy.record: `numpy.record` of above supported data types -> deserialized to `list`
    - numpy.ndarray: `numpy.ndarray` of above supported data types -> deserialized to `np.ndarray`
    - pandas.Series: `pandas.Series` of above supported data types -> deserialized to `pandas.Series`
    - pandas.DataFrame: `pandas.DataFrame` of above supported data types -> deserialized to `pandas.DataFrame`

    :param obj: The object to be serialized.
    :raises SerializorError: If any error occurs.
    :return: `<bytes>` The serialized data.
    """
    try:
        return encode(obj)
    except Exception as err:
        raise SerializorError("<Serializor> %s" % err)


# Decode ---------------------------------------------------------------------------------------------------------------
# Handle types
@cython.cfunc
@cython.inline(True)
def _decode_handler(data: object) -> object:
    if is_list(data):
        return _decode_list(data)
    elif is_dict(data):
        return _decode_dict(data)
    else:
        return data


# Complex types
@cython.cfunc
@cython.inline(True)
def _decode_list(data: list) -> object:
    # Get list length
    _len_: cython.int = len_list(data)
    # Special Key
    if 3 <= _len_ <= 6 and data[0] == UNIQUE_KEY:
        try:
            # Try access key & value
            key: str = data[1]
            val: object = data[2]
            # . pandas.DataFrame
            if key == PDDATAFRAME_KEY and _len_ == 3 and is_list(val):
                dic: dict = {}
                col: list
                for col in val:
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
            if key == PDSERIES_JSON_KEY and _len_ == 4 and is_list(val):
                return pd.Series(val, name=data[3])
            # . pandas.Series (Object)
            if key == PDSERIES_OBJT_KEY and _len_ == 4 and is_list(val):
                return pd.Series([_decode_handler(i) for i in val], name=data[3])
            # . pandas.Series (Timestamp naive)
            if key == PDSERIES_TSNA_KEY and _len_ == 4 and is_list(val):
                return pd.Series(pd.DatetimeIndex(val), name=data[3])
            # . pandas.Series (Timestamp aware)
            if key == PDSERIES_TSAW_KEY and _len_ == 6 and is_list(val):
                tzinfo: object = cydt.gen_timezone(data[4], data[5])
                return pd.Series(pd.DatetimeIndex(val, tz=tzinfo), name=data[3])
            # . pandas.Series (Timedelta)
            if key == PDSERIES_TMDL_KEY and _len_ == 4 and is_list(val):
                return pd.Series(pd.TimedeltaIndex(val), name=data[3])
            # . datetime-niave
            if key == DATETIME_NAIVE_KEY and _len_ == 3 and is_int(val):
                return cydt.dt_fr_microseconds(val)
            # . datetime-aware
            if key == DATETIME_AWARE_KEY and _len_ == 6 and is_int(val):
                tzinfo: object = cydt.gen_timezone(data[3], data[4])
                return cydt.dt_fr_microseconds(val, tzinfo, data[5])
            # . date
            if key == DATE_KEY and _len_ == 3 and is_int(val):
                return cydt.date_fr_ordinal(val)
            # . time-naive
            if key == TIME_NAIVE_KEY and _len_ == 3 and is_int(val):
                return cydt.time_fr_microseconds(val)
            # . time-aware
            if key == TIME_AWARE_KEY and _len_ == 6 and is_int(val):
                tzinfo: object = cydt.gen_timezone(data[3], data[4])
                return cydt.time_fr_microseconds(val, tzinfo, data[5])
            # . timedelta
            if key == TIMEDELTA_KEY and _len_ == 3 and is_int(val):
                return cydt.delta_fr_microseconds(val)
            # . decimal
            if key == DECIMAL_KEY and _len_ == 3 and is_str(val):
                return Decimal(val)
            # . bytes
            if key == BYTES_KEY and _len_ == 3 and is_str(val):
                return val.encode("utf-8")
            # . ndarray
            if key == NDARRAY_KEY and _len_ == 3 and is_list(val):
                return np_from_list([_decode_handler(i) for i in val])

        except Exception:
            # Fallback to normal key
            pass

    # Normal Key
    return [_decode_handler(i) for i in data]


@cython.cfunc
@cython.inline(True)
def _decode_dict(data: dict) -> object:
    return {key: _decode_handler(val) for key, val in data.items()}


# Master decode
@cython.cfunc
@cython.inline(True)
def decode(data: bytes) -> object:
    """cfunction for `loads`."""
    return _decode_handler(orjson.loads(data))


@cython.ccall
def loads(data: bytes) -> object:
    """Deserialize the value to its original (or compatible) python dtype.
    Must be used with the `dumps` function in this module.
    """
    try:
        return decode(data)
    except Exception as err:
        raise SerializorError("<Serializor> %s" % err)


# Exceptions -----------------------------------------------------------------------------------------------------------
class SerializorError(ValueError):
    """The one and only exception this module will raise."""
