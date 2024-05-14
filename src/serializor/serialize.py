# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.libc.math import isnormal  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_Substring as str_substr  # type: ignore
from cython.cimports.cpython.set import PySet_GET_SIZE as set_len  # type: ignore
from cython.cimports.cpython.list import PyList_GET_SIZE as list_len  # type: ignore
from cython.cimports.cpython.tuple import PyTuple_GET_SIZE as tuple_len  # type: ignore
from cython.cimports.cpython.dict import PyDict_Keys as dict_getkeys  # type: ignore
from cython.cimports.cpython.dict import PyDict_Values as dict_getvals  # type: ignore
from cython.cimports.cpython.complex import PyComplex_RealAsDouble as complex_getreal  # type: ignore
from cython.cimports.cpython.complex import PyComplex_ImagAsDouble as complex_getimag  # type: ignore
from cython.cimports.serializor import prefix, typeref  # type: ignore

np.import_array()
np.import_umath()
datetime.import_datetime()

# Python imports
import numpy as np, datetime
from typing import Callable, Iterable
from orjson import dumps
from pandas import Series, DataFrame, DatetimeIndex, TimedeltaIndex
from serializor import prefix, typeref, errors

# Constants -------------------------------------------------------------------------
# . functions
FN_ORJSON_DUMPS: Callable = dumps


# Orjson dumps ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _orjson_dumps(obj: object) -> str:
    """(cfunc) Serialize object using
    'orjson <https://github.com/ijl/orjson>' into JSON `<'str'>`."""
    return decode_bytes(FN_ORJSON_DUMPS(obj))  # type: ignore


# Basic Types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_str(obj: object) -> str:
    """(cfunc) Serialize `<'str'>` to `<'str'>`.

    ### Example:
    >>> obj = "Hello"
        # identifier: {s}5|Hello
        # str length: s{5}|Hello
        # str value:  s5|{Hello}
        return "s5|Hello"
    """
    return "%s%d|%s" % (prefix.STR, str_len(obj), obj)


@cython.cfunc
@cython.inline(True)
def _serialize_float(obj: object) -> str:
    """(cfunc) Serialize `<'float'>` to `<'str'>`.

    ### Example:
    >>> obj = 3.14
        # identifier:  {f}3.14
        # float value: f{3.14}
        return "f3.14"
    """
    # For normal native Python float numbers, orjson performs
    # faster than Python built-in `str` function.
    if isnormal(obj):
        return prefix.FLOAT + _orjson_dumps(obj)
    # For special float numbers such as `inf`, `nan`, etc.,
    # use Python built-in `str` function for serialization.
    return _serialize_float64(obj)


@cython.cfunc
@cython.inline(True)
def _serialize_float64(obj: object) -> str:
    """(cfunc) Serialize `<'numpy.float_'>` to `<'str'>`.

    ### Example:
    >>> obj = np.float64(3.14)
        # identifier:  {f}3.14
        # float value: f{3.14}
        return "f3.14"
    """
    # For numpy.float_ numbers, using Python built-in `str`
    # function performs faster than orjson.
    val: str = str(obj)
    return prefix.FLOAT + val


@cython.cfunc
@cython.inline(True)
def _serialize_int(obj: object) -> str:
    """(cfunc) Serialize `<'int'>` to `<'str'>`.

    ### Example:
    >>> obj = 1024
        # identifier: {i}1024
        # int value:  i{1024}
        return "i1024"
    """
    val: str = str(obj)
    return prefix.INT + val


@cython.cfunc
@cython.inline(True)
def _serialize_bool(obj: object) -> str:
    """(cfunc) Serialize `<'bool'>` to `<'str'>`.

    ### Example:
    >>> obj = True
        # identifier: {o}1
        # bool value: o{1}
        return "o1"
    """
    return prefix.BOOL_TRUE if obj else prefix.BOOL_FALSE


@cython.cfunc
@cython.inline(True)
def _serialize_none() -> str:
    """(cfunc) Serialize `<'None'>` to `<'str'>`.

    ### Example:
    >>> obj = None
        # identifier: {n}
        return "n"
    """
    return prefix.NONE


# Date&Time Types -------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_datetime(obj: object) -> str:
    """(cfunc) Serialize `<'datetime.datetime'>` to `<'str'>`.

    ### Example:
    >>> obj = datetime(2021, 1, 1)
        # identifier: {z}2021-01-01 00:00:00
        # isoformat:  z{2021-01-01 00:00:00}
        return "z2021-01-01 00:00:00"
    """
    # Orjson performs faster than using Python built-in
    # format string for datetime serialization.
    val: str = _orjson_dumps(obj)
    return prefix.DATETIME + str_substr(val, 1, str_len(val) - 1)


@cython.cfunc
@cython.inline(True)
def _serialize_date(obj: object) -> str:
    """(cfunc) Serialize `<'datetime.date'>` to `<'str'>`

    ### Example:
    >>> obj = date(2021, 1, 1)
        # identifier: {d}2021-01-01
        # isoformat:  d{2021-01-01}
        return "d2021-01-01"
    """
    return "%s%04d-%02d-%02d" % (
        prefix.DATE,
        datetime.PyDateTime_GET_YEAR(obj),
        datetime.PyDateTime_GET_MONTH(obj),
        datetime.PyDateTime_GET_DAY(obj),
    )


@cython.cfunc
@cython.inline(True)
def _serialize_time(obj: object) -> str:
    """(cfunc) Serialize `<'datetime.time'>` to `<'str'>`.

    ### Example:
    >>> obj = time(0, 0, 0, 1)
        # identifier: {t}00:00:00.000001
        # isoformat:  t{00:00:00.000001}
        return "t00:00:00.000001"
    """
    microsecond: cython.int = datetime.PyDateTime_TIME_GET_MICROSECOND(obj)
    if microsecond == 0:
        return "%s%02d:%02d:%02d" % (
            prefix.TIME,
            datetime.PyDateTime_TIME_GET_HOUR(obj),
            datetime.PyDateTime_TIME_GET_MINUTE(obj),
            datetime.PyDateTime_TIME_GET_SECOND(obj),
        )
    else:
        return "%s%02d:%02d:%02d.%06d" % (
            prefix.TIME,
            datetime.PyDateTime_TIME_GET_HOUR(obj),
            datetime.PyDateTime_TIME_GET_MINUTE(obj),
            datetime.PyDateTime_TIME_GET_SECOND(obj),
            microsecond,
        )


@cython.cfunc
@cython.inline(True)
def _serialize_timedelta(obj: object) -> str:
    """(cfunc) Serialize `<'datetime.timedelta'>` to `<'str'>`.

    ### Example:
    >>> obj = timedelta(1, 2, 3)
        # identifier: {l}1|2|3
        # days value: l{1}|2|3
        # secs value: l1|{2}|3
        # us value:   l1|2|{3}
        return "l1|2|3"
    """
    days: cython.int = datetime.PyDateTime_DELTA_GET_DAYS(obj)
    secs: cython.int = datetime.PyDateTime_DELTA_GET_SECONDS(obj)
    microseconds: cython.int = datetime.PyDateTime_DELTA_GET_MICROSECONDS(obj)
    return "%s%d|%d|%d" % (prefix.TIMEDELTA, days, secs, microseconds)


@cython.cfunc
@cython.inline(True)
def _serialize_struct_time(obj: object) -> str:
    """(cfunc) Serialize `<'time.struct_time'>` to `<'str'>`.

    ### Example (treats as datetime):
    >>> obj = struct_time(2021, 1, 1, 0, 0, 0, 0, 0, 0)
        # identifier: {z}2021-01-01 00:00:00
        # isoformat:  z{2021-01-01 00:00:00}
        return "z2021-01-01 00:00:00"
    """
    # fmt: off
    return _serialize_datetime(datetime.datetime_new(
        obj.tm_year, obj.tm_mon, obj.tm_mday,
        obj.tm_hour, obj.tm_min, obj.tm_sec,
        0, None, 0) )
    # fmt: on


# Numeric Types ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_decimal(obj: object) -> str:
    """(cfunc) Serialize `<'decimal.Decimal'>` to `<'str'>`.

    ### Example:
    >>> obj = Decimal("3.1415926")
        # identifier:    {e}3.1415926
        # decimal value: e{3.1415926}
        return "e3.1415926"
    """
    val: str = str(obj)
    return prefix.DECIMAL + val


@cython.cfunc
@cython.inline(True)
def _serialize_complex(obj: object) -> str:
    """(cfunc) Serialize `<'complex'>` to `<'str'>`.

    ### Example:
    >>> obj = 1 + 1j
        # identifier: {c}1.0|1.0
        # real value:  c{1.0}|1.0
        # imag value:  c1.0|{1.0}
        return "c1.0|1.0"
    """
    return "%s%s|%s" % (
        prefix.COMPLEX,
        complex_getreal(obj),
        complex_getimag(obj),
    )


# Bytes Types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_bytes(obj: object) -> str:
    """(cfunc) Serialize `<'bytes'>` to `<'str'>`.

    Only supports bytes that can be decoded through 'utf-8'.

    ### Example:
    >>> obj = b"Hello"
        # identifier: {b}5|Hello
        # str length: b{5}|Hello
        # str value:  b5|{Hello}
        return "b5|Hello"
    """
    val: str = decode_bytes(obj)  # type: ignore
    return "%s%d|%s" % (prefix.BYTES, str_len(val), val)


@cython.cfunc
@cython.inline(True)
def _serialize_bytearray(obj: object) -> str:
    """(cfunc) Serialize `<'bytearray'>` to `<'str'>`.

    Only supports bytearray that can be decoded through 'utf-8'.

    ### Example (treats as bytes):
    >>> obj = bytearray(b"Hello")
        # identifier: {b}5|Hello
        # str length: b{5}|Hello
        # str value:  b5|{Hello}
        return "b5|Hello"
    """
    val = decode_bytearray(obj)  # type: ignore
    return "%s%d|%s" % (prefix.BYTES, str_len(val), val)


@cython.cfunc
@cython.inline(True)
def _serialize_memoryview(obj: memoryview) -> str:
    """(cfunc) Serialize `<'memoryview'>` to `<'str'>`.

    Only supports memoryview that can be decoded through 'utf-8'.

    ### Example (treats as bytes):
    >>> obj = memoryview(b"Hello")
        # identifier: {b}5|Hello
        # str length: b{5}|Hello
        # str value:  b5|{Hello}
        return "b5|Hello"
    """
    return _serialize_bytes(obj.tobytes())


# Mapping Types ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_dict(obj: object) -> str:
    """(cfunc) Serialize `<'dict'>` to `<'str'>`.

    ### Example:
    >>> obj = {"1": 1, "2": 1.234, "3": True}
        # identifier:  {D}13["1","2","3"][i1,f1.234,o1,]
        # keys length: D{13}["1","2","3"][i1,f1.234,o1,]
        # keys values: D13{["1","2","3"]}[i1,f1.234,o1,]
        # dict values: D13["1","2","3"]{[i1,f1.234,o1,]}
        return 'D13["1","2","3"][i1,f1.234,o1,]"

    ### Notes:
    - 1. [keys values] section are the dictionary keys serialized
         into json array strings. This approach allows leveraging
         'orjson' to greatly improve serialization & deserialization
         performance.
    - 2. [keys length] section is the unicode length of the json
         array string for [keys values].
    - 3. Difference from dict values, dict keys are restricted to the
         following data types: `<'str'>`, `<'float'>`, `<'int'>` or `<'bool'>`.
    - 4. Empty dict, returns 'D0[]'.
    - 5. Non-empty dict, [dict values] section always end with a comma
         followed by the closing bracket `',]'`.
    """
    # Serialize dict keys into json array
    keys = dict_getkeys(obj)
    try:
        keys_str: str = _orjson_dumps(keys)
    except Exception as err:
        # . invalid dict keys
        keys_str: str = repr(keys)
        if str_len(keys_str) > 30:
            keys_str = "\n" + keys_str
        raise errors.SerializeTypeError(
            "<'Serializor'>\nFailed to serialize <'dict'> keys: %s\n"
            "Only support dict keys of the following data types: "
            "<'str'>, <'float'>, <'int'> or <'bool'>." % keys_str
        ) from err
    keys_len: cython.Py_ssize_t = str_len(keys_str)
    if keys_len == 2:  # empty dict: '[]'
        return "%s0[]" % prefix.DICT  # exit

    # Serialize dict values
    vals: list = [_serialize_common_type(v) for v in dict_getvals(obj)]

    # Format serialization
    return "%s%d%s[%s,]" % (prefix.DICT, keys_len, keys_str, ",".join(vals))


# Sequence Types --------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_list(obj: list) -> str:
    """(cfunc) Serialize `<'list'>` to `<'str'>`.

    ### Example:
    >>> obj = [1, 1.234, True]
        # identifier: {L}[i1,f1.234,o1,]
        # seq size:   L{3}[i1,f1.234,o1,]
        # items:      L3{[i1,f1.234,o1,]}
        return 'L3[i1,f1.234,o1,]'

    ### Notes:
    - 1. [seq size] section represents the total number
         of items in the list.
    - 2. Empty list, returns 'L0[]'.
    - 3. Non-empty list, [items] section always end with
         a comma followed by the closing bracket `',]'`.
    """
    # Get list size
    size: cython.Py_ssize_t = list_len(obj)
    if size == 0:  # list is empty
        return "%s0[]" % prefix.LIST  # exit

    # Serialize list items
    items: list = [_serialize_common_type(i) for i in obj]

    # Format serialization
    return "%s%d[%s,]" % (prefix.LIST, size, ",".join(items))


@cython.cfunc
@cython.inline(True)
def _serialize_tuple(obj: tuple) -> str:
    """(cfunc) Serialize `<'tuple'>` to `<'str'>`.

    ### Example:
    >>> obj = (1, 1.234, True)
        # identifier: {T}3[i1,f1.234,o1,]
        # seq size:   T{3}[i1,f1.234,o1,]
        # items:      T3{[i1,f1.234,o1,]}
        return 'T3[i1,f1.234,o1,]'

    ### Notes:
    - 1. [seq size] section represents the total number
         of items in the tuple.
    - 2. Empty tuple, returns 'T0[]'.
    - 3. Non-empty tuple, [items] section always end with
         a comma followed by the closing bracket `',]'`.
    """
    # Get tuple size
    size: cython.Py_ssize_t = tuple_len(obj)
    if size == 0:  # tuple is empty
        return "%s0[]" % prefix.TUPLE  # exit

    # Serialize tuple items
    items: list = [_serialize_common_type(i) for i in obj]

    # Format serialization
    return "%s%d[%s,]" % (prefix.TUPLE, size, ",".join(items))


@cython.cfunc
@cython.inline(True)
def _serialize_set(obj: set) -> str:
    """(cfunc) Serialize `<'set'>` to `<'str'>`.

    ### Example:
    >>> obj = {1, 2, 3}
        # identifier: {E}3[i1,i2,i3,]
        # set size:   E{3}[i1,i2,i3,]
        # items:      E3{[i1,i2,i3,]}
        return 'E3[i1,i2,i3,]'

    ### Notes:
    - 1. [set size] section represents the total number
         of items in the set.
    - 2. Empty set, returns 'E0[]'.
    - 3. Non-empty set, [items] section always end with
         a comma followed by the closing bracket `',]'`.
    """
    # Get set size
    size: cython.Py_ssize_t = set_len(obj)
    if size == 0:  # set is empty.
        return "%s0[]" % prefix.SET  # exit

    # Serialize set items
    items: list = [_serialize_common_type(i) for i in obj]

    # Format serialization
    return "%s%d[%s,]" % (prefix.SET, size, ",".join(items))


@cython.cfunc
@cython.inline(True)
def _serialize_frozenset(obj: frozenset) -> str:
    """(cfunc) Serialize `<'frozenset'>` to `<'str'>`.

    ### Example (treats as set):
    >>> obj = frozenset({1, 2, 3})
        # identifier: {E}3[i1,i2,i3,]
        # set size:   E{3}[i1,i2,i3,]
        # items:      E3{[i1,i2,i3,]}
        return 'E3[i1,i2,i3,]'

    ### Notes:
    - 1. [set size] section represents the total number
         of items in the set.
    - 2. Empty set, returns 'E0[]'.
    - 3. Non-empty set, [items] section always end with
         a comma followed by the closing bracket `',]'`.
    """
    # Get frozenset size
    size: cython.Py_ssize_t = set_len(obj)
    if size == 0:  # set is empty.
        return "%s0[]" % prefix.SET  # exit

    # Serialize set items
    items: list = [_serialize_common_type(i) for i in obj]

    # Format serialization
    return "%s%d[%s,]" % (prefix.SET, size, ",".join(items))


@cython.cfunc
@cython.inline(True)
def _serialize_sequence(obj: Iterable, pfix: str) -> str:
    """(cfunc) Serialize `<'sequence'>` to `<'str'>`.

    Here 'sequence' means any 1-dimensional iterables
    that supports for-loop iteration, such as:
    'dict_keys', 'dict_values', 'np.record', etc.

    ### Example:
    >>> obj = {"1": 1, "2": 2, "3", 3}.values()
        # identifier: {p*}3[i1,i2,i3,]
        # seq size:   p*{3}[i1,i2,i3,]
        # items:      p*3{[i1,i2,i3,]}
        return 'p*3[i1,i2,i3,]'

    ### Notes:
    - 1. [p*] corresponds to the 'pfix' argument.
    - 2. [seq size] section represents the total number
         of items in the sequence.
    - 3. Empty sequence, returns '{pfix}0[]'.
    - 4. Non-empty sequence, [items] section always end with
         a comma followed by the closing bracket `',]'`.
    """
    # Get iterable size
    size: cython.Py_ssize_t = len(obj)
    if size == 0:  # iterable is empty
        return "%s0[]" % pfix  # exit

    # Serialize iterable items
    items: list = [_serialize_common_type(i) for i in obj]

    # Format serialization
    return "%s%d[%s,]" % (pfix, size, ",".join(items))


# Numpy ndarray ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_datetime64(obj: object) -> str:
    """(cfunc) Serialize `<'numpy.datetime64'>` to `<'str'>`.

    ### Example:
    >>> obj = np.datetime64('2021-01-01 12:00:00')
        # identifier: {M}s1609502400
        # time unit:  M{s}1609502400
        # time value: Ms{1609502400}
        return 'Ms1609502400'
    """
    val: np.npy_datetime = np.get_datetime64_value(obj)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(obj)
    # Common units
    val_str: str = str(val)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return prefix.DATETIME64_NS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return prefix.DATETIME64_US + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return prefix.DATETIME64_MS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return prefix.DATETIME64_S + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return prefix.DATETIME64_MI + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return prefix.DATETIME64_H + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return prefix.DATETIME64_D + val_str
    # Uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return prefix.DATETIME64_PS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return prefix.DATETIME64_FS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return prefix.DATETIME64_AS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return prefix.DATETIME64_Y + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return prefix.DATETIME64_M + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return prefix.DATETIME64_W + val_str
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return prefix.DATETIME64_B + val_str
    raise errors.SerializeTypeError(
        "<'Serializor'>\nFailed to serialize %s: "
        "unknown <'numpy.datetime64'> unit [%s]." % (repr(obj), unit)
    )


@cython.cfunc
@cython.inline(True)
def _serialize_timedelta64(obj: object) -> str:
    """(cfunc) Serialize `<'numpy.timedelta64'>` to `<'str'>`.

    ### Example:
    >>> obj = np.timedelta64(1024, "us")
        # identifier: {m}us1024
        # time unit:  m{us}1024
        # time value: mus{1024}
        return 'mus1024'
    """
    val: np.npy_timedelta = np.get_timedelta64_value(obj)
    unit: np.NPY_DATETIMEUNIT = np.get_datetime64_unit(obj)
    # Common units
    val_str: str = str(val)
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return prefix.TIMEDELTA64_NS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return prefix.TIMEDELTA64_US + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return prefix.TIMEDELTA64_MS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return prefix.TIMEDELTA64_S + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return prefix.TIMEDELTA64_MI + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return prefix.TIMEDELTA64_H + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return prefix.TIMEDELTA64_D + val_str
    # Uncommon units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return prefix.TIMEDELTA64_PS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return prefix.TIMEDELTA64_FS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return prefix.TIMEDELTA64_AS + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return prefix.TIMEDELTA64_Y + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return prefix.TIMEDELTA64_M + val_str
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return prefix.TIMEDELTA64_W + val_str
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return prefix.TIMEDELTA64_B + val_str
    raise errors.SerializeTypeError(
        "<'Serializor'>\nFailed to serialize %s: "
        "unknown <'numpy.timedelta64'> unit [%s]." % (repr(obj), unit)
    )


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray(obj: np.ndarray) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    ### Example:
    >>> obj = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint64)
        # identifier: {N}u2|3|3[1,2,3,4,5,6,7,8,9]
        # dtype:      N{u}2|3|3[1,2,3,4,5,6,7,8,9]
        # dimension:  Nu{2}|3|3[1,2,3,4,5,6,7,8,9]
        # shape:      Nu2|{3|3}[1,2,3,4,5,6,7,8,9]
        # items:      Nu2|3|3{[1,2,3,4,5,6,7,8,9]}
        return 'Nu2|3|3[1,2,3,4,5,6,7,8,9]'

    ### Notes:
    - 1. Supports ndarray up to 4 dimensions.
    - 2. [shape] section changes with the number of dimensions.
    - 3. Empty ndarray, returns 'N{dtype}1|0[]' (1-dimensional).
    - 4. Besides dtype of 'object', other dtypes utilize 'orjson.dumps'
         to serialize items as an 1-dimensional json array for maximum
         performance.
    - 5. For dtype of 'object', items are serialized one by one, similar
         to the serialization process of a list. For more detail, please
         refer to the `_serialize_ndarray_object()` function.
    """
    # Get ndarray dimensions
    ndim: cython.Py_ssize_t = obj.ndim
    if ndim == 0:
        raise errors.SerializeTypeError(
            "<'Serializor'>\nCan't not serialize 0-dimensional <'numpy.ndarray'>."
        )
    if ndim > 4:
        raise errors.SerializeTypeError(
            "<'Serializor'>\nCan't not serialize <'numpy.ndarray'> with more than 4 dimensions."
        )

    # Get ndarray dtype
    dtype: cython.Py_UCS4 = obj.descr.kind

    # Serialize ndarray
    # . ndarray[object]
    if dtype == prefix.NDARRAY_DTYPE_OBJECT_ID:
        return _serialize_ndarray_object(obj, ndim)
    # . ndarray[float]
    if dtype == prefix.NDARRAY_DTYPE_FLOAT_ID:
        if np.PyArray_TYPE(obj) == np.NPY_TYPES.NPY_FLOAT16:
            obj = np.PyArray_Cast(obj, np.NPY_TYPES.NPY_FLOAT32)
        return _serialize_ndarray_common(obj, ndim, prefix.NDARRAY_FLOAT)
    # . ndarray[int]
    if dtype == prefix.NDARRAY_DTYPE_INT_ID:
        return _serialize_ndarray_common(obj, ndim, prefix.NDARRAY_INT)
    # . ndarray[uint]
    if dtype == prefix.NDARRAY_DTYPE_UINT_ID:
        return _serialize_ndarray_common(obj, ndim, prefix.NDARRAY_UINT)
    # . ndarray[bool]
    if dtype == prefix.NDARRAY_DTYPE_BOOL_ID:
        return _serialize_ndarray_bool(obj, ndim)
    # . ndarray[datetime64]
    if dtype == prefix.NDARRAY_DTYPE_DT64_ID:
        return _serialize_ndarray_dt64td64(obj, ndim, True)
    # . ndarray[timedelta64]
    if dtype == prefix.NDARRAY_DTYPE_TD64_ID:
        return _serialize_ndarray_dt64td64(obj, ndim, False)
    # . ndarray[complex]
    if dtype == prefix.NDARRAY_DTYPE_COMPLEX_ID:
        return _serialize_ndarray_complex(obj, ndim)
    # . ndarray[bytes]
    if dtype == prefix.NDARRAY_DTYPE_BYTES_ID:
        return _serialize_ndarray_bytes(obj, ndim)
    # . ndarray[str]
    if dtype == prefix.NDARRAY_DTYPE_UNICODE_ID:
        return _serialize_ndarray_common(obj, ndim, prefix.NDARRAY_UNICODE)
    # . invalid dtype
    raise errors.SerializeTypeError(
        "<'Serializor'>\nFailed to serialize <'numpy.ndarray'>: "
        "unsupported dtype [%s]." % obj.dtype
    )


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray_object(obj: np.ndarray, ndim: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "O" (object).

    ### Example:
    >>> obj = numpy.array([1, 1.234, True, "abc", 1 + 1j], dtype="O")
        # identifier: {N}O1|5[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # dtype:      N{O}1|5[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # dimension:  NO{1}|5[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # shape:      NO1|{5}[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # items:      NO1|5{[i1,f1.234,o1,s3|abc,c1.0|1.0,]}
        return 'NO1|5[i1,f1.234,o1,s3|abc,c1.0|1.0,]'

    ### Notes:
    - 1. Different from other dtypes, [items] section for object
         does not use 'orjson.dumps' for serialization. Instead,
         it performs custom serialization similar to a list.
    - 2. Empty ndarray, returns 'NO1|0[]' (1-dimensional).
    - 3. Non-empty ndarray, [items] section always end with a
         comma followed by the closing bracket `',]'`.
    """
    shape = obj.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    s_k: cython.Py_ssize_t
    s_l: cython.Py_ssize_t
    # 1-dimensional
    if ndim == 1:
        s_i, s_j, s_k, s_l = shape[0], 0, 0, 0
        # . empty ndarray
        if s_i == 0:  # 'NO1|0[]'
            return "%s1|0[]" % prefix.NDARRAY_OBJECT
        # . serialize ndarray[object]
        items = [_serialize_common_type(ndarray_getitem_1d(obj, i)) for i in range(s_i)]  # type: ignore
        # fmt: off
        return "%s1|%d[%s,]" % (  # 'NO1|3[...,]'
            prefix.NDARRAY_OBJECT, s_i, ",".join(items))
        # fmt: on

    # 2-dimensional
    if ndim == 2:
        s_i, s_j, s_k, s_l = shape[0], shape[1], 0, 0
        # . empty prefix.ndarray
        if s_j == 0:  # 'NO2|2|0[]'
            return "%s2|%d|0[]" % (prefix.NDARRAY_OBJECT, s_i)
        # . serialize ndarray[object]
        items = [
            _serialize_common_type(ndarray_getitem_2d(obj, i, j))  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
        ]
        # fmt: off
        return "%s2|%d|%d[%s,]" % (  # 'NO2|2|3[...,]'
            prefix.NDARRAY_OBJECT, s_i, s_j, ",".join(items))
        # fmt: on

    # 3-dimensional
    if ndim == 3:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], 0
        # . empty ndarray
        if s_k == 0:  # 'NO3|2|2|0[]'
            return "%s3|%d|%d|0[]" % (prefix.NDARRAY_OBJECT, s_i, s_j)
        # . serialize ndarray[object]
        items = [
            _serialize_common_type(ndarray_getitem_3d(obj, i, j, k))  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
        ]
        # fmt: off
        return "%s3|%d|%d|%d[%s,]" % (  # 'NO3|2|2|3[...,]'
            prefix.NDARRAY_OBJECT, s_i, s_j, s_k, ",".join(items))
        # fmt: on

    # 4-dimensional
    else:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], shape[3]
        # . empty ndarray
        if s_l == 0:  # 'NO4|2|2|2|0[]'
            return "%s4|%d|%d|%d|0[]" % (prefix.NDARRAY_OBJECT, s_i, s_j, s_k)
        # . serialize ndarray[object]
        items = [
            _serialize_common_type(ndarray_getitem_4d(obj, i, j, k, l))  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
            for l in range(s_l)
        ]
        # fmt: off
        return "%s4|%d|%d|%d|%d[%s,]" % (  # 'NO4|2|2|2|3[...,]'
            prefix.NDARRAY_OBJECT, s_i, s_j, s_k, s_l, ",".join(items))
        # fmt: on


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray_common(
    obj: np.ndarray,
    ndim: cython.Py_ssize_t,
    pfix: str,
) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    This function is specifically for ndarray with dtype of:
    "U" (str), "f" (float), "i" (int) and "u" (uint).

    Identifier is determined by the 'pfix' argument.

    ### Example:
    >>> obj = numpy.array(["1", "2", "3"], dtype="U")  # str
        # identifier: {N}U1|3["1","2","3"]
        # dtype:      N{U}1|3["1","2","3"]
        # dimension:  NU{1}|3["1","2","3"]
        # shape:      NU1|{3}["1","2","3"]
        # items:      NU1|3{["1","2","3"]}
        return 'NU1|3["1","2","3"]'

    >>> obj = numpy.array([1.1, 2.2, 3.3], dtype=np.float64)  # float
        return 'Nf1|3[1.1,2.2,3.3]'

    >>> obj = numpy.array([1, 2, 3], dtype=np.int64)  # int
        return 'Ni1|3[1,2,3]'

    >>> obj = numpy.array([1, 2, 3], dtype=np.uint64)  # unit
        return 'Nu1|3[1,2,3]'
    """
    shape = obj.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    s_k: cython.Py_ssize_t
    s_l: cython.Py_ssize_t

    # 1-dimensional
    if ndim == 1:
        s_i, s_j, s_k, s_l = shape[0], 0, 0, 0
        items = [ndarray_getitem_1d(obj, i) for i in range(s_i)]  # type: ignore
        # fmt: off
        return "%s1|%d%s" % (  # 'Ni1|3[...]'
            pfix, s_i, _orjson_dumps(items) )
        # fmt: on

    # 2-dimensional
    elif ndim == 2:
        s_i, s_j, s_k, s_l = shape[0], shape[1], 0, 0
        items = [
            ndarray_getitem_2d(obj, i, j)  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
        ]
        # fmt: off
        return "%s2|%d|%d%s" % (  # 'Nu2|2|3[...]'
            pfix, s_i, s_j, _orjson_dumps(items) )
        # fmt: on

    # 3-dimensional
    elif ndim == 3:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], 0
        items = [
            ndarray_getitem_3d(obj, i, j, k)  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
        ]
        # fmt: off
        return "%s3|%d|%d|%d%s" % (  # 'Nu3|2|2|3[...]'
            pfix, s_i, s_j, s_k, _orjson_dumps(items) )
        # fmt: on

    # 4-dimensional
    else:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], shape[3]
        items = [
            ndarray_getitem_4d(obj, i, j, k, l)  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
            for l in range(s_l)
        ]
        # fmt: off
        return "%s4|%d|%d|%d|%d%s" % (  # 'Nu4|2|2|2|3[...]'
            pfix, s_i, s_j, s_k, s_l, _orjson_dumps(items) )
        # fmt: on


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray_bool(obj: np.ndarray, ndim: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "b" (bool).

    ### Example:
    >>> obj = numpy.array([True, False, True], dtype=np.bool_)
        # identifier: {N}b1|3[1,0,1]
        # dtype:      N{b}1|3[1,0,1]
        # dimension:  Nb{1}|3[1,0,1]
        # shape:      Nb1|{3}[1,0,1]
        # items:      Nb1|3{[1,0,1]}
        return 'Nb1|3[1,0,1]'
    """
    shape = obj.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    s_k: cython.Py_ssize_t
    s_l: cython.Py_ssize_t

    # 1-dimensional
    if ndim == 1:
        s_i, s_j, s_k, s_l = shape[0], 0, 0, 0
        items = [1 if ndarray_getitem_1d(obj, i) else 0 for i in range(s_i)]  # type: ignore
        # fmt: off
        return "%s1|%d%s" % (  # 'Nb1|3[...]'
            prefix.NDARRAY_BOOL, s_i, _orjson_dumps(items))
        # fmt: on

    # 2-dimensional
    elif ndim == 2:
        s_i, s_j, s_k, s_l = shape[0], shape[1], 0, 0
        items = [
            1 if ndarray_getitem_2d(obj, i, j) else 0  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
        ]
        # fmt: off
        return "%s2|%d|%d%s" % (  # 'Nb2|2|3[...]'
            prefix.NDARRAY_BOOL, s_i, s_j, _orjson_dumps(items))
        # fmt: on

    # 3-dimensional
    elif ndim == 3:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], 0
        items = [
            1 if ndarray_getitem_3d(obj, i, j, k) else 0  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
        ]
        # fmt: off
        return "%s3|%d|%d|%d%s" % (  # 'Nb3|2|2|3[...]'
            prefix.NDARRAY_BOOL, s_i, s_j, s_k, _orjson_dumps(items))
        # fmt: on

    # 4-dimensional
    else:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], shape[3]
        items = [
            1 if ndarray_getitem_4d(obj, i, j, k, l) else 0  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
            for l in range(s_l)
        ]
        # fmt: off
        return "%s4|%d|%d|%d|%d%s" % (  # 'Nb4|2|2|2|3[...]'
            prefix.NDARRAY_BOOL, s_i, s_j, s_k, s_l, _orjson_dumps(items))
        # fmt: on


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray_dt64td64(
    obj: np.ndarray,
    ndim: cython.Py_ssize_t,
    dt64: cython.bint,
) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    This function is specifically for ndarray with
    dtype of: "M" (datetime64) and "m" (timedelta64).

    ### Example:
    >>> obj = numpy.array([1, 2, 3], dtype="timedelta64[ns]")
        # identifier: {N}mns1|3[1,2,3]
        # dtype:      N{m}ns1|3[1,2,3]
        # time unit:  Nm{ns}1|3[1,2,3]
        # dimension:  Nmns{1}|3[1,2,3]
        # shape:      Nmns1|{3}[1,2,3]
        # items:      Nmns1|3{[1,2,3]}
        return 'Nmns1|3[1,2,3]'

    >>> obj = numpy.array(["2023-01-01", "2023-01-02", "2023-01-03"], dtype="datetime64[s]")
        return 'NMs1|3[1672531200,1672617600,1672704000]'
    """
    shape = obj.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    s_k: cython.Py_ssize_t
    s_l: cython.Py_ssize_t

    # 1-dimensional
    if ndim == 1:
        s_i, s_j, s_k, s_l = shape[0], 0, 0, 0
        # . empty ndarray
        if s_i == 0:  # 'Mmns1|0[]'
            dtype = _parse_ndarray_dt64td64_dtype(obj.dtype.str, dt64)
            return "%s%s1|0[]" % (prefix.NDARRAY, dtype)
        # . cast into int64
        dtype = _match_ndarray_dt64td64_dtype(np.get_datetime64_unit(obj[0]), dt64)
        obj = np.PyArray_Cast(obj, np.NPY_TYPES.NPY_INT64)
        # . serialization
        items = [ndarray_getitem_1d(obj, i) for i in range(s_i)]  # type: ignore
        # fmt: off
        return "%s%s1|%d%s" % (  # 'Nmns1|3[...]'
            prefix.NDARRAY, dtype, s_i, _orjson_dumps(items))
        # fmt: on

    # 2-dimensional
    elif ndim == 2:
        s_i, s_j, s_k, s_l = shape[0], shape[1], 0, 0
        # . empty ndarray
        if s_j == 0:  # 'Mmns2|2|0[]'
            dtype = _parse_ndarray_dt64td64_dtype(obj.dtype.str, dt64)
            return "%s%s2|%d|0[]" % (prefix.NDARRAY, dtype, s_i)
        # . cast into int64
        dtype = _match_ndarray_dt64td64_dtype(np.get_datetime64_unit(obj[0, 0]), dt64)
        obj = np.PyArray_Cast(obj, np.NPY_TYPES.NPY_INT64)
        # . serialization
        items = [
            ndarray_getitem_2d(obj, i, j)  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
        ]
        # fmt: off
        return "%s%s2|%d|%d%s" % (  # 'Nmns2|2|3[...]'
            prefix.NDARRAY, dtype, s_i, s_j, _orjson_dumps(items))
        # fmt: on

    # 3-dimensional
    elif ndim == 3:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], 0
        # . empty ndarray
        if s_k == 0:  # 'Mmns3|2|2|0[]'
            dtype = _parse_ndarray_dt64td64_dtype(obj.dtype.str, dt64)
            return "%s%s3|%d|%d|0[]" % (prefix.NDARRAY, dtype, s_i, s_j)
        # . cast into int64
        dtype = _match_ndarray_dt64td64_dtype(
            np.get_datetime64_unit(obj[0, 0, 0]), dt64
        )
        obj = np.PyArray_Cast(obj, np.NPY_TYPES.NPY_INT64)
        # . serialization
        items = [
            ndarray_getitem_3d(obj, i, j, k)  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
        ]
        # fmt: off
        return "%s%s3|%d|%d|%d%s" % (  # 'Nmns3|2|2|3[...]'
            prefix.NDARRAY, dtype, s_i, s_j, s_k, _orjson_dumps(items))
        # fmt: on

    # 4-dimensional
    else:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], shape[3]
        # . empty ndarray
        if s_l == 0:  # 'Mmns4|2|2|2|0[]'
            dtype = _parse_ndarray_dt64td64_dtype(obj.dtype.str, dt64)
            return "%s%s4|%d|%d|%d|0[]" % (prefix.NDARRAY, dtype, s_i, s_j, s_k)
        # . cast into int64
        dtype = _match_ndarray_dt64td64_dtype(
            np.get_datetime64_unit(obj[0, 0, 0, 0]), dt64
        )
        obj = np.PyArray_Cast(obj, np.NPY_TYPES.NPY_INT64)
        # . serialization
        items = [
            ndarray_getitem_4d(obj, i, j, k, l)  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
            for l in range(s_l)
        ]
        # fmt: off
        return "%s%s4|%d|%d|%d|%d%s" % (  # 'Nmns4|2|2|2|3[...]'
            prefix.NDARRAY, dtype, s_i, s_j, s_k, s_l, _orjson_dumps(items))
        # fmt: on


@cython.cfunc
@cython.inline(True)
def _match_ndarray_dt64td64_dtype(unit: np.NPY_DATETIMEUNIT, dt64: cython.bint) -> str:
    """(cfunc) Match `<'numpy.ndarray[datetime64/timedetla64]'>`
    serialization dtype through its time unit `<'str'>`."""
    # Common units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ns:
        return prefix.NDARRAY_DTYPE_DT64_NS if dt64 else prefix.NDARRAY_DTYPE_TD64_NS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_us:
        return prefix.NDARRAY_DTYPE_DT64_US if dt64 else prefix.NDARRAY_DTYPE_TD64_US
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ms:
        return prefix.NDARRAY_DTYPE_DT64_MS if dt64 else prefix.NDARRAY_DTYPE_TD64_MS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_s:
        return prefix.NDARRAY_DTYPE_DT64_S if dt64 else prefix.NDARRAY_DTYPE_TD64_S
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_m:
        return prefix.NDARRAY_DTYPE_DT64_MI if dt64 else prefix.NDARRAY_DTYPE_TD64_MI
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_h:
        return prefix.NDARRAY_DTYPE_DT64_H if dt64 else prefix.NDARRAY_DTYPE_TD64_H
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_D:
        return prefix.NDARRAY_DTYPE_DT64_D if dt64 else prefix.NDARRAY_DTYPE_TD64_D
    # Uncomman units
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_ps:
        return prefix.NDARRAY_DTYPE_DT64_PS if dt64 else prefix.NDARRAY_DTYPE_TD64_PS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_fs:
        return prefix.NDARRAY_DTYPE_DT64_FS if dt64 else prefix.NDARRAY_DTYPE_TD64_FS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_as:
        return prefix.NDARRAY_DTYPE_DT64_AS if dt64 else prefix.NDARRAY_DTYPE_TD64_AS
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_Y:
        return prefix.NDARRAY_DTYPE_DT64_Y if dt64 else prefix.NDARRAY_DTYPE_TD64_Y
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_M:
        return prefix.NDARRAY_DTYPE_DT64_M if dt64 else prefix.NDARRAY_DTYPE_TD64_M
    if unit == np.NPY_DATETIMEUNIT.NPY_FR_W:
        return prefix.NDARRAY_DTYPE_DT64_W if dt64 else prefix.NDARRAY_DTYPE_TD64_W
    # if unit == np.NPY_DATETIMEUNIT.NPY_FR_B:
    #     return prefix.NDARRAY_DTYPE_DT64_B if dt64 else prefix.NDARRAY_DTYPE_TD64_B
    raise errors.SerializeTypeError(
        "<'Serializor'>\nFailed to serialize <'ndarray[%s']>: unknown time unit [%d]."
        % ("datetime64" if dt64 else "timedelta64", unit)
    )


@cython.cfunc
@cython.inline(True)
def _parse_ndarray_dt64td64_dtype(dtype_str: object, dt64: cython.bint) -> str:
    """(cfunc) Parse `<'numpy.ndarray[datetime64/timedelta64]'>`
    dtype from 'ndarray.dtype.str' `<'str'>`."""
    unit: str = str_substr(dtype_str, 4, str_len(dtype_str) - 1)
    if not 1 <= str_len(unit) <= 2:
        raise errors.SerializeTypeError(
            "<'Serializor'>\nFailed to serialize <'ndarray[%s]'>: unknown time unit [%s]."
            % ("datetime64" if dt64 else "timedelta64", unit)
        )
    if dt64:
        return prefix.NDARRAY_DTYPE_DT64 + unit
    else:
        return prefix.NDARRAY_DTYPE_TD64 + unit


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray_complex(obj: np.ndarray, ndim: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "c" (complex).

    ### Example:
    >>> obj = numpy.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
        # identifider: {N}c1|3[1.0,1.0,2.0,2.0,3.0,3.0]
        # dtype:       N{c}1|3[1.0,1.0,2.0,2.0,3.0,3.0]
        # dimension:   Nc{1}|3[1.0,1.0,2.0,2.0,3.0,3.0]
        # shape:       Nc1|{3}[1.0,1.0,2.0,2.0,3.0,3.0]
        # items:       Nc1|3{[1.0,1.0,2.0,2.0,3.0,3.0]}
        return 'Nc1|3[1.0,1.0,2.0,2.0,3.0,3.0]'
    """
    shape = obj.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    s_k: cython.Py_ssize_t
    s_l: cython.Py_ssize_t

    # 1-dimensional
    items: list = []
    if ndim == 1:
        s_i, s_j, s_k, s_l = shape[0], 0, 0, 0
        for i in range(s_i):
            item = ndarray_getitem_1d(obj, i)  # type: ignore
            items.append(complex_getreal(item))
            items.append(complex_getimag(item))
        # fmt: off
        return "%s1|%d%s" % (  # 'Nc1|3[...]'
            prefix.NDARRAY_COMPLEX, s_i, _orjson_dumps(items))
        # fmt: on

    # 2-dimensional
    elif ndim == 2:
        s_i, s_j, s_k, s_l = shape[0], shape[1], 0, 0
        for i in range(s_i):
            for j in range(s_j):
                item = ndarray_getitem_2d(obj, i, j)  # type: ignore
                items.append(complex_getreal(item))
                items.append(complex_getimag(item))
        # fmt: off
        return "%s2|%d|%d%s" % (  # 'Nc2|2|3[...]'
            prefix.NDARRAY_COMPLEX, s_i, s_j, _orjson_dumps(items))
        # fmt: on

    # 3-dimensional
    elif ndim == 3:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], 0
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    item = ndarray_getitem_3d(obj, i, j, k)  # type: ignore
                    items.append(complex_getreal(item))
                    items.append(complex_getimag(item))
        # fmt: off
        return "%s3|%d|%d|%d%s" % (  # 'Nc3|2|2|3[...]'
            prefix.NDARRAY_COMPLEX, s_i, s_j, s_k, _orjson_dumps(items))
        # fmt: on

    # 4-dimensional
    else:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], shape[3]
        for i in range(s_i):
            for j in range(s_j):
                for k in range(s_k):
                    for l in range(s_l):
                        item = ndarray_getitem_4d(obj, i, j, k, l)  # type: ignore
                        items.append(complex_getreal(item))
                        items.append(complex_getimag(item))
        # fmt: off
        return "%s4|%d|%d|%d|%d%s" % (  # 'Nc4|2|2|2|3[...]'
            prefix.NDARRAY_COMPLEX, s_i, s_j, s_k, s_l, _orjson_dumps(items))
        # fmt: on


@cython.cfunc
@cython.inline(True)
def _serialize_ndarray_bytes(obj: np.ndarray, ndim: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'numpy.ndarray'>` to `<'str'>`.

    This function is specifically for ndarray
    with dtype of: "S" (bytes string).

    ### Example:
    >>> obj = numpy.array([b"1", b"2", b"3"], dtype="S")
        # identifier: {N}S1|3["1","2","3"]
        # dtype:      N{S}1|3["1","2","3"]
        # dimension:  NS{1}|3["1","2","3"]
        # shape:      NS1|{3}["1","2","3"]
        # items:      NS1|3{["1","2","3"]}
        return 'NS1|3["1","2","3"]'
    """
    shape = obj.shape
    s_i: cython.Py_ssize_t
    s_j: cython.Py_ssize_t
    s_k: cython.Py_ssize_t
    s_l: cython.Py_ssize_t

    # 1-dimensional
    if ndim == 1:
        s_i, s_j, s_k, s_l = shape[0], 0, 0, 0
        items = [
            decode_bytes(ndarray_getitem_1d(obj, i)) for i in range(s_i)  # type: ignore
        ]
        # fmt: off
        return "%s1|%d%s" % (  # 'NS1|3[...]'
            prefix.NDARRAY_BYTES, s_i, _orjson_dumps(items))
        # fmt: on

    # 2-dimensional
    elif ndim == 2:
        s_i, s_j, s_k, s_l = shape[0], shape[1], 0, 0
        items = [
            decode_bytes(ndarray_getitem_2d(obj, i, j))  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
        ]
        # fmt: off
        return "%s2|%d|%d%s" % (  # 'NS2|2|3[...]'
            prefix.NDARRAY_BYTES, s_i, s_j, _orjson_dumps(items))
        # fmt: on

    # 3-dimensional
    elif ndim == 3:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], 0
        items = [
            decode_bytes(ndarray_getitem_3d(obj, i, j, k))  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
        ]
        # fmt: off
        return "%s3|%d|%d|%d%s" % (  # 'NS3|2|2|3[...]'
            prefix.NDARRAY_BYTES, s_i, s_j, s_k, _orjson_dumps(items))
        # fmt: on

    # 4-dimensional
    else:
        s_i, s_j, s_k, s_l = shape[0], shape[1], shape[2], shape[3]
        items = [
            decode_bytes(ndarray_getitem_4d(obj, i, j, k, l))  # type: ignore
            for i in range(s_i)
            for j in range(s_j)
            for k in range(s_k)
            for l in range(s_l)
        ]
        # fmt: off
        return "%s4|%d|%d|%d|%d%s" % (  # 'NS4|2|2|2|3[...]'
            prefix.NDARRAY_BYTES, s_i, s_j, s_k, s_l, _orjson_dumps(items))
        # fmt: on


# Pandas Series ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_series(obj: Series) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    ### Example:
    >>> obj = pandas.Series([1, 2, 3], dtype=np.int64)
        # identifier: {I}i3[1,2,3]
        # dtype:      I{i}3[1,2,3]
        # ser size:   Ii{3}[1,2,3]
        # items:      Ii3{[1,2,3]}
        return 'Ii3[1,2,3]'

    ### Notes:
    - 1. [ser size] section represents the total number of items in the Series.
    - 2. Empty Series, returns 'I{dtype}0[]'.
    - 3. Besides dtype of 'object', other dtypes utilize 'orjson.dumps'
         to serialize items as a json array for maximum performance.
    - 4. For dtype of 'object', items are serialized one by one, similar
         to the serialization process of a list. For more details, please
         refer to the `_serialize_series_object()` function.
    """
    # Access Series name & values
    name: object = obj.name
    values: np.ndarray = obj.values

    # Get Series dtype
    dtype: cython.Py_UCS4 = values.descr.kind

    # Get Series size
    size: cython.Py_ssize_t = values.shape[0]

    # Serialize Series
    # . Series[object]
    if dtype == prefix.NDARRAY_DTYPE_OBJECT_ID:
        return _serialize_series_object(name, values, size)
    # . Series[float]
    if dtype == prefix.NDARRAY_DTYPE_FLOAT_ID:
        if np.PyArray_TYPE(values) == np.NPY_TYPES.NPY_FLOAT16:
            values = np.PyArray_Cast(values, np.NPY_TYPES.NPY_FLOAT32)
        return _serialize_series_common(name, values, size, prefix.SERIES_FLOAT)
    # . Series[int]
    if dtype == prefix.NDARRAY_DTYPE_INT_ID:
        return _serialize_series_common(name, values, size, prefix.SERIES_INT)
    # . Series[uint]
    if dtype == prefix.NDARRAY_DTYPE_UINT_ID:
        return _serialize_series_common(name, values, size, prefix.SERIES_UINT)
    # . Series[bool]
    if dtype == prefix.NDARRAY_DTYPE_BOOL_ID:
        return _serialize_series_bool(name, values, size)
    # . Series[datetime64]
    if dtype == prefix.NDARRAY_DTYPE_DT64_ID:
        return _serialize_series_dt64td64(name, values, size, True)
    # . Series[timedelta64]
    if dtype == prefix.NDARRAY_DTYPE_TD64_ID:
        return _serialize_series_dt64td64(name, values, size, False)
    # . Series[complex]
    if dtype == prefix.NDARRAY_DTYPE_COMPLEX_ID:
        return _serialize_series_complex(name, values, size)
    # . Series[bytes]
    if dtype == prefix.NDARRAY_DTYPE_BYTES_ID:
        return _serialize_series_bytes(name, values, size)
    # . Series[str]
    if dtype == prefix.NDARRAY_DTYPE_UNICODE_ID:
        return _serialize_series_common(name, values, size, prefix.SERIES_UNICODE)
    # . invalid dtype
    raise errors.SerializeTypeError(
        "<'Serializor'>\nFailed to serialize <'pandas.Series'>: "
        "unsupported dtype [%s]." % obj.dtype
    )


@cython.cfunc
@cython.inline(True)
def _serialize_series_object(
    name: object,
    values: np.ndarray,
    size: cython.Py_ssize_t,
) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    This function is specifically for Series
    with dtype of: "O" (object).

    ### Example:
    >>> obj = pandas.Series([1, 1.234, True, "abc", 1 + 1j])
        # identifier: {I}O5[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # dtype:      I{O}5[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # ser size:   IO{5}[i1,f1.234,o1,s3|abc,c1.0|1.0,]
        # items:      IO5{[i1,f1.234,o1,s3|abc,c1.0|1.0,]}
        return 'IO5[i1,f1.234,o1,s3|abc,c1.0|1.0,]'

    ### Notes:
    - 1. Different from other dtypes, [items] section for object
         does not use 'orjson.dumps' for serialization. Instead,
         it performs custom serialization similar to a list.
    - 2. Empty Series, returns 'IO1|0[]'.
    - 3. Non-empty Series, [items] section always end with a
         comma followed by the closing bracket `',]'`.
    """
    # Empty Series
    if size == 0:  # 'IO1|0[]'
        if name is None:
            return "%s0[]" % prefix.SERIES_OBJECT
        else:
            return "%s0|%s[]" % (prefix.SERIES_OBJECT, name)

    # Serialize Series[object]
    items = [_serialize_common_type(ndarray_getitem_1d(values, i)) for i in range(size)]  # type: ignore
    if name is None:
        return "%s%d[%s,]" % (prefix.SERIES_OBJECT, size, ",".join(items))
    else:
        return "%s%d|%s[%s,]" % (prefix.SERIES_OBJECT, size, name, ",".join(items))


@cython.cfunc
@cython.inline(True)
def _serialize_series_common(
    name: object,
    values: np.ndarray,
    size: cython.Py_ssize_t,
    pfix: str,
) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    This function is specifically for Series with dtype of:
    "U" (str), "f" (float), "i" (int) and "u" (uint).

    Identifier is determined by the 'pfix' argument.

    ### Example:
    >>> obj = pandas.Series([1.1, 2.2, 3.3], dtype=np.float64)  # float
        # identifier: {I}f3[1.1,2.2,3.3]
        # dtype:      I{f}3[1.1,2.2,3.3]
        # ser size:   If{3}[1.1,2.2,3.3]
        # items:      If3{[1.1,2.2,3.3]}
        return 'If3[1.1,2.2,3.3]'

    >>> obj = pandas.Series([1, 2, 3], dtype=np.int64)  # int
        return 'Ii3[1,2,3]'

    >>> obj = pandas.Series([1, 2, 3], dtype=np.uint64)  # uint
        return 'Iu3[1,2,3]'
    """
    items = [ndarray_getitem_1d(values, i) for i in range(size)]  # type: ignore
    if name is None:
        return "%s%d%s" % (pfix, size, _orjson_dumps(items))
    else:
        return "%s%d|%s%s" % (pfix, size, name, _orjson_dumps(items))


@cython.cfunc
@cython.inline(True)
def _serialize_series_bool(
    name: object,
    values: np.ndarray,
    size: cython.Py_ssize_t,
) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    This function is specifically for Series
    with dtype of: "b" (bool).

    ### Example:
    >>> obj = pandas.Series([True, False, True], dtype=np.bool_)
        # identifier: {I}b3[1,0,1]
        # dtype:      I{b}3[1,0,1]
        # ser size:   Ib{3}[1,0,1]
        # items:      Ib3[1,0,1]
        return 'Ib3[1,0,1]'
    """
    items = [1 if ndarray_getitem_1d(values, i) else 0 for i in range(size)]  # type: ignore
    if name is None:
        return "%s%d%s" % (prefix.SERIES_BOOL, size, _orjson_dumps(items))
    else:
        return "%s%d|%s%s" % (prefix.SERIES_BOOL, size, name, _orjson_dumps(items))


@cython.cfunc
@cython.inline(True)
def _serialize_series_dt64td64(
    name: object,
    values: np.ndarray,
    size: cython.Py_ssize_t,
    dt64: cython.bint,
) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    This function is specifically for Series with
    dtype of: "M" (datetime64) and "m" (timedelta64).

    ### Example:
    >>> obj = pandas.Series([1, 2, 3], dtype="timedelta64[ns]")
        # identifier: {I}mns3[1,2,3]
        # dtype:      I{m}ns3[1,2,3]
        # time unit:  Im{ns}3[1,2,3]
        # ser size:   Imns{3}[1,2,3]
        # items:      Imns3{[1,2,3]}
        return 'Imns3[1,2,3]'

    >>> obj = pandas.Series(["2023-01-01", "2023-01-02", "2023-01-03"], dtype="datetime64[s]")
        return IMs3[1672531200,1672617600,1672704000]

    ### Notes:
    - For Series[datetime64] `WITH` timezone, all datetimes will be
      converted to UTC time after serialization and the timezone
      information will be `LOST`.
    """
    # Empty Series
    if size == 0:
        dtype = _parse_ndarray_dt64td64_dtype(values.dtype.str, dt64)
        if name is None:
            return "%s%s0[]" % (prefix.SERIES, dtype)
        else:
            return "%s%s0|%s[]" % (prefix.SERIES, dtype, name)
    # Cast into int64
    dtype = _match_ndarray_dt64td64_dtype(np.get_datetime64_unit(values[0]), dt64)
    values = np.PyArray_Cast(values, np.NPY_TYPES.NPY_INT64)
    # Serialization
    items = [ndarray_getitem_1d(values, i) for i in range(size)]  # type: ignore
    if name is None:
        return "%s%s%d%s" % (prefix.SERIES, dtype, size, _orjson_dumps(items))
    else:
        return "%s%s%d|%s%s" % (prefix.SERIES, dtype, size, name, _orjson_dumps(items))


@cython.cfunc
@cython.inline(True)
def _serialize_series_complex(
    name: object,
    values: np.ndarray,
    size: cython.Py_ssize_t,
) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    This function is specifically for Series
    with dtype of: "c" (complex).

    ### Example
    >>> obj = pandas.Series([1 + 1j, 2 + 2j, 3 + 3j])
        # identifier: {I}c3[1.0,1.0,2.0,2.0,3.0,3.0]
        # dtype:      I{c}3[1.0,1.0,2.0,2.0,3.0,3.0]
        # ser size:   Ic{3}[1.0,1.0,2.0,2.0,3.0,3.0]
        # items:      Ic3{[1.0,1.0,2.0,2.0,3.0,3.0]}
        return 'Ic3[1.0,1.0,2.0,2.0,3.0,3.0]'
    """
    items = []
    for i in range(size):
        item = ndarray_getitem_1d(values, i)  # type: ignore
        items.append(complex_getreal(item))
        items.append(complex_getimag(item))
    if name is None:
        return "%s%d%s" % (prefix.SERIES_COMPLEX, size, _orjson_dumps(items))
    else:
        return "%s%d|%s%s" % (prefix.SERIES_COMPLEX, size, name, _orjson_dumps(items))


@cython.cfunc
@cython.inline(True)
def _serialize_series_bytes(
    name: object,
    values: np.ndarray,
    size: cython.Py_ssize_t,
) -> str:
    """(cfunc) Serialize `<'pandas.Series'>` to `<'str'>`.

    This function is specifically for Series
    with dtype of: "S" (bytes string).

    ### Example:
    >>> obj = pandas.Series([b"1", b"2", b"3"], dtype="S")
        # identifier: {I}S3["1","2","3"]
        # dtype:      I{S}3["1","2","3"]
        # ser size:   IS{3}["1","2","3"]
        # items:      IS3{["1","2","3"]}
        return 'IS3["1","2","3"]'
    """
    items = [decode_bytes(ndarray_getitem_1d(values, i)) for i in range(size)]  # type: ignore
    if name is None:
        return "%s%d%s" % (prefix.SERIES_BYTES, size, _orjson_dumps(items))
    else:
        return "%s%d|%s%s" % (prefix.SERIES_BYTES, size, name, _orjson_dumps(items))


# Pandas DataFrame ------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_dataframe(obj: DataFrame) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` to `<'str'>`.

    ### Example:
    >>> obj = pandas.DataFrame({"i_col": [1, 2, 3], "f_col": [1.1, 2.2, 3.3]})
        # identifier:   {F}3|17["i_col","f_col"][i[1,2,3],f[1.1,2.2,3.3],]
        # rows (len):   F{3}|17["i_col","f_col"][i[1,2,3],f[1.1,2.2,3.3],]
        # names length: F3|{17}["i_col","f_col"][i[1,2,3],f[1.1,2.2,3.3],]
        # names values: F3|17{["i_col","f_col"]}[i[1,2,3],f[1.1,2.2,3.3],]
        # columns:      F3|17["i_col","f_col"]{[i[1,2,3],f[1.1,2.2,3.3],]}
        return 'F3|17["i_col","f_col"][i[1,2,3],f[1.1,2.2,3.3],]'

    ### Notes:
    - 1. [rows (len)] section represents the total number of
         rows in the DataFrame.
    - 2. [names values] section are the DataFrame column names
         serialized into json array string. This approach allows
         leveraging 'orjson' to greatly improve serialization
         & deserialization performance.
    - 3. [names length] section is the unicode length of the
         json array string for [names values].
    - 4. True empty DataFrame, returns 'F0|0[]'.
    - 5. Empty DataFrame with columns, returns 'F0|{names length}[{names values}]'.
    - 6. Non-empty DataFrame, [columns] section always end with
         a comma followed by the closing bracket `',]'`.
    """
    # Get DataFrame rows (length)
    rows: cython.Py_ssize_t = len(obj.index)

    # Serialize DataFrame columns into json array
    cols_str: str = _orjson_dumps(obj.columns.tolist())
    cols_len: cython.Py_ssize_t = str_len(cols_str)
    # . true empty DataFrame
    if cols_len == 2:  # '[]'
        return "%s0|0[]" % prefix.DATAFRAME
    # . empty DataFrame with columns
    if rows == 0:
        return "%s0|%d%s" % (prefix.DATAFRAME, cols_len, cols_str)

    # Serialize DataFrame
    vals: list = []
    for _, col in obj.items():
        # . access column (Series) values
        values: np.ndarray = col.values
        # . get column (Series) dtype
        dtype: cython.Py_UCS4 = values.descr.kind
        # . `<object>`
        if dtype == prefix.NDARRAY_DTYPE_OBJECT_ID:
            v = _serialize_dataframe_object(values, rows)
        # . `<'float'>`
        elif dtype == prefix.NDARRAY_DTYPE_FLOAT_ID:
            if np.PyArray_TYPE(values) == np.NPY_TYPES.NPY_FLOAT16:
                values = np.PyArray_Cast(values, np.NPY_TYPES.NPY_FLOAT32)
            v = _serialize_dataframe_common(values, rows, prefix.DATAFRAME_COL_FLOAT)
        # . `<'int'>`
        elif dtype == prefix.NDARRAY_DTYPE_INT_ID:
            v = _serialize_dataframe_common(values, rows, prefix.DATAFRAME_COL_INT)
        # . `<uint>`
        elif dtype == prefix.NDARRAY_DTYPE_UINT_ID:
            v = _serialize_dataframe_common(values, rows, prefix.DATAFRAME_COL_UINT)
        # . `<'bool'>`
        elif dtype == prefix.NDARRAY_DTYPE_BOOL_ID:
            v = _serialize_dataframe_bool(values, rows)
        # . `<datetime64>`
        elif dtype == prefix.NDARRAY_DTYPE_DT64_ID:
            v = _serialize_dataframe_dt64td64(values, rows, True)
        # . `<timedelta64>`
        elif dtype == prefix.NDARRAY_DTYPE_TD64_ID:
            v = _serialize_dataframe_dt64td64(values, rows, False)
        # . `<'complex'>`
        elif dtype == prefix.NDARRAY_DTYPE_COMPLEX_ID:
            v = _serialize_dataframe_complex(values, rows)
        # . `<'bytes'>`
        elif dtype == prefix.NDARRAY_DTYPE_BYTES_ID:
            v = _serialize_dataframe_bytes(values, rows)
        # . `<'str'>`
        elif dtype == prefix.NDARRAY_DTYPE_UNICODE_ID:
            v = _serialize_dataframe_common(values, rows, prefix.DATAFRAME_COL_UNICODE)
        # . invalid dtype
        else:
            raise errors.SerializeTypeError(
                "<'Serializor'>\nFailed to serialize <'pandas.DataFrame'>: "
                "unsupported column (Series) dtype [%s]." % col.dtype
            )
        vals.append(v)

    # Fromat serialization
    # fmt: off
    return "%s%d|%d%s[%s,]" % (
        prefix.DATAFRAME, rows, cols_len, cols_str, ",".join(vals))
    # fmt: on


@cython.cfunc
@cython.inline(True)
def _serialize_dataframe_object(values: np.ndarray, rows: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` column to `<'str'>`.

    This function is specifically for DataFrame columns
    with dtype of: "O" (object).
    """
    items = [_serialize_common_type(ndarray_getitem_1d(values, i)) for i in range(rows)]  # type: ignore
    return "%s[%s,]" % (prefix.DATAFRAME_COL_OBJECT, ",".join(items))


@cython.cfunc
@cython.inline(True)
def _serialize_dataframe_common(
    values: np.ndarray,
    rows: cython.Py_ssize_t,
    dtype: str,
) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` column to `<'str'>`.

    This function is specifically for DataFrame columns with
    dtype of: "U" (str), "f" (float), "i" (int) and "u" (uint).
    """
    items = [ndarray_getitem_1d(values, i) for i in range(rows)]  # type: ignore
    return dtype + _orjson_dumps(items)


@cython.cfunc
@cython.inline(True)
def _serialize_dataframe_bool(values: np.ndarray, rows: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` column to `<'str'>`.

    This function is specifically for DataFrame columns
    with dtype of: "b" (bool).
    """
    items = [1 if ndarray_getitem_1d(values, i) else 0 for i in range(rows)]  # type: ignore
    return prefix.DATAFRAME_COL_BOOL + _orjson_dumps(items)


@cython.cfunc
@cython.inline(True)
def _serialize_dataframe_dt64td64(
    values: np.ndarray,
    rows: cython.Py_ssize_t,
    dt64: cython.bint,
) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` column to `<'str'>`.

    This function is specifically for DataFrame columns with
    dtype of: "M" (datetime64) and "m" (timedelta64).
    """
    # Cast into int64
    dtype = _match_ndarray_dt64td64_dtype(np.get_datetime64_unit(values[0]), dt64)
    values = np.PyArray_Cast(values, np.NPY_TYPES.NPY_INT64)
    # Serialization
    items = [ndarray_getitem_1d(values, i) for i in range(rows)]  # type: ignore
    return dtype + _orjson_dumps(items)


@cython.cfunc
@cython.inline(True)
def _serialize_dataframe_complex(values: np.ndarray, rows: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` column to `<'str'>`.

    This function is specifically for DataFrame columns
    with dtype of: "c" (complex).
    """
    items = []
    for i in range(rows):
        item = ndarray_getitem_1d(values, i)  # type: ignore
        items.append(complex_getreal(item))
        items.append(complex_getimag(item))
    return prefix.DATAFRAME_COL_COMPLEX + _orjson_dumps(items)


@cython.cfunc
@cython.inline(True)
def _serialize_dataframe_bytes(values: np.ndarray, rows: cython.Py_ssize_t) -> str:
    """(cfunc) Serialize `<'pandas.DataFrame'>` column to `<'str'>`.

    This function is specifically for DataFrame columns
    with dtype of: "S" (bytes string).
    """
    items = [decode_bytes(ndarray_getitem_1d(values, i)) for i in range(rows)]  # type: ignore
    return prefix.DATAFRAME_COL_COMPLEX + _orjson_dumps(items)


# Pandas Datetime/Timedelta Index ---------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_datetime_index(obj: DatetimeIndex) -> str:
    """(cfunc) Serialize `<'pandas.DatetimeIndex'>` to `<'str'>`.

    ### Exmaple:
    >>> obj = pandas.date_range("2021-01-01", periods=2)
        # identifier: {Z}IMns2[1609459200000000000,1609545600000000000]
        # ignore:     Z{IM}ns2[1609459200000000000,1609545600000000000]
        # time unit:  ZIM{ns}2[1609459200000000000,1609545600000000000]
        # ser size:   ZIMns{2}[1609459200000000000,1609545600000000000]
        # items:      ZIMns2{[1609459200000000000,1609545600000000000]}
        return ZIMns2[1609459200000000000,1609545600000000000]

    ### Notes:
    - For DatetimeIndex `WITH` timezone, all datetimes will be
      converted to UTC time after serialization and the timezone
      information will be `LOST`.
    """
    # Get Index name & values & size
    name: object = obj.name
    values: np.ndarray = obj.values
    size: cython.Py_ssize_t = values.shape[0]
    # Serialize DatetimeIndex
    return prefix.DATETIMEINDEX + _serialize_series_dt64td64(name, values, size, True)


@cython.cfunc
@cython.inline(True)
def _serialize_timedelta_index(obj: TimedeltaIndex) -> str:
    """(cfunc) Serialize `<'pandas.TimedeltaIndex'>` to `<'str'>`.

    ### Example:
    >>> obj = pandas.timedelta_range("1 days", periods=2)
        # identifier: {X}Imns2[86400000000000,172800000000000]
        # ignore:     X{Im}ns2[86400000000000,172800000000000]
        # time unit:  XIm{ns}2[86400000000000,172800000000000]
        # ser size:   XImns{2}[86400000000000,172800000000000]
        # items:      XImns2{[86400000000000,172800000000000]}
        return XImns2[86400000000000,172800000000000]
    """
    # Get Index name & values & size
    name: object = obj.name
    values: np.ndarray = obj.values
    size: cython.Py_ssize_t = values.shape[0]
    # Serialize TimedeltaIndex
    return prefix.TIMEDELTAINDEX + _serialize_series_dt64td64(name, values, size, False)


# Serialize -------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _serialize_common_type(obj: object) -> str:
    """(cfunc) Serialize Python object to `<'str'>`.

    The main entry point for serializing object into strings,
    mainly focus on the most common Python types, such as: str,
    float, int, bool, datetime, list, dict, etc.

    If the object date type is not recognized, it will be
    passed to the `_serialize_uncommon_type()` function for
    further serialization.
    """
    # Get data type
    dtype = type(obj)

    ##### Common Types #####
    # Basic Types
    # . <'str'>
    if dtype is str:
        return _serialize_str(obj)
    # . <'float'>
    if dtype is float:
        return _serialize_float(obj)
    # . <'int'>
    if dtype is int:
        return _serialize_int(obj)
    # . <'bool'>
    if dtype is bool:
        return _serialize_bool(obj)
    # . <None>
    if dtype is typeref.NONE:
        return _serialize_none()

    # Date&Time Types
    # . <'datetime.datetime'>
    if dtype is datetime.datetime:
        return _serialize_datetime(obj)
    # . <'datetime.date'>
    if dtype is datetime.date:
        return _serialize_date(obj)
    # . <'datetime.time'>
    if dtype is datetime.time:
        return _serialize_time(obj)
    # . <'datetime.timedelta'>
    if dtype is datetime.timedelta:
        return _serialize_timedelta(obj)

    # Mapping Types
    # . <'dict'>
    if dtype is dict:
        return _serialize_dict(obj)

    # Sequence Types
    # . <'list'>
    if dtype is list:
        return _serialize_list(obj)
    # . <'tuple'>
    if dtype is tuple:
        return _serialize_tuple(obj)

    ##### Uncommon Types #####
    return _serialize_uncommon_type(obj, dtype)


@cython.cfunc
@cython.inline(True)
def _serialize_uncommon_type(obj: object, dtype: type) -> str:
    """(cfunc) Serialize uncommon Python object to `<'str'>`.

    The secondary entry point for serializing object into
    strings, mainly focus on less common Python types, such
    as: complex, Decimal, numpy, pandas, etc.

    If the object date type is not recognized, it will be
    passed to the `_serialize_subclass()` function for
    further serialization.
    """
    ##### Uncommon Types #####
    # Basic Types
    # . <'numpy.float_'>
    if dtype is typeref.FLOAT64 or dtype is typeref.FLOAT32 or dtype is typeref.FLOAT16:
        return _serialize_float64(obj)
    # . <'numpy.int_'>
    if (
        dtype is typeref.INT64
        or dtype is typeref.INT32
        or dtype is typeref.INT16
        or dtype is typeref.INT8
    ):
        return _serialize_int(obj)
    # . <'numpy.uint'>
    if (
        dtype is typeref.UINT64
        or dtype is typeref.UINT32
        or dtype is typeref.UINT16
        or dtype is typeref.UINT8
    ):
        return _serialize_int(obj)
    # . <'numpy.bool_'>
    if dtype is typeref.BOOL_:
        return _serialize_bool(obj)
    # . <'np.NaN'>
    if dtype is typeref.NAN:
        return _serialize_none()

    # Date&Time Types
    # . <'pandas.Timestamp'>
    if dtype is typeref.TIMESTAMP:
        return _serialize_datetime(obj)
    # . <'time.struct_time'>`
    if dtype is typeref.STRUCT_TIME:
        return _serialize_struct_time(obj)
    # . <'pandas.Timedelta'>`
    if dtype is typeref.TIMEDELTA:
        return _serialize_timedelta(obj)
    # . <'cytimes.pydt'>
    if dtype is typeref.PYDT:
        return _serialize_datetime(obj.dt)

    # Numeric Types
    # . <'decimal.Decimal'>
    if dtype is typeref.DECIMAL:
        return _serialize_decimal(obj)
    # . <'complex'>
    if dtype is complex or dtype is typeref.COMPLEX64 or dtype is typeref.COMPLEX128:
        return _serialize_complex(obj)

    # Bytes Types
    # . <'bytes'>
    if dtype is bytes:
        return _serialize_bytes(obj)
    # . <'bytearray'>
    if dtype is bytearray:
        return _serialize_bytearray(obj)
    # . <'memoryview'>
    if dtype is memoryview:
        return _serialize_memoryview(obj)
    # . <'numpy.bytes_'>
    if dtype is typeref.BYTES_:
        return _serialize_bytes(obj)

    # Sequence Types
    # . <'set'>
    if dtype is set:
        return _serialize_set(obj)
    # . <'frozenset'>
    if dtype is frozenset:
        return _serialize_frozenset(obj)
    # . <'dict_keys'> & <'dict_values'>
    if dtype is typeref.DICT_KEYS or dtype is typeref.DICT_VALUES:
        return _serialize_sequence(obj, prefix.LIST)

    # Numpy Types
    # . <'numpy.datetime64'>
    if dtype is typeref.DATETIME64:
        return _serialize_datetime64(obj)
    # . <'numpy.timedelta64'>
    if dtype is typeref.TIMEDELTA64:
        return _serialize_timedelta64(obj)
    # . <'numpy.ndarray'>
    if dtype is np.ndarray:
        return _serialize_ndarray(obj)
    # . <'numpy.record'>
    if dtype is typeref.RECORD:
        return _serialize_sequence(obj, prefix.LIST)

    # Pandas Types
    # . <'pandas.Series'>
    if dtype is typeref.SERIES:
        return _serialize_series(obj)
    # . <'pandas.DataFrame'>
    if dtype is typeref.DATAFRAME:
        return _serialize_dataframe(obj)
    # . <'pandas.DatetimeIndex'>
    if dtype is typeref.DATETIMEINDEX:
        return _serialize_datetime_index(obj)
    # . <'pandas.TimedeltaIndex'>
    if dtype is typeref.TIMEDELTAINDEX:
        return _serialize_timedelta_index(obj)
    # . <'cytimes.pddt'>
    if dtype is typeref.PDDT:
        return _serialize_series(obj.dt)

    ##### Subclass Types #####
    return _serialize_subclass(obj, dtype)


@cython.cfunc
@cython.inline(True)
def _serialize_subclass(obj: object, dtype: type) -> str:
    """(cfunc) Serialize subclass of Python object to `<'str'>`.

    The last resort for serializing object into strings,
    focus on the subclasses of the native Python types.

    If this function failed to serialize the object,
    error will be raised.
    """
    ##### Subclass Types #####
    # Basic Types
    # . subclass of `<'float'>`
    if isinstance(obj, float):
        return _serialize_float(obj)
    # . subclass of `<'int'>`
    if isinstance(obj, int):
        return _serialize_int(obj)
    # . subclass of `<'bool'>`
    if isinstance(obj, bool):
        return _serialize_bool(obj)

    # Date&Time Types
    # . subclass of `<'datetime.datetime'>`
    if isinstance(obj, datetime.datetime):
        return _serialize_datetime(obj)
    # . subclass of `<'datetime.date'>`
    if isinstance(obj, datetime.date):
        return _serialize_date(obj)
    # . subclass of `<'datetime.time'>`
    if isinstance(obj, datetime.time):
        return _serialize_time(obj)
    # . subclass of `<'datetime.timedelta'>`
    if isinstance(obj, datetime.timedelta):
        return _serialize_timedelta(obj)
    # . subclass of `<'cytimes.pydt'>`
    if isinstance(obj, typeref.PYDT):
        return _serialize_datetime(obj.dt)

    # Numeric Types
    # . subclass of `<'decimal.Decimal'>`
    if isinstance(obj, typeref.DECIMAL):
        return _serialize_decimal(obj)
    # . subclass of `<'complex'>`
    if isinstance(obj, complex):
        return _serialize_complex(obj)

    # Mapping Types
    # . subclass of `<'dict'>`
    if isinstance(obj, dict):
        return _serialize_dict(obj)

    # Sequence Types
    # . subclass of `<'list'>`
    if isinstance(obj, list):
        return _serialize_list(obj)
    # . subclass of `<'tuple'>`
    if isinstance(obj, tuple):
        return _serialize_tuple(obj)
    # . subclass of `<'set'>`
    if isinstance(obj, set):
        return _serialize_set(obj)
    # . subclass of `<'frozenset'>`
    if isinstance(obj, frozenset):
        return _serialize_frozenset(obj)

    # Invalid Data Type
    raise errors.SerializeTypeError(
        "<'Serializor'>\nFailed to serialize %s: unsupported data type." % dtype
    )


@cython.ccall
def serialize(obj: object) -> str:
    """Serialize Python object to `<'str'>`.

    Support most of the common Python types and some
    less common types, such as: complex, Decimal, numpy,
    pandas, etc.

    The serialized result should only be deserialized by
    the `deserialize()` function in this package, which will
    re-create the Python object from the serialized string.
    """
    try:
        return _serialize_common_type(obj)
    except errors.SerializeError:
        raise
    except Exception as err:
        raise errors.SerializeError(
            "<'Serializor'>\nFailed to serialize: %s.\nError: %s" % (type(obj), err)
        ) from err
