# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False


# Cython imports
import cython
from cython.cimports import numpy as np  # type: ignore
from cython.cimports.cpython import datetime  # type: ignore
from cython.cimports.cpython.list import PyList_AsTuple as list_to_tuple  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_GET_LENGTH as str_len  # type: ignore
from cython.cimports.cpython.unicode import PyUnicode_FromOrdinal as str_fr_ucs4  # type: ignore
from cython.cimports.cpython.complex import PyComplex_FromDoubles as gen_complex  # type: ignore
from cython.cimports.serializor import prefix, serialize, typeref  # type: ignore

np.import_array()
datetime.import_datetime()

# Python imports
from typing import Callable
import numpy as np, datetime
from orjson import loads
from serializor import prefix, serialize, typeref, errors

# Constants -------------------------------------------------------------------------
# . characters
CHAR_QUOTE: cython.Py_UCS4 = 34  # `"`
CHAR_COMMA: cython.Py_UCS4 = 44  # `,`
CHAR_ONE: cython.Py_UCS4 = 49  # `1`
CHAR_BACKSLASH: cython.Py_UCS4 = 92  # `\`
CHAR_PIPE: cython.Py_UCS4 = 124  # `|`
CHAR_OPEN_BRACKET: cython.Py_UCS4 = 91  # `[`
CHAR_CLOSE_BRACKET: cython.Py_UCS4 = 93  # `]`
# . functions
FN_ORJSON_LOADS: Callable = loads
FN_NUMPY_EMPTY: Callable = np.empty


# Orjson loads ----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _orjson_loads(data: str) -> object:
    """(cfunc) Deserialize JSON token using 'orjson (module)' to Python `<'object'>`."""
    return FN_ORJSON_LOADS(data)


# Basic Types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_str(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the JSON token of the 'data' to `<'str'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...s5|Hello...'
        # Start position: ...{s}5|Hello...
        return "Hello"
    """
    # Parse string unicode length: s{5}|...
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_data_separator(data, idx, pos[1])  # type: ignore
    uni_l: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore

    # Deserialize string: s5|{Hello}
    idx = loc + 1  # skip data separator '|'
    loc = idx + uni_l  # end of the string
    pos[0] = loc + 1  # update position & skip (possible) separator ','
    return slice_to_unicode(data, idx, loc)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_float(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the float token of the 'data' to `<'float'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...f3.14...'
        # Start idx: ...{f}3.14...
        # Ended loc: ...f3.1{4}...
        return 3.14
    """
    return slice_to_float(data, idx + 1, loc)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_int(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the integer token of the 'data' to `<'int'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...i1024...'
        # Start idx: ...{i}1024...
        # Ended loc: ...i102{4}...
        return 1024
    """
    return slice_to_int(data, idx + 1, loc)  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_bool(data: str, idx: cython.Py_ssize_t) -> object:
    """(cfunc) Deserialize the boolean token of the 'data' to `<'bool'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...o1...'
        # Start idx: ...{o}1...
        # Ended loc: ...o{1}...
        return True
    """
    return read_char(data, idx + 1) == CHAR_ONE  # type: ignore


# Date&Time Types -------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_datetime(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the datetime token of the 'data' to `<'datetime.datetime'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...z2021-01-01T00:00:00...'
        # Start idx: ...{z}2021-01-01T00:00:00...
        # Ended loc: ...z2021-01-01T00:00:0{0}...
        return datetime.datetime(2021, 1, 1, 0, 0, 0)
    """
    return datetime.datetime.fromisoformat(slice_to_unicode(data, idx + 1, loc))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_date(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the date section of the 'data' to `<'datetime.date'>`.

    :param data `<'str'>`: The serialized string that contains the date.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...d2021-01-01...'
        # Start idx: ...{d}2021-01-01...
        # Ended loc: ...d2021-01-0{1}...
        return datetime.date(2021, 1, 1)
    """
    return datetime.date.fromisoformat(slice_to_unicode(data, idx + 1, loc))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_time(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the time token of the 'data' to `<'datetime.time'>`.

    :param data `<'str'>`: The serialized string that contains the date.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...t00:00:00.000001...'
        # Start idx: ...{t}00:00:00.000001...
        # Ended loc: ...t00:00:00.00000{1}...
        return datetime.time(0, 0, 0, 1)
    """
    return datetime.time.fromisoformat(slice_to_unicode(data, idx + 1, loc))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_timedelta(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the timedelta token of the 'data' to `<'datetime.timedelta'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...l1|2|3...'
        # Start idx: ...{l}12|3...
        # Ended loc: ...l1|2|{3}...
        return datetime.timedelta(1, 2, 3)
    """
    # Parse days: {1}|2|3
    idx += 1  # skip identifier
    sep: cython.Py_ssize_t = find_data_separator(data, idx, loc)  # type: ignore
    days = slice_to_int(data, idx, sep)  # type: ignore
    idx = sep + 1
    # Parse seconds: 1|{2}|3
    sep: cython.Py_ssize_t = find_data_separator(data, idx, loc)  # type: ignore
    seconds = slice_to_int(data, idx, sep)  # type: ignore
    # Parse microseconds: 1|2|{3}
    microseconds = slice_to_int(data, sep + 1, loc)  # type: ignore
    # Generate timedelta
    return datetime.timedelta_new(days, seconds, microseconds)


# Numeric Types ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_decimal(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the decimal token of the 'data' to `<'decimal.Decimal'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...e3.14...'
        # Start idx: ...{e}3.14...
        # Ended loc: ...e3.1{4}...
        return decimal.Decimal("3.14")
    """
    return typeref.DECIMAL(slice_to_unicode(data, idx + 1, loc))  # type: ignore


@cython.cfunc
@cython.inline(True)
def _deserialize_complex(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the complex token of the 'data' to `<'complex'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...c1.0|1.0...'
        # Start idx: ...{c}1.0|1.0...
        # Ended loc: ...c1.0|1.{0}...
        return complex(1.0, 1.0)
    """
    idx += 1  # skip identifier
    sep: cython.Py_ssize_t = find_data_separator(data, idx, loc)  # type: ignore
    real = slice_to_float(data, idx, sep)  # type: ignore
    imag = slice_to_float(data, sep + 1, loc)  # type: ignore
    return gen_complex(real, imag)


# Bytes Types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_bytes(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the bytes token of the 'data' to `<'bytes'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...b5|Hello...'
        # Start position: ...{b}5|Hello...
        return b"Hello"
    """
    # Parse bytes string unicode length: b{5}|...
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_data_separator(data, idx, pos[1])  # type: ignore
    uni_l: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore

    # Deserialize bytes string: b5|{Hello}
    idx = loc + 1  # skip data separator '|'
    loc = idx + uni_l  # end of the string
    pos[0] = loc + 1  # update position & skip (possible) separator ','
    return slice_to_bytes(data, idx, loc)  # type: ignore


# Numpy Types -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_datetime64(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the datetime64 token of the 'data' to `<'numpy.datetime64'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...Mus3661000001...'
        # Start idx: ...{M}us3661000001...
        # Ended loc: ...Mus366100000{1}...
        return numpy.datetime64(3661000001, 'us')
    """
    # If the 3rd character is ASCII digit [0-9]
    # unit is the 2nd character: M{s}1024.
    if 48 <= read_char(data, idx + 2) <= 57:  # type: ignore
        unit = str_fr_ucs4(read_char(data, idx + 1))  # type: ignore
        val = slice_to_int(data, idx + 2, loc)  # type: ignore
    # Otherwise, unit is the 2nd & 3rd characters: M{us}1024.
    else:
        sep: cython.Py_ssize_t = idx + 3
        unit = slice_to_unicode(data, idx + 1, sep)  # type: ignore
        val = slice_to_int(data, sep, loc)  # type: ignore
    # Generate datetime64
    return typeref.NP_DATETIME64(val, unit)


@cython.cfunc
@cython.inline(True)
def _deserialize_timedelta64(
    data: str,
    idx: cython.Py_ssize_t,
    loc: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize the timedelta64 token of the 'data' to `<'numpy.timedelta64'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param idx `<'Py_ssize_t'>`: The start position of the token.
    :param loc `<'Py_ssize_t'>`: The ended position of the token.

    ### Example:
    >>> data = '...mus1024...'
        # Start idx: ...{m}us1024...
        # Ended loc: ...mus102{4}...
        return numpy.timedelta64(1024, 'us')
    """
    # If the 3rd character is ASCII digit [0-9]
    # unit is the 2nd character: m{s}1024.
    if 48 <= read_char(data, idx + 2) <= 57:  # type: ignore
        unit = str_fr_ucs4(read_char(data, idx + 1))  # type: ignore
        val = slice_to_int(data, idx + 2, loc)  # type: ignore
    # Otherwise, unit is the 2nd & 3rd characters: m{us}1024.
    else:
        sep: cython.Py_ssize_t = idx + 3
        unit = slice_to_unicode(data, idx + 1, sep)  # type: ignore
        val = slice_to_int(data, sep, loc)  # type: ignore
    # Generate timedelta64
    return typeref.NP_TIMEDELTA64(val, unit)


# Mapping Types ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_dict(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the dictionary token of the 'data' to `<'dict'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...D13["0","1","2"][f0.1,f1.1,f2.2,]...'
        # Start position: ...{D}13["0","1","2"][f0.1,f1.1,f2.2,]...
        return {"0": 0.1, "1": 1.1, "2": 2.2}
    """
    # Parse dict keys json array unicode length: D{13}...
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, pos[1])  # type: ignore
    uni_l: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    if uni_l == 0:  # empty dict: ...D0[]...
        pos[0] = loc + 2  # skip '[]'.
        return {}  # exit
    idx = loc  # skip 'D13' to '['
    loc += uni_l  # end of keys ']': 'D13["0","1","2"{]}'.

    # Deserialize dict keys: D13{["0","1","2"]}...
    keys: list = _orjson_loads(slice_to_unicode(data, idx, loc))  # type: ignore

    # Deserialize dict values: D13["0","1","2"]{[f1.1,f1.1,f2.2,]}
    res: dict = {}
    pos[0] = loc + 1  # skip '[' to the 1st identifier: 'f'
    for key in keys:
        res[key] = _deserialize_item(data, pos)
    pos[0] += 2  # skip the ending ']' and (possible) seperator ','.
    return res


# Sequence Types --------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_list(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the list token of the 'data' to `<'list'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...L4[i1,f3.14,o1,n,]...'
        # Start position: ...{L}4[i1,f3.14,o1,n,]...
        return [1, 3.14, True, None]
    """
    # Parse list size: L{4}...
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, pos[1])  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore

    # Deserialize
    res = []
    pos[0] = loc + 1  # skip '[' to the 1st identifier: 'i'
    for _ in range(size):
        res.append(_deserialize_item(data, pos))
    pos[0] += 2  # skip the ending ']' and (possible) seperator ','.
    return res


@cython.cfunc
@cython.inline(True)
def _deserialize_tuple(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the tuple token of the 'data' to `<'tuple'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...T4[i1,f3.14,o1,n,]...'
        # Start position: ...{T}4[i1,f3.14,o1,n,]...
        return (1, 3.14, True, None)
    """
    return list_to_tuple(_deserialize_list(data, pos))


@cython.cfunc
@cython.inline(True)
def _deserialize_set(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the set token of the 'data' to `<'set'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...E4[i1,f3.14,o1,n,]...'
        # Start position: ...{E}4[i1,f3.14,o1,n,]...
        return {1, 3.14, True, None}
    """
    # Parse set size: E{4}...
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, pos[1])  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore

    # Deserialize
    res: set = set()
    pos[0] = loc + 1  # skip '[' to the 1st identifier: 'i'
    for _ in range(size):
        res.add(_deserialize_item(data, pos))
    pos[0] += 2  # skip the ending ']' and (possible) seperator ','.
    return res


# Numpy ndarray ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_ndarray(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize the numpy.ndarray token of the 'data' to `<'numpy.ndarray'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.
    :param item `<'bool'>`: Whether the deserialization is for an item of the object.

    ### Example:
    >>> data = '...Ni1|5[0,1,2,3,4]...'
        # Start position: ...{N}i1|5[0,1,2,3,4]...
        return numpy.array([0, 1, 2, 3, 4], dtype=np.int64)
    """
    # Get ndarray dtype: N{i}1|5[0,1,2,3,4]
    dtype: cython.Py_UCS4 = read_char(data, pos[0] + 1)  # type: ignore

    # Deserialize ndarray
    # . ndarray[object]
    if dtype == prefix.NDARRAY_DTYPE_OBJECT_ID:
        arr = _deserialize_ndarray_object(data, pos)
    # . ndarray[float]
    elif dtype == prefix.NDARRAY_DTYPE_FLOAT_ID:
        arr = _deserialize_ndarray_common(data, pos, item, np.NPY_TYPES.NPY_FLOAT64)
    # . ndarray[int]
    elif dtype == prefix.NDARRAY_DTYPE_INT_ID:
        arr = _deserialize_ndarray_common(data, pos, item, np.NPY_TYPES.NPY_INT64)
    # . ndarray[uint]
    elif dtype == prefix.NDARRAY_DTYPE_UINT_ID:
        arr = _deserialize_ndarray_common(data, pos, item, np.NPY_TYPES.NPY_UINT64)
    # . ndarray[bool]
    elif dtype == prefix.NDARRAY_DTYPE_BOOL_ID:
        arr = _deserialize_ndarray_common(data, pos, item, np.NPY_TYPES.NPY_BOOL)
    # . ndarray[datetime64]
    elif dtype == prefix.NDARRAY_DTYPE_DT64_ID:
        arr = _deserialize_ndarray_dt64td64(data, pos, item, True)
    # . ndarray[timedelta64]
    elif dtype == prefix.NDARRAY_DTYPE_TD64_ID:
        arr = _deserialize_ndarray_dt64td64(data, pos, item, False)
    # . ndarray[complex]
    elif dtype == prefix.NDARRAY_DTYPE_COMPLEX_ID:
        arr = _deserialize_ndarray_complex(data, pos, item)
    # . ndarray[bytes]
    elif dtype == prefix.NDARRAY_DTYPE_BYTES_ID:
        arr = np.PyArray_Cast(
            _deserialize_ndarray_bytes(data, pos, item), np.NPY_TYPES.NPY_STRING
        )
    # . ndarray[str]
    elif dtype == prefix.NDARRAY_DTYPE_UNICODE_ID:
        arr = np.PyArray_Cast(
            _deserialize_ndarray_common(data, pos, item, np.NPY_TYPES.NPY_OBJECT),
            np.NPY_TYPES.NPY_UNICODE,
        )
    # . unrecognized dtype
    else:
        raise errors.DeserializeValueError(
            "<Serializor> Failed to deserialize 'data': "
            "unsupported <'numpy.ndarray'> dtype '%s'." % str_fr_ucs4(dtype)
        )

    # Return ndarray
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_ndarray_common(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
    dtype: cython.int,
) -> object:
    """(cfunc) Deserialize numpy.ndarray token
    of the 'data' to `<'ndarray[str/float/int/uint/bool]'>`.

    This function is specifically for ndarray with dtype of:
    "U" (str), "f" (float), "i" (int), "u" (uint) and "b" (bool).

    ### Example:
    >>> data = '...NU1|3["1","2","3"]...'
        return numpy.array(["1", "2", "3"], dtype="U")

    >>> data = '...Nf1|3[1.1,2.2,3.3]...'
        return numpy.array([1.1, 2.2, 3.3], dtype=np.float64)

    >>> data = '...Ni1|3[1,2,3]...'
        return numpy.array([1, 2, 3], dtype=np.int64)

    >>> data = '...Nu1|3[1,2,3]...'
        return numpy.array([1, 2, 3], dtype=np.uint64)

    >>> data = '...Nb1|3[1,0,1]...'
        return numpy.array([True, False, True], dtype=np.bool_)
    """
    # Parse ndarray shape
    eof: cython.Py_ssize_t = pos[1]
    shape = _parse_ndarray_shape(data, pos[0] + 2, eof)
    ndim: cython.Py_ssize_t = shape.ndim
    idx: cython.Py_ssize_t = shape.loc  # 'Nf1|0{[}....]'
    loc: cython.Py_ssize_t
    arr: np.ndarray

    # Deserialize: 1-dimensional
    if ndim == 1:
        dim1: np.npy_intp[1] = [shape.i]
        arr = np.PyArray_EMPTY(1, dim1, dtype, 0)
        # . empty ndarray: 'Nf1|0[]'
        if shape.i == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        elif dtype == np.NPY_TYPES.NPY_OBJECT:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            serialize.ndarray_setitem_1d(arr, i, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 2-dimensional
    if ndim == 2:
        dim2: np.npy_intp[2] = [shape.i, shape.j]
        arr = np.PyArray_EMPTY(2, dim2, dtype, 0)
        # . empty ndarray: 'Nf2|i|0[]'
        if shape.j == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        elif dtype == np.NPY_TYPES.NPY_OBJECT:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                serialize.ndarray_setitem_2d(arr, i, j, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 3-dimensional
    if ndim == 3:
        dim3: np.npy_intp[3] = [shape.i, shape.j, shape.k]
        arr = np.PyArray_EMPTY(3, dim3, dtype, 0)
        # . empty ndarray: 'Nf3|i|j|0[]'
        if shape.k == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        elif dtype == np.NPY_TYPES.NPY_OBJECT:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    serialize.ndarray_setitem_3d(arr, i, j, k, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 4-dimensional
    if ndim == 4:
        dim4: np.npy_intp[4] = [shape.i, shape.j, shape.k, shape.l]
        arr = np.PyArray_EMPTY(4, dim4, dtype, 0)
        # . empty ndarray: 'Nf4|i|j|k|0[]'
        if shape.l == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        elif dtype == np.NPY_TYPES.NPY_OBJECT:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    for l in range(shape.l):
                        serialize.ndarray_setitem_4d(arr, i, j, k, l, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Invalid dimension
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unsupported <'numpy.ndarray'> dimension [%d]." % ndim
    )


@cython.cfunc
@cython.inline(True)
def _deserialize_ndarray_dt64td64(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
    dt64: cython.bint,
) -> object:
    """(cfunc) Deserialize numpy.ndarray token
    of the 'data' to `<'ndarray[datetime64/timedelta64]'>`.

    This function is specifically for ndarray with
    dtype of: "M" (datetime64) and "m" (timedelta64).

    ### Example:
    >>> data = '...NMs1|3[1672531200,1672617600,1672704000]...'
        return numpy.array(["2023-01-01", "2023-01-02", "2023-01-03"], dtype="datetime64[s]")

    >>> data = '...Nms1|3[1,2,3]...'
        return numpy.array([1, 2, 3], dtype="timedelta64[s]")
    """
    # If the 4th character is ASCII digit [0-9]
    # unit is the 3rd character: NM[s]1
    idx: cython.Py_ssize_t = pos[0] + 2  # skip 'NM'
    if 48 <= read_char(data, idx + 1) <= 57:  # type: ignore
        unit: str = str_fr_ucs4(read_char(data, idx))  # type: ignore
        idx += 1
    # Otherwise, unit is the 3rd & 4th characters: NM[us]1
    else:
        unit: str = slice_to_unicode(data, idx, idx + 2)  # type: ignore
        idx += 2
    dtype = "datetime64[%s]" % unit if dt64 else "timedelta64[%s]" % unit

    # Parse ndarray shape
    eof: cython.Py_ssize_t = pos[1]
    shape = _parse_ndarray_shape(data, idx, eof)
    ndim: cython.Py_ssize_t = shape.ndim
    idx = shape.loc  # 'NMs1|0{[}....]'
    loc: cython.Py_ssize_t
    arr: np.ndarray

    # Deserialize: 1-dimensional
    if ndim == 1:
        arr = FN_NUMPY_EMPTY(shape.i, dtype=dtype)
        # . empty ndarray: 'NMs1|0[]'
        if shape.i == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            serialize.ndarray_setitem_1d(arr, i, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 2-dimensional
    if ndim == 2:
        arr = FN_NUMPY_EMPTY([shape.i, shape.j], dtype=dtype)
        # . empty ndarray: 'NMs2|j|0[]'
        if shape.j == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                serialize.ndarray_setitem_2d(arr, i, j, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 3-dimensional
    if ndim == 3:
        arr = FN_NUMPY_EMPTY([shape.i, shape.j, shape.k], dtype=dtype)
        # . empty ndarray: 'NMs3|j|k|0[]'
        if shape.k == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    serialize.ndarray_setitem_3d(arr, i, j, k, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 4-dimensional
    if ndim == 4:
        arr = FN_NUMPY_EMPTY([shape.i, shape.j, shape.k, shape.l], dtype=dtype)
        # . empty ndarray: 'NMs4|j|k|l|0[]'
        if shape.l == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    for l in range(shape.l):
                        serialize.ndarray_setitem_4d(arr, i, j, k, l, next(items))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Invalid dimension
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unsupported <'numpy.ndarray'> dimension [%d]." % ndim
    )


@cython.cfunc
@cython.inline(True)
def _deserialize_ndarray_complex(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize numpy.ndarray token
    of the 'data' to `<'ndarray[complex]'>`.

    This function is specifically for ndarray
    with dtype of: "c" (complex).

    ### Example:
    >>> data = '...Nc1|3[1.0,1.0,2.0,2.0,3.0,3.0]...'
        return numpy.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    """
    # Parse ndarray shape
    eof: cython.Py_ssize_t = pos[1]
    shape = _parse_ndarray_shape(data, pos[0] + 2, eof)
    ndim: cython.Py_ssize_t = shape.ndim
    idx: cython.Py_ssize_t = shape.loc  # 'Nc1|0{[}....]'
    loc: cython.Py_ssize_t
    arr: np.ndarray

    # Deserialize: 1-dimensional
    if ndim == 1:
        dim1: np.npy_intp[1] = [shape.i]
        arr = np.PyArray_EMPTY(1, dim1, np.NPY_TYPES.NPY_COMPLEX128, 0)
        # . empty ndarray: 'Nc1|0[]'
        if shape.i == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            serialize.ndarray_setitem_1d(arr, i, gen_complex(next(items), next(items)))
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 2-dimensional
    if ndim == 2:
        dim2: np.npy_intp[2] = [shape.i, shape.j]
        arr = np.PyArray_EMPTY(2, dim2, np.NPY_TYPES.NPY_COMPLEX128, 0)
        # . empty ndarray: 'Nc2|i|0[]'
        if shape.j == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                serialize.ndarray_setitem_2d(
                    arr, i, j, gen_complex(next(items), next(items))
                )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 3-dimensional
    if ndim == 3:
        dim3: np.npy_intp[3] = [shape.i, shape.j, shape.k]
        arr = np.PyArray_EMPTY(3, dim3, np.NPY_TYPES.NPY_COMPLEX128, 0)
        # . empty ndarray: 'Nc3|i|j|0[]'
        if shape.k == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    serialize.ndarray_setitem_3d(
                        arr, i, j, k, gen_complex(next(items), next(items))
                    )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 4-dimensional
    if ndim == 4:
        dim4: np.npy_intp[4] = [shape.i, shape.j, shape.k, shape.l]
        arr = np.PyArray_EMPTY(4, dim4, np.NPY_TYPES.NPY_COMPLEX128, 0)
        # . empty ndarray: 'Nc4|i|j|k|0[]'
        if shape.l == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    for l in range(shape.l):
                        serialize.ndarray_setitem_4d(
                            arr, i, j, k, l, gen_complex(next(items), next(items))
                        )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Invalid dimension
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unsupported <'numpy.ndarray'> dimension [%d]." % ndim
    )


@cython.cfunc
@cython.inline(True)
def _deserialize_ndarray_bytes(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize numpy.ndarray token
    of the 'data' to `<'ndarray[bytes]'>`.

    This function is specifically for ndarray
    with dtype of: "S" (bytes string).

    ### Example:
    >>> data = '...NS1|3["1","2","3"]...'
        return numpy.array([b"1", b"2", b"3"], dtype="S")
    """
    # Parse ndarray shape
    eof: cython.Py_ssize_t = pos[1]
    shape = _parse_ndarray_shape(data, pos[0] + 2, eof)
    ndim: cython.Py_ssize_t = shape.ndim
    idx: cython.Py_ssize_t = shape.loc  # 'NS1|0{[}....]'
    loc: cython.Py_ssize_t
    arr: np.ndarray

    # Deserialize: 1-dimensional
    if ndim == 1:
        dim1: np.npy_intp[1] = [shape.i]
        arr = np.PyArray_EMPTY(1, dim1, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NS1|0[]'
        if shape.i == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            serialize.ndarray_setitem_1d(
                arr, i, serialize.bytes_encode_utf8(next(items))
            )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 2-dimensional
    if ndim == 2:
        dim2: np.npy_intp[2] = [shape.i, shape.j]
        arr = np.PyArray_EMPTY(2, dim2, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NS2|i|0[]'
        if shape.j == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                serialize.ndarray_setitem_2d(
                    arr, i, j, serialize.bytes_encode_utf8(next(items))
                )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 3-dimensional
    if ndim == 3:
        dim3: np.npy_intp[3] = [shape.i, shape.j, shape.k]
        arr = np.PyArray_EMPTY(3, dim3, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NS3|i|j|0[]'
        if shape.k == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    serialize.ndarray_setitem_3d(
                        arr, i, j, k, serialize.bytes_encode_utf8(next(items))
                    )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Deserialize: 4-dimensional
    if ndim == 4:
        dim4: np.npy_intp[4] = [shape.i, shape.j, shape.k, shape.l]
        arr = np.PyArray_EMPTY(4, dim4, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NS4|i|j|k|0[]'
        if shape.l == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . find ndarray end
        if not item:
            loc = eof
        else:
            loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
        # . deserialize: ndarray
        items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    for l in range(shape.l):
                        serialize.ndarray_setitem_4d(
                            arr, i, j, k, l, serialize.bytes_encode_utf8(next(items))
                        )
        pos[0] = loc + 1  # update position & skip ','
        return arr

    # Invalid dimension
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unsupported <'numpy.ndarray'> dimension [%d]." % ndim
    )


@cython.cfunc
@cython.inline(True)
def _deserialize_ndarray_object(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize numpy.ndarray token
    of the 'data' to `<'ndarray[object]'>`.

    This function is specifically for ndarray
    with dtype of: "O" (object).

    ### Example:
    >>> '...NO1|5[i1,f1.234,o1,"abc",c1.0|1.0,]...'
        return numpy.array([1, 1.234, True, "abc", 1 + 1j], dtype="O")
    """
    # Parse ndarray shape
    shape = _parse_ndarray_shape(data, pos[0] + 2, pos[1])
    ndim: cython.Py_ssize_t = shape.ndim
    idx: cython.Py_ssize_t = shape.loc  # 'NO1|0{[}....]'
    arr: np.ndarray

    # Deserialize: 1-dimensional
    if ndim == 1:
        dim1: np.npy_intp[1] = [shape.i]
        arr = np.PyArray_EMPTY(1, dim1, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NO1|0[]'
        if shape.i == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . deserialize: ndarray
        pos[0] = idx + 1  # skip '[' to the 1st identifier: 'i'
        for i in range(shape.i):
            serialize.ndarray_setitem_1d(arr, i, _deserialize_item(data, pos))
        pos[0] += 2  # skip the ending ']' & ','
        return arr

    # Deserialize: 2-dimensional
    if ndim == 2:
        dim2: np.npy_intp[2] = [shape.i, shape.j]
        arr = np.PyArray_EMPTY(2, dim2, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NO2|i|0[]'
        if shape.j == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . deserialize: ndarray
        pos[0] = idx + 1  # skip '[' to the 1st identifier: 'i'
        for i in range(shape.i):
            for j in range(shape.j):
                serialize.ndarray_setitem_2d(arr, i, j, _deserialize_item(data, pos))
        pos[0] += 2  # skip the ending ']' & ','
        return arr

    # Deserialize: 3-dimensional
    if ndim == 3:
        dim3: np.npy_intp[3] = [shape.i, shape.j, shape.k]
        arr = np.PyArray_EMPTY(3, dim3, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NO3|i|j|0[]'
        if shape.k == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . deserialize: ndarray
        pos[0] = idx + 1  # skip '[' to the 1st identifier: 'i'
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    serialize.ndarray_setitem_3d(
                        arr, i, j, k, _deserialize_item(data, pos)
                    )
        pos[0] += 2  # skip the ending ']' & ','
        return arr

    # Deserialize: 4-dimensional
    if ndim == 4:
        dim4: np.npy_intp[4] = [shape.i, shape.j, shape.k, shape.l]
        arr = np.PyArray_EMPTY(4, dim4, np.NPY_TYPES.NPY_OBJECT, 0)
        # . empty ndarray: 'NO4|i|j|k|0[]'
        if shape.l == 0:
            pos[0] = idx + 3  # skip '[]' & ','
            return arr
        # . deserialize: ndarray
        pos[0] = idx + 1  # skip '[' to the 1st identifier: 'i'
        for i in range(shape.i):
            for j in range(shape.j):
                for k in range(shape.k):
                    for l in range(shape.l):
                        serialize.ndarray_setitem_4d(
                            arr, i, j, k, l, _deserialize_item(data, pos)
                        )
        pos[0] += 2  # skip the ending ']' & ','
        return arr

    # Invalid dimension
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unsupported <'numpy.ndarray'> dimension [%d]." % ndim
    )


@cython.cfunc
@cython.inline(True)
def _parse_ndarray_shape(
    data: str,
    idx: cython.Py_ssize_t,
    eof: cython.Py_ssize_t,
) -> shape:  # type: ignore
    """(cfunc) Parse numpy.ndarray `<'shape'>`.

    :param data `<'str'>`: The serialized data that contains the ndarray token.
    :param pos_s `<'Py_ssize_t'>`: The start position of the ndarray shape.

    ### Example:
    >>> data = '...Ni2|2|2|2|3[1,2,3,...]...'
        # Start position '2': ...Ni{2}|2|2|2|3[1,2,3,...]...
    """
    # Parse dimensions: 'Ni{2}|'
    loc: cython.Py_ssize_t = find_data_separator(data, idx, eof)  # type: ignore
    ndim: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc + 1  # skip '|'

    # Parse 1-dim shape: 'Ni2|{2}|
    if ndim == 1:  # 'Ni1|{3}['
        loc = find_open_bracket(data, idx, eof)  # type: ignore
        i = slice_to_int(data, idx, loc)  # type: ignore
        return shape(ndim, i, 0, 0, 0, loc)  # type: ignore
    loc = find_data_separator(data, idx, eof)  # type: ignore
    i = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc + 1  # skip '|'

    # Parse 2-dim shape: 'Ni2|2|{2}|
    if ndim == 2:  # 'Ni2|2|{3}['
        loc = find_open_bracket(data, idx, eof)  # type: ignore
        j = slice_to_int(data, idx, loc)  # type: ignore
        return shape(ndim, i, j, 0, 0, loc)  # type: ignore
    loc = find_data_separator(data, idx, eof)  # type: ignore
    j = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc + 1  # skip '|'

    # Parse 3-dim shape: 'Ni2|2|2|{2}|
    if ndim == 3:  # 'Ni2|2|2|{3}['
        loc = find_open_bracket(data, idx, eof)  # type: ignore
        k = slice_to_int(data, idx, loc)  # type: ignore
        return shape(ndim, i, j, k, 0, loc)  # type: ignore
    loc = find_data_separator(data, idx, eof)  # type: ignore
    k = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc + 1  # skip '|'

    # Parse 4-dim shape: 'Ni2|2|2|2|{3}['
    loc = find_open_bracket(data, idx, eof)  # type: ignore
    l = slice_to_int(data, idx, loc)  # type: ignore
    return shape(ndim, i, j, k, l, loc)  # type: ignore


# Pandas Series ---------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_series(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize the pandas.Series token
    of the 'data' to `<'pandas.Series'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.
    :param item `<'bool'>`: Whether the deserialization is for an item of the object.

    ### Example:
    >>> data = '...Ii5[0,1,2,3,4]...'
        # Start position: ...{I}i1|5[0,1,2,3,4]...
        return pandas.Series([0, 1, 2, 3, 4], dtype=np.int64)
    """
    # Get Series dtype: I{i}5[0,1,2,3,4]
    dtype: cython.Py_UCS4 = read_char(data, pos[0] + 1)  # type: ignore

    # Deserialize Series
    # . Series[object]
    if dtype == prefix.NDARRAY_DTYPE_OBJECT_ID:
        arr = _deserialize_series_object(data, pos)
    # . Series[float]
    elif dtype == prefix.NDARRAY_DTYPE_FLOAT_ID:
        arr = _deserialize_series_common(data, pos, item, np.NPY_TYPES.NPY_FLOAT64)
    # . Series[int]
    elif dtype == prefix.NDARRAY_DTYPE_INT_ID:
        arr = _deserialize_series_common(data, pos, item, np.NPY_TYPES.NPY_INT64)
    # . Series[uint]
    elif dtype == prefix.NDARRAY_DTYPE_UINT_ID:
        arr = _deserialize_series_common(data, pos, item, np.NPY_TYPES.NPY_UINT64)
    # . Series[bool]
    elif dtype == prefix.NDARRAY_DTYPE_BOOL_ID:
        arr = _deserialize_series_common(data, pos, item, np.NPY_TYPES.NPY_BOOL)
    # . Series[datetime64]
    elif dtype == prefix.NDARRAY_DTYPE_DT64_ID:
        arr = _deserialize_series_dt64td64(data, pos, item, True)
    # . Series[timedelta64]
    elif dtype == prefix.NDARRAY_DTYPE_TD64_ID:
        arr = _deserialize_series_dt64td64(data, pos, item, False)
    # . Series[complex]
    elif dtype == prefix.NDARRAY_DTYPE_COMPLEX_ID:
        arr = _deserialize_series_complex(data, pos, item)
    # . Series[bytes]
    elif dtype == prefix.NDARRAY_DTYPE_BYTES_ID:
        arr = np.PyArray_Cast(
            _deserialize_series_bytes(data, pos, item),
            np.NPY_TYPES.NPY_STRING,
        )
    # . Series[str]
    elif dtype == prefix.NDARRAY_DTYPE_UNICODE_ID:
        arr = np.PyArray_Cast(
            _deserialize_series_common(data, pos, item, np.NPY_TYPES.NPY_OBJECT),
            np.NPY_TYPES.NPY_UNICODE,
        )
    # . unrecognized dtype
    else:
        raise errors.DeserializeValueError(
            "<Serializor> Failed to deserialize 'data': "
            "unknown <'pandas.Series'> dtype '%s'." % str_fr_ucs4(dtype)
        )

    # Generate Series
    return typeref.PD_SERIES(arr, copy=False)


@cython.cfunc
@cython.inline(True)
def _deserialize_series_common(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
    dtype: cython.int,
) -> object:
    """(cfunc) Deserialize pandas.Series token
    of the 'data' to `<'ndarray[str/float/int/uint/bool]'>`.

    This function is specifically for Series with dtype of:
    "U" (str), "f" (float), "i" (int), "u" (uint) and "b" (bool).

    ### Example:
    >>> data = '...If3[1.1,2.2,3.3]...'
        return numpy.array([1.1, 2.2, 3.3], dtype=np.float64)

    >>> data = '...Ii3[1,2,3]...'
        return numpy.array([1, 2, 3], dtype=np.int64)

    >>> data = '...Iu3[1,2,3]...'
        return numpy.array([1, 2, 3], dtype=np.uint64)

    >>> data = '...Ib3[1,0,1]...'
        return numpy.array([True, False, True], dtype=np.bool_)
    """
    # Parse Series size
    idx: cython.Py_ssize_t = pos[0] + 2  # skip 'If'
    eof: cython.Py_ssize_t = pos[1]
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, eof)  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc  # 'If3{[}....]'

    # Deserialize: Series
    dim: np.npy_intp[1] = [size]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, dtype, 0)
    # . empty Series # 'If0[]'
    if size == 0:
        pos[0] = idx + 3  # skip '[]' & ','
        return arr
    #  . find Series end
    if not item:
        loc = eof
    elif dtype == np.NPY_TYPES.NPY_OBJECT:
        loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
    else:
        loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(size):
        serialize.ndarray_setitem_1d(arr, i, next(items))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_series_dt64td64(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
    dt64: cython.bint,
) -> object:
    """(cfunc) Deserialize pandas.Series token
    of the 'data' to `<'ndarray[datetime64/timedelta64]'>`.

    This function is specifically for Series with
    dtype of: "M" (datetime64) and "m" (timedelta64).

    ### Example:
    >>> data = '...IMs3[1672531200,1672617600,1672704000]...'
        return numpy.array([1672531200,1672617600,1672704000], dtype="datetime64[s]")

    >>> data = '...Ims3[1,2,3]...'
        return numpy.array([1, 2, 3], dtype="timedelta64[s]")
    """
    # If the 4th character is ASCII digit [0-9]
    # unit is the 3rd character: IM{s}3
    idx: cython.Py_ssize_t = pos[0] + 2  # skip 'IM'
    if 48 <= read_char(data, idx + 1) <= 57:  # type: ignore
        unit: str = str_fr_ucs4(read_char(data, idx))  # type: ignore
        idx += 1
    # Otherwise, unit is the 3rd & 4th characters: IM{us}3
    else:
        unit: str = slice_to_unicode(data, idx, idx + 2)  # type: ignore
        idx += 2
    dtype = "datetime64[%s]" % unit if dt64 else "timedelta64[%s]" % unit

    # Parse Series size
    eof: cython.Py_ssize_t = pos[1]
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, eof)  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc  # 'IMs3{[}....]'

    # Deserialize: Series
    arr: np.ndarray = FN_NUMPY_EMPTY(size, dtype=dtype)
    # . empty Series # 'IMs0[]'
    if size == 0:
        pos[0] = idx + 3  # skip '[]' & ','
        return arr
    # . find Series end
    if not item:
        loc = eof
    else:
        loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
    # . dserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(size):
        serialize.ndarray_setitem_1d(arr, i, next(items))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_series_complex(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize pandas.Series token
    of the 'data' to `<'ndarray[complex]'>`.

    This function is specifically for Series
    with dtype of: "c" (complex).

    ### Example:
    >>> data = '...Ic3[1.0,1.0,2.0,2.0,3.0,3.0]...'
        return numpy.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=np.complex128)
    """
    # Parse Series size
    idx: cython.Py_ssize_t = pos[0] + 2  # skip 'Ic'
    eof: cython.Py_ssize_t = pos[1]
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, eof)  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc  # 'Ic3{[}....]'

    # Deserialize: Series
    dim: np.npy_intp[1] = [size]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_COMPLEX128, 0)
    # . empty Series: 'Ic0[]'
    if size == 0:
        pos[0] = idx + 3  # skip '[]' & ','
        return arr
    # . find Series end
    if not item:
        loc = eof
    else:
        loc = find_close_bracket(data, idx, eof) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(size):
        serialize.ndarray_setitem_1d(arr, i, gen_complex(next(items), next(items)))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_series_bytes(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize pandas.Series token
    of the 'data' to `<'ndarray[bytes]'>`.

    This function is specifically for Series
    with dtype of: "S" (bytes string).

    ### Example:
    >>> data = '...IS3["1","2","3"]...'
        return numpy.array([b"1", b"2", b"3"], dtype="S")
    """
    # Parse Series size
    idx: cython.Py_ssize_t = pos[0] + 2  # skip 'IS'
    eof: cython.Py_ssize_t = pos[1]
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, eof)  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc  # 'IS3{[}....]'

    # Deserialize: Series
    dim: np.npy_intp[1] = [size]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_OBJECT, 0)
    # . empty Series: 'IS0[]'
    if size == 0:
        pos[0] = idx + 3  # skip '[]' & ','
        return arr
    # . find Series end
    if not item:
        loc = eof
    else:
        loc = find_close_bracketq(data, idx, eof) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(size):
        serialize.ndarray_setitem_1d(arr, i, serialize.bytes_encode_utf8(next(items)))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_series_object(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize pandas.Series token
    of the 'data' to `<'ndarray[object]'>`.

    This function is specifically for Series
    with dtype of: "O" (object).

    ### Example:
    >>> data = '...IO3[i1,f1.234,o1]...'
        return numpy.array([1, 1.234, True], dtype="O")
    """
    # Parse Series size
    idx: cython.Py_ssize_t = pos[0] + 2  # skip 'IO'
    loc: cython.Py_ssize_t = find_open_bracket(data, idx, pos[1])  # type: ignore
    size: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc  # 'IO3{[}....]'

    # Deserialize: Series
    dim: np.npy_intp[1] = [size]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_OBJECT, 0)
    # . empty Series: 'IO0[]'
    if size == 0:
        pos[0] = idx + 3  # skip '[]' & ','
        return arr
    # Deserialize: ndarray
    pos[0] = idx + 1  # skip '[' to the 1st identifier: 'i'
    for i in range(size):
        serialize.ndarray_setitem_1d(arr, i, _deserialize_item(data, pos))
    pos[0] += 2  # skip the ending ']' & ','
    return arr


# Pandas DataFrame ------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_dataframe(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the pandas.DataFrame token
    of the 'data' to `<'pandas.DataFrame'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.

    ### Example:
    >>> data = '...F3|9["i_col","f_col"][i[1,2,3],f[1.1,2.2,3.3],]...'
        # Start position: ...{F}3|9["i_col","f_col"][i[1,2,3],f[1.1,2.2,3.3],]...
        return pandas.DataFrame({"i_col": [1, 2, 3], "f_col": [1.1, 2.2, 3.3]})
    """
    # Parse DataFrame rows: 'F{3}|'
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    eof: cython.Py_ssize_t = pos[1]
    loc: cython.Py_ssize_t = find_data_separator(data, idx, eof)  # type: ignore
    rows: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    idx = loc + 1  # skip '|'

    # Parse column names json array unicode length: 'F3|{9}[...]'
    loc = find_open_bracket(data, idx, eof)  # type: ignore
    uni_l: cython.Py_ssize_t = slice_to_int(data, idx, loc)  # type: ignore
    if uni_l == 0:
        # . true empty DataFrame: 'F3|0[]'
        pos[0] = loc + 3  # skip '[]' & ','
        return typeref.PD_DATAFRAME({}, copy=False)

    # Deserialize column names: F3|9{["i_col","f_col"]}
    idx = loc + uni_l  # end of json array
    cols: list = _orjson_loads(slice_to_unicode(data, loc, idx))  # type: ignore
    pos[0] = idx + 1  # update position
    if rows == 0:
        # . empty DataFrame with columns: 'F0|9["i_col","f_col"]'
        arr_empty = np.array([], dtype="O")
        return typeref.PD_DATAFRAME({col: arr_empty for col in cols}, copy=False)

    # Deserialize: DataFrame
    res: dict = {}
    for col in cols:
        # . get Series dtype: [{i}[1,2,3],f[1.1,2.2,3.3],]
        dtype: cython.Py_UCS4 = read_char(data, pos[0])  # type: ignore
        # . `<'object'>`
        if dtype == prefix.NDARRAY_DTYPE_OBJECT_ID:
            val = _deserialize_dataframe_object(data, pos, rows)
        # . `<'float'>`
        elif dtype == prefix.NDARRAY_DTYPE_FLOAT_ID:
            val = _deserialize_dataframe_common(
                data, pos, rows, np.NPY_TYPES.NPY_FLOAT64
            )
        # . `<'int'>`
        elif dtype == prefix.NDARRAY_DTYPE_INT_ID:
            val = _deserialize_dataframe_common(data, pos, rows, np.NPY_TYPES.NPY_INT64)
        # . `<uint>`
        elif dtype == prefix.NDARRAY_DTYPE_UINT_ID:
            val = _deserialize_dataframe_common(
                data, pos, rows, np.NPY_TYPES.NPY_UINT64
            )
        # . `<'bool'>`
        elif dtype == prefix.NDARRAY_DTYPE_BOOL_ID:
            val = _deserialize_dataframe_common(data, pos, rows, np.NPY_TYPES.NPY_BOOL)
        # . `<datetime64>`
        elif dtype == prefix.NDARRAY_DTYPE_DT64_ID:
            val = _deserialize_dataframe_dt64td64(data, pos, rows, True)
        # . `<timedelta64>`
        elif dtype == prefix.NDARRAY_DTYPE_TD64_ID:
            val = _deserialize_dataframe_dt64td64(data, pos, rows, False)
        # . `<'complex'>`
        elif dtype == prefix.NDARRAY_DTYPE_COMPLEX_ID:
            val = _deserialize_dataframe_complex(data, pos, rows)
        # . `<'bytes'>`
        elif dtype == prefix.NDARRAY_DTYPE_BYTES_ID:
            val = np.PyArray_Cast(
                _deserialize_dataframe_bytes(data, pos, rows),
                np.NPY_TYPES.NPY_STRING,
            )
        # . `<'str'>`
        elif dtype == prefix.NDARRAY_DTYPE_UNICODE_ID:
            val = np.PyArray_Cast(
                _deserialize_dataframe_common(data, pos, rows, np.NPY_TYPES.NPY_OBJECT),
                np.NPY_TYPES.NPY_UNICODE,
            )
        # . unrecognized dtype
        else:
            raise errors.DeserializeValueError(
                "<Serializor> Failed to deserialize 'data': unknown "
                "<'pandas.DataFrame'> column (Series) dtype '%s'." % str_fr_ucs4(dtype)
            )
        # . collect
        res[col] = val

    # Update position
    pos[0] += 2  # skip the ending ']' & ','

    # Generate DataFrame
    return typeref.PD_DATAFRAME(res, copy=False)


@cython.cfunc
@cython.inline(True)
def _deserialize_dataframe_common(
    data: str,
    pos: cython.Py_ssize_t[2],
    rows: cython.Py_ssize_t,
    dtype: cython.int,
) -> object:
    """(cfunc) Deserialize pandas.DataFrame column token
    of the 'data' to `<'ndarray[str/float/int/uint/bool]'>`.

    This function is specifically for DataFrame columns with dtype of:
    "U" (str), "f" (float), "i" (int), "u" (uint) and "b" (bool).
    """
    # Deserialize: Series
    dim: np.npy_intp[1] = [rows]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, dtype, 0)
    # . find Series end
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    if dtype == np.NPY_TYPES.NPY_OBJECT:
        loc: cython.Py_ssize_t = find_close_bracketq(data, idx, pos[1]) + 1  # type: ignore
    else:
        loc: cython.Py_ssize_t = find_close_bracket(data, idx, pos[1]) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(rows):
        serialize.ndarray_setitem_1d(arr, i, next(items))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_dataframe_dt64td64(
    data: str,
    pos: cython.Py_ssize_t[2],
    rows: cython.Py_ssize_t,
    dt64: cython.bint,
) -> object:
    """(cfunc) Deserialize pandas.DataFrame column token
    of the 'data' to `<'ndarray[datetime64/timedelta64]'>`.

    This function is specifically for DataFrame columns with
    dtype of: "M" (datetime64) and 'm" (timedelta64).
    """
    # If the 3rd character is '['
    # unit is the 2nd character: M{s}[...]
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    if read_char(data, idx + 1) == CHAR_OPEN_BRACKET:  # type: ignore
        unit: str = str_fr_ucs4(read_char(data, idx))  # type: ignore
        idx += 1
    # Otherwise, unit is the 2nd & 3rd characters: M{us}[...]
    else:
        unit: str = slice_to_unicode(data, idx, idx + 2)  # type: ignore
        idx += 2
    dtype = "datetime64[%s]" % unit if dt64 else "timedelta64[%s]" % unit

    # Deserialize Series
    arr: np.ndarray = FN_NUMPY_EMPTY(rows, dtype=dtype)
    # . find Series end
    loc: cython.Py_ssize_t = find_close_bracket(data, idx, pos[1]) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(rows):
        serialize.ndarray_setitem_1d(arr, i, next(items))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_dataframe_complex(
    data: str,
    pos: cython.Py_ssize_t[2],
    rows: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize pandas.DataFrame column token
    of the 'data' to `<'ndarray[complex]'>`.

    This function is specifically for DataFrame columns
    with dtype of: "c" (complex).
    """
    # Deserialize: Series
    dim: np.npy_intp[1] = [rows]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_COMPLEX128, 0)
    # . find Series end
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_close_bracket(data, idx, pos[1]) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(rows):
        serialize.ndarray_setitem_1d(arr, i, gen_complex(next(items), next(items)))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_dataframe_bytes(
    data: str,
    pos: cython.Py_ssize_t[2],
    rows: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize pandas.DataFrame column token
    of the 'data' to `<'ndarray[bytes]'>`.

    This function is specifically for DataFrame columns
    with dtype of: "S" (bytes string).
    """
    # Deserialize: Series
    dim: np.npy_intp[1] = [rows]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_OBJECT, 0)
    # . find Series end
    idx: cython.Py_ssize_t = pos[0] + 1  # skip identifier
    loc: cython.Py_ssize_t = find_close_bracketq(data, idx, pos[1]) + 1  # type: ignore
    # . deserialize: ndarray
    items = iter(_orjson_loads(slice_to_unicode(data, idx, loc)))  # type: ignore
    for i in range(rows):
        serialize.ndarray_setitem_1d(arr, i, serialize.bytes_encode_utf8(next(items)))
    pos[0] = loc + 1  # update position & skip ','
    return arr


@cython.cfunc
@cython.inline(True)
def _deserialize_dataframe_object(
    data: str,
    pos: cython.Py_ssize_t[2],
    rows: cython.Py_ssize_t,
) -> object:
    """(cfunc) Deserialize pandas.DataFrame column token
    of the 'data' to `<'ndarray[object]'>`.

    This function is specifically for DataFrame columns
    with dtype of: "O" (object).
    """
    # Deserialize: Series
    dim: np.npy_intp[1] = [rows]
    arr: np.ndarray = np.PyArray_EMPTY(1, dim, np.NPY_TYPES.NPY_OBJECT, 0)
    # Deserialize: Series
    pos[0] += 2  # skip 'O[' to the 1st identifier: 'i'
    for i in range(rows):
        serialize.ndarray_setitem_1d(arr, i, _deserialize_item(data, pos))
    pos[0] += 2  # skip the ending ']' & ','
    return arr


# Pandas Datetime/Timedelta Index ---------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_datetime_index(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize pandas.DatetimeIndex token
    of the 'data' to `<'pandas.DatetimeIndex'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.
    :param item `<'bool'>`: Whether the deserialization is for an item of the object.

    ### Example:
    >>> data = '...ZIMs|3[1609459200,1609545600,1609632000]...'
        # Start position: ...{Z}IMs|3[1609459200,1609545600,1609632000]...
        return pandas.DatetimeIndex(
            numpy.array([1609459200, 1609545600, 1609632000],
            dtype="datetime64[s]")
        )
    """
    pos[0] += 1  # skip identifier
    val = _deserialize_series_dt64td64(data, pos, item, True)
    # Generate DatetimeIndex
    return typeref.PD_DATETIMEINDEX(val)


@cython.cfunc
@cython.inline(True)
def _deserialize_timedelta_index(
    data: str,
    pos: cython.Py_ssize_t[2],
    item: cython.bint,
) -> object:
    """(cfunc) Deserialize pandas.TimedeltaIndex token
    of the 'data' to `<'pandas.TimedeltaIndex'>`.

    :param data `<'str'>`: The serialized data that contains the token.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the data position.
        - pos[0]: The start position of the token.
        - pos[1]: The end (total length) of the data.
    :param item `<'bool'>`: Whether the deserialization is for an item of the object.

    ### Example:
    >>> data = '...XIms|3[86400,172800,259200]...'
        # Start position: ...{X}IMs|3[86400,172800,259200]...
        return pandas.TimedeltaIndex(
            numpy.array([86400, 172800, 259200],
            dtype="timedelta64[s]")
        )
    """
    pos[0] += 1  # skip identifier
    val = _deserialize_series_dt64td64(data, pos, item, False)
    # Generate TimedeltaIndex
    return typeref.PD_TIMEDELTAINDEX(val)


# Deserialize -----------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _deserialize_obj(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the serialized 'data' to a Python `<'object'>`."""
    # Peak Identifier
    obj_id: cython.Py_UCS4 = read_char(data, 0)  # type: ignore

    # Mapping Types
    # . <'dict'>: D13["0","1","2"][f0.1,f1.1,f2.2,]
    if obj_id == prefix.DICT_ID:
        return _deserialize_dict(data, pos)

    # Sequence Types
    # . <'list'>: L4[i1,f3.14,o1,n,]
    if obj_id == prefix.LIST_ID:
        return _deserialize_list(data, pos)
    # . <'tuple'>: T4[i1,f3.14,o1,n,]
    if obj_id == prefix.TUPLE_ID:
        return _deserialize_tuple(data, pos)
    # . <'set'>: S4[i1,f3.14,o1,n,]
    if obj_id == prefix.SET_ID:
        return _deserialize_set(data, pos)

    # Basic Types
    # . <'str'>: 's5|Hello'
    if obj_id == prefix.STR_ID:
        return _deserialize_str(data, pos)
    # . <'float'>: 'f3.14'
    if obj_id == prefix.FLOAT_ID:
        return _deserialize_float(data, 0, pos[1])  # type: ignore
    # . <'int'>: 'i1024'
    if obj_id == prefix.INT_ID:
        return _deserialize_int(data, 0, pos[1])  # type: ignore
    # . <'bool'>: 'o1'
    if obj_id == prefix.BOOL_ID:
        return _deserialize_bool(data, 0)  # type: ignore
    # . <'None'>: 'n'
    if obj_id == prefix.NONE_ID:
        return None

    # Date&Time Types
    # . <'datetime.datetime'>: 'z2021-01-01T00:00:00'
    if obj_id == prefix.DATETIME_ID:
        return _deserialize_datetime(data, 0, pos[1])
    # . <'datetime.date'>: 'd2021-01-01'
    if obj_id == prefix.DATE_ID:
        return _deserialize_date(data, 0, pos[1])
    # . <'datetime.time'>: 't00:00:00.000001'
    if obj_id == prefix.TIME_ID:
        return _deserialize_time(data, 0, pos[1])
    # . <'datetime.timedelta'>: 'l1|2|3'
    if obj_id == prefix.TIMEDELTA_ID:
        return _deserialize_timedelta(data, 0, pos[1])

    # Numeric Types
    # . <'decimal.Decimal'>: 'e3.14'
    if obj_id == prefix.DECIMAL_ID:
        return _deserialize_decimal(data, 0, pos[1])
    # . <'complex'>: 'c1.0|1.0'
    if obj_id == prefix.COMPLEX_ID:
        return _deserialize_complex(data, 0, pos[1])

    # Bytes Types
    # . <'bytes'>: 'b5|Hello'
    if obj_id == prefix.BYTES_ID:
        return _deserialize_bytes(data, pos)

    # Numpy Types
    # . <'numpy.datetime64'>: 'Mus3661000001'
    if obj_id == prefix.DATETIME64_ID:
        return _deserialize_datetime64(data, 0, pos[1])
    # . <'numpy.timedelta64'>: 'mus1024'
    if obj_id == prefix.TIMEDELTA64_ID:
        return _deserialize_timedelta64(data, 0, pos[1])
    # . <'numpy.ndarray'>
    if obj_id == prefix.NDARRAY_ID:
        return _deserialize_ndarray(data, pos, False)

    # Pandas Types
    # . <'pandas.Series'>
    if obj_id == prefix.SERIES_ID:
        return _deserialize_series(data, pos, False)
    # . <'pandas.DataFrame'>
    if obj_id == prefix.DATAFRAME_ID:
        return _deserialize_dataframe(data, pos)
    # . <'pandas.DatetimeIndex'>
    if obj_id == prefix.DATETIMEINDEX_ID:
        return _deserialize_datetime_index(data, pos, False)
    # . <'pandas.TimedeltaIndex'>
    if obj_id == prefix.TIMEDELTAINDEX_ID:
        return _deserialize_timedelta_index(data, pos, False)

    # Invalid 'data'
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unknown object identifer '%s'." % str_fr_ucs4(obj_id)
    )


@cython.cfunc
@cython.inline(True)
def _deserialize_item(data: str, pos: cython.Py_ssize_t[2]) -> object:
    """(cfunc) Deserialize the next item of the 'data' to an `<'object'>`.

    :param data `<'str'>`: The serialized data that contains the items.
    :param pos `<'Py_ssize_t[2]'>`: The pointer of the item position.
        - pos[0]: The start position of the item.
        - pos[1]: The end (total length) of the data.
    """
    # Peek object identifier
    idx: cython.Py_ssize_t
    loc: cython.Py_ssize_t
    obj_id: cython.Py_UCS4 = read_char(data, pos[0])  # type: ignore

    ##### Common Types #####
    # Basic Types
    # . <'str'>: '...s5|Hello,...'
    if obj_id == prefix.STR_ID:
        return _deserialize_str(data, pos)
    # . <'float'>: '...f3.14,...'
    if obj_id == prefix.FLOAT_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_float(data, idx, loc)
    # . <'int'>: '...i1024,...'
    if obj_id == prefix.INT_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_int(data, idx, loc)
    # . <'bool'>: '...o1,...'
    if obj_id == prefix.BOOL_ID:
        idx = pos[0]
        pos[0] = idx + 3  # update position
        return _deserialize_bool(data, idx)
    # . <'None'>: '...n,...'
    if obj_id == prefix.NONE_ID:
        pos[0] += 2  # skip identifier & ','
        return None

    # Date&Time Types
    # . <'datetime.datetime'>: '...z2021-01-01T00:00:00,...'
    if obj_id == prefix.DATETIME_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_datetime(data, idx, loc)
    # . <'datetime.date'>: '...d2021-01-01,...'
    if obj_id == prefix.DATE_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_date(data, idx, loc)
    # . <'datetime.time'>: '...t00:00:00.000001,...'
    if obj_id == prefix.TIME_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_time(data, idx, loc)
    # . <'datetime.timedelta'>: '...l1|2|3,...'
    if obj_id == prefix.TIMEDELTA_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_timedelta(data, idx, loc)

    # Mapping Types
    # . <'dict'>: '...D13["0","1","2"][f0.1,f1.1,f2.2,],...'
    if obj_id == prefix.DICT_ID:
        return _deserialize_dict(data, pos)

    # Sequence Types
    # . <'list'>: '...L4[i1,f3.14,o1,n,],...'
    if obj_id == prefix.LIST_ID:
        return _deserialize_list(data, pos)
    # . <'tuple'>: '...T4[i1,f3.14,o1,n,],...'
    if obj_id == prefix.TUPLE_ID:
        return _deserialize_tuple(data, pos)
    # . <'set'>: '...S4[i1,f3.14,o1,n,],...'
    if obj_id == prefix.SET_ID:
        return _deserialize_set(data, pos)

    #### Uncommon Types #####
    # Numeric Types
    # . <'decimal.Decimal'>: '...e3.14,...'
    if obj_id == prefix.DECIMAL_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_decimal(data, idx, loc)
    # . <'complex'>: '...c1.0|1.0,...'
    if obj_id == prefix.COMPLEX_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_complex(data, idx, loc)

    # Bytes Types
    # . <'bytes'>: '...b5|Hello,...'
    if obj_id == prefix.BYTES_ID:
        return _deserialize_bytes(data, pos)

    # Numpy Types
    # . <'numpy.datetime64'>: '...Mus3661000001,...'
    if obj_id == prefix.DATETIME64_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_datetime64(data, idx, loc)
    # . <'numpy.timedelta64'>: '...mus1024,...'
    if obj_id == prefix.TIMEDELTA64_ID:
        idx = pos[0]
        loc = find_item_separator(data, idx + 1, pos[1])  # type: ignore
        pos[0] = loc + 1  # update position
        return _deserialize_timedelta64(data, idx, loc)
    # . <'numpy.ndarray'>
    if obj_id == prefix.NDARRAY_ID:
        return _deserialize_ndarray(data, pos, True)

    # Pandas Types
    # . <'pandas.Series'>
    if obj_id == prefix.SERIES_ID:
        return _deserialize_series(data, pos, True)
    # . <'pandas.DataFrame'>
    if obj_id == prefix.DATAFRAME_ID:
        return _deserialize_dataframe(data, pos)
    # . <'pandas.DatetimeIndex'>
    if obj_id == prefix.DATETIMEINDEX_ID:
        return _deserialize_datetime_index(data, pos, True)
    # . <'pandas.TimedeltaIndex'>
    if obj_id == prefix.TIMEDELTAINDEX_ID:
        return _deserialize_timedelta_index(data, pos, True)

    ##### Invalid 'data' #####
    raise errors.DeserializeValueError(
        "<Serializor> Failed to deserialize 'data': "
        "unknown object[item] identifer '%s'." % str_fr_ucs4(obj_id)
    )


@cython.cfunc
def capi_deserialize(data: str) -> object:
    """(cfunc) Deserialize the serialized 'data' to a Python `<'object'>`.

    The given 'data' must be the result of the `serialize()`
    function in this package, which will be used to re-create
    the original Python object.
    """
    # Validate data
    if data is None:
        raise errors.DeserializeValueError(
            "<Serializor> Failed to deserialize 'data': %s." % type(data)
        )
    eof: cython.Py_ssize_t = str_len(data)
    if eof == 0:
        raise errors.DeserializeValueError(
            "<Serializor> Failed to deserialize 'data': '%s'." % data
        )
    pos: cython.Py_ssize_t[2] = [0, eof]

    # Deserialize data
    try:
        return _deserialize_obj(data, pos)
    except errors.DeserializeError:
        raise
    except MemoryError as err:
        raise MemoryError("<Serializor> %s" % err) from err
    except Exception as err:
        raise errors.DeserializeError(
            "<Serializor> Failed to deserialize:\n'%s'\nError: %s" % (data, err)
        ) from err


def deserialize(data: str) -> object:
    """Deserialize the serialized 'data' to a Python `<'object'>`.

    The given 'data' must be the result of the `serialize()`
    function in this package, which will be used to re-create
    the original Python object.
    """
    return capi_deserialize(data)
