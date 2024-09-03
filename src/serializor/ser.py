# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.serializor.binary.ser import serialize as _ser_bin  # type: ignore
from cython.cimports.serializor.unicode.ser import serialize as _ser_uni  # type: ignore

# Python imports
from serializor.binary.ser import serialize as _ser_bin
from serializor.unicode.ser import serialize as _ser_uni

__all__ = ["serialize"]


# Serialize -----------------------------------------------------------------------------------
@cython.ccall
def serialize(obj: object, to_bytes: cython.bint = True) -> object:
    """Serialize an object to `<'bytes'>` or `<'str'>`.

    :param obj `<'object'>`: The object to serialize.
    :param to_bytes `<'bool'>`: Whether to serialize to `<'bytes'>` (recommended). Defaults to `True`.

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
    return _ser_bin(obj) if to_bytes else _ser_uni(obj)
