# cython: language_level=3

# Python imports
import numpy as np
from decimal import Decimal
from time import struct_time
from pandas import Series, DataFrame
from pandas import Timestamp, DatetimeIndex
from pandas import Timedelta, TimedeltaIndex

# Constants -------------------------------------------------------------------------
# . python types
NONE: type = type(None)
DICT_KEYS: type = type(dict().keys())
DICT_VALUES: type = type(dict().values())
DECIMAL: type = Decimal
STRUCT_TIME: type = struct_time
# . numpy types
NP_FLOAT16: type = np.float16
NP_FLOAT32: type = np.float32
NP_FLOAT64: type = np.float64
NP_INT8: type = np.int8
NP_INT16: type = np.int16
NP_INT32: type = np.int32
NP_INT64: type = np.int64
NP_UINT8: type = np.uint8
NP_UINT16: type = np.uint16
NP_UINT32: type = np.uint32
NP_UINT64: type = np.uint64
NP_COMPLEX64: type = np.complex64
NP_COMPLEX128: type = np.complex128
NP_DATETIME64: type = np.datetime64
NP_TIMEDELTA64: type = np.timedelta64
NP_BOOL: type = np.bool_
NP_BYTES: type = np.bytes_
NP_NAN: type = np.nan
NP_RECORD: type = np.record
NP_NDARRAY: type = np.ndarray
# . pandas types
PD_SERIES: type = Series
PD_DATAFRAME: type = DataFrame
PD_TIMESTAMP: type = Timestamp
PD_DATETIMEINDEX: type = DatetimeIndex
PD_TIMEDELTA: type = Timedelta
PD_TIMEDELTAINDEX: type = TimedeltaIndex
