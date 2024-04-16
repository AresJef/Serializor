# cython: language_level=3

# Constants
cdef:
    # . python types
    object NONE, DICT_KEYS, DICT_VALUES
    object DECIMAL, STRUCT_TIME
    # . numpy types
    object NP_FLOAT16, NP_FLOAT32, NP_FLOAT64
    object NP_INT8, NP_INT16, NP_INT32, NP_INT64
    object NP_UINT8, NP_UINT16, NP_UINT32, NP_UINT64
    object NP_COMPLEX64, NP_COMPLEX128
    object NP_DATETIME64, NP_TIMEDELTA64
    object NP_BOOL, NP_BYTES, NP_NAN
    object NP_RECORD, NP_NDARRAY
    # . pandas types
    object PD_SERIES, PD_DATAFRAME
    object PD_TIMESTAMP, PD_DATETIMEINDEX
    object PD_TIMEDELTA, PD_TIMEDELTAINDEX
    