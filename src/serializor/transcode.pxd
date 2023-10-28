# cython: language_level=3

# Constants
cdef:
    str UNIQUE_KEY, DECIMAL_KEY, BYTES_KEY, DATE_KEY, DATETIME_NAIVE_KEY
    str DATETIME_AWARE_KEY, TIME_NAIVE_KEY, TIME_AWARE_KEY, TIMEDELTA_KEY
    str NDARRAY_KEY, PDSERIES_JSON_KEY, PDSERIES_OBJT_KEY, PDSERIES_TSNA_KEY
    str PDSERIES_TSAW_KEY, PDSERIES_TMDL_KEY, PDDATAFRAME_KEY

# Encode
cdef dict ENCODERS
cdef bytes encode(object obj) except *
# Decode
cdef object decode(bytes data) except *
