# cython: language_level=3

cdef:
    # Basic Types
    char STR
    char BOOL
    char FLOAT
    char INT
    char NONE

    # Date&Time Types
    char DATE
    char TIME
    char DATETIME
    char TIMEDELTA
    char STRUCT_TIME

    # Numeric Types
    char DECIMAL
    char COMPLEX

    # Bytes Types
    char BYTES
    char BYTEARRAY

    # Sequence Types
    char LIST
    char TUPLE
    char SET
    char FROZENSET
    char RANGE

    # Mapping Types
    char DICT

    # NumPy Types
    char DATETIME64
    char TIMEDELTA64
    char NDARRAY
    char NDARRAY_OBJECT
    char NDARRAY_INT
    char NDARRAY_UINT
    char NDARRAY_FLOAT
    char NDARRAY_BOOL
    char NDARRAY_DT64
    char NDARRAY_TD64
    char NDARRAY_COMPLEX
    char NDARRAY_BYTES
    char NDARRAY_UNICODE

    # Pandas Types
    char SERIES
    char DATAFRAME
    char PD_TIMESTAMP
    char PD_TIMEDELTA
    char DATETIMEINDEX
    char TIMEDELTAINDEX

    ### Test for duplicates ###
    bint _duplicates_test
