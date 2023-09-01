# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

# Encode
cdef bytes encode(object obj) except *
# Decode
cdef object decode(bytes data) except *
