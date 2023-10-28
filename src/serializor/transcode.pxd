# cython: language_level=3

# Encode
cdef bytes encode(object obj) except *
# Decode
cdef object decode(bytes data) except *
