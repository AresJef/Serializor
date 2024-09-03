# cython: language_level=3

# Encrypt
cpdef bytes encrypt(object obj, object key)

# Decrypt
cpdef object decrypt(bytes data, object key)
