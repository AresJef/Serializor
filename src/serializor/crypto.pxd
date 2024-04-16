# cython: language_level=3

# Encrypt
cdef bytes capi_encrypt(object obj, object key)

# Decrypt
cdef object capi_decrypt(bytes enc, object key)
