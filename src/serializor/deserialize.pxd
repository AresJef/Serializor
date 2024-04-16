# cython: language_level=3

from libc.stdlib cimport malloc, free, strtod, strtoll
from cpython.unicode cimport PyUnicode_READ_CHAR as read_char
from cpython.unicode cimport PyUnicode_Substring, PyUnicode_FindChar
from serializor cimport serialize

# Constants
cdef:
    # . characters
    Py_UCS4 CHAR_QUOTE, CHAR_COMMA, CHAR_ONE, CHAR_BACKSLASH
    Py_UCS4 CHAR_PIPE, CHAR_OPEN_BRACKET, CHAR_CLOSE_BRACKET
    # . functions
    object FN_ORJSON_LOADS, FN_NUMPY_EMPTY

# Struct
ctypedef struct shape:
    Py_ssize_t ndim
    Py_ssize_t i
    Py_ssize_t j
    Py_ssize_t k
    Py_ssize_t l
    Py_ssize_t loc

# Utils
cdef inline object slice_to_bytes(str data, Py_ssize_t start, Py_ssize_t end):
    """Slice data (str) from 'start' to 'end', and convert to `<bytes>`."""
    return serialize.bytes_encode_utf8(PyUnicode_Substring(data, start, end))

cdef inline unicode slice_to_unicode(str data, Py_ssize_t start, Py_ssize_t end):
    """Slice data (str) from 'start' to 'end', and convert to `<unicode>`."""
    return PyUnicode_Substring(data, start, end)

cdef inline double slice_to_float(str data, Py_ssize_t start, Py_ssize_t end):
    """Slice data (str) from 'start' to 'end', and convert to `<double>`."""
    # Calculate the size of the token.
    cdef Py_ssize_t size = end - start
    # Allocate memory for the slice.
    cdef char* buffer = <char*>malloc(size + 1)
    if buffer == NULL:
        raise MemoryError("Failed to allocate memory for the 'data' slice.")
    cdef Py_ssize_t i
    try:
        # Assign the slice to the buffer.
        for i in range(size):
            buffer[i] = read_char(data, start + i)
        # Null-terminate the buffer.
        buffer[size] = 0
        # Convert to double.
        return strtod(buffer, NULL)
    finally:
        # Free the memory.
        free(buffer)

cdef inline long long slice_to_int(str data, Py_ssize_t start, Py_ssize_t end):
    """Slice data (str) from 'start' to 'end', and convert to `<long long>`."""
    # Calculate the size of the token.
    cdef Py_ssize_t size = end - start
    # Allocate memory for the slice.
    cdef char* buffer = <char*>malloc(size + 1)
    if buffer == NULL:
        raise MemoryError("Failed to allocate memory for the 'data' slice.")
    cdef Py_ssize_t i
    try:
        # Assign the slice to the buffer.
        for i in range(size):
            buffer[i] = read_char(data, start + i)
        # Null-terminate the buffer.
        buffer[size] = 0
        # Convert to long long.
        return strtoll(buffer, NULL, 10)
    finally:
        # Free the memory.
        free(buffer)

cdef inline Py_ssize_t find_data_separator(str data, Py_ssize_t start, Py_ssize_t end) except -1:
    """Find the next data separator `'|'` and returns its position `<Py_ssize_t>`."""
    cdef Py_ssize_t loc = start
    while loc < end:
        if read_char(data, loc) == CHAR_PIPE:
            return loc
        loc += 1
    raise ValueError(
        "Failed to locate the next data separator '|' "
        "from:\n'%s'" % slice_to_unicode(data, start, end) )

cdef inline Py_ssize_t find_item_separator(str data, Py_ssize_t start, Py_ssize_t end) except -1:
    """Find the next item separator `','` and returns its position `<Py_ssize_t>`."""
    cdef Py_ssize_t loc = start
    while loc < end:
        if read_char(data, loc) == CHAR_COMMA:
            return loc
        loc += 1
    raise ValueError(
        "Failed to locate the next item separator ',' "
        "from:\n'%s'" % slice_to_unicode(data, start, end) )

cdef inline Py_ssize_t find_open_bracket(str data, Py_ssize_t start, Py_ssize_t end) except -1:
    """Find the next open bracket `'['` and returns its position `<Py_ssize_t>`."""
    cdef Py_ssize_t loc = start
    while loc < end:
        if read_char(data, loc) == CHAR_OPEN_BRACKET:
            return loc
        loc += 1
    raise ValueError(
        "Failed to locate the next open bracket '[' "
        "from:\n'%s'" % slice_to_unicode(data, start, end) )

cdef inline Py_ssize_t find_close_bracket(str data, Py_ssize_t start, Py_ssize_t end) except -1:
    """Find the next close bracket `']'` and returns its position `<Py_ssize_t>`."""
    cdef Py_ssize_t loc = PyUnicode_FindChar(data, CHAR_CLOSE_BRACKET, start, end, 1)
    if loc < 0:
        raise ValueError(
            "Failed to locate the next close bracket ']' "
            "from:\n'%s'" % slice_to_unicode(data, start, end) )
    return loc

cdef inline Py_ssize_t find_close_bracketq(str data, Py_ssize_t start, Py_ssize_t end) except -1:
    """Find the next close bracket `']'` that is preceded by a non-escaped `'"'` 
    and returns its position `<Py_ssize_t>`."""
    cdef Py_ssize_t loc = start
    while loc < end:
        # Find closing bracket ']'.
        loc = PyUnicode_FindChar(data, CHAR_CLOSE_BRACKET, loc, end, 1)
        if loc < 0:
            break # not found
        # Ensure it is preceded by a non-escaped '"'.
        if (
            read_char(data, loc - 1) == CHAR_QUOTE  # '"]"
            and read_char(data, loc - 2) != CHAR_BACKSLASH  # '"]'
        ):
            return loc # exit
        # Continue searching.
        loc += 1
    # Not found.
    raise ValueError(
        "Failed to locate the next close bracket ']' "
        "that is preceded by a non-escaped '\"' "
        "from:\n%s" % slice_to_unicode(data, start, end) )

# Deserialize
cdef object capi_deserialize(str obj)