import cython
from numpy import ndarray

# Constants
AVAILABLE_TIMEZONES: set[str]

# Utils: encode / decode
def encode_str(obj: str, encoding: cython.pchar) -> bytes:
    """(cfunc) Encode string to bytes using the 'encoding' with
    'surrogateescape' error handling `<'bytes'>`."""

def decode_bytes(data: bytes, encoding: cython.pchar) -> str:
    """(cfunc) Decode bytes to string using the 'encoding' with
    "surrogateescape" error handling `<'str'>`."""

def decode_bytes_utf8(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'utf-8' encoding with
    'surrogateescape' error handling `<'str'>`."""

def decode_bytes_ascii(data: bytes) -> str:
    """(cfunc) Decode bytes to string using 'ascii' encoding with
    'surrogateescape' error handling `<'str'>`."""

# NumPy: ndarray get item
def arr_getitem_1d(arr: ndarray, i: int) -> object:
    """(cfunc) Get item from 1-dimensional numpy ndarray as `<'object'>`."""

def arr_getitem_2d(arr: ndarray, i: int, j: int) -> object:
    """(cfunc) Get item from 2-dimensional numpy ndarray as `<'object'>`."""

def arr_getitem_3d(arr: ndarray, i: int, j: int, k: int) -> object:
    """(cfunc) Get item from 3-dimensional numpy ndarray as `<'object'>`."""

def arr_getitem_4d(arr: ndarray, i: int, j: int, k: int, l: int) -> object:
    """(cfunc) Get item from 4-dimensional numpy ndarray as `<'object'>`."""

# NumPy: ndarray set item
def arr_setitem_1d(arr: ndarray, item: object, i: int) -> bool:
    """(cfunc) Set item for 1-dimensional numpy ndarray."""

def arr_setitem_2d(arr: ndarray, item: object, i: int, j: int) -> bool:
    """(cfunc) Set item for 2-dimensional numpy ndarray."""

def arr_setitem_3d(arr: ndarray, item: object, i: int, j: int, k: int) -> bool:
    """(cfunc) Set item for 3-dimensional numpy ndarray."""

def arr_setitem_4d(arr: ndarray, item: object, i: int, j: int, k: int, l: int) -> bool:
    """(cfunc) Set item for 4-dimensional numpy ndarray."""

# NumPy: ndarray flatten
def arr_flatten(arr: ndarray) -> ndarray:
    """(cfunc) Flatten multi-dimensional numpy ndarray into
    1-dimension with 'C-ORDER' `<'np.ndarray'>`."""

# NumPy: nptime unit
def map_nptime_unit_int2str(unit: int) -> str:
    """(cfunc) Map ndarray[datetime64/timedelta64] time unit from integer
    to the corresponding string representation `<'str'>`."""

def map_nptime_unit_str2int(unit: str) -> int:
    """(cfunc) Map ndarray[datetime64/timedelta64] time unit from string
    representation to the corresponding integer `<'int'>`."""

def parse_arr_nptime_unit(arr: ndarray) -> int:
    """(cfunc) Parse numpy datetime64/timedelta64 time unit from the
    given 'arr', returns the unit in `<'int'>`."""
