# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.serializor.binary.des import deserialize as _des_bin  # type: ignore
from cython.cimports.serializor.unicode.des import deserialize as _des_uni  # type: ignore

# Python imports
from serializor.binary.des import deserialize as _des_bin
from serializor.unicode.des import deserialize as _des_uni
from serializor import errors

__all__ = ["deserialize"]


# Deserialize ---------------------------------------------------------------------------------
@cython.ccall
def deserialize(data: bytes | str) -> object:
    """Deserialize the data (bytes/str) back to an `<'object'>`.

    :param data `<'bytes/str'>`: The data to deserialize.
    """
    dtype = type(data)
    if dtype is bytes:
        return _des_bin(data)
    if dtype is str:
        return _des_uni(data)
    raise errors.DeserializeError(
        "<'Serializor'>\nInvalid 'data' to deserialize: "
        "expects <'bytes/str'>, got %s." % dtype
    )
