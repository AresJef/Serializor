# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.serializor.binary.ser import serialize as _ser_bin  # type: ignore
from cython.cimports.serializor.unicode.ser import serialize as _ser_uni  # type: ignore

# Python imports
from serializor.binary.ser import serialize as _ser_bin
from serializor.unicode.ser import serialize as _ser_uni

__all__ = ["serialize"]


# Serialize -----------------------------------------------------------------------------------
def serialize(obj: object, to_bytes: bool = True) -> bytes | str:
    """Serialize an object to `<bytes>` or `<str>`.
    
    :param obj `<'object'>`: The object to serialize.
    :param to_bytes `<'bool'>`: Whether to serialize to `<bytes>`. Default is `True`.
    """
    return _ser_bin(obj) if bool(to_bytes) else _ser_uni(obj)
