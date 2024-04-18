# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.serializor import serialize, deserialize  # type: ignore

# Python imports
from hashlib import sha256
from base64 import urlsafe_b64encode
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from serializor import serialize, deserialize, errors


# Utils -----------------------------------------------------------------------------
@cython.cfunc
@cython.inline(True)
def _convert_to_bytes(val: object) -> bytes:
    """(cfunc) Convert the given object to `<'bytes'>`."""
    dtype = type(val)
    if dtype is str:
        return serialize.bytes_encode_utf8(val)
    if dtype is bytes:
        return val
    if dtype is bytearray:
        return bytes(val)
    if dtype is memoryview:
        return val.tobytes()
    raise errors.CryptoTypeError(
        "<Serializor> Expects <'str'> or <'bytes'>, "
        "instead got: %s %s." % (dtype, repr(val))
    )


@cython.cfunc
@cython.inline(True)
def _generate_salt(key: bytes) -> bytes:
    """(cfunc) Generate a salt form the given key `<'bytes'>`."""
    array: bytearray = bytearray()
    byte: cython.uint
    shift: cython.uint = 1
    for byte in key:
        array.append((byte + shift) % 256)
        shift = 1 if shift > 10 else shift + 1
    return sha256(array[::-1]).digest()


@cython.cfunc
@cython.inline(True)
def _create_fernet(key: bytes) -> object:
    """(cfunc) Create a Fernet from the given key `<'Fernet'>`."""
    salt = _generate_salt(key)
    return Fernet(
        urlsafe_b64encode(
            PBKDF2HMAC(
                algorithm=SHA256(),
                length=32,
                salt=salt,
                iterations=32,
                backend=default_backend(),
            ).derive(key)
        )
    )


# Encrypt ----------------------------------------------------------------------------
@cython.cfunc
def capi_encrypt(obj: object, key: str | bytes) -> bytes:
    """(cfunc) Serialize and encrypt the Python object with the given key into `<'bytes'>`."""
    # Serialize
    val = serialize.bytes_encode_utf8(serialize.capi_serialize(obj))

    # Encrypt
    fernet: Fernet = _create_fernet(_convert_to_bytes(key))
    try:
        return fernet.encrypt(val)
    except Exception as err:
        raise errors.CryptoError(
            "<Serializor> Object encryption failed: %s" % err
        ) from err


def encrypt(obj: object, key: str | bytes) -> bytes:
    """Serialize and encrypt the Python object with the given key into `<'bytes'>`."""
    return capi_encrypt(obj, key)


# Decrypt ----------------------------------------------------------------------------
@cython.cfunc
def capi_decrypt(enc: bytes, key: str | bytes) -> object:
    """(cfunc) Decrypt and deserialize data with the given key back to a Python `<'object'>`."""
    # Decrypt
    fernet: Fernet = _create_fernet(_convert_to_bytes(key))
    try:
        val = fernet.decrypt(enc)
    except Exception as err:
        raise errors.CryptoError(
            "<Serializor> Data decryption failed: %s" % err
        ) from err

    # Deserialize
    return deserialize.capi_deserialize(serialize.bytes_decode_utf8(val))


def decrypt(enc: bytes, key: str | bytes) -> object:
    """Decrypt and deserialize data with the given key back to a Python `<'object'>`."""
    return capi_decrypt(enc, key)
