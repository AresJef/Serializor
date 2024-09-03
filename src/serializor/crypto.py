# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.serializor.binary.ser import serialize as _ser_bin  # type: ignore
from cython.cimports.serializor.binary.des import deserialize as _des_bin  # type: ignore

# Python imports
from hashlib import sha256 as _sha256
from base64 import urlsafe_b64encode as _urlsafe_b64encode
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend as _default_backend
from serializor.binary.ser import serialize as _ser_bin
from serializor.binary.des import deserialize as _des_bin
from serializor import errors

__all__ = ["encrypt", "decrypt"]


# Utils -----------------------------------------------------------------------------
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
    return _sha256(array[::-1]).digest()


@cython.cfunc
@cython.inline(True)
def _create_fernet(key: bytes) -> object:
    """(cfunc) Create a Fernet from the given key `<'Fernet'>`."""
    return Fernet(
        _urlsafe_b64encode(
            PBKDF2HMAC(
                algorithm=SHA256(),
                length=32,
                salt=_generate_salt(key),
                iterations=32,
                backend=_default_backend(),
            ).derive(key)
        )
    )


# Encrypt ----------------------------------------------------------------------------
@cython.ccall
def encrypt(obj: object, key: object) -> bytes:
    """Serialize and encrypt an object with the given 'key' into `<'bytes'>`.

    :param obj `<'object'>`: The object to encrypt.
    :param key `<'object'>`: The key to encrypt the object with.
    """
    # Serialize & Encrypt
    val: bytes = _ser_bin(obj)
    fernet: Fernet = _create_fernet(_ser_bin(key))
    try:
        return fernet.encrypt(val)
    except Exception as err:
        raise errors.EncryptError(
            "<'Serializor'>\nObject encryption failed: %s" % err
        ) from err


# Decrypt ----------------------------------------------------------------------------
@cython.ccall
def decrypt(data: bytes, key: object) -> object:
    """Decrypt and deserialize encrypted data (bytes)
    with the given 'key' back to an `<'object'>`.

    :param enc `<'bytes'>`: The encrypted data.
    :param key `<'object'>`: The key to decrypt the data with.
    """
    # Decrypt & Deserialize
    fernet: Fernet = _create_fernet(_ser_bin(key))
    try:
        val = fernet.decrypt(data)
    except Exception as err:
        raise errors.DecryptError(
            "<'Serializor'>\nData decryption failed: %s" % err
        ) from err
    return _des_bin(val)
