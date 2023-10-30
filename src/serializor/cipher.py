# cython: language_level=3
# cython: wraparound=False
# cython: boundscheck=False

# Cython imports
import cython
from cython.cimports.cpython.string import PyString_Check as is_str  # type: ignore
from cython.cimports.cpython.bytes import PyBytes_Check as is_bytes  # type: ignore
from cython.cimports.cpython.bytearray import PyByteArray_Check as is_bytearray  # type: ignore
from cython.cimports.serializor.transcode import encode, decode  # type: ignore

# Python imports
from hashlib import sha256
from typing import Any, Union
from base64 import urlsafe_b64encode
from cryptography.fernet import Fernet
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from serializor.transcode import SerializorError


__all__ = ["Cipher", "encrypt", "decrypt", "CipherError"]


# Cipher ---------------------------------------------------------------------------------------------------------------
@cython.cclass
class Cipher:
    """Cipher class for encrypting and decrypting python object."""

    _key: bytes
    _salt: bytes
    _cipher: Fernet

    def __init__(self, key: Union[str, bytes]) -> None:
        """Cipher class for encrypting and decrypting python object.

        :param key: `<str/bytes>` The key to use for encryption and decryption.
        """
        self._key = self._convert_to_bytes(key)
        self._salt = self._gen_salt(key)
        self._cipher = Fernet(
            urlsafe_b64encode(
                PBKDF2HMAC(
                    algorithm=SHA256(),
                    length=32,
                    salt=self._salt,
                    iterations=32,
                    backend=default_backend(),
                ).derive(self._key)
            )
        )

    # Cipher API ----------------------------------------------------------------------------
    def encrypt(self, obj: Any) -> bytes:
        "Encrypts the given python object `<bytes>`."
        return self._encrypt(obj)

    @cython.cfunc
    @cython.inline(True)
    def _encrypt(self, obj: object) -> bytes:
        "(cfunc) Encrypts the given python object `<bytes>`."
        try:
            value: bytes = encode(obj)
            return self._cipher.encrypt(value)
        except Exception as err:
            raise CipherError(
                "<Cipher> Failed to encrypt object {}.\n"
                "Error: {}".format(repr(obj), err)
            )

    def decrypt(self, crypt: bytes) -> Any:
        "Decrypts the given encrypted data `<Any>`."
        return self._decrypt(crypt)

    @cython.cfunc
    @cython.inline(True)
    def _decrypt(self, crypt: bytes) -> object:
        "(cfunc) Decrypts the given encrypted data `<Any>`."
        try:
            crypt = self._cipher.decrypt(crypt)
            return decode(crypt)
        except Exception as err:
            raise CipherError(
                "<Cipher> Failed to decrypt data {}.\n"
                "Error: {}".format(repr(crypt), err)
            )

    # Utils ---------------------------------------------------------------------------------
    @cython.cfunc
    @cython.inline(True)
    def _convert_to_bytes(self, value: object) -> bytes:
        "(cfunc) Convert the value to bytes `<bytes>`."
        if is_bytes(value):
            return value
        elif is_str(value):
            return value.encode("utf-8", "ignore")
        elif is_bytearray(value):
            return bytes(value)
        else:
            raise CipherError(
                "<Cipher> Value {} must be a string or bytes: {}".format(
                    repr(value), type(value)
                )
            )

    @cython.cfunc
    @cython.inline(True)
    def _gen_salt(self, value: object) -> bytes:
        "(cfunc) Generate salt `<bytes>`."
        b_value: bytes
        if is_bytes(value):
            b_value = self._randomize_bytes(value)
        elif is_str(value):
            b_value = self._randomize_str(value).encode("utf-8", "ignore")
        elif is_bytearray(value):
            b_value = self._randomize_bytes(bytes(value))
        else:
            raise CipherError(
                "<Cipher> Value {} must be a string or bytes: {}".format(
                    repr(value), type(value)
                )
            )
        return sha256(b_value).digest()

    @cython.cfunc
    @cython.inline(True)
    def _randomize_str(self, value: str) -> str:
        "(cfunc) Randomize the string `<str>`."
        shift: cython.int = 1
        max: cython.int = 10
        randomized_chars: list = []
        for c in value:
            if shift >= max:
                shift = 1
            randomized_chars.append(chr((ord(c) - 32 + shift) % 95 + 32))
            shift += 1
        return "".join(randomized_chars)[::-1]

    @cython.cfunc
    @cython.inline(True)
    def _randomize_bytes(self, value: bytes) -> bytes:
        "(cfunc) Randomize the bytes `<bytes>`."
        shift: cython.int = 1
        max: cython.int = 10
        randomized_bytes: bytearray = bytearray()
        for i in value:
            if shift >= max:
                shift = 1
            i = (i + shift) % 256
            randomized_bytes.append(i)
            shift += 1
        return bytes(randomized_bytes[::-1])

    def __del__(self) -> None:
        self._key = None
        self._salt = None
        self._cipher = None


def encrypt(obj: Any, key: Union[str, bytes]) -> bytes:
    "Encrypts the given python object `<bytes>`."
    return Cipher(key)._encrypt(obj)


def decrypt(crypt: bytes, key: Union[str, bytes]) -> Any:
    "Decrypts the given encrypted data `<Any>`."
    return Cipher(key)._decrypt(crypt)


# Exceptions -----------------------------------------------------------------------------------------------------------
class CipherError(SerializorError):
    """The one and only exception this module will raise."""
