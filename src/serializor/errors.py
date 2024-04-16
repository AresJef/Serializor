# Base Exceptions ---------------------------------------------------------------------------------
class SerializorError(Exception):
    """Base class for all exceptions raised by serializor."""


# Serialize Errors --------------------------------------------------------------------------------
class SerializeError(SerializorError):
    """Base class for all exceptions raised by serializor during serialization."""


class SerializeTypeError(SerializeError, TypeError):
    """Raised when the type of the object to serialize is not supported."""


# Deserialize Errors ------------------------------------------------------------------------------
class DeserializeError(SerializorError):
    """Base class for all exceptions raised by serializor during deserialization."""


class DeserializeValueError(DeserializeError, ValueError):
    """Raised when the value of the object to deserialize is not supported."""


# Crypto Errors -----------------------------------------------------------------------------------
class CryptoError(SerializorError):
    """Base class for all exceptions raised by serializor during encryption and decryption."""


class CryptoTypeError(CryptoError, TypeError):
    """Raised when the type of the object to encrypt or decrypt is not supported."""
