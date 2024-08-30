# Base Exceptions ---------------------------------------------------------------------------------
class SerializorError(Exception):
    """Base class for all exceptions raised by serializor."""


# Serialize Errors --------------------------------------------------------------------------------
class SerializeError(SerializorError):
    """Base class for all exceptions raised by serializor during serialization."""


# Deserialize Errors ------------------------------------------------------------------------------
class DeserializeError(SerializorError):
    """Base class for all exceptions raised by serializor during deserialization."""


# Crypto Errors -----------------------------------------------------------------------------------
class CryptoError(SerializorError):
    """Base class for all exceptions raised by serializor during encryption and decryption."""


class EncryptError(CryptoError, SerializeError):
    """Raised when the object encryption fails."""


class DecryptError(CryptoError, DeserializeError):
    """Raised when the data decryption fails."""
