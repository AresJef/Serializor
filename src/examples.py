import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
import numpy as np
import serializor

# Serialize & Deserialize
obj = {
    "str": "Hello World!\n中国\n한국어\nにほんご\nEspañol",
    "int": 1234567890,
    "float": 3.141592653589793,
    "bool": True,
    "none": None,
    "datetime": datetime.datetime(2012, 1, 2, 3, 4, 5, 6),
    "datetime.tz1": datetime.datetime.now(ZoneInfo("CET")),
    "datetime.tz2": datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9))
    ),
    "date": datetime.date(2012, 1, 2),
    "time": datetime.time(3, 4, 5, 6),
    "timedelta": datetime.timedelta(1, 2, 3),
    "decimal": Decimal("3.1234"),
    "complex": 2.12345 + 3.12345j,
    "bytes": "Hello World!\n中国\n한국어\nにほんご\nEspañol".encode("utf-8"),
    "datetime64": np.datetime64("2012-06-30 12:00:00.000000010"),
    "timedelta64": np.timedelta64(-datetime.timedelta(1, 2, 3)),
    "complex64": np.complex64(1 + 1j),
    "complex128": np.complex128(-1 + -1j),
}

se = serializor.serialize(obj)
print(se)
de = serializor.deserialize(se)
assert obj == de
print(de)
print()
