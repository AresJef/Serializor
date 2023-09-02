## Serialize common python/numpy/pandas objects.

Created to be used in a project, this package is published to github 
for ease of management and installation across different modules.

### Features
This package is designed to serialize common python objects, and
deserializes them back to their original (or compatiable) python
data type.

Supported data types includes:
- boolean: `bool` & `numpy.bool_` -> deserialize to `bool`
- integer: `int` & `numpy.int` & `numpy.uint` -> deserialize to `int`
- float: `float` & `numpy.float_` -> deserialize to `float`
- decimal: `decimal.Decimal` -> deserialize to `decimal.Decimal`
- string: `str` -> deserialize to `str`
- bytes: `bytes` -> deserialize to `bytes`
- date: `datetime.date` -> deserialize to `datetime.date`
- time: `datetime.time` -> deserialize to `datetime.time`
- datetime: `datetime.datetime` & `pandas.Timestamp` -> deserialize to `datetime.datetime`
- datetime64*: `numpy.datetime64` & `time.struct_time` -> deserialize to `datetime.datetime`
- timedelta: `datetime.timedelta` & `pandas.Timedelta` -> deserialize to `datetime.timedelta`
- timedelta64: `numpy.timedelta64` -> deserialize to `datetime.timedelta`
- None: `None` & `numpy.nan` -> deserialize to `None`
- list: `list` of above supported data types -> deserialize to `list`
- tuple: `tuple` of above supported data types -> deserialize to `list`
- set: `set` of above supported data types -> deserialize to `list`
- frozenset: `frozenset` of above supported data types -> deserialize to `list`
- dict: `dict` of above supported data types -> deserialize to `dict`
- numpy.record: `numpy.record` of above supported data types -> deserialize to `list`
- numpy.ndarray: `numpy.ndarray` of above supported data types -> deserialize to `np.ndarray`
- pandas.Series: `pandas.Series` of above supported data types -> deserialize to `pandas.Series`
- pandas.DataFrame: `pandas.DataFrame` of above supported data types -> deserialize to `pandas.DataFrame`

The core serialization is based on `orjson`, which is a fast, superb JSON
library for Python. As a result, the `dumps` result is in `bytes` format
(just as `orjson` does). However, in order to serialize non-JSON objects,
and `loads` the serialized data back to its original python date type, `dumps` 
result can not be simply treated as serialized JSON. Instead, requires the 
`loads` function in this package to deserialize it properly.

Also, in order to support data types such as `Decimal`, `pandas.Series`, etc.,
The serialization performance will be slower than `orjson` when non-standard
data types are involved. The deserialization performance will always be slower
than `orjson` due to the process of validation and reconstruction of the 
serialized data to its original (or compatible) python data type.

### Installation
Install from `PyPi`
``` bash
pip install serializor
```

Install from `github`
``` bash
pip install git+https://github.com/AresJef/Serializor.git
```

### Compatibility
Only support for python 3.10 and above.

### Boolean
All supported boolean data types will be deserialized to python 
native `bool` type.

``` python
import serializor as sr, numpy as np

## Python native boolean
obj = True
s = sr.dumps(obj)
d = sr.loads(s)
# True <class 'bool'>

## Numpy boolean
obj = np.bool_(True)
s = sr.dumps(obj)
d = sr.loads(s)
# True <class 'bool'>
```

### Integer
All supported integer data types will be deserialized to python 
native `int` type.

``` python
import serializor as sr, numpy as np

## Python native integer
obj = 1
s = sr.dumps(obj)
d = sr.loads(s)
# 1 <class 'int'>

## Numpy integer
obj = np.int64(1)
s = sr.dumps(obj)
d = sr.loads(s)
# 1 <class 'int'>

## Numpy unsigned integer
obj = np.uint64(1)
s = sr.dumps(obj)
d = sr.loads(s)
# 1 <class 'int'>
```

### Float
All supported float data types will be deserialized to python 
native `float` type.

``` python
import serializor as sr, numpy as np

## Python native float
obj = 1.0
s = sr.dumps(obj)
d = sr.loads(s)
# 1.0 <class 'float'>

## Numpy float
obj = np.float64(1.0)
s = sr.dumps(obj)
d = sr.loads(s)
# 1.0 <class 'float'>
```

### Decimal
Decimal data type will be deserialized back to `decimal.Decimal` type.

``` python
import serializor as sr
from decimal import Decimal

obj = Decimal('1.0')
s = sr.dumps(obj)
d = sr.loads(s)
# 1.0 <class 'decimal.Decimal'>
```

### String
String data type will be deserialized back to `str` type.

``` python
import serializor as sr

obj = 'STRING'
s = sr.dumps(obj)
d = sr.loads(s)
# 'STRING' <class 'str'>
```

### Bytes
Bytes data type will be deserialized back to `bytes` type.

``` python
import serializor as sr

obj = b'BYTES'
s = sr.dumps(obj)
d = sr.loads(s)
# b'BYTES' <class 'bytes'>
```

### Date
Date data type will be deserialized back to `datetime.date` type.

``` python
import serializor as sr, datetime

obj = datetime.date(2023, 8, 27)
s = sr.dumps(obj)
d = sr.loads(s)
# 2023-08-27 <class 'datetime.date'>
```

### Time
Time data type will be deserialized back to `datetime.time` type.

``` python
import serializor as sr, datetime

# Time (timezone-naive)
obj = datetime.time(21, 12, 31, 354654)
s = sr.dumps(obj)
d = sr.loads(s)
# 21:12:31.354654 <class 'datetime.time'>

# Time (timezone-aware)
tzinfo = datetime.timezone(datetime.timedelta(hours=8), "CST")
obj = datetime.time(21, 12, 31, 354654, tzinfo=tzinfo)
s = sr.dumps(obj)
d = sr.loads(s)
# 21:12:31.354654+08:00 <class 'datetime.time'>
```

### Datetime
All supported datetime data types will be deserialized to python 
native `datetime.datetime` type.

``` python
import serializor as sr, datetime, numpy as np, time

# Datetime (timezone-naive) / pandas.Timestamp
obj = datetime.datetime(2023, 8, 27, 21, 16, 4, 780600)
s = sr.dumps(obj)
d = sr.loads(s)
# 2023-08-27 21:16:04.780600 <class 'datetime.datetime'>

# Datetime (timezone-aware) / pandas.Timestamp
tzinfo = datetime.timezone(datetime.timedelta(hours=8), "CST")
obj = datetime.datetime(2023, 8, 27, 21, 16, 4, 780600, tzinfo=tzinfo)
s = sr.dumps(obj)
d = sr.loads(s)
# 2023-08-27 21:16:04.780600+08:00 <class 'datetime.datetime'>

# Numpy datetime64
obj = np.datetime64('2023-08-27T21:16:04.780600')
s = sr.dumps(obj)
d = sr.loads(s)
# 2023-08-27 21:16:04.780600 <class 'datetime.datetime'>

# Time struct_time
obj = time.struct_time((2023, 8, 27, 21, 16, 4, 0, 0, 0))
s = sr.dumps(obj)
d = sr.loads(s)
# 2023-08-27 21:16:04 <class 'datetime.datetime'>
```

### Timedelta
All supported timedelta data types will be deserialized to python
native `datetime.timedelta` type.

``` python
import serializor as sr, datetime, numpy as np

# Timedelta / pandas.Timedelta
obj = datetime.timedelta(days=1)
s = sr.dumps(obj)
d = sr.loads(s)
# 1 day, 0:00:00 <class 'datetime.timedelta'>

# Numpy timedelta64
obj = np.timedelta64(1, 'D')
s = sr.dumps(obj)
d = sr.loads(s)
# 1 day, 0:00:00 <class 'datetime.timedelta'>
```

### Sequence (list, tuple, set, frozenset, np.record)
All supported sequence data types will be deserialized to python
native `list` type.

``` python
import serializor as sr, datetime, numpy as np, pandas as pd

## List / Tuple / Set / Frozenset / Numpy record
tzinfo=datetime.timezone(datetime.timedelta(hours=8), 'CST')
obj = [
    True, np.bool_(False),
    1, np.int8(-2), np.int16(-3), np.int32(-4), np.int64(-5), 
    np.uint16(6), np.uint32(7), np.uint64(8),
    1.1, np.float16(2.2), np.float32(3.3), np.float64(4.4),
    Decimal('3.3'), 
    'STRING', 
    b'BYTES', 
    datetime.date(2023, 8, 27), 
    datetime.datetime(2023, 1, 1, 1, 1, 1, 1), datetime.datetime(2023, 1, 1, 1, 1, 1, 1, tzinfo=tzinfo), 
    pd.Timestamp('2023-01-01 01:01:01.000001'), pd.Timestamp('2023-01-01 01:01:01.000001+0800'),
    np.datetime64('2023-01-01T01:01:01.000001'),
    datetime.time(1, 1, 1, 1), datetime.time(1, 1, 1, 1, tzinfo=tzinfo), 
    datetime.timedelta(days=1), pd.Timedelta(2, unit='D'), np.timedelta64(3,'D'), 
    None,
]
s = sr.dumps(obj)
d = sr.loads(s)
# [
#   True, False, 
#   1, -2, -3, -4, -5, 6, 7, 8, 
#   1.1, 2.19921875, 3.299999952316284, 4.4, 
#   Decimal('3.3'), 
#   'STRING', 
#   b'BYTES', 
#   datetime.date(2023, 8, 27), 
#   datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
#   datetime.datetime(2023, 1, 1, 1, 1, 1, 1, tzinfo=...), 
#   datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
#   datetime.datetime(2023, 1, 1, 1, 1, 1, 1, tzinfo=...), 
#   datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
#   datetime.time(1, 1, 1, 1), 
#   datetime.time(1, 1, 1, 1, tzinfo=...), 
#   datetime.timedelta(days=1), datetime.timedelta(days=2), datetime.timedelta(days=3), 
#   None
# ]
```

### Dictionary
Dictionary data type will be deserialized to python native `dict` type. 
Notice, the key of the dictionary must be of `str` type only.

``` python
import serializor as sr, datetime, numpy as np, pandas as pd

## Dictionary
tzinfo=datetime.timezone(datetime.timedelta(hours=8), 'CST')
obj = {
    "bool": True, "np_bool": np.bool_(False),
    "int": 1, 
    "np_int8": np.int8(-2), "np_int16": np.int16(-3), "np_int32": np.int32(-4), "np_int64": np.int64(-5), "np_uint16": np.uint16(6), "np_uint32": np.uint32(7), "np_uint64": np.uint64(8),
    "float": 1.1, "np_float16": np.float16(2.2), "np_float32": np.float32(3.3), "np_float64": np.float64(4.4),
    "decimal": Decimal('3.3'),
    "str": 'STRING',
    "bytes": b'BYTES',
    "date": datetime.date(2023, 8, 27),
    "datetime": datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
    "datetime_tz": datetime.datetime(2023, 1, 1, 1, 1, 1, 1, tzinfo=tzinfo),
    "pd_timestamp": pd.Timestamp('2023-01-01 01:01:01.000001'),
    "pd_timestamp_tz": pd.Timestamp('2023-01-01 01:01:01.000001+0800'),
    "np_datetime64": np.datetime64('2023-01-01T01:01:01.000001'),
    "time": datetime.time(1, 1, 1, 1),
    "time_tz": datetime.time(1, 1, 1, 1, tzinfo=tzinfo),
    "timedelta": datetime.timedelta(days=1),
    "pd_timedelta": pd.Timedelta(2, unit='D'),
    "np_timedelta64": np.timedelta64(3,'D'),
    "none": None,
}
s = sr.dumps(obj)
d = sr.loads(s)
# {
#   'bool': True, 'np_bool': False, 
#   'int': 1, 'np_int8': -2, 'np_int16': -3, 'np_int32': -4, 'np_int64': -5, 
#   'np_uint16': 6, 'np_uint32': 7, 'np_uint64': 8, 
#   'float': 1.1, 'np_float16': 2.19921875, 'np_float32': 3.299999952316284, 'np_float64': 4.4, 
#   'decimal': Decimal('3.3'), 
#   'str': 'STRING', 
#   'bytes': b'BYTES', 
#   'date': datetime.date(2023, 8, 27), 
#   'datetime': datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
#   'datetime_tz': datetime.datetime(2023, 1, 1, 1, 1, 1, 1, tzinfo=...), 
#   'pd_timestamp': datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
#   'pd_timestamp_tz': datetime.datetime(2023, 1, 1, 1, 1, 1, 1, tzinfo=...), 
#   'np_datetime64': datetime.datetime(2023, 1, 1, 1, 1, 1, 1), 
#   'time': datetime.time(1, 1, 1, 1), 
#   'time_tz': datetime.time(1, 1, 1, 1, tzinfo=...), 
#   'timedelta': datetime.timedelta(days=1), 
#   'pd_timedelta': datetime.timedelta(days=2), 
#   'np_timedelta64': datetime.timedelta(days=3), 
#   'none': None
# }
```

### Numpy ndarray
Numpy ndarray data type will be deserialized to  `numpy.ndarray` type.

``` python
import serializor as sr, numpy as np
obj = np.array([[1,2,3], [4,5,6]])
s = sr.dumps(obj)
d = sr.loads(s)
# [[1 2 3], [4 5 6]] <class 'numpy.ndarray'>
```

### Pandas Series
Pandas Series data type will be deserialized to  `pandas.Series` type.

``` python
import serializor as sr, pandas as pd
# Series of integers
obj = pd.Series([1,2,3], name='Integers')
s = sr.dumps(obj)
d = sr.loads(s)
# <pandas.Series>
# 0    1
# 1    2
# 2    3
# Name: Integers, dtype: int64

# Series of Timestamps
obj = pd.Series(pd.date_range("2023-01-01", periods=3, freq="D"), name="Dates")
s = sr.dumps(obj)
d = sr.loads(s)
# <pandas.Series>
# 0   2023-01-01
# 1   2023-01-02
# 2   2023-01-03
# Name: Dates, dtype: datetime64[ns]
```

### Pandas DataFrame
Pandas DataFrame data type will be deserialized to  `pandas.DataFrame` type.

``` python
import serializor as sr, pandas as pd, numpy as np, datetime
dt = datetime.datetime(2023, 8, 1, 12, 34, 56, 789)
val = {
    "bool": True,
    "int": -1,
    "float": 1.1,
    "decimal": Decimal("3.3"),
    "str": "STRING",
    "bytes": b"BYTES",
    "datetime": dt,
    "time": dt.time(),
    "timedelta": datetime.timedelta(1),
}
obj = pd.DataFrame([val] * 3)
s = sr.dumps(obj)
d = sr.loads(s)
# <pandas.DataFrame>
# bool int float decimal str bytes datetime time timedelta
# 0 True -1 1.1 3.3 STRING b'BYTES' 2023-08-01 12:34:56.000789 12:34:56.000789 1 days
# 1 True -1 1.1 3.3 STRING b'BYTES' 2023-08-01 12:34:56.000789 12:34:56.000789 1 days
# 2 True -1 1.1 3.3 STRING b'BYTES' 2023-08-01 12:34:56.000789 12:34:56.000789 1 days
```

### Acknowledgements
serializor is based on several open-source repositories.
- [cytimes](https://github.com/AresJef/cyTimes)
- [numpy](https://github.com/numpy/numpy)
- [orjson](https://github.com/pandas-dev/pandas)
- [pandas](https://github.com/pandas-dev/pandas)

