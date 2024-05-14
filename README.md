## Serialization for common Python objects.

Created to be used in a project, this package is published to github for ease of management and installation across different modules.

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
Supports Python 3.10 and above.

### Features
This package is designed to serialize Python object, and deserializes it back to the original (or compatiable) Python object.

Supported data types:
- string: `str` -> deserialize to `str`
- float: `float` & `numpy.float_` -> deserialize to `float`
- integer: `int` & `numpy.int` & `numpy.uint` -> deserialize to `int`
- boolean: `bool` & `numpy.bool_` -> deserialize to `bool`
- None: `None` & `numpy.nan` -> deserialize to `None`
- datetime: `datetime.datetime`, `pandas.Timestamp` & `time.struct_time` -> deserialize to `datetime.datetime`
- datetime64: `numpy.datetime64` -> deserialize to `numpy.datetime64`
- date: `datetime.date` -> deserialize to `datetime.date`
- time: `datetime.time` -> deserialize to `datetime.time`
- timedelta: `datetime.timedelta` & `pandas.Timedelta` -> deserialize to `datetime.timedelta`
- timedelta64: `numpy.timedelta64` -> deserialize to `numpy.timedelta64`
- decimal: `decimal.Decimal` -> deserialize to `decimal.Decimal`
- complex: `complex` & `numpy.complex_` -> deserialize to `complex`
- bytes: `bytes`, `bytearray` & `numpy.bytes_` -> deserialize to `bytes`
- list: `list` of above supported data types -> deserialize to `list`
- tuple: `tuple` of above supported data types -> deserialize to `tuple`
- set: `set` & `frozenset` of above supported data types -> deserialize to `set`
- dict: `dict` of above supported data types -> deserialize to `dict`
- numpy.ndarray: `numpy.ndarray` of above supported data types -> deserialize to `np.ndarray`
- pandas.Series: `pandas.Series` of above supported data types -> deserialize to `pandas.Series`
- pandas.DataFrame: `pandas.DataFrame` of above supported data types -> deserialize to `pandas.DataFrame`

### Usage (Serialization & Deserialization)
```python
from pandas import Series
from serializor import serialize, deserialize

obj = Series([1, 2, 3, 4, 5], name="test")
en = serialize(obj)
de = deserialize(en)
print(de)
```
```
0    1
1    2
2    3
3    4
4    5
Name: test, dtype: int64
```

### Acknowledgements
serializor is based on several open-source repositories.
- [numpy](https://github.com/numpy/numpy)
- [orjson](https://github.com/ijl/orjson)
- [pandas](https://github.com/pandas-dev/pandas)

