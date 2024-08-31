## Serialization/Deserialization for Python native and NumPy/Pandas objects.

Created to be used in a project, this package is published to github for ease of management and installation across different modules.

### Installation

Install from `PyPi`

```bash
pip install serializor
```

Install from `github`

```bash
pip install git+https://github.com/AresJef/Serializor.git
```

### Requirements

- Python 3.10 or higher.

### Features

This package is designed to serialize most of Python objects into <'bytes'>, and then deserializes them back to the original (or compatiable) object.

- Python natives:
  - Ser <'str'> -> Des <'str'>
  - Ser <'int'> -> Des <'int'>
  - Ser <'float'> -> Des <'float'>
  - Ser <'bool'> -> Des <'bool'>
  - Ser <'datetime.datetime'> -> Des <'datetime.datetime'> `[Supports Timezone]`
  - Ser <'datetime.date'> -> Des <'datetime.date'>
  - Ser <'datetime.time'> -> Des <'datetime.time'> `[Supports Timezone]`
  - Ser <'datetime.timedelta'> -> Des <'datetime.timedelta'>
  - Ser <'time.struct_time'> -> Des <'time.struct_time'>
  - Ser <'decimal.Decimal'> -> Des <'decimal.Decimal'>
  - Ser <'complex'> -> Des <'comples'>
  - Ser <'bytes'> -> Des <'bytes'>
  - Ser <'bytearray'> -> Des <'bytearray'>
  - Ser <'memoryview'> -> Des <'bytes'>
  - Ser <'list'> -> Des <'list'>
  - Ser <'tuple'> -> Des <'tuple'>
  - Ser <'set'> -> Des <'set'>
  - Ser <'frozenset'> -> Des <'frozenset'>
  - Ser <'range'> -> Des <'range'>
  - Ser <'deque'> -> Des <'list'>
  - Ser <'dict'> -> Des <'dict'>
- NumPy objects:
  - Ser <'np.str\_'> -> Des <'str'>
  - Ser <'np.int\*'> -> Des <'int'>
  - Ser <'np.uint\*'> -> Des <'int'>
  - Ser <'np.float\*'> -> Des <'float'>
  - Ser <'np.bool\_'> -> Des <'bool'>
  - Ser <'np.datetime64'> -> Des <'np.datetime64'>
  - Ser <'np.timedelta64'> -> Des <'np.timedelta64'>
  - Ser <'np.complex\*'> -> Des <'complex'>
  - Ser <'np.bytes\_'> -> Des <'bytes'>
  - Ser <'np.ndarray'> -> Des <'np.ndarray'> `[1-4 dimemsional]`
- Pandas objects:
  - Ser <'pd.Timestamp'> -> Des <'pd.Timestamp'> `[Supports Timezone]`
  - Ser <'pd.Timedelta'> -> Des <'pd.Timedelta'>
  - Ser <'pd.Series'> -> Des <'pd.Series'>
  - Ser <'pd.DatetimeIndex'> -> Des <'pd.DatetimeIndex'> `[Supports Timezone]`
  - Ser <'pd.TimedeltaIndex'> -> Des <'pd.TimedeltaIndex'>
  - Ser <'pd.DataFrame'> -> Des <'pd.DataFrame'>

### Benchmark

The following result comes from [benchmark](./src/benchmark.py):

- Device: MacbookPro M1Pro(2E8P) 32GB
- Python: 3.11.6

```
method      description                 rounds      ser_time    des_time    total_time
serializor  datetime.date               1,000,000   0.111606    0.054381    0.165986
serializor  datetime.datetime           1,000,000   0.156439    0.058737    0.215176
serializor  datetime.datetime.tz        1,000,000   0.279386    0.146563    0.425948
serializor  datetime.time               1,000,000   0.130660    0.053886    0.184547
serializor  datetime.timedelta          1,000,000   0.137054    0.058629    0.195683
serializor  decimal.Decimal             1,000,000   0.151619    0.164718    0.316336
serializor  native.None                 1,000,000   0.029402    0.032072    0.061473
serializor  native.bool                 1,000,000   0.029300    0.032557    0.061856
serializor  native.bytearray            1,000,000   0.163795    0.204271    0.368066
serializor  native.bytes.ascci          1,000,000   0.090823    0.059227    0.150049
serializor  native.bytes.utf-8          1,000,000   0.086925    0.063321    0.150246
serializor  native.complex              1,000,000   0.340286    0.109108    0.449394
serializor  native.float                1,000,000   0.148515    0.075095    0.223610
serializor  native.int                  1,000,000   0.126915    0.046609    0.173524
serializor  native.memoryview           1,000,000   0.116604    0.062738    0.179342
serializor  native.str.ascii            1,000,000   0.103631    0.065440    0.169071
serializor  native.str.utf-8            1,000,000   0.119890    0.103121    0.223011
serializor  np.bool_                    1,000,000   0.038970    0.032415    0.071385
serializor  np.bytes_                   1,000,000   0.085287    0.056964    0.142251
serializor  np.complex128               1,000,000   0.346779    0.104836    0.451615
serializor  np.datetime64               1,000,000   0.108729    0.131430    0.240160
serializor  np.float64                  1,000,000   0.175410    0.076764    0.252173
serializor  np.int64                    1,000,000   0.144910    0.046478    0.191387
serializor  np.int64                    1,000,000   0.149947    0.062591    0.212537
serializor  np.str_                     1,000,000   0.113398    0.066752    0.180150
serializor  np.timedelta64              1,000,000   0.108617    0.120959    0.229576
serializor  pd.Timedelta                1,000,000   0.102173    2.273358    2.375531
serializor  pd.Timestamp                1,000,000   0.143575    1.037143    1.180718
serializor  pd.Timestamp.tz             1,000,000   0.243033    2.138342    2.381375
serializor  time.struct_time            1,000,000   1.443414    1.181920    2.625334
serializor  native.dict[mixed]          100,000     0.166442    0.095385    0.261826
serializor  native.frozenset[mixed]     100,000     0.086782    0.105518    0.192300
serializor  native.list[mixed]          100,000     0.086708    0.049511    0.136219
serializor  native.range                100,000     0.044731    0.018418    0.063149
serializor  native.set[mixed]           100,000     0.088746    0.078731    0.167477
serializor  native.tuple[mixed]         100,000     0.084207    0.050072    0.134279
serializor  np.ndarray.bool.1dim        100,000     0.059088    0.038294    0.097381
serializor  np.ndarray.bool.2dim        100,000     0.095351    0.055688    0.151039
serializor  np.ndarray.bytes.1dim       100,000     0.067917    0.042592    0.110509
serializor  np.ndarray.bytes.2dim       100,000     0.125669    0.061103    0.186772
serializor  np.ndarray.complex128.1dim  100,000     0.101098    0.066359    0.167458
serializor  np.ndarray.complex128.2dim  100,000     0.182839    0.114994    0.297833
serializor  np.ndarray.datetime64.1dim  100,000     0.077447    0.062867    0.140315
serializor  np.ndarray.datetime64.2dim  100,000     0.117511    0.081945    0.199457
serializor  np.ndarray.float64.1dim     100,000     0.079962    0.043448    0.123410
serializor  np.ndarray.float64.2dim     100,000     0.151705    0.062965    0.214670
serializor  np.ndarray.int64.1dim       100,000     0.040866    0.040583    0.081449
serializor  np.ndarray.int64.2dim       100,000     0.075535    0.057880    0.133414
serializor  np.ndarray.object.1dim      100,000     0.120009    0.035457    0.155466
serializor  np.ndarray.object.2dim      100,000     0.228742    0.055797    0.284538
serializor  np.ndarray.timedelta64.1dim 100,000     0.078657    0.060962    0.139619
serializor  np.ndarray.timedelta64.2dim 100,000     0.118687    0.079886    0.198573
serializor  np.ndarray.unicode.1dim     100,000     0.032552    0.076028    0.108581
serializor  np.ndarray.unicode.2dim     100,000     0.071057    0.116673    0.187730
serializor  pd.DataFrame                10,000      0.241303    2.152159    2.393463
serializor  pd.DatetimeIndex            10,000      0.012195    0.078967    0.091162
serializor  pd.Series.bool              10,000      0.012649    0.107103    0.119751
serializor  pd.Series.bytes             10,000      0.012066    0.097440    0.109506
serializor  pd.Series.complex128        10,000      0.015063    0.112606    0.127668
serializor  pd.Series.datetime64        10,000      0.172289    0.193961    0.366250
serializor  pd.Series.float64           10,000      0.013266    0.180588    0.193854
serializor  pd.Series.int64             10,000      0.009387    0.109114    0.118501
serializor  pd.Series.object            10,000      0.017119    0.107831    0.124949
serializor  pd.Series.timedelta64       10,000      0.013114    0.170046    0.183160
serializor  pd.Series.unicode           10,000      0.013292    0.126546    0.139838
serializor  pd.TimedeltaIndex           10,000      0.010394    0.060776    0.071169
```

### Usage

#### Serialize & Deserialize

```python
import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
import numpy as np
import serializor

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
```

```python
se = serializor.serialize(obj)
print(se)
```

```
b'D\x12s\x03strs3Hello World!\n\xe4\xb8\xad\xe5\x9b\xbd\n\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4\n\xe3\x81\xab\xe3\x81\xbb\xe3\x82\x93\xe3\x81\x94\nEspa\xc3\xb1ols\x03inti\n1234567890s\x05floatf\x113.141592653589793s\x04boolo\x01s\x04nonens\x08datetimez\x00\xdc\x07\x01\x02\x03\x04\x05\x06\x00\x00\x00s\x0cdatetime.tz1z\x01\xe8\x07\x08\x1f\x05/\x15\x11t\x0e\x00\x03CETs\x0cdatetime.tz2z\x02\xe8\x07\x08\x1f\x0c/\x15\x1at\x0e\x00\x00\x00\x00\x00\x90~\x00\x00\x00\x00\x00\x00s\x04dated\xdc\x07\x01\x02s\x04timet\x00\x03\x04\x05\x06\x00\x00\x00s\ttimedeltal\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00s\x07decimale\x063.1234s\x07complexc\x072.12345\x073.12345s\x05bytesb3Hello World!\n\xe4\xb8\xad\xe5\x9b\xbd\n\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4\n\xe3\x81\xab\xe3\x81\xbb\xe3\x82\x93\xe3\x81\x94\nEspa\xc3\xb1ols\ndatetime64M\n\n\x80V/\xc8d\x9c\x12s\x0btimedelta64m\t}\x1b\n\xe2\xeb\xff\xff\xffs\tcomplex64c\x031.0\x031.0s\ncomplex128c\x04-1.0\x04-1.0'
```

```python
de = serializor.deserialize(se)
assert obj == de
print(de)
```

```
{'str': 'Hello World!\n中国\n한국어\nにほんご\nEspañol', 'int': 1234567890, 'float': 3.141592653589793, 'bool': True, 'none': None, 'datetime': datetime.datetime(2012, 1, 2, 3, 4, 5, 6), 'datetime.tz1': datetime.datetime(2024, 8, 31, 5, 47, 21, 947217, tzinfo=zoneinfo.ZoneInfo(key='CET')), 'datetime.tz2': datetime.datetime(2024, 8, 31, 12, 47, 21, 947226, tzinfo=datetime.timezone(datetime.timedelta(seconds=32400))), 'date': datetime.date(2012, 1, 2), 'time': datetime.time(3, 4, 5, 6), 'timedelta': datetime.timedelta(days=1, seconds=2, microseconds=3), 'decimal': Decimal('3.1234'), 'complex': (2.12345+3.12345j), 'bytes': b'Hello World!\n\xe4\xb8\xad\xe5\x9b\xbd\n\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4\n\xe3\x81\xab\xe3\x81\xbb\xe3\x82\x93\xe3\x81\x94\nEspa\xc3\xb1ol', 'datetime64': numpy.datetime64('2012-06-30T12:00:00.000000010'), 'timedelta64': numpy.timedelta64(-86402000003,'us'), 'complex64': (1+1j), 'complex128': (-1-1j)}
```

### Acknowledgements

SQLCyCli is based on the following open-source repositories:

- [cryptography](https://github.com/pyca/cryptography)
- [numpy](https://github.com/numpy/numpy)
- [orjson](https://github.com/ijl/orjson)
- [pandas](https://github.com/pandas-dev/pandas)
