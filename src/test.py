import time, unittest, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd


class TestCase(unittest.TestCase):
    name: str = "Case"
    # fmt: off
    data = {
        "str": "Hello World!\n中国\n한국어\nにほんご\nEspañol",
        "int": 1234567890,
        "float": 3.141592653589793,
        "bool": True,
        "none": None,
        "datetime": datetime.datetime(2012, 1, 2, 3, 4, 5, 6),
        "datetime.tz1": datetime.datetime.now(datetime.timezone.utc),
        "datetime.tz2": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))),
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
    # fmt: on

    def test_all(self) -> None:
        pass

    # Utils
    def log_start(self, msg: str) -> None:
        msg = "START TEST '%s': %s" % (self.name, msg)
        print(msg.ljust(60), end="\r")
        self._start_time = time.perf_counter()

    def log_ended(self, msg: str, skip: bool = False) -> None:
        self._ended_time = time.perf_counter()
        msg = "%s TEST '%s': %s" % ("SKIP" if skip else "PASS", self.name, msg)
        if self._start_time is not None:
            msg += " (%.6fs)" % (self._ended_time - self._start_time)
        print(msg.ljust(60))


class Test_Ser_Des(TestCase):
    name: str = "Ser/Des"

    def test_all(self) -> None:
        self.test_str()
        self.test_int()
        self.test_float()
        self.test_bool()
        self.test_datetime()
        self.test_date()
        self.test_time()
        self.test_timedelta()
        self.test_struct_time()
        self.test_decimal()
        self.test_complex()
        self.test_bytes()
        self.test_sequence()
        self.test_mapping()
        self.test_ndarray_series()
        self.test_series_like()
        self.test_dataframe()

    def test_str(self) -> None:
        test = "STR"
        self.log_start(test)

        for val in ("Hello World!", "中文", "한국어", "にほんご", "Español"):
            for dtype in (str, np.str_):
                se = serialize(dtype(val))
                de = deserialize(se)
                self.assertEqual(str(val), de)

        self.log_ended(test)

    def test_int(self) -> None:
        test = "INT"
        self.log_start(test)

        # Signed Int
        for val in (-1, 0, 1):
            for dtype in (int, np.int8, np.int16, np.int32, np.int64):
                se = serialize(dtype(val))
                de = deserialize(se)
                self.assertEqual(int(val), de)

        # Unsigned Int
        for val in (0, 1):
            for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
                se = serialize(dtype(val))
                de = deserialize(se)
                self.assertEqual(int(val), de)

        # Extreme
        for val in (
            -9223372036854775808,
            -9223372036854775807,
            0,
            9223372036854775806,
            9223372036854775807,
            18446744073709551613,
            18446744073709551614,
            18446744073709551615,
            18446744073709551615 * 1000,
        ):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)

        self.log_ended(test)

    def test_float(self) -> None:
        test = "FLOAT"
        self.log_start(test)

        # Var-types
        for val in (-1.1, 0, 1.1):
            for dtype in (float, np.float16, np.float32, np.float64):
                se = serialize(dtype(val))
                de = deserialize(se)
                self.assertEqual(float(val), de)

        # Extreme
        for val in (-2.718281828459045123, 3.141592653589793123):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)

        # Infinity
        for val in (float("inf"), float("-inf"), np.inf, -np.inf):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)

        # NaN
        for val in (float("nan"), np.nan):
            se = serialize(val)
            de = deserialize(se)
            self.assertTrue(np.isnan(de))

        self.log_ended(test)

    def test_bool(self) -> None:
        test = "BOOL"
        self.log_start(test)

        for val in (True, False):
            for dtype in (bool, np.bool_):
                se = serialize(dtype(val))
                de = deserialize(se)
                self.assertEqual(bool(val), de)

        self.log_ended(test)

    def test_none(self) -> None:
        test = "NONE"
        self.log_start(test)

        se = serialize(None)
        de = deserialize(se)
        self.assertEqual(None, de)

        self.log_ended(test)

    def test_datetime(self) -> None:
        from zoneinfo import ZoneInfo

        test = "DATETIME"
        self.log_start(test)

        # datetime
        for val in [
            datetime.datetime(1, 1, 2, 3, 4, 5),
            datetime.datetime(2013, 1, 2, 3, 4, 5, 6),
            datetime.datetime(9999, 1, 2, 3, 4, 5, 60000),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), datetime.datetime)

        # datetime with timezone
        offset = datetime.timedelta(hours=9)
        for val in [
            # fmt: off
            datetime.datetime.now(datetime.UTC),
            datetime.datetime.now(ZoneInfo("CET")),
            datetime.datetime(1970, 1, 2, 3, 4, 5, 6, datetime.timezone(offset), fold=0),
            datetime.datetime(1970, 1, 2, 3, 4, 5, 6, datetime.timezone(offset), fold=1),
            # fmt: on
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), datetime.datetime)

        # numpy.datetime64
        for val, unit in [
            (i, u)
            for i in (-2, -1, 0, 1, 2)
            for u in (
                "Y",
                "M",
                "W",
                "D",
                "h",
                "m",
                "s",
                "ms",
                "us",
                "ns",
                "ps",
                "fs",
                "as",
            )
        ]:
            se = serialize(np.datetime64(val, unit))
            de = deserialize(se)
            self.assertEqual(np.datetime64(val, unit), de)

        # pandas.Timestamp w/o timezone
        for val in [
            pd.Timestamp(pd.Timestamp.min),
            pd.Timestamp("2013-01-02 03:04:05"),
            pd.Timestamp(pd.Timestamp.max),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), pd.Timestamp)

        # pandas.Timestamp w/t timezone
        offset = datetime.timedelta(hours=9)
        for val in [
            pd.Timestamp(datetime.datetime.now(), tz="CET"),
            pd.Timestamp(datetime.datetime.now(), tz=ZoneInfo("CET")),
            pd.Timestamp(datetime.datetime.now(), tz=datetime.timezone(offset), fold=0),
            pd.Timestamp(datetime.datetime.now(), tz=datetime.timezone(offset), fold=1),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), pd.Timestamp)

        self.log_ended(test)

    def test_date(self) -> None:
        test = "DATE"
        self.log_start(test)

        for val in [datetime.date(y, 1, 1) for y in range(1, 10000)]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)

        self.log_ended(test)

    def test_time(self) -> None:
        test = "TIME"
        self.log_start(test)

        # Time w/o timezone
        for val in [
            datetime.time(1, 2, 3),
            datetime.time(1, 2, 3, 4),
            datetime.time(1, 2, 3, 40),
            datetime.time(1, 2, 3, 400),
            datetime.time(1, 2, 3, 4000),
            datetime.time(1, 2, 3, 40000),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), datetime.time)

        # Time w/t timezone
        offset = datetime.timedelta(hours=9)
        for val in [
            datetime.time(1, 2, 3, 4, datetime.UTC),
            datetime.time(1, 2, 3, 4, ZoneInfo("CET")),
            datetime.time(1, 2, 3, 4, datetime.timezone(offset), fold=0),
            datetime.time(1, 2, 3, 4, datetime.timezone(offset), fold=1),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), datetime.time)

        self.log_ended(test)

    def test_timedelta(self) -> None:
        test = "TIMEDELTA"
        self.log_start(test)

        # datetime.timedelta
        for val in [
            datetime.timedelta(1, 2, 3),
            datetime.timedelta(1, 2, 3, 4),
            datetime.timedelta(1, 2, 3, 40),
            datetime.timedelta(1, 2, 3, 400),
            datetime.timedelta(1, 2, 3, 4000),
            datetime.timedelta(1, 2, 3, 40000),
            -datetime.timedelta(1, 2, 3),
            -datetime.timedelta(1, 2, 3, 4),
            -datetime.timedelta(1, 2, 3, 40),
            -datetime.timedelta(1, 2, 3, 400),
            -datetime.timedelta(1, 2, 3, 4000),
            -datetime.timedelta(1, 2, 3, 40000),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), datetime.timedelta)

        # numpy.timedelta64
        for val, unit in [
            (i, u)
            for i in (-2, -1, 0, 1, 2)
            for u in (
                "Y",
                "M",
                "W",
                "D",
                "h",
                "m",
                "s",
                "ms",
                "us",
                "ns",
                "ps",
                "fs",
                "as",
            )
        ]:
            se = serialize(np.timedelta64(val, unit))
            de = deserialize(se)
            self.assertEqual(np.timedelta64(val, unit), de)
        self.assertEqual(type(de), np.timedelta64)

        # pandas.Timedelta
        for val in [
            pd.Timedelta(pd.Timedelta.min),
            pd.Timedelta("1 days 2 hours 3 minutes 4 seconds"),
            pd.Timedelta(pd.Timedelta.max),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
        self.assertEqual(type(de), pd.Timedelta)

        self.log_ended(test)

    def test_struct_time(self) -> None:
        test = "STRUCT_TIME"
        self.log_start(test)

        value = time.localtime()
        for val in [value, time.struct_time(value[:9])]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
            self.assertEqual(type(de), time.struct_time)
            self.assertEqual(val.tm_zone, de.tm_zone)
            self.assertEqual(val.tm_gmtoff, de.tm_gmtoff)

        self.log_ended(test)

    def test_decimal(self) -> None:
        test = "DECIMAL"
        self.log_start(test)

        for val in [
            Decimal("-1"),
            Decimal("0"),
            Decimal("1"),
            Decimal(
                "-3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408"
            ),
            Decimal(
                "3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067982148086513282306647093844609550582231725359408"
            ),
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)

        self.log_ended(test)

    def test_complex(self) -> None:
        test = "COMPLEX"
        self.log_start(test)

        # complex
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                val = complex(i, j)
                se = serialize(val)
                de = deserialize(se)
                self.assertEqual(val, de)

        # numpy.complex_
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for dtype in (np.complex64, np.complex128):
                    val = dtype(complex(i, j))
                    se = serialize(val)
                    de = deserialize(se)
                    self.assertEqual(val, de)

        self.log_ended(test)

    def test_bytes(self) -> None:
        test = "BYTES"
        self.log_start(test)

        for val in [
            b"Hello World!",
            "中国".encode("utf8"),
            "한국어".encode("utf8"),
            "にほんご".encode("utf8"),
            "Español".encode("utf8"),
        ]:
            for dtype in (bytes, bytearray, memoryview, np.bytes_):
                se = serialize(dtype(val))
                de = deserialize(se)
                self.assertEqual(val, de)
                if dtype is bytearray:
                    self.assertEqual(type(de), bytearray)
                else:
                    self.assertEqual(type(de), bytes)
        self.log_ended(test)

    def test_sequence(self) -> None:
        test = "SEQUENCE"
        self.log_start(test)

        # List & Tuple & Set & Frozenset
        seq = self.data.values()
        for seq in (self.data.values(), []):
            for dtype in (list, tuple, set, frozenset):
                val = dtype(seq)
                se = serialize(val)
                de = deserialize(se)
                self.assertEqual(val, de)
                self.assertEqual(type(val), type(de))
        # . nested
        for dtype in (list, tuple):
            v = dtype(self.data.values())
            val = dtype([v, v])
            se = serialize(dtype(val))
            de = deserialize(se)
            self.assertEqual(val, de)

        # Sequence dict.keys() & dict.values()
        for val in (
            self.data.keys(),
            self.data.values(),
            # . empty
            dict().keys(),
            dict().values(),
        ):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(list(val), de)

        # Range
        for val in (
            range(100),
            range(-100, 100),
            range(2, 100, 2),
            range(100, 1, -2),
        ):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val, de)
            self.assertEqual(type(val), type(de))

        self.log_ended(test)

    def test_mapping(self) -> None:
        from collections import OrderedDict

        test = "MAPPING"
        self.log_start(test)

        # Dict & dict_items & OrderedDict [Fast]
        for val in (self.data, self.data.items(), OrderedDict(self.data)):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(self.data, de)
        # . nested
        val = [self.data, self.data]
        se = serialize(val)
        de = deserialize(se)
        self.assertEqual(val, de)
        val = {str(i): self.data for i in range(2)}
        se = serialize(val)
        de = deserialize(se)
        self.assertEqual(val, de)

        # Dict [uncommon keys]
        val = {(k,): v for k, v in self.data.items()}
        se = serialize(val)
        de = deserialize(se)
        self.assertEqual(val, de)

        # . empty
        for val in (dict(), dict().items(), OrderedDict()):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(dict(), de)

        self.log_ended(test)

    def test_ndarray_series(self) -> None:
        test = "NDARRAY & SERIES"
        self.log_start(test)

        # fmt: off
        for l, dtype in [
            (list(self.data.values()), "O"),  # Object: 'O'
            *[([-1, 0, 1], d) for d in (np.int8, np.int16, np.int32, np.int64)],  # Integer: 'i'
            *[([0, 1, 10], d) for d in (np.uint8, np.uint16, np.uint32, np.uint64)],  # Unsigned Integer: 'u'
            *[([-1.1, 0, 1.1], d) for d in (np.float16, np.float32, np.float64)],  # Float: 'f'
            ([True, False, False, True], np.bool_),  # Boolean: 'b'
            *[([-2, -1, 0, 1, 2], d) for d in (
                "datetime64[%s]" % u for u in (
                    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ms", "ps", "fs", "as"))],  # Datetime64: 'M'
            *[([-2, -1, 0, 1, 2], d) for d in (
                "timedelta64[%s]" % u for u in (
                    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ms", "ps", "fs", "as"))],  # Timedelta64 'm'
            *[([1 + 1j, -1 + -1j], d) for d in (np.complex64, np.complex128)],  # Complex: 'c'
            ([b"Hello", b"World", "你好".encode("utf8")], "S"),  # Bytes: 'S'
            (["Hello", "World", "你好"], "U"),  # String: 'U'
            # . empty
            ([], "O"),  # Object: 'O'
            *[([], d) for d in (np.int8, np.int16, np.int32, np.int64)],  # Integer: 'i'
            *[([], d) for d in (np.uint8, np.uint16, np.uint32, np.uint64)],  # Unsigned Integer: 'u'
            *[([], d) for d in (np.float16, np.float32, np.float64)],  # Float: 'f'
            ([], np.bool_),  # Boolean: 'b'
            *[([], d) for d in (
                "datetime64[%s]" % u for u in (
                    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ms", "ps", "fs", "as"))],  # Datetime64: 'M'
            *[([], d) for d in (
                "timedelta64[%s]" % u for u in (
                    "Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ms", "ps", "fs", "as"))],  # Timedelta64 'm'
            *[([], d) for d in (np.complex64, np.complex128)],  # Complex: 'c'
            ([], "S"),  # Bytes: 'S'
            ([], "U"),  # String: 'U'
        ]:
            # . 1-dimensional
            arr = np.array(l, dtype=dtype)
            se = serialize(arr)
            de = deserialize(se)
            self.assertTrue(np.array_equal(arr, de))
            # . pd.Series
            s = pd.Series(arr)
            se = serialize(s)
            de = deserialize(se)
            self.assertTrue(s.equals(de))
            # . 2-dimensional
            arr = np.array([l, l], dtype=dtype)
            se = serialize(arr)
            de = deserialize(se)
            self.assertTrue(np.array_equal(arr, de))
            # . 3-dimensional
            arr = np.array([[l, l], [l, l]], dtype=dtype)
            se = serialize(arr)
            de = deserialize(se)
            self.assertTrue(np.array_equal(arr, de))
            # . 4-dimensional
            arr = np.array([[[l, l], [l, l]], [[l, l], [l, l]]], dtype=dtype)
            se = serialize(arr)
            de = deserialize(se)
            self.assertTrue(np.array_equal(arr, de))
        # fmt: on

        self.log_ended(test)

    def test_series_like(self) -> None:
        test = "SERIES LIKE"
        self.log_start(test)

        # pd.DatetimeIndex
        for name in [None, "", "s", b"", b"b", 1, 1.1]:
            for val in [
                # fmt: off
                # . DatetimeIndex
                pd.date_range("2021-01-01", periods=10, name=name),
                pd.date_range("2021-01-01", periods=10, name=name, tz="CET"),
                pd.date_range("2021-01-01", periods=10, name=name, tz=datetime.timezone.utc),
                pd.date_range("2021-01-01", periods=10, name=name, tz=datetime.timezone(datetime.timedelta(hours=9))),
                pd.DatetimeIndex([], name=name),
                # . TimdeltaIndex
                pd.timedelta_range("1 days", periods=10, name=name),
                pd.TimedeltaIndex([], name=name),
                # . Series Datetime
                pd.Series(pd.date_range("2021-01-01", periods=10, name=name)),
                pd.Series(pd.date_range("2021-01-01", periods=10, name=name, tz="CET")),
                pd.Series(pd.date_range("2021-01-01", periods=10, name=name, tz=datetime.timezone.utc)),
                pd.Series(pd.date_range("2021-01-01", periods=10, name=name, tz=datetime.timezone(datetime.timedelta(hours=9)))),
                pd.Series([], name=name),
                # fmt: on
            ]:
                se = serialize(val)
                de = deserialize(se)
                self.assertTrue(val.equals(de))
                self.assertEqual(val.name, de.name)
                self.assertEqual(type(val), type(de))

        # pd.Series[datetime64]
        for val in [
            pd.Series(np.array([1, 2, 3], dtype="datetime64[%s]" % u))
            for u in ["D", "h", "m", "s", "ms", "us", "ns"]
        ]:
            se = serialize(val)
            de = deserialize(se)
            self.assertTrue(val.equals(de))
            self.assertEqual(val.dtype, de.dtype)
            self.assertEqual(type(val), type(de))

        self.log_ended(test)

    def test_dataframe(self) -> None:
        test = "DATAFRAME"
        self.log_start(test)

        df = pd.DataFrame({k: [v] * 10 for k, v in self.data.items()})
        for val in (df, pd.DataFrame(columns=df.columns), pd.DataFrame()):
            se = serialize(val)
            de = deserialize(se)
            self.assertEqual(val.columns.tolist(), de.columns.tolist())
            self.assertTrue(val.equals(de))
            self.assertEqual(type(val), type(de))
        self.log_ended(test)


if __name__ == "__main__":
    # Test binary
    print(" Test 'binary' ".center(80, "-"))
    from serializor.binary import serialize, deserialize

    Test_Ser_Des().test_all()
    print()

    # Test unicode
    print(" Test 'unicode' ".center(80, "-"))
    from serializor.unicode import serialize, deserialize

    Test_Ser_Des().test_all()
    print()

    # Test unifunction
    print(" Test unifunction ".center(80, "-"))
    from serializor import serialize, deserialize

    obj = TestCase.data
    se = serialize(obj, to_bytes=True)
    assert type(se) is bytes
    assert deserialize(se) == obj

    se = serialize(obj, to_bytes=False)
    assert type(se) is str
    assert deserialize(se) == obj
    print("* PASS")
    print()

    # Test encrypt/decrypt
    print(" Test encrypt/decrypt ".center(80, "-"))
    from serializor import encrypt, decrypt

    obj = TestCase.data
    key = "show me the money!"
    en = encrypt(obj, key)
    assert type(en) is bytes
    assert decrypt(en, key) == obj
    print("* PASS")
    print()

    # Test utils
    print(" Test utils ".center(80, "-"))
    from serializor.utils import _test_utils

    _test_utils()

    print(" Test utils 'binary' ".center(80, "-"))
    from serializor.binary.ser import _test_utils

    _test_utils()

    from serializor.binary.des import _test_utils

    _test_utils()

    print(" Test utils 'unicode' ".center(80, "-"))
    from serializor.unicode.ser import _test_utils

    _test_utils()

    from serializor.unicode.des import _test_utils

    _test_utils()
