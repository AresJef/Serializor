import time
from decimal import Decimal
from zoneinfo import ZoneInfo
import datetime, numpy as np, pandas as pd
from json import dumps as json_dumps, loads as json_loads
from orjson import dumps as orjson_dumps, loads as orjson_loads, OPT_SERIALIZE_NUMPY
from serializor.binary import serialize as ser_bin, deserialize as des_bin
from serializor.unicode import serialize as ser_uni, deserialize as des_uni


class Benchmark:
    def __init__(self) -> None:
        self._descr: str = None
        self._stats: list[dict] = []

    @property
    def stats(self) -> pd.DataFrame:
        stats = (
            pd.DataFrame(self._stats)
            .sort_values(
                by=["rounds", "description", "total_time"],
                ascending=[False, True, True],
            )
            .reset_index(drop=True)
        )
        stats["rounds"] = stats["rounds"].map("{:,}".format)
        return stats

    def run(
        self,
        obj: object,
        descr: str,
        rounds: int = 1_000_000,
    ) -> None:
        self._descr = descr
        rounds = int(rounds)
        # self._run_json(obj, rounds)
        # self._run_orjson(obj, rounds)
        self._run_serializor_binary(obj, rounds)
        # self._run_serializor_unicode(obj, rounds)

    def _run_json(self, obj: object, rounds: int) -> None:
        try:
            en = json_dumps(obj)
        except Exception:
            return None
        # Serialize
        ser_t = time.perf_counter()
        [json_dumps(obj) for _ in range(rounds)]
        ser_t = time.perf_counter() - ser_t
        # Deserialize
        des_t = time.perf_counter()
        [json_loads(en) for _ in range(rounds)]
        des_t = time.perf_counter() - des_t
        # Record
        self._record_stats("json", obj, rounds, ser_t, des_t)

    def _run_orjson(self, obj: object, rounds: int) -> None:
        def _run_orjson_native(en: object) -> None:
            # Serialize
            ser_t = time.perf_counter()
            [orjson_dumps(obj) for _ in range(rounds)]
            ser_t = time.perf_counter() - ser_t
            # Deserialize
            des_t = time.perf_counter()
            [orjson_loads(en) for _ in range(rounds)]
            des_t = time.perf_counter() - des_t
            # Record
            self._record_stats("orjson (default)", obj, rounds, ser_t, des_t)

        def _run_orjson_numpy(en: object) -> None:
            # Serialize
            ser_t = time.perf_counter()
            [orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY) for _ in range(rounds)]
            ser_t = time.perf_counter() - ser_t
            # Deserialize
            des_t = time.perf_counter()
            [orjson_loads(en) for _ in range(rounds)]
            des_t = time.perf_counter() - des_t
            # Record
            self._record_stats("orjson (numpy)", obj, rounds, ser_t, des_t)

        try:
            en = orjson_dumps(obj)
            _run_orjson_native(en)
        except Exception:
            try:
                en = orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY)
                _run_orjson_numpy(en)
            except Exception:
                return None

    def _run_serializor_binary(self, obj: object, rounds: int) -> None:
        try:
            en = ser_bin(obj)
        except Exception:
            return None
        # Serialize
        ser_t = time.perf_counter()
        [ser_bin(obj) for _ in range(rounds)]
        ser_t = time.perf_counter() - ser_t
        # Deserialize
        des_t = time.perf_counter()
        [des_bin(en) for _ in range(rounds)]
        des_t = time.perf_counter() - des_t
        # Record
        self._record_stats("serializor (binary)", obj, rounds, ser_t, des_t)

    def _run_serializor_unicode(self, obj: object, rounds: int) -> None:
        try:
            en = ser_uni(obj)
        except Exception:
            return None
        # Serialize
        ser_t = time.perf_counter()
        [ser_uni(obj) for _ in range(rounds)]
        ser_t = time.perf_counter() - ser_t
        # Deserialize
        des_t = time.perf_counter()
        [des_uni(en) for _ in range(rounds)]
        des_t = time.perf_counter() - des_t
        # Record
        self._record_stats("serializor (unicode)", obj, rounds, ser_t, des_t)

    def _record_stats(
        self,
        method: str,
        obj: object,
        rounds: int,
        ser_t: float,
        des_t: float,
    ) -> None:
        self._stats.append(
            {
                "method": method,
                "description": self._descr,
                "rounds": rounds,
                "ser_time": ser_t,
                "des_time": des_t,
                "total_time": ser_t + des_t,
            }
        )


if __name__ == "__main__":
    data = {
        "str": "Hello World!\n中国\n한국어\nにほんご\nEspañol",
        "int": 1234567890,
        "float": 3.141592653589793,
        "bool": True,
        "none": None,
        "datetime": datetime.datetime(2012, 1, 2, 3, 4, 5, 6),
        "date": datetime.date(2012, 1, 2),
        "time": datetime.time(3, 4, 5, 6),
    }
    arr = list(range(10))
    r1m = 1_000_000
    r100k = 100_000
    r10k = 10_000

    # fmt: off
    benchmark = Benchmark()
    for obj, descr, rounds in [
        # . native single
        ("Hello, World!", "native.str.ascii", r1m),
        ("中文한국어にほんご", "native.str.utf-8", r1m),
        (1, "native.int", r1m),
        (1.1, "native.float", r1m),
        (True, "native.bool", r1m),
        (None, "native.None", r1m),
        (datetime.datetime.now(), "datetime.datetime", r1m),
        (datetime.datetime.now().astimezone(ZoneInfo("CET")), "datetime.datetime.tz", r1m),
        (datetime.datetime.now().date(), "datetime.date", r1m),
        (datetime.datetime.now().time(), "datetime.time", r1m),
        (datetime.timedelta(1, 2, 3), "datetime.timedelta", r1m),
        (time.localtime(), "time.struct_time", r1m),
        (Decimal("1.23"), "decimal.Decimal", r1m),
        (1 + 2j, "native.complex", r1m),
        (b"Hello, World!", "native.bytes.ascci", r1m),
        ("中文한국어にほんご".encode("utf8"), "native.bytes.utf-8", r1m),
        (bytearray(b"Hello, World!"), "native.bytearray", r1m),
        (memoryview(b"Hello, World!"), "native.memoryview", r1m),
        # . sequence
        (list(data.values()), "native.list[mixed]", r100k),
        (tuple(data.values()), "native.tuple[mixed]", r100k),
        (set(data.values()), "native.set[mixed]", r100k),
        (frozenset(data.values()), "native.frozenset[mixed]", r100k),
        (range(1, 100, 2), "native.range", r100k),
        # . mapping
        (data, "native.dict[mixed]", r100k),
        # . ndarray
        (np.str_("Hello, World!"), "np.str_", r1m),
        (np.int64(-1), "np.int64", r1m),
        (np.uint64(1), "np.int64", r1m),
        (np.float64(1.1), "np.float64", r1m),
        (np.bool_(True), "np.bool_", r1m),
        (np.datetime64("2021-01-01"), "np.datetime64", r1m),
        (np.timedelta64(1, "D"), "np.timedelta64", r1m),
        (np.complex128(1 + 2j), "np.complex128", r1m),
        (np.bytes_("Hello, World!"), "np.bytes_", r1m),
        (np.array(arr, dtype="O"), "np.ndarray.object.1dim", r100k),
        (np.array([arr,arr], dtype="O"), "np.ndarray.object.2dim", r100k),
        (np.array(arr, dtype=np.int64), "np.ndarray.int64.1dim", r100k),
        (np.array([arr,arr], dtype=np.int64), "np.ndarray.int64.2dim", r100k),
        (np.array(arr, dtype=np.float64), "np.ndarray.float64.1dim", r100k),
        (np.array([arr,arr], dtype=np.float64), "np.ndarray.float64.2dim", r100k),
        (np.array(arr, dtype=np.bool_), "np.ndarray.bool.1dim", r100k),
        (np.array([arr,arr], dtype=np.bool_), "np.ndarray.bool.2dim", r100k),
        (np.array(arr, dtype="datetime64[ns]"), "np.ndarray.datetime64.1dim", r100k),
        (np.array([arr,arr], dtype="datetime64[ns]"), "np.ndarray.datetime64.2dim", r100k),
        (np.array(arr, dtype="timedelta64[ns]"), "np.ndarray.timedelta64.1dim", r100k),
        (np.array([arr,arr], dtype="timedelta64[ns]"), "np.ndarray.timedelta64.2dim", r100k),
        (np.array(arr, dtype=np.complex128), "np.ndarray.complex128.1dim", r100k),
        (np.array([arr,arr], dtype=np.complex128), "np.ndarray.complex128.2dim", r100k),
        (np.array(arr, dtype="S"), "np.ndarray.bytes.1dim", r100k),
        (np.array([arr,arr], dtype="S"), "np.ndarray.bytes.2dim", r100k),
        (np.array(arr, dtype="U"), "np.ndarray.unicode.1dim", r100k),
        (np.array([arr,arr], dtype="U"), "np.ndarray.unicode.2dim", r100k),
        # . Pandas date&time
        (pd.Timestamp(datetime.datetime.now()), "pd.Timestamp", r1m),
        (pd.Timestamp(datetime.datetime.now(), tz="CET"), "pd.Timestamp.tz", r1m),
        (pd.Timedelta(1, unit="D"), "pd.Timedelta", r1m),
        # . Series
        (pd.Series(np.array(arr, dtype="O")), "pd.Series.object", r10k),
        (pd.Series(np.array(arr, dtype=np.int64)), "pd.Series.int64", r10k),
        (pd.Series(np.array(arr, dtype=np.float64)), "pd.Series.float64", r10k),
        (pd.Series(np.array(arr, dtype=np.bool_)), "pd.Series.bool", r10k),
        (pd.Series(np.array(arr, dtype="datetime64[ns]")), "pd.Series.datetime64", r10k),
        (pd.Series(np.array(arr, dtype="timedelta64[ns]")), "pd.Series.timedelta64", r10k),
        (pd.Series(np.array(arr, dtype=np.complex128)), "pd.Series.complex128", r10k),
        (pd.Series(np.array(arr, dtype="S")), "pd.Series.bytes", r10k),
        (pd.Series(np.array(arr, dtype="U")), "pd.Series.unicode", r10k),
        (pd.date_range("2021-01-01", periods=10), "pd.DatetimeIndex", r10k),
        (pd.timedelta_range("1 days", periods=10), "pd.TimedeltaIndex", r10k),
        # . DataFrame
        (pd.DataFrame({k: [v] * 10 for k, v in data.items()}), "pd.DataFrame", r10k),
    ]:
        benchmark.run(obj, descr, rounds)
    # fmt: on

    pd.options.display.max_rows = 1_000_000
    print(benchmark.stats)
