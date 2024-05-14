import time
from timeit import timeit
from decimal import Decimal
from zoneinfo import ZoneInfo
import datetime, numpy as np, pandas as pd
from json import dumps as json_dumps, loads as json_loads
from orjson import dumps as orjson_dumps, loads as orjson_loads, OPT_SERIALIZE_NUMPY
from serializor.serialize import serialize
from serializor.deserialize import deserialize
from serializor.crypto import encrypt, decrypt


def diff(base_t: float, comp_t: float) -> str:
    """Calculate performance difference."""
    if base_t < comp_t:
        res = (comp_t - base_t) / base_t
    else:
        res = -(base_t - comp_t) / comp_t
    return ("" if res < 0 else "+") + f"{res:.6f}x"


def benchmark_simple_object(
    obj: object,
    rounds: int,
    mode: int = 0,
    validate: bool = True,
) -> None:
    """Benchmark for simple python object."""
    # Display ------------------------------------------------------
    print(f" Benchmark for {type(obj)} ".center(100, "-"))
    print(f"- OBJECT: {repr(obj)} {type(obj)}")
    print()

    # Encodeing ----------------------------------------------------
    print(f"[encode] - rounds:\t# {rounds:,}")
    t_ser1 = timeit(lambda: serialize(obj), number=rounds)
    r_ser1 = serialize(obj)
    print(f"[encode] - serializor:\t{t_ser1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(
            lambda: orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY), number=rounds
        )
        r_orjs = orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY)
        print(f"[encode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_ser1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_dumps(obj), number=rounds)
        r_json = json_dumps(obj)
        print(f"[encode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_ser1, t_json)}")
    print(f"- Serialized:\t{repr(r_ser1)}")
    print()

    # Decodeing ----------------------------------------------------
    print(f"[decode] - rounds:\t# {rounds:,}")
    t_des1 = timeit(lambda: deserialize(r_ser1), number=rounds)
    r_des1 = deserialize(r_ser1)
    print(f"[decode] - serializor:\t{t_des1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(lambda: orjson_loads(r_orjs), number=rounds)
        r_orjs = orjson_loads(r_orjs)
        print(f"[decode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_des1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_loads(r_json), number=rounds)
        r_json = json_loads(r_json)
        print(f"[decode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_des1, t_json)}")
    print(f"- Deserialized:\t{type(r_des1)}\n{repr(r_des1)} ")
    print()

    # Validate -----------------------------------------------------
    print("- EQUALS:", eq := obj == r_des1)
    if validate:
        assert eq
    print()


def benchmark_sequence(
    obj: object,
    rounds: int,
    mode: int = 0,
    validate: bool = True,
    debug: bool = True,
) -> None:
    """Benchmark for python sequence."""
    # Data types ---------------------------------------------------
    dtypes = {type(i) for i in obj}
    if (length := len(dtypes)) == 1:
        dtype = dtypes.pop()
    elif length == 0:
        dtype = "empty"
    elif (
        dict in dtypes
        or list in dtypes
        or set in dtypes
        or tuple in dtypes
        or np.ndarray in dtypes
        or pd.Series in dtypes
        or pd.DataFrame in dtypes
    ):
        dtype = "nested"
    else:
        dtype = "mixed"

    # Display ------------------------------------------------------
    print(f" Benchmark for {type(obj)}[{dtype}] ".center(100, "-"))
    print(f"- OBJECT: {type(obj)}\n{repr(obj)}")
    print()

    # Encodeing ----------------------------------------------------
    print(f"[encode] - rounds:\t# {rounds:,}")
    t_ser1 = timeit(lambda: serialize(obj), number=rounds)
    r_ser1 = serialize(obj)
    print(f"[encode] - serializor:\t{t_ser1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(
            lambda: orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY), number=rounds
        )
        r_orjs = orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY)
        print(f"[encode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_ser1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_dumps(obj), number=rounds)
        r_json = json_dumps(obj)
        print(f"[encode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_ser1, t_json)}")
    if debug:
        print(f"- Serialized:\n{repr(r_ser1)}")
    print()

    # Decodeing ----------------------------------------------------
    print(f"[decode] - rounds:\t# {rounds:,}")
    t_des1 = timeit(lambda: deserialize(r_ser1), number=rounds)
    r_des1 = deserialize(r_ser1)
    print(f"[decode] - serializor:\t{t_des1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(lambda: orjson_loads(r_orjs), number=rounds)
        r_orjs = orjson_loads(r_orjs)
        print(f"[decode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_des1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_loads(r_json), number=rounds)
        r_json = json_loads(r_json)
        print(f"[decode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_des1, t_json)}")
    if debug:
        print(f"- Deserialized:\t{type(r_des1)}\n{repr(r_des1)} ")
    print()

    # Validate -----------------------------------------------------
    print("- EQUALS:", eq := obj == r_des1)
    if validate:
        assert eq
    print()


def benchmark_dict(
    obj: dict,
    rounds: int,
    mode: int = 0,
    validate: bool = True,
    debug: bool = True,
) -> None:
    """Benchmark for python dict."""
    # Data types ---------------------------------------------------
    dtypes = {type(v) for v in obj.values()}
    if (length := len(dtypes)) == 1:
        dtype = dtypes.pop()
    elif length == 0:
        dtype = "empty"
    elif (
        dict in dtypes
        or list in dtypes
        or set in dtypes
        or tuple in dtypes
        or np.ndarray in dtypes
        or pd.Series in dtypes
        or pd.DataFrame in dtypes
    ):
        dtype = "nested"
    else:
        dtype = "mixed"

    # Display ------------------------------------------------------
    print(f" Benchmark for {type(obj)}[{dtype}] ".center(100, "-"))
    print(f"- OBJECT: {type(obj)}\n{repr(obj)}")
    print()

    # Encodeing ----------------------------------------------------
    print(f"[encode] - rounds:\t# {rounds:,}")
    t_ser1 = timeit(lambda: serialize(obj), number=rounds)
    r_ser1 = serialize(obj)
    print(f"[encode] - serializor:\t{t_ser1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(
            lambda: orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY), number=rounds
        )
        r_orjs = orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY)
        print(f"[encode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_ser1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_dumps(obj), number=rounds)
        r_json = json_dumps(obj)
        print(f"[encode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_ser1, t_json)}")
    if debug:
        print(f"- Serialized:\n{repr(r_ser1)}")
    print()

    # Decodeing ----------------------------------------------------
    print(f"[decode] - rounds:\t# {rounds:,}")
    t_des1 = timeit(lambda: deserialize(r_ser1), number=rounds)
    r_des1 = deserialize(r_ser1)
    print(f"[decode] - serializor:\t{t_des1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(lambda: orjson_loads(r_orjs), number=rounds)
        r_orjs = orjson_loads(r_orjs)
        print(f"[decode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_des1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_loads(r_json), number=rounds)
        r_json = json_loads(r_json)
        print(f"[decode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_des1, t_json)}")
    if debug:
        print(f"- Deserialized:\t{type(r_des1)}\n{repr(r_des1)} ")
    print()

    # Validate -----------------------------------------------------
    print("- EQUALS:", eq := obj == r_des1)
    if validate:
        assert eq
    print()


def benchmark_ndarray(
    obj: np.ndarray,
    rounds: int,
    mode: int = 0,
    validate: bool = True,
    debug: bool = True,
) -> None:
    """Benchmark for python sequence."""
    # Display ------------------------------------------------------
    print(f" Benchmark for {type(obj)}[{obj.dtype}] ".center(100, "-"))
    print(f"- OBJECT: {type(obj)}[{obj.dtype}]\n{repr(obj)}")
    print()

    # Encodeing ----------------------------------------------------
    print(f"[encode] - rounds:\t# {rounds:,}")
    t_ser1 = timeit(lambda: serialize(obj), number=rounds)
    r_ser1 = serialize(obj)
    print(f"[encode] - serializor:\t{t_ser1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(
            lambda: orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY), number=rounds
        )
        r_orjs = orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY)
        print(f"[encode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_ser1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_dumps(obj), number=rounds)
        r_json = json_dumps(obj)
        print(f"[encode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_ser1, t_json)}")
    if debug:
        print(f"- Serialized:\n{repr(r_ser1)}")
    print()

    # Decodeing ----------------------------------------------------
    print(f"[decode] - rounds:\t# {rounds:,}")
    t_des1 = timeit(lambda: deserialize(r_ser1), number=rounds)
    r_des1 = deserialize(r_ser1)
    print(f"[decode] - serializor:\t{t_des1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(lambda: orjson_loads(r_orjs), number=rounds)
        r_orjs = orjson_loads(r_orjs)
        print(f"[decode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_des1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_loads(r_json), number=rounds)
        r_json = json_loads(r_json)
        print(f"[decode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_des1, t_json)}")
    if debug:
        print(f"- Deserialized:\t{type(r_des1)}\n{repr(r_des1)} ")
    print()

    # Validate -----------------------------------------------------
    print("- EQUALS:", eq := (obj == r_des1).all())
    if validate:
        assert eq
    print()


def benchmark_dataframe(
    obj: np.ndarray,
    rounds: int,
    mode: int = 0,
    validate: bool = True,
    debug: bool = True,
) -> None:
    """Benchmark for python sequence."""
    # Display ------------------------------------------------------
    print(f" Benchmark for {type(obj)} ".center(100, "-"))
    print(f"- OBJECT: {type(obj)}\n{repr(obj)}")
    print()

    # Encodeing ----------------------------------------------------
    print(f"[encode] - rounds:\t# {rounds:,}")
    t_ser1 = timeit(lambda: serialize(obj), number=rounds)
    r_ser1 = serialize(obj)
    print(f"[encode] - serializor:\t{t_ser1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(
            lambda: orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY), number=rounds
        )
        r_orjs = orjson_dumps(obj, option=OPT_SERIALIZE_NUMPY)
        print(f"[encode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_ser1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_dumps(obj), number=rounds)
        r_json = json_dumps(obj)
        print(f"[encode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_ser1, t_json)}")
    if debug:
        print(f"- Serialized:\n{repr(r_ser1)}")
    print()

    # Decodeing ----------------------------------------------------
    print(f"[decode] - rounds:\t# {rounds:,}")
    t_des1 = timeit(lambda: deserialize(r_ser1), number=rounds)
    r_des1 = deserialize(r_ser1)
    print(f"[decode] - serializor:\t{t_des1:.9f}s")
    if mode >= 1:
        t_orjs = timeit(lambda: orjson_loads(r_orjs), number=rounds)
        r_orjs = orjson_loads(r_orjs)
        print(f"[decode] - orjson:\t{t_orjs:.9f}s\tdiff: {diff(t_des1, t_orjs)}")
    if mode >= 2:
        t_json = timeit(lambda: json_loads(r_json), number=rounds)
        r_json = json_loads(r_json)
        print(f"[decode] - python.json:\t{t_json:.9f}s\tdiff: {diff(t_des1, t_json)}")
    if debug:
        print(f"- Deserialized:\t{type(r_des1)}\n{repr(r_des1)} ")
    print()

    # Validate -----------------------------------------------------
    print("- EQUALS:", eq := obj.equals(r_des1))
    if validate:
        assert eq
    print()


def benchmark() -> None:
    timeit(lambda: orjson_dumps(1024), number=1_000_000)

    run_basic_types = 1
    run_datetime_types = 1
    run_numeric_types = 1
    run_bytes_types = 1
    run_list = 1
    run_sequence_types = 1
    run_dict = 1
    run_ndarray = 1
    run_series = 1
    run_dataframe = 1
    run_pdindex = 1
    run_nested = 1
    run_crypto = 1

    # Basic Types
    if run_basic_types:
        rounds = 1_000_000

        benchmark_simple_object("Hello World", rounds, 2)  # `<str>`
        benchmark_simple_object(3.1415926, rounds, 2)  # `<float>`
        benchmark_simple_object(np.float64(-3.1415926), rounds, 2)  # `<np.float64>`
        benchmark_simple_object(float("inf"), rounds, 2)  # `<float> inf`
        benchmark_simple_object(float("nan"), rounds, 2, False)  # `<float> nan`
        benchmark_simple_object(np.nan, rounds, 2, False)  # `<np.nan>`
        benchmark_simple_object(1024, rounds, 2)  # `<int>`
        benchmark_simple_object(np.int64(-1024), rounds, 1)  # `<np.int64>`
        benchmark_simple_object(np.uint64(1024), rounds, 1)  # `<np.int64>`
        benchmark_simple_object(True, rounds, 2)  # `<bool>`
        benchmark_simple_object(None, rounds, 2)  # `<NoneType>`

    # Date&Time Types
    if run_datetime_types:
        rounds = 1_000_000

        # fmt: off
        benchmark_simple_object(
            datetime.datetime(1970, 1, 1, 1, 1, 1, 1), rounds, 1)  # `<datetime.datetime>`
        benchmark_simple_object(
            datetime.datetime(1970, 1, 1, 1, 1, 1, 1, ZoneInfo("CET")), rounds, 1)  # `<datetime.datetime>` & tzinfo
        benchmark_simple_object(
            np.datetime64("1970-01-01 01:01:01.000001"), rounds, 1)  # `<np.datetime64>`
        benchmark_simple_object(datetime.date(1970, 1, 1), rounds, 1) # `<datetime.date>`
        benchmark_simple_object(datetime.time(1, 1, 1, 1), rounds, 1) # `<datetime.time>`
        benchmark_simple_object(datetime.timedelta(1, 1, 1, 1, 1, 1, 1), rounds, 0) # `<datetime.timedelta>`
        benchmark_simple_object(np.timedelta64(1024, "us"), rounds, 0) # `<np.timedelta64>`
        benchmark_simple_object(
            time.struct_time((1970, 1, 1, 1, 1, 1, 1, 1, 0)), rounds, 0, False) # `<time.struct_time>`
        # fmt: on

    # Numeric Types
    if run_numeric_types:
        rounds = 1_000_000

        benchmark_simple_object(Decimal("3.1415926"), rounds, 0)  # `<decimal.Decimal>`
        benchmark_simple_object(1 + 1j, rounds, 0)  # `<complex>`
        benchmark_simple_object(np.complex128(1 + 1j), rounds, 0)  # `<np.complex128>`

    # Bytes Types
    if run_bytes_types:
        rounds = 1_000_000

        # fmt: off
        val = b"serializor's bytes"
        val = "中国".encode("utf-8")
        benchmark_simple_object(val, rounds, 0) # `<bytes>`
        benchmark_simple_object(bytearray(val), rounds, 0) # `<bytearray>`
        benchmark_simple_object(memoryview(val), rounds, 0) # `<memoryview>`
        benchmark_simple_object(np.bytes_(val), rounds, 0) # `<memoryview>`
        # fmt: on

    # List
    if run_list:
        rounds = 1_000_000

        # fmt: off
        benchmark_sequence([], rounds, 2)  # empty
        obj = [i for i in range(10)]
        benchmark_sequence([str(i) for i in obj], rounds, 2)  # `<str>`
        benchmark_sequence([float(i) for i in obj], rounds, 2)  # `<float>`
        benchmark_sequence([np.float64(i) for i in obj], rounds, 2) # `<np.float64>`
        benchmark_sequence([int(i) for i in obj], rounds, 2)  # `<int>`
        benchmark_sequence([np.int64(i) for i in obj], rounds, 1) # `<np.int64>`
        benchmark_sequence([np.uint64(i) for i in obj], rounds, 1) # `<np.uint64>`
        benchmark_sequence([bool(i) for i in obj], rounds, 2)  # `<bool>`
        benchmark_sequence([None] * 10, rounds, 2)  # `<NoneType>`
        obj = [datetime.datetime(1970, 1, 1, 1, 1, 1, 1)] * 10
        benchmark_sequence(obj, rounds, 1) # `<datetime.datetime>`
        obj = [datetime.datetime(1970, 1, 1, 1, 1, 1, 1, ZoneInfo("CET"))] * 10
        benchmark_sequence(obj, rounds, 1, debug=False) # `<datetime.datetime>` & tzinfo
        benchmark_sequence([datetime.date(1970, 1, 1)] * 10, rounds, 1) # `<datetime.date>`
        benchmark_sequence([datetime.time(1, 1, 1, 1)] * 10, rounds, 1) # `<datetime.time>`
        benchmark_sequence([datetime.timedelta(1, 2, 3)] * 10, rounds, 0) # `<datetime.timedelta>`
        benchmark_sequence([Decimal("3.1415926")] * 10, rounds, 0)  # `<decimal.Decimal>`
        benchmark_sequence([1 + 1j] * 10, rounds, 0)  # `<complex>`
        benchmark_sequence([b"Hello"] * 10, rounds, 0)  # `<bytes>`
        benchmark_sequence([bytearray(b"Hello")] * 10, rounds, 0) # `<bytearray>`
        benchmark_sequence([memoryview(b"Hello")] * 10, rounds, 0) # `<memoryview>`
        obj = [np.datetime64("1970-01-01 01:01:01.000001")] * 10
        benchmark_sequence(obj, rounds, 1) # `<np.datetime64>`
        obj = [np.timedelta64(1024, "us")] * 10
        benchmark_sequence(obj, rounds, 0) # `<np.timedelta64>`
        # fmt: on
        obj = [
            "serializor's string",
            3.1415926,
            np.float64(-3.1415926),
            1024,
            np.int64(-1024),
            np.uint64(1024),
            True,
            None,
            datetime.datetime(1970, 1, 1, 1, 1, 1, 1),
            datetime.datetime(1970, 1, 1, 1, 1, 1, 1, ZoneInfo("CET")),
            np.datetime64("1970-01-01 01:01:01.000001"),
            datetime.date(1970, 1, 1),
            datetime.time(1, 1, 1, 1),
            datetime.timedelta(1, 1, 1, 1, 1, 1, 1),
            np.timedelta64(1024, "us"),
            Decimal("3.1415926"),
            1 + 1j,
            np.complex128(1 + 1j),
            b"serializor's bytes",
            bytearray(b"serializor's bytes"),
            memoryview(b"serializor's bytes"),
        ]
        benchmark_sequence(obj, rounds, 0)  # mixed
        obj = [obj] * 10
        benchmark_sequence(obj, int(rounds / 10), 0, debug=False)  # nested

    # Sequence Types
    if run_sequence_types:
        rounds = 1_000_000

        obj = [
            "serializor's string",
            3.1415926,
            np.float64(-3.1415926),
            1024,
            np.int64(-1024),
            np.uint64(1024),
            True,
            datetime.datetime(1970, 1, 1, 1, 1, 1, 1),
            datetime.datetime(1970, 1, 1, 1, 1, 1, 1, ZoneInfo("CET")),
            np.datetime64("1970-01-01 01:01:01.000001"),
            datetime.date(1970, 1, 1),
            datetime.time(1, 1, 1, 1),
            datetime.timedelta(1, 1, 1, 1, 1, 1, 1),
            np.timedelta64(1024, "us"),
            Decimal("3.1415926"),
            1 + 1j,
            np.complex128(1 + 1j),
        ]
        benchmark_sequence(tuple(), rounds, 2)  # empty
        benchmark_sequence(set(), rounds, 0)  # empty
        benchmark_sequence(frozenset(), rounds, 0)  # empty
        benchmark_sequence({}.keys(), rounds, 0, False)  # empty
        benchmark_sequence(tuple(obj), rounds, 0)  # `<tuple>`
        benchmark_sequence(set(obj), rounds, 0)  # `<set>`
        benchmark_sequence(frozenset(obj), rounds, 0)  # `<frozenset>`
        benchmark_sequence(
            {i: v for i, v in enumerate(obj)}.values(), rounds, 0, False, False
        )  # `<dick_values>`

    # Dict
    if run_dict:
        rounds = 1_000_000

        # fmt: off
        benchmark_dict({}, rounds, 2)  # `<str>`
        obj = {str(i): i for i in range(10)}
        benchmark_dict({k: str(v) for k, v in obj.items()}, rounds, 2)  # `<str>`
        benchmark_dict({k: float(v) for k, v in obj.items()}, rounds, 2)  # `<float>`
        benchmark_dict({k: np.float64(v) for k, v in obj.items()}, rounds, 2)  # `<np.float64>`
        benchmark_dict({k: int(v) for k, v in obj.items()}, rounds, 2)  # `<int>`
        benchmark_dict({k: np.int64(v) for k, v in obj.items()}, rounds, 1)  # `<np.int64>`
        benchmark_dict({k: np.uint64(v) for k, v in obj.items()}, rounds, 1)  # `<np.uint64>`
        benchmark_dict({k: bool(v) for k, v in obj.items()}, rounds, 2)  # `<bool>`
        benchmark_dict({k: None for k, v in obj.items()}, rounds, 2)  # `<NoneType>`
        obj = {k: datetime.datetime(1970, 1, 1, 1, 1, 1, 1) for k in obj}
        benchmark_dict(obj, rounds, 1)  # `<datetime.datetime>`
        obj = {k: datetime.datetime(1970, 1, 1, 1, 1, 1, 1, ZoneInfo("CET")) for k in obj}
        benchmark_dict(obj, rounds, 1, debug=False)  # `<datetime.datetime>` & tzinfo
        benchmark_dict({k: datetime.date(1970, 1, 1) for k in obj}, rounds, 1)  # `<datetime.date>`
        benchmark_dict({k: datetime.time(1, 1, 1, 1) for k in obj}, rounds, 1)  # `<datetime.time>`
        benchmark_dict({k: datetime.timedelta(1, 2, 3) for k in obj}, rounds, 0)  # `<datetime.timedelta>`
        benchmark_dict({k: Decimal("3.1415926") for k in obj}, rounds, 0)  # `<datetime.Decimal>`
        benchmark_dict({k: 1 + 1j for k in obj}, rounds, 0)  # `<complex>`
        benchmark_dict({k: b"Hello" for k in obj}, rounds, 0)  # `<bytes>`
        benchmark_dict({k: bytearray(b"Hello") for k in obj}, rounds, 0)  # `<bytearray>`
        benchmark_dict({k: memoryview(b"Hello") for k in obj}, rounds, 0)  # `<memoryview>`
        obj = {k: np.datetime64("1970-01-01 01:01:01.000001") for k in obj}
        benchmark_dict(obj, rounds, 1)  # `<np.datetime64>`
        obj = {k: np.timedelta64(1024, "us") for k in obj}
        benchmark_dict(obj, rounds, 0)  # `<np.timedelta64>`
        # fmt: on
        obj = {
            "strx": ["apple's|banana's|\"apple\"'s|" for i in range(10)],
            "str": [str(i) for i in range(10)],
            "float": [float(i) for i in range(10)],
            "np.float64": [np.float64(i) for i in range(10)],
            "int": [i for i in range(10)],
            "np.int64": [np.int64(i) for i in range(10)],
            "np.uint64": [np.uint64(i) for i in range(10)],
            "bool": [bool(i) for i in range(10)],
            "None": [None] * 10,
            "datetime": [datetime.datetime(1970, 1, 1, 1, 1, 1, 1) for _ in range(10)],
            "datetime_tz": [
                datetime.datetime(1970, 1, 1, 1, 1, 1, 1, ZoneInfo("CET"))
                for _ in range(10)
            ],
            "np.datetime64": [
                np.datetime64("1970-01-01 01:01:01.000001") for _ in range(10)
            ],
            "date": [datetime.date(1970, 1, 1) for _ in range(10)],
            "time": [datetime.time(1, 1, 1, 1) for _ in range(10)],
            "timedelta": [datetime.timedelta(1, 1, 1, 1, 1, 1, 1) for _ in range(10)],
            "np.timedelta64": [np.timedelta64(1024, "us") for _ in range(10)],
            "decimal": [Decimal("3.1415926") for _ in range(10)],
            "complex": [1 + 1j for _ in range(10)],
            "np.complex128": [np.complex128(1 + 1j) for _ in range(10)],
            "bytes": [b"serializor's bytes" for _ in range(10)],
            "bytearray": [bytearray(b"serializor's bytes") for _ in range(10)],
            "memoryview": [memoryview(b"serializor's bytes") for _ in range(10)],
        }
        benchmark_dict(obj, 100_000, 0, debug=False)  # nested

    # Numpy Types
    if run_ndarray:
        rounds = 1_000_000

        # fmt: off
        benchmark_ndarray(np.array([], dtype="U"), rounds, 0)  # empty str
        benchmark_ndarray(np.array([], dtype=np.float64), rounds, 0)  # empty float
        benchmark_ndarray(np.array([], dtype=np.int64), rounds, 0)  # empty int
        benchmark_ndarray(np.array([], dtype=np.uint64), rounds, 0)  # empty uint
        benchmark_ndarray(np.array([], dtype=np.bool_), rounds, 0)  # empty bool
        benchmark_ndarray(np.array([], dtype="datetime64[s]"), rounds, 0)  # empty datetime64
        benchmark_ndarray(np.array([], dtype="timedelta64[s]"), rounds, 0)  # empty timedelta64
        benchmark_ndarray(np.array([], dtype=np.complex128), rounds, 0)  # empty complex128
        benchmark_ndarray(np.array([], dtype="S"), rounds, 0)  # empty bytes
        benchmark_ndarray(np.array([], dtype="O"), rounds, 0)  # empty object
        l = ['[apple"]', '[banana"]'] * 5
        benchmark_ndarray(np.array([l, l], dtype="U"), rounds, 0)  # `<np.ndarray>` str
        l = [i for i in range(10)]
        benchmark_ndarray(np.array([l, l], dtype=np.float64), rounds, 1)  # `<np.ndarray>` float
        benchmark_ndarray(np.array([l, l], dtype=np.int64), rounds, 1)  # `<np.ndarray>` int
        benchmark_ndarray(np.array([l, l], dtype=np.uint64), rounds, 1)  # `<np.ndarray>` uint
        benchmark_ndarray(np.array([l, l], dtype=np.bool_), rounds, 1)  # `<np.ndarray>` bool
        l = [np.datetime64(f"%d-01-02 03:04:05.0006007" % i) for i in range(1970, 1980)]
        benchmark_ndarray(np.array([l, l], dtype="M"), rounds, 1)  # `<np.ndarray>` datetime64
        l = [np.timedelta64(1, "D") for _ in range(10)]
        benchmark_ndarray(np.array(l, dtype="m"), rounds, 0)  # `<np.ndarray>` timedelta64
        l = [1 + 1j for _ in range(10)]
        benchmark_ndarray(np.array([l, l], dtype=np.complex128), rounds, 0)  # `<np.ndarray>` complex128
        l = [b'[apple"]' for _ in range(10)]
        benchmark_ndarray(np.array([l, l], dtype="S"), rounds, 0)  # `<np.ndarray>` bytes
        l = [1, 1.234, True, Decimal("3.14"), '[apple"]'] * 2
        benchmark_ndarray(np.array([l, l]), rounds, 0)  # `<np.ndarray>` mixed
        # fmt: on

    # Pandas Series
    if run_series:
        rounds = 100_000

        # fmt: off
        benchmark_ndarray(pd.Series(np.array([], dtype="U")), rounds, 0)  # empty str
        benchmark_ndarray(pd.Series(np.array([], dtype=np.float64)), rounds, 0)  # empty float
        benchmark_ndarray(pd.Series(np.array([], dtype=np.int64)), rounds, 0)  # empty int
        benchmark_ndarray(pd.Series(np.array([], dtype=np.uint64)), rounds, 0)  # empty uint
        benchmark_ndarray(pd.Series(np.array([], dtype=np.bool_)), rounds, 0)  # empty bool
        benchmark_ndarray(pd.Series(np.array([], dtype="datetime64[s]")), rounds, 0)  # empty datetime64
        benchmark_ndarray(pd.Series(np.array([], dtype="timedelta64[s]")), rounds, 0)  # empty timedelta64
        benchmark_ndarray(pd.Series(np.array([], dtype=np.complex128)), rounds, 0)  # empty complex128
        benchmark_ndarray(pd.Series(np.array([], dtype="S")), rounds, 0)  # empty bytes
        benchmark_ndarray(pd.Series(np.array([], dtype="O")), rounds, 0)  # empty object
        # fmt: on
        obj = np.array(['[apple"]', '[banana"]'] * 5, dtype="U")
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` str
        obj = np.array([i for i in range(10)], dtype=np.float64)
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` float
        obj = np.array([i for i in range(10)], dtype=np.int64)
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` int
        obj = np.array([i for i in range(10)], dtype=np.uint64)
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` uint
        obj = np.array([i for i in range(10)], dtype=np.bool_)
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` bool
        l = [np.datetime64(f"%d-01-02 03:04:05.0006007" % i) for i in range(1970, 1980)]
        obj = np.array(l, dtype="M")
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` datetime64
        obj = np.array([np.timedelta64(1, "D") for _ in range(10)], dtype="m")
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` timedelta64
        obj = np.array([1 + 1j for _ in range(10)], dtype=np.complex128)
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` complex128
        obj = np.array([b'[apple"]' for _ in range(10)], dtype="S")
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` bytes
        obj = np.array([1, 1.234, True, Decimal("3.14"), '[apple"]'] * 2)
        benchmark_ndarray(pd.Series(obj), rounds, 0)  # `<pd.Series>` mixed

    # Pandas DataFrame
    if run_dataframe:
        rounds = 10_000

        c1 = np.array(['[apple"]', '[banana"]'] * 5, dtype="U")  # str
        c2 = np.array([i for i in range(10)], dtype=np.float64)  # float
        c3 = np.array([i for i in range(10)], dtype=np.int64)  # int
        c4 = np.array([i for i in range(10)], dtype=np.uint64)  # uint
        c5 = np.array([i for i in range(10)], dtype=np.bool_)  # bool
        l = [np.datetime64(f"%d-01-02 03:04:05.0006007" % i) for i in range(1970, 1980)]
        c6 = np.array(l, dtype="M")  # datetime64
        l = [np.timedelta64(1, "D") for _ in range(10)]
        c7 = np.array(l, dtype="m")  # timedelta64
        c8 = np.array([1 + 1j for _ in range(10)], dtype=np.complex128)  # complex128
        c9 = np.array([b'[apple"]' for _ in range(10)], dtype="S")  # bytes
        c0 = np.array([1, 1.234, True, Decimal("3.14"), '[apple"]'] * 2)  # mixed
        obj = pd.DataFrame(
            {
                "str": c1,
                "float": c2,
                "int": c3,
                "uint": c4,
                "bool": c5,
                "datetime": c6,
                "timedelta": c7,
                "complex": c8,
                "bytes": c9,
                "mixed": c0,
            }
        )
        benchmark_dataframe(pd.DataFrame(), rounds, 0)  # `<pd.DataFrame>` empty
        # fmt: off
        benchmark_dataframe(pd.DataFrame(columns=list(obj)), rounds, 0)  # `<pd.DataFrame>` columns only
        # fmt: on
        benchmark_dataframe(obj, rounds, 0)  # `<pd.DataFrame>`

    # Pandas Index
    if run_pdindex:
        rounds = 100_000

        # fmt: off
        obj = pd.DatetimeIndex(np.array([], dtype="datetime64[s]"))
        benchmark_ndarray(obj, rounds, 0)  # `<pd.DatetimeIndex>` datetime64
        l = [np.datetime64(f"%d-01-02 03:04:05.0006007" % i) for i in range(1970, 1980)]
        obj = pd.DatetimeIndex(np.array(l, dtype="M"))
        benchmark_ndarray(obj, rounds, 0)  # `<pd.DatetimeIndex>` datetime64
        obj = pd.TimedeltaIndex(np.array([], dtype="timedelta64[s]"))
        benchmark_ndarray(obj, rounds, 0)  # `<pd.TimedeltaIndex>` timedelta64
        l = [np.timedelta64(1, "D") for _ in range(10)]
        obj = pd.TimedeltaIndex(np.array(l, dtype="m"))
        benchmark_ndarray(obj, rounds, 0)  # `<pd.TimedeltaIndex>` timedelta64

    # Nested Objects
    if run_nested:
        print(" Validate nested objects ".center(100, "-"))
        l = [1, 1.234, True, Decimal("3.14"), '[apple"]'] * 2
        # list[list]
        obj = [l] * 10
        de = deserialize(serialize(obj))
        eq = all([obj[i] == de[i] for i in range(len(obj))])
        print("- list[list]\tEQUALS:", eq)
        assert eq

        # list[dict]
        d = {str(i): l for i in range(10)}
        obj = [d] * 10
        de = deserialize(serialize(obj))
        eq = all([obj[i] == de[i] for i in range(len(obj))])
        print("- list[dict]\tEQUALS:", eq)
        assert eq

        # list[ndarray]
        arr = np.array([l, l])
        obj = [arr] * 10
        de = deserialize(serialize(obj))
        eq = all([(obj[i] == de[i]).all() for i in range(len(obj))])
        print("- list[ndarray]\tEQUALS:", eq)
        assert eq

        # dict[np.ndarray]
        obj = {str(i): arr for i in range(10)}
        de = deserialize(serialize(obj))
        eq = all([(obj[k] == de[k]).all() for k in obj])
        print("- dict[ndarray]\tEQUALS:", eq)
        assert eq

        # list[Series]
        ser = pd.Series(l)
        obj = [ser] * 10
        de = deserialize(serialize(obj))
        eq = all([obj[i].equals(de[i]) for i in range(len(obj))])
        print("- list[Series]\tEQUALS:", eq)
        assert eq

        # dict[Series]
        obj = {str(i): ser for i in range(10)}
        de = deserialize(serialize(obj))
        eq = all([obj[k].equals(de[k]) for k in obj])
        print("- dict[Series]\tEQUALS:", eq)
        assert eq

        # list[DataFrame]
        c1 = np.array(['[apple"]', '[banana"]'] * 5, dtype="U")  # str
        c2 = np.array([i for i in range(10)], dtype=np.float64)  # float
        c3 = np.array([i for i in range(10)], dtype=np.int64)  # int
        c4 = np.array([i for i in range(10)], dtype=np.uint64)  # uint
        c5 = np.array([i for i in range(10)], dtype=np.bool_)  # bool
        l = [np.datetime64(f"%d-01-02 03:04:05.0006007" % i) for i in range(1970, 1980)]
        c6 = np.array(l, dtype="M")  # datetime64
        l = [np.timedelta64(1, "D") for _ in range(10)]
        c7 = np.array(l, dtype="m")  # timedelta64
        c8 = np.array([1 + 1j for _ in range(10)], dtype=np.complex128)  # complex128
        c9 = np.array([b'[apple"]' for _ in range(10)], dtype="S")  # bytes
        c0 = np.array([1, 1.234, True, Decimal("3.14"), '[apple"]'] * 2)  # mixed
        df = pd.DataFrame(
            {
                "str": c1,
                "float": c2,
                "int": c3,
                "uint": c4,
                "bool": c5,
                "datetime": c6,
                "timedelta": c7,
                "complex": c8,
                "bytes": c9,
                "mixed": c0,
            }
        )
        obj = [df] * 10
        de = deserialize(serialize(obj))
        eq = all([obj[i].equals(de[i]) for i in range(len(obj))])
        print("- list[df]\tEQUALS:", eq)
        assert eq

        # dict[DataFrame]
        obj = {str(i): df for i in range(10)}
        de = deserialize(serialize(obj))
        eq = all([obj[k].equals(de[k]) for k in obj])
        print("- dict[df]\tEQUALS:", eq)
        assert eq

    # Encryption
    if run_crypto:
        print(" Validate Crypto ".center(100, "-"))
        obj = [3.13, 1024, True, False, "apple", b"hello", 1 + 1j, None] * 10
        key = "hello world's"
        en = encrypt(obj, key)
        de = decrypt(en, key)
        eq = obj == de
        print(" - Crypto EQUALS:", eq)
        assert obj == de


if __name__ == "__main__":
    benchmark()

    # ---------------------------------------------------------------------------------------------------
    if False:
        import pstats, cProfile

        rounds = 1_000_000
        obj = [None for i in range(10)]

        en = serialize(obj)
        print(en)
        cProfile.runctx(
            "timeit(lambda: serialize(obj), number=rounds)",
            globals(),
            locals(),
            "Profile.prof",
        )
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
        print()

        de = deserialize(en)
        print(de)
        cProfile.runctx(
            "timeit(lambda: deserialize(en), number=rounds)",
            globals(),
            locals(),
            "Profile.prof",
        )
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
