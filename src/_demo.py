# /usr/bin/python
# -*- coding: UTF-8 -*-
# from typing import Any
from typing import Any
from timeit import timeit
import datetime, time
from decimal import Decimal
import numpy as np, pandas as pd
from serializor import dumps, loads


def gen_data(rows: int = 1, offset: int = 0) -> list[dict]:
    tz = datetime.timezone(datetime.timedelta(hours=8), "CUS")
    dt = datetime.datetime.now()
    dt = datetime.datetime(2023, 1, 1, 1, 1, 1, 1)
    # fmt: off
    val = {
        "bool": True,
        "np_bool": np.bool_(False),
        "int": 1 + offset,
        "int8": np.int8(2 + offset),
        "int16": np.int16(3 + offset),
        "int32": np.int32(4 + offset),
        "int64": np.int64(5 + offset),
        "unit": np.uint(5 + offset),
        "unit16": np.uint16(5 + offset),
        "unit32": np.uint32(5 + offset),
        "unit64": np.uint64(5 + offset),
        "float": 1.1 + offset,
        "float16": np.float16(2.2 + offset),
        "float32": np.float32(3.3 + offset),
        "float64": np.float64(4.4 + offset),
        "decimal": Decimal("3.3"),
        "str": "STRING",
        "bytes": b"BYTES",
        "date": datetime.date.today() + datetime.timedelta(offset),
        "datetime": dt + datetime.timedelta(offset),
        "datetime_tz": (dt + datetime.timedelta(offset)).replace(tzinfo=tz),
        "Timestamp": pd.Timestamp(dt + datetime.timedelta(offset)),
        "Timestamp_tz": (pd.Timestamp(dt + datetime.timedelta(offset))).replace(tzinfo=tz),
        "time": (dt + datetime.timedelta(hours=offset)).time(),
        "time_tz": (dt + datetime.timedelta(hours=offset)).time().replace(tzinfo=tz),
        "timedelta": datetime.timedelta(1 + offset),
        "Timedelta": pd.Timedelta(1 + offset, "D"),
        "datetime64": np.datetime64(dt + datetime.timedelta(offset)),
        "timedelta64": np.timedelta64(2 + offset, "D"),
        "None": None,
    }
    # fmt: on
    return [val for _ in range(rows)]


def pkg_tester(val: Any, rounds: int) -> Any:
    # fmt: off
    print("-" * 80)
    print(f"Value: {type(val)}\t{val}")
    en = dumps(val)
    print(f"Encode Value".ljust(10), en, type(en), sep="\t")
    de = loads(en)
    print(f"Decode Value:".ljust(10), de, type(de), sep="\t")
    print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
    print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
    # fmt: on
    return de


def pkg_tester_validate(val: Any, rounds: int) -> Any:
    de = pkg_tester(val, rounds)
    print("#Validate#:".ljust(10), de == val, sep="\t")
    print()


def test_boolean() -> None:
    print(" Test boolean serialization ".center(80, "="))

    # fmt: off
    def base_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), bool(val) == de, sep="\t")
        print()

    def seqc_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), [bool(i) for i in val] == de, sep="\t")
        print()

    def dict_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), {k: bool(v) for k, v in val.items()} == de, sep="\t")
        print()

    rounds = 1_000_000
    base_tester(True, rounds)  # `<bool>`
    base_tester(False, rounds)  # `<bool>`
    base_tester(np.bool_(True), rounds)  # `<np.bool_>`
    base_tester(np.bool_(False), rounds)  # `<np.bool_>`

    count, rounds = 10, 1_000_000
    seqc_tester([True for _ in range(count)], rounds)  # `<list[bool]>`
    seqc_tester([False for _ in range(count)], rounds)  # `<list[bool]>`
    seqc_tester([np.bool_(True) for _ in range(count)], rounds)  # `<list[np.bool_]>`
    seqc_tester([np.bool_(False) for _ in range(count)], rounds)  # `<list[np.bool_]>`

    count, rounds = 10, 1_000_000
    dict_tester({str(i): True for i in range(count)}, rounds)  # `<dict[int, bool]>`
    dict_tester({str(i): False for i in range(count)}, rounds)  # `<dict[int, bool]>`
    dict_tester({str(i): np.bool_(True) for i in range(count)}, rounds)  # `<dict[int, np.bool_]>`
    dict_tester({str(i): np.bool_(False) for i in range(count)}, rounds)  # `<dict[int, np.bool_]>`
    # fmt: on


def test_integer() -> None:
    print(" Test integer serialization ".center(80, "="))

    # fmt: off
    def base_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), int(val) == de, sep="\t")
        print()

    def seqc_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), [int(i) for i in val] == de, sep="\t")
        print()

    def dict_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), {k: int(v) for k, v in val.items()} == de, sep="\t")
        print()

    rounds = 1_000_000
    base_tester(0, rounds)  # `<int>`
    base_tester(np.int_(-1), rounds)  # `<np.int_>`
    base_tester(np.int8(2), rounds)  # `<np.int8>`
    base_tester(np.int16(-3), rounds)  # `<np.int16>`
    base_tester(np.int32(4), rounds)  # `<np.int32>`
    base_tester(np.int64(-5), rounds)  # `<np.int64>`
    base_tester(np.uint(1), rounds)  # `<np.uint>`
    base_tester(np.uint16(2), rounds)  # `<np.uint16>`
    base_tester(np.uint32(3), rounds)  # `<np.uint32>`
    base_tester(np.uint64(4), rounds)  # `<np.uint64>`

    count, rounds = 10, 1_000_000
    seqc_tester([i for i in range(count)], rounds)  # `<list[int]>`
    seqc_tester([np.int_(i) for i in range(count)], rounds)  # `<list[np.int_]>`
    seqc_tester([np.int64(i) for i in range(count)], rounds)  # `<list[np.int64]>`
    seqc_tester([np.uint(i) for i in range(count)], rounds)  # `<list[np.uint]>`
    seqc_tester([np.uint64(i) for i in range(count)], rounds)  # `<list[np.uint64]>`

    count, rounds = 10, 1_000_000
    dict_tester({str(i): i for i in range(count)}, rounds)  # `<dict[int, int]>`
    dict_tester({str(i): np.int_(i) for i in range(count)}, rounds)  # `<dict[int, np.int_]>`
    dict_tester({str(i): np.int64(i) for i in range(count)}, rounds)  # `<dict[int, np.int64]>`
    dict_tester({str(i): np.uint(i) for i in range(count)}, rounds)  # `<dict[int, np.uint]>`
    dict_tester({str(i): np.uint64(i) for i in range(count)}, rounds)  # `<dict[int, np.uint64]>`
    # fmt: on


def test_float() -> None:
    print(" Test float serialization ".center(80, "="))

    # fmt: off
    def base_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), float(val) == de, sep="\t")
        print()

    def seqc_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), [float(i) for i in val] == de, sep="\t")
        print()

    def dict_tester(val: Any, rounds: int):
        de = pkg_tester(val, rounds)
        print("#Validate#:".ljust(10), {k: float(v) for k, v in val.items()} == de, sep="\t")
        print()

    rounds = 1_000_000
    base_tester(0.0, rounds)  # `<float>`
    base_tester(np.float_(-1.1), rounds)  # `<np.float_>`
    base_tester(np.float16(2.2), rounds)  # `<np.float16>`
    base_tester(np.float32(-3.3), rounds)  # `<np.float32>`
    base_tester(np.float64(4.4), rounds)  # `<np.float64>`

    count, rounds = 10, 1_000_000
    seqc_tester([i / 10 for i in range(count)], rounds)  # `<list[float]>`
    seqc_tester([np.float_(i / 10) for i in range(count)], rounds)  # `<list[np.float_]>`
    seqc_tester([np.float64(i / 10) for i in range(count)], rounds)  # `<list[np.float64]>`

    count, rounds = 10, 1_000_000
    dict_tester({str(i): i / 10 for i in range(count)}, rounds)  # `<dict[int, float]>`
    dict_tester({str(i): np.float_(i / 10) for i in range(count)}, rounds)  # `<dict[int, np.float_]>`
    dict_tester({str(i): np.float64(i / 10) for i in range(count)}, rounds)  # `<dict[int, np.float64]>`
    # fmt: on


def test_decimal() -> None:
    print(" Test decimal serialization ".center(80, "="))

    # fmt: off
    rounds = 1_000_000
    pkg_tester_validate(Decimal("0.0"), rounds)  # `<Decimal>`
    pkg_tester_validate(Decimal("-1.1"), rounds)  # `<Decimal>`
    pkg_tester_validate(Decimal("2.2"), rounds)  # `<Decimal>`
    pkg_tester_validate(Decimal("-3.3"), rounds)  # `<Decimal>`
    pkg_tester_validate(Decimal("4.4"), rounds)  # `<Decimal>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate([Decimal(str(i / 10)) for i in range(count)], rounds)  # `<list[Decimal]>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate({str(i): Decimal(str(i / 10)) for i in range(count)}, rounds)  # `<dict[int, Decimal]>`
    # fmt: on


def test_string() -> None:
    print(" Test string serialization ".center(80, "="))

    # fmt: off
    rounds = 1_000_000
    pkg_tester_validate("", rounds)  # `<str>`
    pkg_tester_validate("STRING", rounds)  # `<str>`
    pkg_tester_validate("中文", rounds)  # `<str>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate([str(i) for i in range(count)], rounds)  # `<list[str]>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate({str(i): str(i) for i in range(count)}, rounds)  # `<dict[int, str]>`


def test_bytes() -> None:
    print(" Test bytes serialization ".center(80, "="))

    # fmt: off
    rounds = 1_000_000
    pkg_tester_validate(b"", rounds)  # `<bytes>`
    pkg_tester_validate(b"BYTES", rounds)  # `<bytes>`
    
    count, rounds = 10, 1_000_000
    pkg_tester_validate([str(i).encode() for i in range(count)], rounds)  # `<list[bytes]>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate({str(i): str(i).encode() for i in range(count)}, rounds)  # `<dict[int, bytes]>`


def test_date() -> None:
    print(" Test date serialization ".center(80, "="))

    # fmt: off
    today = datetime.date.today()
    rounds = 1_000_000
    pkg_tester_validate(today, rounds)  # `<datetime.date>`
    pkg_tester_validate(datetime.date(2021, 1, 1), rounds)  # `<datetime.date>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate([today for _ in range(count)], rounds)  # `<list[datetime.date]>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate({str(i): today for i in range(count)}, rounds)  # `<dict[int, datetime.date]>`
    # fmt: on


def test_datetime() -> None:
    print(" Test datetime serialization ".center(80, "="))

    # fmt: off
    def compare_tester(val: Any, compare: Any,rounds: int) -> Any:
        print("-" * 80)
        print(f"Value: {type(val)}\t{val}")
        en = dumps(val)
        print(f"Encode Value".ljust(10), en, type(en), sep="\t")
        de = loads(en)
        print(f"Decode Value:".ljust(10), de, type(de), sep="\t")
        print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
        print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
        print("#Validate#:".ljust(10), de == compare, sep="\t")

    # fmt: off
    tzinfo = datetime.timezone(datetime.timedelta(hours=8), "CUS")
    dt = datetime.datetime.now()
    dt_tz = dt.replace(tzinfo=tzinfo)

    rounds = 1_000_000
    pkg_tester_validate(dt, rounds)  # `<datetime.datetime>`
    pkg_tester_validate(dt_tz, rounds)  # `<datetime.datetime>`
    ts, ts_tz = pd.Timestamp(dt), pd.Timestamp(dt_tz)
    compare_tester(ts, dt, rounds)  # `<pandas.Timestamp>`
    compare_tester(ts_tz, dt_tz, rounds)  # `<pandas.Timestamp>`
    dt64 = np.datetime64(dt)
    pkg_tester_validate(dt64, rounds)  # `<numpy.datetime64>`
    stm = time.localtime(dt.timestamp())
    stm_dt = datetime.datetime(*stm[:6])
    compare_tester(stm, stm_dt, rounds)

    count, rounds = 10, 1_000_000
    dt_lst = [dt for _ in range(count)]
    dt_tz_lst = [dt_tz for _ in range(count)]
    pkg_tester_validate(dt_lst, rounds)  # `<list[datetime.datetime]>`
    pkg_tester_validate(dt_tz_lst, rounds)  # `<list[datetime.datetime]>`
    ts_lst = [ts for _ in range(count)]
    ts_tz_lst = [ts_tz for _ in range(count)]
    compare_tester(ts_lst, dt_lst, rounds)  # `<list[pandas.Timestamp]>`
    compare_tester(ts_tz_lst, dt_tz_lst, rounds)  # `<list[pandas.Timestamp]>`
    dt64_lst = [dt64 for _ in range(count)]
    compare_tester(dt64_lst, dt_lst, rounds)  # `<list[numpy.datetime64]>`
    stm_lst = [stm for _ in range(count)]
    stm_dt_lst = [stm_dt for _ in range(count)]
    compare_tester(stm_lst, stm_dt_lst, rounds)

    count, rounds = 10, 1_000_000
    dt_dict = {str(i): dt for i in range(count)}
    dt_tz_dict = {str(i): dt_tz for i in range(count)}
    pkg_tester_validate(dt_dict, rounds)  # `<dict[int, datetime.datetime]>`
    pkg_tester_validate(dt_tz_dict, rounds)  # `<dict[int, datetime.datetime]>`
    ts_dict = {str(i): ts for i in range(count)}
    ts_tz_dict = {str(i): ts_tz for i in range(count)}
    compare_tester(ts_dict, dt_dict, rounds)  # `<dict[int, pandas.Timestamp]>`
    compare_tester(ts_tz_dict, dt_tz_dict, rounds)  # `<dict[int, pandas.Timestamp]>`
    dt64_dict = {str(i): dt64 for i in range(count)}
    compare_tester(dt64_dict, dt_dict, rounds)  # `<dict[int, numpy.datetime64]>`
    stm_dict = {str(i): stm for i in range(count)}
    stm_dt_dict = {str(i): stm_dt for i in range(count)}
    compare_tester(stm_dict, stm_dt_dict, rounds)


def test_time() -> None:
    print(" Test time serialization ".center(80, "="))

    # fmt: off
    tzinfo = datetime.timezone(datetime.timedelta(hours=8), "CUS")
    dt = datetime.datetime.now()
    dt_zt = dt.replace(tzinfo=tzinfo)
    tm, tm_tz = dt.time(), dt_zt.timetz()

    rounds = 1_000_000
    pkg_tester_validate(tm, rounds)  # `<datetime.time>`
    pkg_tester_validate(tm_tz, rounds)  # `<datetime.time>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate([tm for _ in range(count)], rounds)  # `<list[datetime.time]>`
    pkg_tester_validate([tm_tz for _ in range(count)], rounds)  # `<list[datetime.time]>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate({str(i): tm for i in range(count)}, rounds)  # `<dict[int, datetime.time]>`
    pkg_tester_validate({str(i): tm_tz for i in range(count)}, rounds)  # `<dict[int, datetime.time]>`
    # fmt: on


def test_timedelta() -> None:
    print(" Test timedelta serialization ".center(80, "="))

    # fmt: off
    def compare_tester(val: Any, compare: Any,rounds: int) -> Any:
        print("-" * 80)
        print(f"Value: {type(val)}\t{val}")
        en = dumps(val)
        print(f"Encode Value".ljust(10), en, type(en), sep="\t")
        de = loads(en)
        print(f"Decode Value:".ljust(10), de, type(de), sep="\t")
        print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
        print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
        print("#Validate#:".ljust(10), de == compare, sep="\t")

    rounds = 1_000_000
    dl1 = datetime.timedelta(1)
    dl2 = datetime.timedelta(1, 1)
    dl3 = datetime.timedelta(1, 1, 1)
    pkg_tester_validate(dl1, rounds)  # `<datetime.timedelta>`
    pkg_tester_validate(dl2, rounds)  # `<datetime.timedelta>`
    pkg_tester_validate(dl3, rounds)  # `<datetime.timedelta>`
    dt64 = np.timedelta64(dl3)
    compare_tester(dt64, dl3, rounds)  # `<numpy.timedelta64>`

    count, rounds = 10, 1_000_000
    dl_lst = [dl3 for _ in range(count)]
    pkg_tester_validate(dl_lst, rounds)  # `<list[datetime.timedelta]>`
    dt64_lst = [dt64 for _ in range(count)]
    compare_tester(dt64_lst, dl_lst, rounds)  # `<list[numpy.timedelta64]>`

    count, rounds = 10, 1_000_000
    dl_dict = {str(i): dl3 for i in range(count)}
    pkg_tester_validate(dl_dict, rounds)  # `<dict[int, datetime.timedelta]>`
    dt64_dict = {str(i): dt64 for i in range(count)}
    compare_tester(dt64_dict, dl_dict, rounds)  # `<dict[int, numpy.timedelta64]>`


def test_None() -> None:
    print(" Test None serialization ".center(80, "="))

    # fmt: off
    rounds = 1_000_000
    pkg_tester_validate(None, rounds)  # `<NoneType>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate([None for _ in range(count)], rounds)  # `<list[NoneType]>`

    count, rounds = 10, 1_000_000
    pkg_tester_validate({str(i): None for i in range(count)}, rounds)  # `<dict[int, NoneType]>`
    # fmt: on


def test_mixed() -> None:
    print(" Test mix serialization ".center(80, "="))

    # fmt: off
    def mixed_tester(val: Any, rounds: int) -> Any:
        print("-" * 80)
        print(f"Value: {type(val)}, {val[0]}")
        en = dumps(val)
        print(f"Encode Value".ljust(10), en[:100], type(en), sep="\t")
        de = loads(en)
        print(f"Decode Value:".ljust(10), de[0], type(de), sep="\t")
        print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
        print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
        print("#Validate#:".ljust(10), de == val, sep="\t")
        print()


    data = gen_data(10000)
    columns = ["bool", "np_bool", "int", "int64", "float", "float64", "decimal", "str", 
               "bytes", "date", "datetime", "datetime_tz", "Timestamp", "Timestamp_tz", 
               "time", "time_tz", "timedelta", "Timedelta", "datetime64", "timedelta64", "None"
            ]
    data = [{k: v for k, v in d.items() if k in columns} for d in data]

    rounds = 100
    mixed_tester(data, rounds)  # `<list[dict]>`


def test_ndarray() -> None:
    print(" Test numpy.ndarray serialization ".center(80, "="))

    # fmt: off
    def ndarray_tester(val: np.ndarray, rounds: int) -> Any:
        print("-" * 80)
        print(f"Value:\n{val[0]}", {type(val)})
        en = dumps(val)
        print(f"Encode Value".ljust(10), en[:100], type(en), sep="\t")
        de = loads(en)
        print(f"Decode Value:\n{de[0]}", type(de))
        print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
        print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
        print("#Validate#:".ljust(10), (de == val).all(), sep="\t")
        print()


    df = pd.DataFrame(gen_data(10000))
    columns = ["bool", "np_bool", "int", "int64", "float", "float64", "decimal", "str", 
               "bytes", "date", "datetime", "datetime_tz", "Timestamp", "Timestamp_tz", 
               "time", "time_tz", "timedelta", "Timedelta", "datetime64", "timedelta64", "None"
            ]
    df = df[columns]
    arr = df.to_numpy()

    rounds = 100
    ndarray_tester(arr, rounds)  # `<numpy.ndarray>`


def test_psSeries() -> None:
    print(" Test pandas.Series serialization ".center(80, "="))

    # fmt: off
    def series_tester(val: pd.Series, rounds: int) -> Any:
        print("-" * 80)
        print(f"Value: {val.dtype}\t{val.values[:10]}")
        en = dumps(val)
        print(f"Encode Value".ljust(10), en[:100], type(en), sep="\t")
        de = loads(en)
        print(f"Decode Value:".ljust(10), de.values[:10], type(de), sep="\t")
        print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
        print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
        print("#Validate#:".ljust(10), val.equals(de), sep="\t")

    df = pd.DataFrame(gen_data(10000))
    rounds = 100
    series_tester(df["bool"], rounds)  # `<pandas.Series>`
    series_tester(df["np_bool"], rounds)  # `<pandas.Series>`
    series_tester(df["int"], rounds)  # `<pandas.Series>`
    series_tester(df["int64"], rounds)  # `<pandas.Series>`
    series_tester(df["float"], rounds)  # `<pandas.Series>`
    series_tester(df["float64"], rounds)  # `<pandas.Series>`
    series_tester(df["decimal"], rounds)  # `<pandas.Series>`
    series_tester(df["str"], rounds)  # `<pandas.Series>`
    series_tester(df["bytes"], rounds)  # `<pandas.Series>`
    series_tester(df["date"], rounds)  # `<pandas.Series>`
    series_tester(df["datetime"], rounds)  # `<pandas.Series>`
    series_tester(df["datetime_tz"], rounds)  # `<pandas.Series>`
    series_tester(df["Timestamp"], rounds)  # `<pandas.Series>`
    series_tester(df["Timestamp_tz"], rounds)  # `<pandas.Series>`
    series_tester(df["time"], rounds)  # `<pandas.Series>`
    series_tester(df["time_tz"], rounds)  # `<pandas.Series>`
    series_tester(df["timedelta"], rounds)  # `<pandas.Series>`
    series_tester(df["Timedelta"], rounds)  # `<pandas.Series>`
    series_tester(df["datetime64"], rounds)  # `<pandas.Series>`
    series_tester(df["timedelta64"], rounds)  # `<pandas.Series>`
    series_tester(df["None"], rounds)  # `<pandas.Series>`
    # fmt: on


def test_pdDataFrame() -> None:
    print(" Test pandas.DataFrame serialization ".center(80, "="))

    # fmt: off
    def df_tester(val: pd.DataFrame, rounds: int) -> Any:
        print("-" * 80)
        print(f"Value:\n{val}", {type(val)})
        en = dumps(val)
        print(f"Encode Value".ljust(10), en[:100], type(en), sep="\t")
        de = loads(en)
        print(f"Decode Value:\n{de}", type(de))
        print(f"Encode Perf.:".ljust(10), timeit(lambda: dumps(val), number=rounds), sep="\t")
        print(f"Decode Perf.:".ljust(10), timeit(lambda: loads(en), number=rounds), sep="\t")
        print("#Validate#:".ljust(10), val.equals(de), sep="\t")

    df = pd.DataFrame(gen_data(10000))
    columns = ["bool", "np_bool", "int", "int64", "float", "float64", "decimal", "str", 
               "bytes", "date", "datetime", "datetime_tz", "Timestamp", "Timestamp_tz", 
               "time", "time_tz", "timedelta", "Timedelta", "datetime64", "timedelta64", "None"
            ]
    df = df[columns]
    
    rounds = 100
    df_tester(df, rounds)  # `<pandas.DataFrame>`
    # fmt: on


def test_cipher() -> None:
    from serializor.cipher import encrypt, decrypt

    key = b"show me the money xasdasdasd"
    data = {"a": 1.23, "c": {"d": 123}}

    a = encrypt(data, key)
    print(a, type(a))
    print()

    b = decrypt(a, key)
    print(b, type(b))
    print()


if __name__ == "__main__":
    test_boolean()
    test_integer()
    test_float()
    test_decimal()
    test_string()
    test_bytes()
    test_date()
    test_datetime()
    test_time()
    test_timedelta()
    test_None()
    test_mixed()
    test_ndarray()
    test_psSeries()
    test_pdDataFrame()
    test_cipher()
