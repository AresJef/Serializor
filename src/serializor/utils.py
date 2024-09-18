# cython: language_level=3

# Python imports
from zoneinfo import available_timezones as _available_timezones

# Constants -----------------------------------------------------------------------------------
AVAILABLE_TIMEZONES: set[str] = _available_timezones()
del _available_timezones


########## All rest utility functions are in the utils.pxd file ##########
########## The following functions are for testing purpose only ##########


def _test_utils() -> None:
    _test_encode_decode_utf8()
    _test_encode_decode_ascii()
    _test_numpy_time_unit()


def _test_encode_decode_utf8() -> None:
    val = "中国\n한국어\nにほんご\nEspañol"
    # encode
    n = val.encode("utf-8")
    x = encode_str(val, "utf-8")  # type: ignore
    assert n == x, f"{n} | {x}"
    # decode
    i = n.decode("utf-8")
    j = decode_bytes(n, "utf-8")  # type: ignore
    k = decode_bytes_utf8(n)  # type: ignore
    assert i == j == k == val, f"{i} | {j} | {k} | {val}"

    print("Pass Encode/Decode UTF-8".ljust(80))


def _test_encode_decode_ascii() -> None:
    val = "hello\nworld"
    # encode
    n = val.encode("ascii")
    x = encode_str(val, "ascii")  # type: ignore
    assert n == x, f"{n} | {x}"
    # decode
    i = n.decode("ascii")
    j = decode_bytes(n, "ascii")  # type: ignore
    k = decode_bytes_ascii(n)  # type: ignore
    assert i == j == k == val, f"{i} | {j} | {k} | {val}"

    print("Pass Encode/Decode ASCII".ljust(80))


def _test_numpy_time_unit() -> None:
    import numpy as np

    units = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")

    for unit in units:
        unit == map_nptime_unit_int2str(map_nptime_unit_str2int(unit))  # type: ignore

    for unit in units:
        arr = np.array([], dtype="datetime64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore
        arr = np.array([1, 2, 3], dtype="datetime64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore
        arr = np.array([], dtype="timedelta64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore
        arr = np.array([1, 2, 3], dtype="timedelta64[%s]" % unit)
        assert unit == map_nptime_unit_int2str(parse_arr_nptime_unit(arr))  # type: ignore

    print("Passed: numpy_time_unit")

    del np
