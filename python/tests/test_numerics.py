import math

from helpers import INF, NAN, bits, same

from jneuro.numerics import bitnot, canon, compensated_mean, from_bits, ieee_div, to_bits


def test_canon():
    assert same(canon(NAN), 0.0)
    assert same(canon(-0.0), 0.0)
    assert same(canon(0.0), 0.0)
    assert canon(INF) == 1.7976931348623157e308 and canon(-INF) == -1.7976931348623157e308
    assert same(canon(-2.5), -2.5)


def test_bits_roundtrip():
    assert to_bits(1.0) == 0x3FF0000000000000
    assert same(from_bits(0xC00FFFFFFFFFFFFF), -3.9999999999999996)


def test_bitnot():
    assert same(bitnot(1.0), -3.9999999999999996)
    assert same(bitnot(2.0), -1.9999999999999998)
    assert math.isnan(bitnot(0.0))
    assert same(bitnot(-INF), from_bits(0x000FFFFFFFFFFFFF))
    for x in (1.0, -3.5, 1e-300, 7e300):
        assert same(bitnot(bitnot(x)), x)


def test_ieee_div():
    assert ieee_div(1.0, 0.0) == INF
    assert ieee_div(-1.0, 0.0) == -INF
    assert ieee_div(1.0, -0.0) == -INF
    assert math.isnan(ieee_div(0.0, 0.0))
    assert math.isnan(ieee_div(NAN, 2.0))
    assert ieee_div(6.0, 3.0) == 2.0
    assert same(ieee_div(1.0, -INF), -0.0)


def test_compensated_mean():
    assert compensated_mean([1e16, 1.0, -1e16]) == 1.0 / 3.0
    assert compensated_mean([1e308, 1e308, -1e308]) == INF
    assert math.isnan(compensated_mean([INF, -INF, 0.0]))
    assert compensated_mean([0.0, 0.0, 1.0]) == 1.0 / 3.0
    assert compensated_mean([2.0]) == 2.0
    assert bits(compensated_mean([0.1] * 10)) == bits(0.1)
