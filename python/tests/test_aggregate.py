from helpers import INF, same

from jneuro import AggOp
from jneuro.aggregate import aggregate

T, F = True, False


def test_mean_counts_empties():
    assert aggregate(AggOp.MEAN, [3.0, 5.0, 0.0, 0.0, 0.0], [T, T, F, F, F]) == 8.0 / 5.0


def test_median_counts_empties():
    assert same(aggregate(AggOp.MEDIAN, [3.0, 5.0, 0.0, 0.0, 0.0], [T, T, F, F, F]), 0.0)
    assert aggregate(AggOp.MEDIAN, [9.0, -1.0, 4.0], [T, T, T]) == 4.0


def test_min_skips_empties():
    assert aggregate(AggOp.MIN, [3.0, 5.0, 0.0], [T, T, F]) == 3.0
    assert aggregate(AggOp.MIN, [3.0, 0.0, 5.0], [T, T, T]) == 0.0
    assert same(aggregate(AggOp.MIN, [0.0, 0.0, 0.0], [F, F, F]), 0.0)


def test_max_counts_empties():
    assert same(aggregate(AggOp.MAX, [-3.0, -5.0, 0.0], [T, T, F]), 0.0)
    assert aggregate(AggOp.MAX, [-3.0, -5.0, -1.0], [T, T, T]) == -1.0


def test_infinities():
    assert aggregate(AggOp.MIN, [INF, -INF, 0.0], [T, T, F]) == -INF
    assert aggregate(AggOp.MAX, [INF, -INF, 0.0], [T, T, F]) == INF
