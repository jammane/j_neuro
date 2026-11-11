"""Shared aggregation for s_neurons and m_neurons (spec/04)."""
from .numerics import compensated_mean
from .types import AggOp


def aggregate(op: AggOp, values, present) -> float:
    """values: canonical values in order; present: connected slot / filled history entry.
    Non-present values are +0.0 and count for mean, median and max; min skips them."""
    if op == AggOp.MEAN:
        return compensated_mean(values)
    if op == AggOp.MEDIAN:
        return sorted(values)[len(values) // 2]
    if op == AggOp.MIN:
        m = None
        for v, p in zip(values, present):
            if p and (m is None or v < m):
                m = v
        return 0.0 if m is None else m
    m = values[0]
    for v in values[1:]:
        if v > m:
            m = v
    return m
