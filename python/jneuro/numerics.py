"""Exact binary64 helpers (spec/02)."""
import math
import struct

_MASK64 = (1 << 64) - 1
MAX_FINITE = 1.7976931348623157e308


def to_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & _MASK64))[0]


def canon(x: float) -> float:
    """NaN and -0.0 become +0.0, +-inf become +-MAX_FINITE; finite values are unchanged."""
    if x != x or x == 0.0:
        return 0.0
    if x == math.inf:
        return MAX_FINITE
    if x == -math.inf:
        return -MAX_FINITE
    return x


def bitnot(x: float) -> float:
    """Flip all 64 bits of the binary64 pattern."""
    return from_bits(~to_bits(x))


def ieee_div(a: float, b: float) -> float:
    """IEEE-754 division; never raises ZeroDivisionError."""
    if b == 0.0:
        if a != a or a == 0.0:
            return math.nan
        sign_neg = (math.copysign(1.0, a) < 0) != (math.copysign(1.0, b) < 0)
        return -math.inf if sign_neg else math.inf
    return a / b


def compensated_mean(values) -> float:
    """Neumaier-compensated mean in the given order (spec/04)."""
    s = 0.0
    c = 0.0
    for v in values:
        t = s + v
        if abs(s) >= abs(v):
            c = c + ((s - t) + v)
        else:
            c = c + ((v - t) + s)
        s = t
    total = s + c if math.isfinite(s) else s
    return ieee_div(total, float(len(values)))
