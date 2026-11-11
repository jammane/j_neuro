"""Shared PRNG and distributions (spec/08). Bit-identical to the C++ implementation."""
import math

_M = (1 << 64) - 1
_TWO_M53 = 2.0 ** -53


def _rotl(v: int, r: int) -> int:
    return ((v << r) | (v >> (64 - r))) & _M


def _splitmix64(x: int):
    x = (x + 0x9E3779B97F4A7C15) & _M
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
    return x, z ^ (z >> 31)


class Rng:
    __slots__ = ("_s",)

    def __init__(self, seed: int = 0):
        x = seed & _M
        s = []
        for _ in range(4):
            x, z = _splitmix64(x)
            s.append(z)
        self._s = s

    def next_u64(self) -> int:
        s = self._s
        result = (_rotl((s[1] * 5) & _M, 7) * 9) & _M
        t = (s[1] << 17) & _M
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def u(self) -> float:
        return float(self.next_u64() >> 11) * _TWO_M53

    def below(self, n: int) -> int:
        if n < 1 or n > (1 << 64):
            raise ValueError("below(n) requires 1 <= n <= 2^64")
        if n == (1 << 64):
            return self.next_u64()
        threshold = ((1 << 64) - n) % n
        while True:
            r = self.next_u64()
            if r >= threshold:
                return r % n

    def choice(self, k: int) -> int:
        return self.below(k)

    def weighted(self, weights) -> int:
        total = 0.0
        for w in weights:
            total = total + w
        x = self.u() * total
        acc = 0.0
        for i, w in enumerate(weights):
            acc = acc + w
            if x < acc:
                return i
        for i in range(len(weights) - 1, -1, -1):
            if weights[i] > 0:
                return i
        raise ValueError("weighted() needs a positive total weight")

    def normal(self) -> float:
        s = 0.0
        for _ in range(12):
            s = s + self.u()
        return s - 6.0

    def prob_round(self, x: float) -> int:
        f = math.floor(x)
        return int(f) + (1 if self.u() < (x - f) else 0)

    def get_state(self):
        return tuple(self._s)

    def set_state(self, state):
        self._s = [int(v) & _M for v in state]


def round_half_away(x: float) -> int:
    r = math.floor(abs(x) + 0.5)
    return int(r) if x >= 0 else -int(r)
