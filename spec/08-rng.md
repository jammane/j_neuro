# 08 — Random numbers

Both implementations MUST use exactly these algorithms. Platform generators and
distributions (`std::mt19937`, `std::*_distribution`, `random`, `numpy.random`) MUST NOT be
used: their output differs between libraries and versions.

All integer arithmetic below is on unsigned 64-bit integers, wrapping modulo 2⁶⁴
(Python: mask with `& 0xFFFFFFFFFFFFFFFF` after every operation that can overflow).

## Seeding — splitmix64

```
splitmix64(x):                       # x is the 64-bit state, updated in place
    x = x + 0x9E3779B97F4A7C15
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)
```

`seed(s)`: set `x = s`, then `state[0..3] = splitmix64(x)` four times in order.

## Generator — xoshiro256**

```
rotl(v, r) = (v << r) | (v >> (64 - r))

next_u64():
    result = rotl(state[1] * 5, 7) * 9
    t = state[1] << 17
    state[2] ^= state[0]
    state[3] ^= state[1]
    state[1] ^= state[2]
    state[0] ^= state[3]
    state[2] ^= t
    state[3] = rotl(state[3], 45)
    return result
```

## Derived procedures

### u() — uniform binary64 in [0, 1)

```
u() = (next_u64() >> 11) * 2^-53          # exact: 53-bit integer times a power of two
```

### below(n) — uniform integer in [0, n)

`n` is a 64-bit unsigned integer `≥ 1`. For `n = 2⁶⁴` (only arises as a range size of a full
`U64` span) use `next_u64()` directly.

```
below(n):
    threshold = (2^64 - n) mod n            # computed as (0 - n) mod n in u64
    loop:
        r = next_u64()
        if r >= threshold: return r mod n
```

### choice(k) — uniform index in [0, k)

`choice(k) = below(k)`. Used with the enumeration orders of [04](04-neurons.md).

### weighted(w) — index by weights

`w` is an ordered list of non-negative binary64 weights with positive total.

```
total = w[0] + w[1] + ... (left to right)
x = u() * total
acc = 0.0
for i in order:
    acc = acc + w[i]
    if x < acc: return i
return last index with w[i] > 0
```

Exactly one `u()` draw.

### normal() — approximately standard normal

Irwin–Hall with 12 uniforms (mean 0, variance 1, range [−6, 6]):

```
s = 0.0
repeat 12 times: s = s + u()
return s - 6.0
```

Exactly twelve `u()` draws. Chosen because it needs only additions, so it is bit-identical
everywhere (Box–Muller would need `log`, `cos`, `sin`, which differ between math libraries).
The bounded range also prevents extreme mutation jumps.

### prob_round(x) — probabilistic rounding, x ≥ 0

```
f = floor(x)
return f + (1 if u() < (x - f) else 0)
```

Always exactly one `u()` draw, even when `x` is an integer. Expected value equals `x`.

### round_half_away(x)

Not random. `sign(x) * floor(|x| + 0.5)`, returned as a signed integer.

## Streams

A mutation call receives one RNG and draws from it in the order specified in
[06](06-mutation.md) and [07](07-removal.md). How a population derives one RNG per offspring
is out of scope for v0.1; implementations SHOULD expose `seed`, `next_u64` and state
save/restore so callers can control it.
