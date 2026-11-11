# 02 — Numerics

Accuracy is the first priority of j_neuro. This document makes every computation exactly
reproducible.

## Number format

- All node values, weights and intermediate results are **IEEE-754 binary64** (`double`).
- Implementations MUST NOT use any other precision (no `float`, `float32`, `long double`,
  x87 extended precision, or fused multiply-add).
- The bitwise inversion `~` ([04](04-neurons.md#v_neuron)) is defined on the binary64 bit
  pattern, so changing precision would change results. Precision is therefore not a setting.

## Allowed operations

Evaluation uses only operations that IEEE-754 defines as correctly rounded or exact:
`+`, `−`, `×`, `÷`, comparison, negation, `floor`, and bit reinterpretation. Mutation uses the
same set ([08](08-rng.md) defines its random distributions without transcendental functions).

Implementations MUST NOT:

- Rewrite expressions (`x / w` is never computed as `x * (1/w)`; `a + b + c` is evaluated
  left to right as written).
- Call transcendental functions (`exp`, `log`, `sin`, `cos`, `pow`, `tanh`, …) anywhere in
  evaluation or mutation. Their results differ between math libraries.
- Fuse multiply and add.

## Canonical values

Every value stored in a node is **canonical**:

```
canon(x) = +0.0   if x is NaN
         = +0.0   if x == 0   (covers −0.0)
         = x      otherwise
```

- `canon` is applied to **every neuron output** and to **every caller-supplied input value**.
- ±infinity is a valid canonical value and passes through unchanged.
- Consequences: nodes never hold NaN or −0.0; equal canonical values have identical bits, so
  ties in `min`, `max` and `median` never depend on order.

Inside a single neuron computation, intermediate values may be NaN, −0.0 or infinite; only
the neuron's final output is canonicalized.

## Division

Division follows IEEE-754: `x / 0.0` is `±inf` or NaN; `1.0 / +0.0 = +inf`.
In Python, the plain `/` operator on `float` raises `ZeroDivisionError` and MUST NOT be used
for values that can be zero; use NumPy `float64` arithmetic or an explicit IEEE division
helper.

## Bitwise inversion

```
bitnot(x) = from_bits( ~to_bits(x) )   # all 64 bits flipped
```

- C++: `std::bit_cast<std::uint64_t>(x)` (C++20). MUST NOT use union type-punning.
- Python: `numpy.float64(x).view(numpy.uint64)`, or `struct.pack('<d', x)` /
  `struct.unpack('<Q', …)`.
- Informative: for normal `x`, `bitnot(x) ≈ −4/x` (within a factor of about 1.125);
  `bitnot(+0.0)` is a NaN; inputs near the subnormal range map to ±inf or NaN; ±inf and NaN
  map to tiny values or zero.

## Summation

Sums are computed left to right in the order the spec defines (slot index order for
m_neurons, oldest-to-newest for s_neuron histories). The mean uses compensated summation as
defined in [04](04-neurons.md#aggregate).

## Integer arithmetic

- Depth values are unsigned integers of the net's `DEPTH` width. Depth arithmetic MUST be
  exact integer arithmetic and MUST NOT pass through binary64.
- Python implementations MUST enforce the `DEPTH` range explicitly (Python integers are
  unbounded).

## Build rules (C++)

- MUST NOT compile with `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`,
  `-ffinite-math-only`, or MSVC `/fp:fast`. The build MUST fail if they are present.
- MUST compile with `-ffp-contract=off` (MSVC: `/fp:precise` or `/fp:strict`).
- MUST target 64-bit platforms using SSE2 or equivalent scalar binary64 hardware; x87 FPU
  code generation is not allowed.
