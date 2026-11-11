# 10 — Conformance tests

Golden cases live in `tests/golden/` at the repository root and are run unchanged by both
the C++ and the Python test suites. A case passes only on an **exact** match.

## Value encoding

binary64 values that are compared (inputs, outputs) are written as 16-digit lowercase hex
strings of their bit pattern, e.g. `"3ff0000000000000"` for `1.0`, so that ±inf, signed zero
and NaN payloads are unambiguous. Weights inside genomes use the normal encoding of
[09](09-serialization.md).

## Case kinds

Each case is one JSON file with a `kind` field.

### `rng`

```json
{"kind": "rng", "seed": 42,
 "next_u64": ["…16 hex digits…", "…"],
 "u": ["…bits…"], "below": [{"n": 10, "values": [3, 7]}], "normal": ["…bits…"]}
```

Each list is generated from a fresh `seed(seed)`.

### `eval`

```json
{"kind": "eval", "genome": { "…": "…" },
 "steps": [{"inputs": ["…bits…"], "outputs": ["…bits…"]}],
 "priorities": ["SPEED", "MEMORY", "BALANCED"]}
```

Steps run in order without reset. The same expected outputs MUST be produced under every
listed priority.

### `create`

```json
{"kind": "create", "seed": 1, "n_inputs": 7, "n_outputs": 2,
 "depth_type": "U16", "in_qty_max": 3, "priority": "MEMORY",
 "expected_genome": { "…": "…" }}
```

### `mutate`

```json
{"kind": "mutate", "seed": 9, "generation": 12,
 "genome": { "…": "…" }, "profile": { "…": "…" },
 "expected_genome": { "…": "…" }}
```

The serialized result MUST be byte-identical to the serialized `expected_genome`.

### `invalid`

```json
{"kind": "invalid", "genome": { "…": "…" }, "reason": "path rule"}
```

Loading MUST fail.

## Required coverage

The golden set SHOULD include at least:

- Every v inversion × op, including division by zero, `bitnot(0)`, ±inf inputs and NaN inputs.
- Every aggregation op for m_neurons with empty slots, and for s_neurons during warm-up
  (`min` skipping unfilled entries).
- Compensated mean cases where naive summation would round differently, and overflow cases.
- Minimal-net creation for `n_inputs` = 1, ≤ `in_qty_max`, and > `in_qty_max`.
- Every addition method, including skipped attempts (no gap, no eligible receiver, cap
  reached).
- Bypass, silenced removal with downstream cascade stopping at a `min` reader and at a
  nonzero constant, immunity skip, garbage collection and compaction.
- Every `DEPTH` type, including `U64` depths above 2⁵³.

## Generating golden files

The first implementation to support a feature generates its golden files; the second
implementation must then reproduce them. Any disagreement is resolved against the spec, never
by editing the expected values to match one implementation.
