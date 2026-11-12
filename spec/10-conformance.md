# 10 — Conformance tests

Golden cases live in `tests/golden/` at the repository root and are run unchanged by both
the C++ and the Python test suites. A case passes only on an **exact** match.

## Value encoding

binary64 values that are compared (inputs, outputs) are written as 16-digit lowercase hex
strings of their bit pattern, e.g. `"3ff0000000000000"` for `1.0`, so that ±inf, signed zero
and NaN payloads are unambiguous. Nets are stored as the lowercase hex of their `.jnet`
bytes ([09](09-serialization.md)); "byte-identical" means the `.jnet` bytes are equal.

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
{"kind": "eval", "jnet": "4a4e4554…",
 "steps": [{"inputs": ["…bits…"], "outputs": ["…bits…"]}],
 "priorities": ["SPEED", "MEMORY", "BALANCED"]}
```

Steps run in order without reset. The same expected outputs MUST be produced under every
listed priority.

### `create`

```json
{"kind": "create", "seed": 1, "n_inputs": 7, "n_outputs": 2,
 "depth_type": "U16", "in_qty_max": 3, "priority": "MEMORY", "path_mix": 2, "max_run": 3,
 "expected_jnet": "4a4e4554…"}
```

### `mutate`

```json
{"kind": "mutate", "seed": 9, "generation": 12,
 "jnet": "4a4e4554…", "profile": { "…": "…" },
 "expected_jnet": "4a4e4554…"}
```

The result MUST serialize to exactly `expected_jnet`.

### `diversity`

```json
{"kind": "diversity", "seed": 3, "pct": 40.0, "sigma_reset": 1.0,
 "jnet": "4a4e4554…", "expected_jnet": "4a4e4554…"}
```

`inject_diversity(net, pct, seed(seed), sigma_reset)` MUST serialize to exactly
`expected_jnet`.

### `modify`

```json
{"kind": "modify", "op": "add_input", "args": {"position": 0, "active": false, "generation": 30},
 "jnet": "4a4e4554…", "expected_jnet": "4a4e4554…"}
```

`op` is one of the operations of [12](12-modification.md), called on `jnet` with `args`:
`add_input(position, active, generation)`, `add_output_clone(source, position, generation)`,
`add_output_fresh(seed(seed), position, generation)`, `remove_input(index)` or
`remove_output(index)`. The result MUST serialize to exactly `expected_jnet`.

### `population`

```json
{"kind": "population", "log": { "…": "run log" },
 "expected": {"generation": 1, "ids": [1, 2], "jnets": {"1": "4a4e…", "2": "4a4e…"}},
 "save": {"name": "pop", "jpnet": "4a504f50…"},
 "evaluate": {"inputs": [["…bits…"]], "outputs": {"1": [["…bits…"]], "2": [["…bits…"]]}}}
```

`Population.replay(log)` MUST yield exactly these ids, generation and nets (byte-identical),
`evaluate` with `skip_warmup = true` MUST yield exactly these outputs, and `save` to
`<dir>/pop.jpnet` MUST write exactly the `jpnet` bytes (and the listed `.jnet` files).

### `invalid`

```json
{"kind": "invalid", "jnet": "4a4e4554…", "reason": "path rule"}
```

Loading MUST fail. Cases cover every structural rule plus binary damage: wrong magic, unknown
format version, wrong spec version, truncation, trailing bytes, checksum mismatch and
out-of-range enumeration bytes.

## Required coverage

The golden set SHOULD include at least:

- Every v inversion × op, including division by zero, `bitnot(0)`, ±inf inputs and NaN inputs.
- Every aggregation op for m_neurons with empty slots, and every s op (including `lag`) for
  s_neurons during warm-up (`min` skipping unfilled entries).
- Compensated mean cases where naive summation would round differently, and overflow cases.
- Minimal-net creation for `n_inputs` = 1, ≤ `in_qty_max`, and > `in_qty_max`, with
  `path_mix` 0 and above.
- Every mix-rule skip: rewire, slot connect, slot fill and bypass.
- `growth_budget = exact`, and constant folding including the path-rule restore.
- Run-rule skips (path insertion, slot fill, taps), the run rule on load, and minimal nets with
  `max_run` 0 and above, including identity-v tree levels and chain separators.
- m_clone, m_insert, links, slot retarget, `insert_depth` both ways, every `width_bias`.
- Every addition method (including the three input taps), including skipped attempts (no gap, no eligible receiver, cap
  reached).
- Bypass, silenced removal with downstream cascade stopping at a `min` reader and at a
  nonzero constant, immunity skip, garbage collection and compaction.
- Every `DEPTH` type, including `U64` depths above 2⁵³.
- Infinite and overflowing intermediates clamped to ±MAX; v/s rewiring including the orphan
  and path-rule skips; inversions on new v_neurons; the node cap; `get_warmup_steps()`;
  `inject_diversity`.
- Every modification operation of [12](12-modification.md), at the first and last position,
  including `add_input` both ways with median insertion (nets whose m_neurons are all `min`).

## Generating golden files

The first implementation to support a feature generates its golden files; the second
implementation must then reproduce them. Any disagreement is resolved against the spec, never
by editing the expected values to match one implementation.
