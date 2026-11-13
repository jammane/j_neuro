# 04 — Neurons

All values read by a neuron are canonical ([02](02-numerics.md#canonical-values)). Every
neuron output is passed through `canon` before being stored.

Enumeration orders below are normative; they define what an RNG choice index means
([08](08-rng.md#derived-procedures)).

Every connection reads a canonical value: the node's value for position 0, the length node's for
position 1, history node `j`'s for position `j + 1` ([03](03-net-structure.md#connections-and-history-nodes),
[s_neuron](#s_neuron)). `value(c)` below means the value of connection `c`.

## aggregate

Shared by s_neurons and m_neurons.

```
aggregate(op, values, present) -> binary64
```

- `values`: ordered list of `n` canonical values (slot order, or oldest-to-newest history).
- `present`: for each value, whether it comes from a connected slot / filled history entry
  (`true`) or from an empty slot / unfilled history entry (`false`). Non-present values are
  `+0.0`.
- Aggregation ops, in enumeration order: `[mean, median, min, max]`.

| op | Participating values | Result |
|---|---|---|
| `mean` | all `n` | compensated mean (below) |
| `median` | all `n` (`n` is always odd) | sort ascending, take element at index `n // 2` |
| `min` | **present values only** | smallest; `+0.0` if none are present |
| `max` | all `n` | largest |

Empty slots and unfilled history entries therefore count as 0 for `mean`, `median` and
`max`, and are skipped by `min`.

**Compensated mean** (Neumaier), over the participating values `v[0..n-1]` in order:

```
s = 0.0; c = 0.0
for v in values:
    t = s + v
    if |s| >= |v|: c = c + ((s - t) + v)
    else:          c = c + ((v - t) + s)
    s = t
if s is finite: total = s + c
else:           total = s            # infinities/overflow: plain IEEE sum
mean = total / n                      # n converted to binary64
```

## v_neuron

| Field | Values |
|---|---|
| `input` | connection, or missing |
| `inversion` | `[none, bitnot, recip]` (enumeration order) |
| `op` | `[mul, div, add, sub]` (enumeration order) |
| `weight` | finite binary64 |

```
x = value(input)  or  +0.0 if missing
y = x             if inversion == none
    bitnot(x)     if inversion == bitnot     # see 02-numerics
    1.0 / x       if inversion == recip
z = y * w | y / w | y + w | y - w            # by op
output = canon(z)
```

**Identity weights** (used for new neurons and after op changes):

| op | identity weight |
|---|---|
| `mul` | `1.0` |
| `div` | `1.0` |
| `add` | `0.0` |
| `sub` | `0.0` |

With inversion `none` and the identity weight, a v_neuron returns its (canonical) input
exactly.

## s_neuron

| Field | Values |
|---|---|
| `input` | connection, or missing |
| `op` | `[mean, median, min, max, lag]` (enumeration order): the four aggregation ops plus `lag`, which only s_neurons have |
| `k` | integer, `1 ≤ k ≤ in_qty_max`; **odd if `op == median`** |
| state: `history` | ring of `k` values, initially all `+0.0` |
| state: `filled` | number of entries written since reset, capped at `k` |

The history holds the **previous** `k` input values; it never includes the current one.

```
x = value(input)  or  +0.0 if missing
h = history, oldest to newest (k values; unfilled entries are +0.0)
present[i] = i is among the `filled` most recent entries
output = canon(h[0])                            if op == lag   # the input from k steps ago
         canon(x - aggregate(op, h, present))   otherwise
push x into history (dropping the oldest), filled = min(filled + 1, k)
```

**Length and history nodes.** Both hold their value as it is at the **start** of the step, before
this step's push; every reader sees that value, whether it is evaluated before or after the
s_neuron itself.

- The length node (position 1) holds `filled` as a binary64: 0 right after a reset, then one more
  each step up to `k`. With `mean` it recovers the sum of the filled history.
- History node `j` (position `j + 1`, `1 ≤ j ≤ k`) holds `h[k − j]`: the input from `j` steps
  ago, or `+0.0` while that entry is unfilled (`j > filled`). History node `k` is what `lag`
  outputs.

On the first step after a reset all entries are unfilled, so `aggregate` returns 0 and the
output equals `x`. A `lag` s_neuron outputs the raw input from `k` steps ago, and `+0.0` for
the first `k` steps after a reset (its oldest entry is still unfilled); `k` is its position
in the history.

## m_neuron

| Field | Values |
|---|---|
| `slots` | list of exactly `in_qty_max` entries, each a connection or empty |
| `op` | aggregation op |

```
values[i]  = value(slots[i]) if connected else +0.0
present[i] = slots[i] is connected
output = canon(aggregate(op, values, present))
```

`median` always sees `in_qty_max` values, which is odd.

## Informative notes

- **Nonlinearity** comes from `recip`, `bitnot`, `min`, `max` and `median`. An m_neuron with
  `max` and at least one empty slot computes `max(inputs…, 0)`, i.e. a built-in ReLU.
- **Weighted sums** are built as v_neurons (`* wᵢ`) feeding an m_neuron (`mean`); the mean
  divides by `in_qty_max`, which a following v_neuron (`* in_qty_max`) can undo.
- **Constants**: a v_neuron or s_neuron with a missing input outputs a constant (for example
  `+ w` gives `w`), acting as a bias source.
