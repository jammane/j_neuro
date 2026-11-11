# 06 — Mutation

```
mutate(parent, profile, generation, rng) -> offspring
```

The offspring starts as an exact copy of the parent's genome, in the reset state
([05](05-evaluation.md)). The parent is not modified. Every random draw comes from `rng`
using the procedures of [08](08-rng.md), in exactly the order this document lists them.

## Principles

1. **Accuracy first.** Additions start with the least disruptive values available; removal
   only removes neurons whose removal leaves outputs exactly unchanged.
2. **"Up to" budgets.** Every rate is a maximum. The actual amount is drawn in `[0, max]`.
3. **Skip rule.** An attempt that is unavailable or would violate a rule is skipped: nothing
   changes, no further random draws are made for that attempt, and mutation continues with
   the next attempt.
4. **Path rule.** No mutation may leave an input without a path to an output
   ([03](03-net-structure.md#path-rule)).

## Mutation profile

A profile has an **overall** section and optional **per-type** sections for `v`, `s` and `m`.
A per-type value overrides the overall value for neurons of that type.

| Field | Default | Per-type override | Meaning |
|---|---|---|---|
| `op_change_max_pct` | 1.0 | yes | Max % of neurons whose operation changes |
| `edit_max_pct` | 5.0 | yes | Max % of neurons whose parameters are edited |
| `growth_max_pct` | 1.5 | yes | Max % growth (new neurons) |
| `removal_max_pct` | 1.0 | yes | Max % of neurons removed (triggers only, see [07](07-removal.md)) |
| `depth_shift_max_pct` | 5.0 | yes | Max % of neurons whose node depth is shifted |
| `depth_shift_mode` | `uniform` | no | `uniform` or `center` |
| `sigma_weight` | 0.1 | yes (v) | Weight mutation scale |
| `sign_flip_prob` | 0.02 | yes (v) | Probability a weight edit also flips its sign |
| `sigma_k` | 1.0 | yes (s) | History-length mutation scale |
| `sigma_slots` | 1.0 | yes (m) | Scale of how many slots change per m edit |
| `min_age_for_removal` | 5 | yes | Generations a neuron must exist before it can trigger a removal |
| `add_method_weights` | `slot_fill 40, v_insert 30, s_insert 20, m_after_m 10` | no | Odds of each addition method |
| `slot_fill_type_weights` | `v 1, s 1, m 1` | no | Odds of each neuron type created by slot fill |

All percentages are of the **whole net's neuron count** (non-input nodes). Profiles may be
changed between generations; every change is recorded in the run log
([09](09-serialization.md#run-log)).

## Phases

Mutation runs these phases in this order:

1. Operation changes
2. Parameter edits
3. Growth (additions)
4. Removal ([07](07-removal.md))
5. Depth shifts
6. Garbage collection and compaction ([07](07-removal.md))

Removed nodes are only **marked** (tombstoned) until phase 6; ids do not change before then.
A **live** node is one not marked removed. The **live neuron list** is all live non-input
nodes in ascending id order.

## Budgets

At the start of each of phases 1–5:

```
N = number of live neurons
B = prob_round((u() * overall.max_pct) * N / 100.0)          # attempts this phase
for T in [v, s, m]:
    if section T overrides max_pct:
        cap[T] = prob_round((u() * T.max_pct) * N / 100.0)
    else:
        cap[T] = unlimited
    used[T] = 0
```

`max_pct` is the phase's field (`op_change_max_pct`, …). `N` is converted to binary64.
The phase then makes `B` attempts. An attempt whose neuron type `T` has `used[T] == cap[T]`
is skipped. A successful attempt increments `used[T]`.

**Picking a neuron:** `L` = live neuron list (current); if empty, skip; otherwise take
`L[below(len(L))]`.

## Phase 1 — Operation changes

Each attempt picks a neuron and changes its operation to a different one, uniformly:

- **v:** the operation is the pair (inversion, op), indexed `inversion * 4 + op` (12 pairs).
  Draw `j = below(11)`; if `j >= current index`, `j += 1`. Set the pair and reset the weight
  to the new op's identity weight ([04](04-neurons.md#v_neuron)).
- **s:** draw `j = below(3)` over the other three aggregation ops (same skip-current rule).
  If the new op is `median` and `k` is even, set `k = k − 1`.
- **m:** draw `j = below(3)` over the other three aggregation ops.

## Phase 2 — Parameter edits

Each attempt picks a neuron and edits it by type.

**v — weight.**

```
z = normal()
if op in (add, sub):  w2 = w + sigma_weight * z
else:                 a = sigma_weight * z
                      f = 1.0 + a       if a >= 0
                          1.0 / (1.0 - a) otherwise
                      w2 = w * f
if u() < sign_flip_prob: w2 = -w2
if w2 is not finite: skip
w = w2
```

Multiplicative edits keep a weight of 0 at 0 (a silenced neuron stays silenced).

**s — history length.**

```
z  = normal()
d  = round_half_away(sigma_k * z)
k2 = clamp(k + d, 1, in_qty_max)
if op == median and k2 is even:
    k2 = k2 - 1 if k2 > k else k2 + 1
if k2 == k: skip
k = k2
```

**m — slots.**

```
z = normal()
c = min(in_qty_max, 1 + floor(|sigma_slots * z|))
repeat c times:
    i = below(in_qty_max)
    if slot i is empty:
        C = live nodes with depth < depth(this node), not connected to any slot of this
            neuron, ascending id
        if C is empty: continue
        slot i = C[below(len(C))]
    else:
        if emptying slot i would break the path rule: continue
        slot i = empty
```

The attempt counts as successful if at least one slot changed.

## Phase 3 — Growth

Each attempt:

```
method = weighted(add_method_weights)   # order: slot_fill, v_insert, s_insert, m_after_m
```

then performs that method. If the method's new neuron type has reached its cap, or the
method is unavailable, the attempt is skipped.

New neurons receive a new `birth` id (`next_birth++`) and `born_gen = generation`.
New v_neurons always have inversion `none`.

### Path insertion (v_insert, s_insert, m_after_m)

Inserts a new neuron **X** between an existing neuron **N** and the node **A** it writes:
after insertion N writes a new node **B**, and X reads B and writes A. Readers of A are
unchanged.

1. **Choose A.** `v_insert`, `s_insert`: any live non-input node. `m_after_m`: any live node
   written by an m_neuron. Pick `A = list[below(len)]` (ascending id). None → skip.
2. **Find the gap.** `lo` = (largest depth among N's connections) + 1, or 1 if N has none.
   `hi = depth(A) − 1`.
   If `lo > hi` and A is hidden: let `ub` = (smallest depth among A's readers) − 1, or
   `D_MAX − 1` if A has no readers; if `ub > depth(A)`, set `depth(A) = ub` and recompute
   `hi`. If still `lo > hi` → skip.
3. **Place B.** `depth(B) = lo + below(hi − lo + 1)`. B is appended as a new node. N, with its
   `birth` and `born_gen`, moves to B.
4. **Create X** at A:
   - `v_insert`: v_neuron, input B, `op = choice(4)`, identity weight. **Exactly neutral.**
   - `s_insert`: s_neuron, input B, `op = choice(4)`, `k = 1`. Disruptive by nature.
   - `m_after_m`: m_neuron, slot 0 = B, other slots empty, op `max`. Exactly neutral when B
     is never negative (e.g. N is a `max` m_neuron with an empty slot); otherwise clamps
     negative values to 0. Later slot fills into it are exactly neutral.

RNG order: method, A, depth(B), op (if drawn).

### Slot fill

Adds a new neuron **Q** with random inputs into an empty slot of an m_neuron **R**, behind a
silencing v_neuron **W** (`mul`, weight `0.0`) so the net's outputs are unchanged.

1. **Choose R.** Live m_neurons with op ≠ `min` and at least one empty slot, ascending id.
   None → skip. `R = list[below(len)]`.
2. **Choose slot.** `s = empty_slots(R)[below(count)]` (ascending slot index).
3. **Choose type.** `T = weighted(slot_fill_type_weights)` (order v, s, m). Cap reached → skip.
4. **Depths.** If `depth(R) < 3` → skip.
   `depth(W) = 2 + below(depth(R) − 2)`; `depth(Q) = 1 + below(depth(W) − 1)`.
5. **Build Q.** `C` = live nodes with depth < depth(Q), ascending id (never empty: inputs
   qualify).
   - v: `input = C[below(len(C))]`, `op = choice(4)`, identity weight.
   - s: `input = C[below(len(C))]`, `op = choice(4)`, `k = 1`.
   - m: `count = 1 + below(min(in_qty_max, len(C)))`; for each of `count` slots in order,
     pick a node from `C` not yet chosen (`below` over the remaining list, ascending id);
     `op = choice(4)`.
6. **Connect.** Append Q, then W (input Q). Set `R.slots[s] = W`.

Why not into `min` receivers: `min` skips empty slots but not a real 0, so a silenced filler
would change its output.

## Phase 4 — Removal

Defined in [07](07-removal.md#removal-phase).

## Phase 5 — Depth shifts

Each attempt picks a node from the live **hidden** nodes (ascending id; none → skip). Its
neuron's type is used for caps.

```
lo = (largest depth among its neuron's connections) + 1, or 1 if none
hi = (smallest depth among its readers) − 1, or D_MAX − 1 if none
if lo == hi: skip
uniform: d = lo + below(hi − lo + 1)
center:  d = lo + (hi − lo) / 2           # integer division
if d == depth: skip
depth = d
```

Depth shifts never change outputs. They redistribute gaps so later insertions find room.
