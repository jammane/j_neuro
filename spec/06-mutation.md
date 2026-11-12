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
5. **Mix and run rules.** No mutation may leave a path with fewer than `path_mix` neurons of
   any type ([03](03-net-structure.md#mix-rule)), or with more than `max_run` 1:1 or many:1
   neurons in a row ([03](03-net-structure.md#run-rule)). Together they are the **structural
   rules**. Every operation that adds or moves a connection, or removes a neuron from a path,
   checks them right after it changes the net and is undone (and skipped) if one is broken:
   rewiring, connecting a slot, retargeting a slot, every growth method, and bypass.

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
| `rewire_prob` | 0.05 | yes (v, s) | Probability a v/s edit rewires the input instead of editing a parameter |
| `insert_inversion_prob` | 0.05 | yes (v) | Probability a newly added v_neuron starts with an inversion |
| `min_age_for_removal` | 5 | yes | Generations a neuron must exist before it can trigger a removal |
| `add_method_weights` | `slot_fill 0, v_insert 10, s_insert 5, m_after_m 5, v_tap 10, s_tap 10, m_tap 5, m_clone 60, m_insert 20, v_link 24, s_link 24` | no | Odds of each addition method; a weight left out counts as its default |
| `slot_fill_type_weights` | `v 1, s 1, m 1` | no | Odds of each neuron type created by slot fill |
| `rng_mode` | `fresh` | no | Where a population's `mutate` draws random numbers: `fresh` or `shared` ([11](11-population.md#rng-modes)). Ignored by the standalone `mutate` function, which takes its RNG directly. |
| `growth_budget` | `up_to` | no | `up_to` or `exact`: whether the growth budget is a random fraction of its cap or the whole cap ([budgets](#budgets)). |
| `constant_folding` | `off` | no | `off` or `on`: replace constant neurons by constants before garbage collection ([07](07-removal.md#constant-folding)). |
| `insert_depth` | `midpoint` | no | `uniform` or `midpoint`: where a new node goes within its allowed depth range ([growth](#phase-3--growth)) |
| `slot_retarget_prob` | 0.05 | yes (m) | Probability an m edit retargets one slot instead of the usual slot changes |
| `width_bias` | `deficit` | no | `none`, `center` or `deficit`: how growth chooses the m_neuron that receives a new branch ([receivers](#choosing-a-receiver)) |
| `width_peak` | 0.5 | no | Depth fraction in (0, 1) where receivers are favored most (slot fill, m_clone, links) |
| `tap_peak` | 0.25 | no | The same, for input taps |
| `link_max_span` | 0.25 | no | Longest connection a link, rewire or retarget may make, as a fraction of `D_MAX`: the new source is at most `floor(link_max_span × D_MAX)` shallower than the neuron that reads it |
| `prune` | `static` | no | `any` or `static`: with `static`, edits never cut a live branch — an m slot is emptied only if its source is constant ([07](07-removal.md#constant-folding)), and a retarget never leaves its old source without readers |

All percentages are of the **whole net's neuron count** (non-input nodes). Profiles may be
changed between generations; every change is recorded in the run log
([09](09-serialization.md#run-log-json)).

## Phases

Mutation runs these phases in this order:

1. Operation changes
2. Parameter edits
3. Growth (additions)
4. Removal ([07](07-removal.md))
5. Depth shifts
6. Constant folding if `constant_folding` is `on`, then garbage collection and compaction
   ([07](07-removal.md))

Removed nodes are only **marked** (tombstoned) until phase 6; ids do not change before then.
A **live** node is one not marked removed. The **live neuron list** is all live non-input
nodes in ascending id order.

## Budgets

At the start of each of phases 1–5:

```
N = number of live neurons
limit(pct) = 0.0                      if pct == 0
             max(1.0, pct * N / 100.0) otherwise              # a nonzero cap is at least 1
draw(pct) = prob_round(limit(pct))          if exact
            prob_round(u() * limit(pct))    otherwise
B = draw(overall.max_pct)                                     # attempts this phase
for T in [v, s, m]:
    if section T overrides max_pct:
        cap[T] = draw(T.max_pct)
    else:
        cap[T] = unlimited
    used[T] = 0
```

`exact` is true only in phase 3 when `growth_budget` is `exact`; it then draws no `u()`.
For example, a tiny `growth_max_pct` with `growth_budget = exact` makes exactly one growth
attempt (attempts can still be skipped).
`max_pct` is the phase's field (`op_change_max_pct`, …). `N` is converted to binary64 and
`pct * N / 100.0` is evaluated left to right. Because a nonzero cap is never below 1, small
nets can always mutate: a 1-neuron net with a 1% cap still gets up to 1 change.
The phase then makes `B` attempts. An attempt whose neuron type `T` has `used[T] == cap[T]`
is skipped. A successful attempt increments `used[T]`.

**Picking a neuron:** `L` = live neuron list (current); if empty, skip; otherwise take
`L[below(len(L))]`.

## Phase 1 — Operation changes

Each attempt picks a neuron and changes its operation to a different one, uniformly:

- **v:** the operation is the pair (inversion, op), indexed `inversion * 4 + op` (12 pairs).
  Draw `j = below(11)`; if `j >= current index`, `j += 1`. Set the pair and reset the weight
  to the new op's identity weight ([04](04-neurons.md#v_neuron)).
- **s:** draw `j = below(4)` over the other four s ops `[mean, median, min, max, lag]` (same
  skip-current rule).
  If the new op is `median` and `k` is even, set `k = k − 1`.
- **m:** draw `j = below(3)` over the other three aggregation ops (m_neurons have no `lag`).

## Phase 2 — Parameter edits

Each attempt picks a neuron and edits it by type.

**v and s — rewire or parameter.** The attempt first draws `x = u()`. If
`x < rewire_prob` (effective for the neuron's type) it **rewires** the input; otherwise it
edits the parameter as described below.

```
rewire:
    old = current input (may be missing)
    if old is connected and this neuron is old's only reader: skip   # would orphan old
    C = live nodes with span_floor(depth(this node)) <= depth < depth(this node), ≠ old, ascending id
    if C is empty: skip
    input = C[below(len(C))]
    if the path rule or a structural rule is now broken: restore old; skip
```

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

**m — slot retarget.** If `slot_retarget_prob` (effective for m) is greater than 0, the attempt
first draws `x = u()`; if `x < slot_retarget_prob` it retargets one slot and does nothing else.
(With `slot_retarget_prob = 0` nothing is drawn.)

```
retarget:
    S = connected slots of this neuron, ascending index; if empty: skip
    i = S[below(len(S))]; old = slot i
    if prune == static and slot i is the only connection to old: skip   # would strand old
    if emptying slot i would break the path rule: skip        # slot i is on a critical path
    C = live nodes with span_floor(depth(this node)) <= depth <= depth(old), ≠ old, not connected
        to any slot of this neuron, ascending id
    if C is empty: skip
    slot i = C[below(len(C))]
    if a structural rule is now broken: slot i = old; skip
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
        if a structural rule is now broken: slot i = empty; continue
    else:
        if prune == static and old's constant value is unknown: continue   # live source
        if emptying slot i would break the path rule: continue
        slot i = empty
```

The attempt counts as successful if at least one slot changed.

## Phase 3 — Growth

Each attempt:

```
method = weighted(add_method_weights)   # order: slot_fill, v_insert, s_insert, m_after_m,
                                        #        v_tap, s_tap, m_tap, m_clone, m_insert,
                                        #        v_link, s_link
```

then performs that method. Immediately after the method is drawn, the attempt is skipped
(with no further draws) if it would exceed the node cap ([03](03-net-structure.md#per-net-parameters)):
slot fill adds 2 nodes, every other method 1. It is also skipped if the method's new neuron type
has reached its cap (m for m_clone and m_insert, v for v_link, s for s_link), or the method is
unavailable. Every method ends by checking the structural rules; if one is broken, the attempt
is undone completely (new nodes removed, connections and depths restored, `next_birth`
restored) and skipped. The draws already made stay drawn.

**Span floor.** `span_floor(d) = d − floor(link_max_span × D_MAX)`, or 0 if that would be
negative (binary64 product, truncated): the shallowest source a link, rewire or retarget may use
for a neuron at depth `d`.

**New node depth.** Where a method below places a new node "in `[lo, hi]`":

```
place(lo, hi) = lo + (hi − lo) / 2        if insert_depth == midpoint   (integer division; no draw)
                lo + below(hi − lo + 1)   if insert_depth == uniform
```

### Choosing a receiver

Slot fill, the input taps, m_clone and the links each add a branch that ends in an empty slot of
an existing m_neuron **R**, chosen from a candidate list (ascending id) as follows. `peak` is
`tap_peak` for input taps and `width_peak` otherwise; links also multiply each weight by `e(R)`,
the number of R's empty slots.

```
none:     R = list[below(len)]                         (links: weighted(e(R)))
center:   x = depth(R) / D_MAX                          (binary64)
          tri = x / peak if x < peak else (1 − x) / (1 − peak)
          w(R) = tri + 1e-6                             (links: × e(R))
deficit:  as center, but tri = tri / max(1, width(depth(R))), where width(d) is the number of
          connections c → i between live nodes with depth(c) < d <= depth(i)
R = list[weighted(w)]
```

`center` favors receivers near `peak`; `deficit` also favors depths where the net is still
narrow, so new branches spread toward a bell-shaped width profile.

New neurons receive a new `birth` id (`next_birth++`) and `born_gen = generation`.

**Inversion of new v_neurons.** Right after a new v_neuron's `op` is drawn (v_insert, or a
v-type Q in slot fill; never the silencing wrapper W):

```
if u() < insert_inversion_prob: inversion = [bitnot, recip][below(2)]
else:                           inversion = none
```

The weight stays the op's identity weight. With an inversion the insertion is no longer
neutral (except behind the slot-fill wrapper), which the low default probability reflects.

### Path insertion (v_insert, s_insert, m_after_m, m_insert)

Inserts a new neuron **X** between an existing neuron **N** and the node **A** it writes:
after insertion N writes a new node **B**, and X reads B and writes A. Readers of A are
unchanged.

1. **Choose A.** `v_insert`, `s_insert`, `m_insert`: any live non-input node. `m_after_m`: any
   live node written by an m_neuron. Pick `A = list[below(len)]` (ascending id). None → skip.
2. **Find the gap.** `lo` = (largest depth among N's connections) + 1, or 1 if N has none.
   `hi = depth(A) − 1`.
   If `lo > hi` and A is hidden: let `ub` = (smallest depth among A's readers) − 1, or
   `D_MAX − 1` if A has no readers. If `lo ≤ ub − 1`, set `depth(A) = ub` and
   `hi = ub − 1`; otherwise skip (A is not moved).
3. **Place B.** `depth(B) = place(lo, hi)`. B is appended as a new node. N, with its
   `birth` and `born_gen`, moves to B.
4. **Create X** at A:
   - `v_insert`: v_neuron, input B, `op = choice(4)`, identity weight, inversion as above.
     **Exactly neutral** when no inversion is drawn.
   - `s_insert`: s_neuron, input B, `op = choice(5)` (s ops, including `lag`), `k = 1`.
     Disruptive by nature.
   - `m_after_m`: m_neuron, slot 0 = B, other slots empty, op `max`. Exactly neutral when B
     is never negative (e.g. N is a `max` m_neuron with an empty slot); otherwise clamps
     negative values to 0. Later slot fills into it are exactly neutral.
   - `m_insert`: m_neuron, slot 0 = B, other slots empty, op `min`. **Exactly neutral** (a `min`
     over one connected slot returns it). Its empty slots are where links, taps and clones can
     attach new branches.
5. **Structural rules.** If a run would now be too long, undo (A's neuron, `birth`, `born_gen`
   and depth restored, B removed, `next_birth` restored) and skip.

RNG order: method, A, depth(B) (if drawn), op (if drawn), inversion draws (v only). m_after_m
and m_insert draw no op.

### Slot fill

Adds a new neuron **Q** with random inputs into an empty slot of an m_neuron **R**, behind a
silencing v_neuron **W** (`mul`, weight `0.0`) so the net's outputs are unchanged.

1. **Choose R.** Live m_neurons with op ≠ `min` and at least one empty slot, ascending id.
   None → skip. R is chosen as in [choosing a receiver](#choosing-a-receiver).
2. **Choose slot.** `s = empty_slots(R)[below(count)]` (ascending slot index).
3. **Choose type.** `T = weighted(slot_fill_type_weights)` (order v, s, m). Cap reached → skip.
4. **Depths.** If `depth(R) < 3` → skip.
   `depth(W) = place(2, depth(R) − 1)`; `depth(Q) = place(1, depth(W) − 1)`.
5. **Build Q.** `C` = live nodes with depth < depth(Q), ascending id (never empty: inputs
   qualify).
   - v: `input = C[below(len(C))]`, `op = choice(4)`, inversion as above, identity weight.
   - s: `input = C[below(len(C))]`, `op = choice(5)` (s ops), `k = 1`.
   - m: `count = 1 + below(min(in_qty_max, len(C)))`; for each of `count` slots in order,
     pick a node from `C` not yet chosen (`below` over the remaining list, ascending id);
     `op = choice(4)`.
6. **Connect.** Append Q, then W (input Q). Set `R.slots[s] = W`.
7. **Structural rules.** If one is now broken, undo the attempt (remove Q and W, empty the slot
   again, restore `next_birth`) and skip. The draws already made stay drawn.

Why not into `min` receivers: `min` skips empty slots but not a real 0, so a silenced filler
would change its output.

### Input taps (v_tap, s_tap, m_tap)

Adds a new neuron **X** that reads the net's **input nodes** directly and feeds an empty slot
of an existing m_neuron **R**. Unlike slot fill, X is not silenced: R's output changes at once.

1. **Choose R.** Live m_neurons with at least one empty slot and at least one connected slot,
   ascending id. None → skip. R is chosen as in [choosing a receiver](#choosing-a-receiver),
   with `tap_peak`.
2. **Choose slot.** `s = empty_slots(R)[below(count)]` (ascending slot index).
3. **Depth.** `hi` = largest depth among the nodes R reads. If `hi < 1` (R reads only inputs)
   → skip. `depth(X) = place(1, hi)`, so X is never deeper than what R already reads.
4. **Build X.**
   - `v_tap`: v_neuron, `op = choice(4)`, `input = below(n_inputs)`, inversion as for new
     v_neurons, identity weight.
   - `s_tap`: s_neuron, `input = below(n_inputs)`, `op = choice(5)` (s ops), `k = 1`.
   - `m_tap`: m_neuron. `filled = (in_qty_max + 1) / 2` slots (half, rounded up);
     `from_inputs = min((filled + 1) / 2, n_inputs)` of them (half again, rounded up) are
     distinct input nodes, each `I[below(len(I))]` from the remaining inputs in ascending
     order; the rest, up to `filled`, are distinct nodes from
     `C` = live nodes with depth < depth(X) not yet chosen, ascending id, each
     `C[below(len(C))]` (fewer if `C` runs out). They fill slots `0…` in that order; the rest
     stay empty. Then `op = choice(4)`.
5. **Connect.** Append X (new `birth`, `born_gen = generation`) and set `R.slots[s] = X`.
6. **Structural rules.** If one is now broken, undo the attempt (remove X, empty the slot again,
   restore `next_birth`) and skip.

RNG order: method, R, slot, depth (if drawn), then for `v_tap` op, input, inversion draws; for
`s_tap` input, op; for `m_tap` the inputs, the other nodes, op.

### m_clone

Copies an m_neuron **U** that feeds an m_neuron **R** and attaches the copy to an empty slot of R,
so the branch through U is doubled.

1. **Choose R.** Live m_neurons with at least one empty slot that read at least one node written
   by an m_neuron, ascending id. None → skip. R is chosen as in
   [choosing a receiver](#choosing-a-receiver).
2. **Choose U.** The distinct m-written nodes R reads, ascending id: `U = list[below(len)]`.
3. **Choose slot.** `s = empty_slots(R)[below(count)]`.
4. **Copy.** Append a node at `depth(U)` with a copy of U's m_neuron (same op, same slots), a new
   `birth`, `born_gen = generation`; set `R.slots[s]` to it.

Every new path mirrors a path through U, so the path, mix and run rules always hold. The copy
changes R's output unless R's op is `min` or `max`.

### Links (v_link, s_link)

A new 1:1 neuron **X** connects an existing node to an empty slot of an m_neuron **R**; the open
slots m_insert leaves are the main targets.

1. **Choose R.** Live m_neurons with at least one empty slot and at least one connected slot,
   ascending id. None → skip. R is chosen as in [choosing a receiver](#choosing-a-receiver),
   each weight multiplied by R's number of empty slots.
2. **Choose slot.** `s = empty_slots(R)[below(count)]`.
3. **Choose the source.** `hi` = largest depth among the nodes R reads;
   `span = floor(link_max_span × D_MAX)` (binary64 product, truncated); `lo = depth(R) − span`, or
   0 if `depth(R) ≤ span`. `C` = live nodes with `lo ≤ depth ≤ hi` not connected to any slot of R,
   ascending id (inputs included). None → skip. `src = C[below(len(C))]`.
4. **Depth.** `lo = depth(src) + 1`, `hi2 = depth(R) − 1`; if `lo > hi2` → skip.
   `depth(X) = place(lo, hi2)`.
5. **Build X.** `v_link`: v_neuron, input `src`, `op = choice(4)`, inversion as for new v_neurons,
   identity weight. `s_link`: s_neuron, input `src`, `op = choice(5)`, `k = 1`.
6. **Connect** and check the structural rules as for input taps.

RNG order: method, R, slot, source, depth (if drawn), op, inversion draws (v only).

Informative: under `path_mix ≥ 1`, a tap's new paths skip every neuron between the inputs and
R, so taps into m_neurons close to the inputs are often undone by the mix check.

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

## Diversity injection

```
inject_diversity(parent, pct, rng, sigma_reset = 1.0) -> offspring
```

An **explicit** call, never part of `mutate`. It gives a percentage of v_neurons a new random
weight, to restore diversity when a population has converged. The parent is not modified;
the offspring starts in the reset state.

```
V     = v_neurons in ascending id order
count = min(len(V), prob_round(pct * len(V) / 100.0))     # pct * len(V) evaluated first
repeat count times:
    v = V.pop(below(len(V)))                               # distinct neurons
    v.weight = normal() * sigma_reset
```

Only weights change: operations, inversions, inputs and structure are untouched. Each call is
recorded in the run log ([09](09-serialization.md#run-log-json)).
