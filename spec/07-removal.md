# 07 — Removal, immunity and garbage collection

Removal shrinks a net **without changing its outputs**. Only neurons proven not to matter
are removed, and the cleanup they cause cascades both downstream and upstream.

## Candidates

A neuron is a removal **candidate** if all of the following hold:

1. It writes a **hidden** node (output neurons are never removed; inputs have no neuron).
2. `generation − born_gen ≥ min_age_for_removal` (effective value for its type). This grace
   period lets newly inserted neutral neurons survive long enough to be tuned.
3. It is an **identity neuron** or a **silenced neuron**:

**Identity neuron:** a v_neuron with a connected input, inversion `none`, and the identity
weight of its op (`mul 1.0`, `div 1.0`, `add 0.0`, `sub 0.0`). It returns its input exactly.

**Silenced neuron:** a neuron whose output is always `+0.0`:

- a v_neuron with op `mul` and weight `0.0` (any inversion, any input) — `y * 0` is ±0 or
  NaN, both canonicalized to `+0.0`; or
- a neuron with **no connections** (missing input, or all slots empty) whose output, evaluated
  from the reset state, is `+0.0`:
  - s_neuron: always (input 0, history 0, result 0);
  - m_neuron: always (all aggregations of empty slots give 0);
  - v_neuron: when `canon(op(inversion(+0.0), w)) == +0.0` (for example `mul`, `div` with a
    nonzero weight, or any op after `bitnot`, which yields NaN).

Connection-less neurons whose constant output is **not** 0 (for example `add w`) are
**constants** (bias sources) and are never candidates.

## Removal phase

Phase 4 of mutation ([06](06-mutation.md#phases)). The budget is drawn as for other phases
with `removal_max_pct`. Each attempt:

```
X = pick a live neuron                      # 06: picking a neuron
if cap reached for type(X): skip
if X is not a candidate: skip
if X is an identity neuron:
    bypass(X)
    if a structural rule (mix or run) is now broken: undo the bypass; skip
else:                                        # silenced
    if any reader of X is an m_neuron with op min: skip
    if X is immune: skip
    remove_silenced(X)
```

Only the triggering neuron counts toward the budget and cap. Cascaded removals are free.

### Bypass (identity neurons)

Let `A` be X's node and `I` its input node. Every connection to `A` (m slots and v/s inputs)
is redirected to `I`, then `A` is marked removed. Readers of `A` are deeper than `A`, which is
deeper than `I`, so depth rules hold. Paths are only shortened, so the path rule holds.
Outputs are unchanged because X returned its input exactly. Every path through `A` loses one
v_neuron, which can break the mix rule ([03](03-net-structure.md#mix-rule)) or join two runs into
one too long for the run rule ([03](03-net-structure.md#run-rule)); the attempt is
then undone and skipped.

### Removing a silenced neuron (downstream cascade)

```
queue = [X]
while queue not empty:
    Y = queue.pop_front()
    R = readers of Y, ascending id
    for each connection to Y: m slot -> empty; v/s input -> missing
    mark Y removed
    for each r in R (ascending id, each once):
        if r writes a hidden node
           and r now has no connections and is silenced
           and no reader of r is an m_neuron with op min:
            queue.push_back(r)
```

Exactness: Y always produced `+0.0`, so every reader that loses Y reads the same `+0.0` as
before (an empty slot or missing input reads 0). The one exception is a `min` m_neuron, which
skips empty slots but not real zeros, so the cascade **stops** there. A reader that becomes a
nonzero constant stays as a bias source, and the cascade stops there too.

The upstream half of the cascade — nodes that no longer reach any output — is handled by
garbage collection in phase 6.

## Immunity

A neuron X is **immune** if removing every connection that reads X's node would break the
path rule, i.e. some input would have no path left to some output. Equivalently, X dominates
some output from some input.

- Only the trigger of a silenced removal needs the check. Bypass never breaks paths. Cascaded
  neurons have no remaining connections, so no path passes through them. Upstream dead nodes
  reach no output, so no required path passes through them.
- The same test protects m slots: emptying a slot is refused if it would break the path rule
  ([06](06-mutation.md#phase-2--parameter-edits)).
- Immunity is **derived state**. It is never serialized. It can change after any structural
  mutation: additions can only remove immunity (they add paths), removals can only add it.
- Implementations MAY compute it with dominator trees (one per input), cached and updated
  incrementally (`SPEED`) or recomputed on demand (`MEMORY`). The result MUST equal the
  definition above.

Informative: over generations a net can reroute a required path — grow an alternative route,
which lifts the old route's immunity, then remove the old route once it is silenced.

## Constant folding (phase 6)

Runs only when the profile's `constant_folding` is `on`, before garbage collection. It draws
no random numbers and has no budget or cap.

**Constant values.** Visit live nodes in ascending depth. A node's constant value `c` is known
when its output is provably the same at every step from the reset state:

| Neuron | `c` is known when | `c` |
|---|---|---|
| v_neuron | input missing | `eval_v(+0.0)` |
| | input's `c` known | `eval_v(c(input))` |
| | otherwise, op `mul` and weight `0.0` | `+0.0` |
| s_neuron | input missing, or input's `c` is `+0.0` | `+0.0` |
| m_neuron | every connected slot's `c` is known (also when all slots are empty) | `canon(aggregate(op, values, present))`, empty slots `+0.0` and not present |

Input nodes have no constant value. (An s_neuron of any other constant is not constant: it
outputs the constant on the first step and then the constant minus its history.)

**Folding.** Candidates are live **hidden** nodes with a known `c`, at least one connection,
and `generation − born_gen ≥ min_age_for_removal` (effective for the neuron's type). Visit
them by descending depth, ties by ascending id. Replace the neuron by a constant v_neuron
(input missing, inversion `none`, op `add`, weight `c`); if the path rule is now broken,
restore it. `birth` and `born_gen` are kept.

The replacement outputs `canon(+0.0 + c) = c` at every step, so outputs are unchanged and
`min` readers are unaffected (the slot stays connected). Nodes upstream that no longer reach
an output are removed by garbage collection. Folding only removes connections, so the mix
rule still holds.

Informative: the silencing wrapper W of a slot fill is a constant (`mul 0`); once it is
`min_age_for_removal` generations old it is folded and its filler removed, unless an operation
change has made it non-constant.

## Garbage collection (phase 6)

1. Mark every live node from which some output node is reachable through connections
   (outputs themselves included). Input nodes are always kept.
2. Mark every unmarked live hidden node as removed.

Such nodes cannot influence any output, so removing them never changes results.

## Compaction

After garbage collection, removed nodes are deleted. Remaining nodes keep their relative
order and are renumbered `0, 1, 2, …`; every connection is updated to the new ids. `birth`
ids and `next_birth` are unchanged.

The returned offspring always satisfies all invariants of [03](03-net-structure.md) and
contains no removed or dead nodes.
