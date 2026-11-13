# 12 — Modification: adding and removing inputs and outputs

A net's `n_inputs` and `n_outputs` are fixed by [`create`](03-net-structure.md#minimal-net)
and by mutation. The five operations below change them on an existing, usually trained, net:

```
remove_input(net, index)                                   -> net
remove_output(net, index)                                  -> net
add_input(net, position = n_inputs, active = false, generation = 0)  -> net
add_output_clone(net, source, position = n_outputs, generation = 0)  -> net
add_output_fresh(net, rng, position = n_outputs, generation = 0)     -> net
```

Each one checks that the given net is valid ([09](09-serialization.md)), leaves it unchanged, and
returns a new net in the reset state ([05](05-evaluation.md)). The new net satisfies every
invariant of [03](03-net-structure.md): the depth, path, mix and run rules, the node cap, and no
dead nodes. All other parameters are kept, including `priority`, `depth_type`, `in_qty_max`,
`path_mix` and `max_run`. Every operation is deterministic apart from the RNG that
`add_output_fresh` takes.

A population only holds nets of one shape ([11](11-population.md)). To keep training a
modified net, put it in a population of the new shape with `add(net)`.

## Common rules

**Errors.** An operation fails with a validation error, and returns nothing, when:

- an index or position is out of range (`index < count`, `position ≤ count`);
- a removal would leave zero inputs or zero outputs;
- the result would exceed the node cap;
- `add_input` finds no depth room for its chain (below).

**Positions.** Inputs keep caller order, and so do outputs ([03](03-net-structure.md#nodes)).

- A new input takes input index `position`. A new output takes output index `position`.
- The node is inserted at that id, and every node id at or above it moves up by 1.
- A removed input or output is deleted, and the nodes after it move down. This is the
  [compaction](07-removal.md#compaction) of spec 07.

**New nodes.** Every new node gets `birth = next_birth++`, in the creation order this document
gives, and `born_gen = generation`. Choose `generation` as the trainer's current generation so
that the grace period of [07](07-removal.md#candidates) protects the new nodes.

**Reachability.** "X reaches Y" means a sequence of connections leads from X to Y
([03](03-net-structure.md#path-rule)).

## remove_output

1. Mark output `n_inputs + index` removed.
2. Run [garbage collection](07-removal.md#garbage-collection-phase-6). It removes every hidden
   node that no longer reaches any output. These are exactly the nodes that only fed the
   removed output; shared nodes still reach another output and stay.
3. Compact, then set `n_outputs −= 1`.

The other outputs are unchanged, because nothing on any path to them was removed.

## remove_input

Let `x` be input `index`. Then:

```
A = nodes reachable from x
B = nodes reachable from any other input
doomed = {x} ∪ (A \ B)                   # only x could reach these
for each live node not in doomed:
    every connection to a doomed node (any position): m slot -> empty; v/s input -> missing
mark every doomed node removed
garbage collection; compact; n_inputs −= 1
```

- **Downstream.** `doomed` holds the hidden neurons that depended only on `x`. Neurons shared
  with another input stay; they just lose the slot that read a doomed node.
- **Upstream.** Garbage collection then removes whatever fed only the doomed neurons, for
  example the bias constants they read.
- **Outputs.** By the path rule every output is reachable from the other inputs, so outputs are
  never doomed.
- **Kept v and s neurons.** A kept v or s neuron never loses its input: its only predecessor is
  reachable from another input.

Rules:

- The path rule still holds: no path from a remaining input passed through a doomed node.
- Removing predecessors can only raise the mix rule's fewest counts and shorten runs, so the
  mix and run rules still hold.

Outputs change, since the input is gone. Nothing is preserved exactly.

## add_output_clone

The new output starts as an exact copy of output `source`.

1. Copy the neuron of output `source`. `source` is an index in the original net.
2. Insert a new output node with depth `D_MAX` and that neuron at output index `position`.
3. Shift ids. The copy reads the same nodes as the original.

The copy has the same connections and neuron type as its source, so it gets the same paths,
the same mix triple and the same runs, and every rule holds. The existing outputs are unchanged.
The new output's values equal output `source`'s until mutation changes either one.

## add_output_fresh

The new output gets its own minimal chain from the inputs, independent of the trained part of the
net.

1. `G = create(n_inputs, 1, depth_type, in_qty_max, priority, rng, path_mix, max_run)`
   ([03](03-net-structure.md#minimal-net)), drawing from the given `rng`.
2. Insert a new output node at output index `position`. It takes the neuron of `G`'s output.
3. Append `G`'s hidden nodes after all existing nodes, in `G`'s id order, keeping their
   depths and neurons.
4. Map references: `G`'s inputs → the net's inputs, `G`'s output → the new output, `G`'s
   hidden nodes → their appended ids.

Births are assigned to the new output first, then to the appended nodes in order.

`G` satisfies every rule by construction, and its nodes read only inputs and each other, so the
existing outputs are unchanged.

## add_input

Insert a new input node `I` at input index `position`, with depth 0. Its birth is assigned
first. Then connect `I` to every output through one shared **chain** of new neurons that ends in
a v_neuron **W**. `W` feeds an empty slot of one or more existing m_neurons, the **receivers**,
which between them reach every output.

**W.** With `active = false` (the default), W is a silencing v_neuron: op `mul`, inversion
`none`, weight `0.0`, as in [slot fill](06-mutation.md#slot-fill).

- W always outputs `+0.0`. A receiver whose op is `mean`, `median` or `max` reads an empty slot
  as 0 too, so **every existing output stays bit-for-bit unchanged** for any value fed to `I`.
- Mutation activates the new input later. An operation change gives W an identity weight;
  weight edits keep 0 at 0.
- Removal cannot delete W while it is the new input's only route: W is
  [immune](07-removal.md#immunity), and folding it would break the path rule.

With `active = true`, W is an identity v_neuron (`mul`, `1.0`). The new input then affects
outputs at once.

The steps, all on the net after `I` is inserted:

**1. Receivers.** The candidates are live m_neurons, hidden or output, that have at least one
empty slot, and whose op is not `min` unless `active`. `outs(R)` is the set of outputs reachable
from `R`. Greedy cover:

```
uncovered = all outputs
repeat:
    R = candidate with the most outputs in outs(R) ∩ uncovered;
        ties: smaller depth, then smaller id
    if that count is 0: stop
    add R to the receivers; uncovered −= outs(R)
```

**2. Median insertion.** This happens for each output still uncovered, in ascending output
order. Let `N` be the output's neuron and `O` the output node.

1. Compute the gap. `lo = floor(N)` ([06](06-mutation.md#principles); N's own s gap included),
   and `hi = D_MAX − 1`. If `lo > hi`, fail.
2. Place B. A new node B gets depth `lo + (hi − lo) / 2` (integer division). N moves to B, and so
   do O's `birth` and `born_gen`. B is appended.
3. Separator, only if N is an m_neuron and `max_run > 0`: append a node C with depth
   `depth(B) + 1 + (hi − depth(B) − 1) / 2`, written by an identity v_neuron (`mul`, `1.0`)
   that reads B. This needs `depth(B) < hi`; otherwise fail. Let `S = C`, or `S = B` when there
   is no separator.
4. Give O the median neuron. O gets a new m_neuron with op `median`: slots
   `0 … (in_qty_max − 1) / 2` read S, and the remaining `(in_qty_max − 1) / 2` slots are empty.
   O's new birth is assigned now, before C's.
5. O becomes a receiver.

The median is exact. Its `(in_qty_max + 1) / 2` copies of S's value are a majority, so the
median at index `in_qty_max // 2` is that value, and canonical values are finite. A 0 in an
empty slot, or from W, does not change that.

The separator keeps the run rule. Without it, a median after an m_neuron could lengthen an m run;
with it, the median's m run is 1.

**3. Chain contents.** Let `fewest(R)` be, for each type t in (v, s, m), the minimum number of
type-t neurons on any path from R (R's own neuron included) to an output. It is computed by
descending structural depth ([03](03-net-structure.md#connections-and-history-nodes)): an output counts only its own neuron, and a hidden node adds its own neuron to
the element-wise minimum over its readers. With `P = path_mix`:

```
need[t] = max over receivers R of max(0, P − fewest(R)[t])
parts   = round robin m, s, v (each while some remain) over
          need[m] × m,  need[s] × s,  max(0, need[v] − 1) × v
parts  += [W]
```

If `max_run = K > 0`, separators are inserted as in `create`. Start with the run counters at
`1:1 = 0` and `many:1 = 0`, then walk the parts:

- before an m part, if `many:1 == K`, insert an identity v and reset `many:1` to 0;
- before a 1:1 part (s, v or W), if `1:1 == K`, insert an m part and reset `1:1` to 0.

**4. Build.** Neurons by part:

| Part | Neuron |
|---|---|
| m and m separators | m_neuron, op `min`, slot 0 = previous, other slots empty |
| s | s_neuron, op `mean`, `k = 1` |
| v and v separators | identity v_neuron (`mul`, `1.0`) reading previous |
| W | as described under **W** above, reading previous |

"Previous" is I for the first part. Each part is appended as a new node, in chain order.

Let `L` be the number of parts and `d` the smallest receiver depth. With
`step = d / (L + 1)` (integer division), part `j = 1 … L` gets depth `step × j`. If
`step = 0`, or the chain has an s part and `step ≤ in_qty_max + 1` (the s gap,
[03](03-net-structure.md#depth-rules)), fail with no depth room.

**5. Connect.** For each receiver, in the order they were chosen, set its lowest empty slot to W.

**Why the rules hold.**

- Path rule. I reaches every output through W and the receivers. Existing paths are untouched,
  since every new node reads only I and the chain.
- Mix rule. Each new path is the chain plus a path from a receiver R, and it has at least
  `need[t] + fewest(R)[t] ≥ P` neurons of each type t. Existing paths are unchanged, and the
  median insertions only add neurons to them.
- Run rule. Separators bound the runs inside the chain. W is 1:1 and every receiver is many:1,
  so the run of 1:1 neurons ends at W and no receiver's run of m neurons grows.
