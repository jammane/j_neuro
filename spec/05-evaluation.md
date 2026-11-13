# 05 — Evaluation

## API

| Operation | Effect |
|---|---|
| `step(inputs) -> outputs` | Evaluate the net once. `inputs` has `n_inputs` binary64 values; returns `n_outputs` values in output order. |
| `reset()` | Clear all state: every s_neuron history becomes all `+0.0` with `filled = 0`. |

A newly created, loaded, copied or mutated net starts in the reset state. State is never part
of the genome and is never inherited by offspring.

## Step

1. Store `canon(inputs[i])` in input node `i`.
2. Evaluate every non-input node in **ascending depth**, breaking ties by ascending node id,
   computing its neuron as defined in [04](04-neurons.md) and storing the canonical result.
3. Return the values of the output nodes in output order.

Because every neuron reads only strictly shallower nodes, all values it reads were computed
earlier in the same step. A history node holds a value from an earlier step; it is read as it
stood before this step's pushes ([04](04-neurons.md#s_neuron)), so implementations MAY push
every s_neuron's input at the end of the step, or read the ring with the offset of a push already
made. Nodes of equal depth are independent; implementations MAY evaluate
them in any order or in parallel. The tie-break rule only fixes a reference order.

## State and sequences

s_neurons make a net stateful: `step` results depend on the sequence of earlier inputs since
the last `reset`. Callers evaluating independent episodes MUST call `reset()` between them.
A single net instance MUST NOT be stepped concurrently.

## Warm-up

s_neuron histories start empty after a reset, so early outputs do not yet reflect a full
history. Because s_neurons can be chained, a net is fully warmed up only after

```
get_warmup_steps() = max over output nodes o of W(o)
W(n) = (k if n is an s_neuron else 0) + max(W(c) for c in connections of n), or the own term
       alone if n has no connections; W(input) = 0
W(c) = W(node)                                  for a connection (node, 0)
       0                                        for a length node (S, 1)
       j + W(input of S), or j if it is missing  for history node j, a connection (S, j + 1)
```

steps. A net without s_neurons has 0 warm-up steps. Fitness evaluation SHOULD ignore the
outputs of the first `get_warmup_steps()` steps after every `reset()`. Both implementations
MUST provide `get_warmup_steps()` on the net.

## Priority invariance

For any genome and input sequence, outputs MUST be bit-identical under `SPEED`, `MEMORY` and
`BALANCED`. The priority MAY affect only:

- Storage layout (dense vs. sparse, struct-of-arrays vs. compact records).
- Cached derived data (evaluation order, reader lists, input-reachability sets, immunity).
- Buffer lifetime (keeping all node values vs. freeing a value after its last reader).

It MUST NOT change the arithmetic, its order, or the values of any node.

## Nodes that do not affect outputs

Nets returned by `create`, `load` and `mutate` contain no nodes that fail to reach an output
([07](07-removal.md#garbage-collection-phase-6)), so every node is evaluated.
