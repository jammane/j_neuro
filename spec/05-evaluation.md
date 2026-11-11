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
earlier in the same step. Nodes of equal depth are independent; implementations MAY evaluate
them in any order or in parallel. The tie-break rule only fixes a reference order.

## State and sequences

s_neurons make a net stateful: `step` results depend on the sequence of earlier inputs since
the last `reset`. Callers evaluating independent episodes MUST call `reset()` between them.
A single net instance MUST NOT be stepped concurrently.

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
