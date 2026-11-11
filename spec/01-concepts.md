# 01 — Concepts

## The model in one paragraph

A **net** is a set of **nodes** (values) connected by **neurons** (operations). Every node
has an integer **depth**. Input nodes sit at depth 0, output nodes at the maximum depth of the
net's `DEPTH` type, and every other node somewhere between. Each non-input node is written by
exactly one neuron, and a neuron may only read nodes strictly shallower than the node it
writes, so a net is always a directed acyclic graph that can be evaluated in depth order.
Nets are not trained by gradients; they change by **mutation**.

## Glossary

| Term | Definition |
|---|---|
| **Net** | One network: its parameters, nodes and neurons. Also called a genome when serialized. |
| **Node** | A slot holding one binary64 value per evaluation step. Has an `id`, a `depth`, a `birth` id and a `born_gen`. |
| **Input node** | Node at depth 0 whose value is supplied by the caller. Never written by a neuron. Never removed. |
| **Output node** | Node at depth `D_MAX` whose value is returned to the caller. Never removed. Never read by a neuron. |
| **Hidden node** | Any node that is neither input nor output. Depth in `[1, D_MAX − 1]`. |
| **Neuron** | The operation writing a non-input node. One per non-input node; identified with the node it writes. |
| **v_neuron** | Single input → single output. Optional inversion, then `* / + −` by a weight. |
| **s_neuron** | Single input → single output, with a history ring of the previous `k` inputs. Output `x − agg(history)`. |
| **m_neuron** | `IN_QTY_MAX` slots → single output. Output `agg(slot values)`. |
| **Slot** | One input position of an m_neuron. Either connected to a node or **empty**. |
| **Missing input** | A v_neuron or s_neuron input that is not connected. Reads as 0. |
| **Connection** | A reference from a neuron's input (or slot) to a node. Empty slots and missing inputs are not connections. |
| **Reader** | Neuron that has a connection to a given node. |
| **Path** | Sequence of connections from one node to a deeper node. |
| **Path rule** | Every input node has at least one path to every output node ([03](03-net-structure.md#path-rule)). |
| **Aggregation op** | One of `mean`, `median`, `min`, `max` ([04](04-neurons.md#aggregate)). |
| **Priority** | Per-net secondary goal: `SPEED`, `MEMORY` or `BALANCED`. Affects storage and caching only. |
| **DEPTH** | Per-net integer type of depth values: `U16`, `U32`, `U64`. `D_MAX` is its maximum value. |
| **IN_QTY_MAX** | Per-net odd integer ≥ 3: number of slots per m_neuron and maximum s_neuron history length. |
| **Mutation profile** | Configuration controlling mutation rates, sigmas and addition-method weights ([06](06-mutation.md)). |
| **Generation** | Integer counter supplied by the caller to each mutation; used for neuron ages. |
| **Identity neuron** | v_neuron that returns its input unchanged ([07](07-removal.md#candidates)). |
| **Silenced neuron** | Neuron whose output is always 0 ([07](07-removal.md#candidates)). |
| **Immune neuron** | Neuron whose removal would break the path rule ([07](07-removal.md#immunity)). |
| **Canonical value** | Finite binary64 value that is not −0.0 ([02](02-numerics.md#canonical-values)). |
| **Warm-up steps** | Steps after a reset before every s_neuron history is filled ([05](05-evaluation.md#warm-up)). |
