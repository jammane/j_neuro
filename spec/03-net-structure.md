# 03 — Net structure

## Per-net parameters

| Parameter | Values | Meaning |
|---|---|---|
| `priority` | `SPEED`, `MEMORY`, `BALANCED` | Secondary goal after accuracy. Changes storage, caching and memory reuse only. MUST NOT change any result. |
| `depth_type` | `U16`, `U32`, `U64` | Unsigned integer type of depth values. `D_MAX` = 2¹⁶−1, 2³²−1 or 2⁶⁴−1. Fixed width on every platform. |
| `in_qty_max` | odd integer ≥ 3 | Slots per m_neuron; maximum s_neuron history length `k`. |
| `n_inputs` | integer ≥ 1 | Number of input nodes. |
| `n_outputs` | integer ≥ 1 | Number of output nodes. |

`depth_type`, `in_qty_max`, `n_inputs` and `n_outputs` are fixed when the net is created.
`priority` MAY be changed at any time without affecting results.

**Node cap.** A net MUST NOT have more than `D_MAX` nodes in total (inputs, outputs and
hidden). `depth_type` therefore bounds net size directly: at most 65,535 nodes for `U16`,
about 4.3 billion for `U32`, and effectively unbounded for `U64`. It also bounds how often a
new node can be inserted between two existing ones.

## Nodes

Each node has:

| Field | Meaning |
|---|---|
| `id` | Position in the node list (0-based, dense). Changes on compaction ([07](07-removal.md#compaction)). |
| `birth` | Permanent unique id, assigned from the net's `next_birth` counter. Never reused. For lineage tracking only. |
| `born_gen` | Generation in which the node was created. Used for removal age. |
| `depth` | Unsigned integer of the net's `depth_type`. |
| `neuron` | The neuron that writes this node. Absent for input nodes. |

**Node order.** Node ids `0 … n_inputs−1` are the inputs, in caller order. Ids
`n_inputs … n_inputs+n_outputs−1` are the outputs, in caller order. Hidden nodes follow.
New nodes are appended at the end.

## Depth rules

1. Input nodes have depth `0`. Output nodes have depth `D_MAX`. Hidden nodes have depth in
   `[1, D_MAX − 1]`.
2. Every connection of a neuron MUST refer to a node with depth **strictly less than** the
   depth of the node the neuron writes.
3. Any number of nodes may share a depth.

Consequences: nets are acyclic; output nodes have no readers; nodes at the same depth never
depend on each other.

## Neuron rules

- Every non-input node has exactly one neuron; input nodes have none.
- An m_neuron has exactly `in_qty_max` slots, each empty or connected. Several slots MAY
  connect to the same node (this can result from identity bypass, [07](07-removal.md)).
- A v_neuron or s_neuron input MAY be missing. Such a neuron reads 0 and acts as a constant.

## Path rule

**Every input node MUST have at least one path to every output node.**

- A path is made of connections only; empty slots and missing inputs are not part of any
  path.
- The rule is structural: a connection counts even if its value cannot influence the output
  (for example through a v_neuron `* 0`).
- Every path from an input to an output contains at least one neuron, because inputs and
  outputs are distinct nodes.
- The rule is checked on load ([09](09-serialization.md)) and is preserved by every mutation
  ([06](06-mutation.md), [07](07-removal.md)).

## Minimal net

`create(n_inputs, n_outputs, depth_type, in_qty_max, priority, rng)` builds the smallest net
satisfying the path rule. The caller supplies only the input and output counts; no layers or
starting neurons are specified by hand. All other arguments are optional, with these defaults
(identical in both implementations): `depth_type = U32`, `in_qty_max = 5`,
`priority = BALANCED`, `rng = seed(0)`. All new nodes get `born_gen = 0` and consecutive `birth` ids equal
to their `id`.

Let `n = n_inputs`, `Q = in_qty_max`.

**Case n = 1.** Each output is written by a v_neuron reading input 0, inversion `none`, a
random op ([08](08-rng.md#derived-procedures)) from `[mul, div, add, sub]` and that op's identity weight
([04](04-neurons.md#v_neuron)).

**Case 2 ≤ n ≤ Q.** Each output is written by an m_neuron with a random aggregation op from
`[mean, median, min, max]`, slots `0 … n−1` connected to inputs `0 … n−1` in order, remaining
slots empty.

**Case n > Q.** Build a merge tree:

1. Level 0 is the list of input nodes.
2. While the current level has more than `Q` nodes: split it, in order, into consecutive
   chunks of `Q` (last chunk may be shorter). Each chunk becomes one new hidden node written
   by an m_neuron (random op) whose slots `0…` connect to the chunk's nodes in order. The new
   nodes, in order, form the next level.
3. Each output is written by an m_neuron (random op) whose slots connect, in order, to the
   nodes of the final level.

With `L` hidden levels, nodes of hidden level `ℓ` (1-based) get depth
`(D_MAX / (L + 1)) * ℓ`, using integer division first.

If the resulting net would exceed the node cap, `create` fails with a validation error.

**RNG order.** Exactly one op choice is drawn per neuron, in this order: hidden neurons level
by level (each level in chunk order), then output neurons in output order. Hidden node ids are
assigned in the same order, after the output ids.
