# 03 — Net structure

## Per-net parameters

| Parameter | Values | Meaning |
|---|---|---|
| `priority` | `SPEED`, `MEMORY`, `BALANCED` | Secondary goal after accuracy. Changes storage, caching and memory reuse only. MUST NOT change any result. |
| `depth_type` | `U16`, `U32`, `U64` | Unsigned integer type of depth values. `D_MAX` = 2¹⁶−1, 2³²−1 or 2⁶⁴−1. Fixed width on every platform. |
| `in_qty_max` | odd integer ≥ 3 | Slots per m_neuron; maximum s_neuron history length `k`. |
| `n_inputs` | integer ≥ 1 | Number of input nodes. |
| `n_outputs` | integer ≥ 1 | Number of output nodes. |
| `path_mix` | integer 0–255 | Minimum number of v, of s and of m neurons on every input-to-output path ([mix rule](#mix-rule)). |

`depth_type`, `in_qty_max`, `n_inputs`, `n_outputs` and `path_mix` are fixed when the net is
created.
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

## Mix rule

**Every path from an input node to an output node MUST pass through at least `path_mix`
v_neurons, at least `path_mix` s_neurons and at least `path_mix` m_neurons.**

- Paths are as in the path rule. A path's neurons are the neurons writing its non-input
  nodes, the output node included.
- `path_mix = 0` disables the rule. [`create`](#minimal-net) defaults to `path_mix = 2`.
- The rule is checked on load ([09](09-serialization.md)) and is preserved by every mutation:
  an attempt that would break it is skipped ([06](06-mutation.md), [07](07-removal.md)).
- Check (informative): visit live nodes in ascending depth; give each input `(0, 0, 0)`; give
  every other node the element-wise minimum of its connected nodes' triples, plus 1 in the
  position of its own neuron type (v, s, m). Nodes with no connected predecessor get none. The
  rule holds when every output's triple has all entries `≥ path_mix`.

## Minimal net

`create(n_inputs, n_outputs, depth_type, in_qty_max, priority, rng, path_mix)` builds the
smallest net satisfying the path rule and the mix rule. The caller supplies only the input and
output counts; no layers or starting neurons are specified by hand. All other arguments are
optional, with these defaults (identical in both implementations): `depth_type = U32`,
`in_qty_max = 5`, `priority = BALANCED`, `rng = seed(0)`, `path_mix = 2`. All new nodes get
`born_gen = 0` and consecutive `birth` ids equal to their `id`.

Let `n = n_inputs`, `Q = in_qty_max`, `P = path_mix`.

**1. Merge tree.** Level 0 is the list of input nodes. While the current level has more than
`Q` nodes: split it, in order, into consecutive chunks of `Q` (last chunk may be shorter). Each
chunk becomes one new hidden node written by an m_neuron (random op from
`[mean, median, min, max]`, [08](08-rng.md#derived-procedures)) whose slots `0…` connect to
the chunk's nodes in order, remaining slots empty. The new nodes, in order, form the next
level. Let `L` be the number of tree levels built (0 if `n ≤ Q`) and `F` the final level.

**2. Chain.** Every output gets its own chain of neurons, in this order:

| Part | Count | Neuron | Reads |
|---|---|---|---|
| merge | 1 if `n ≥ 2`, else 0 | m_neuron, random op | slots `0…` = `F` in order |
| extra m | `max(0, P − m_have)`, where `m_have = L + 1` if `n ≥ 2`, else 0 | m_neuron, op `min` | slot 0 = previous |
| s | `P` | s_neuron, op `mean`, `k = 1` | previous |
| v | `P`; but 1 if `P = 0` and the chain would otherwise be empty | v_neuron, inversion `none`, random op from `[mul, div, add, sub]`, that op's identity weight ([04](04-neurons.md#v_neuron)) | previous |

"Previous" is the node written by the chain's preceding neuron, or input 0 for the first one.
The last neuron of the chain writes the output node; each other one writes a new hidden node.
Every chain has the same length `K`.

**Depths.** There are `H = L + K − 1` hidden levels: tree levels `1 … L`, then chain positions
`L + 1 … L + K − 1`. A hidden node of level `ℓ` gets depth `(D_MAX / (H + 1)) * ℓ`, using
integer division first.

If the resulting net would exceed the node cap, `create` fails with a validation error.

**RNG order and ids.** One op choice is drawn for each merge-tree neuron (level by level, each
level in chunk order), then for each output in output order, for each neuron of its chain in
chain order: one op choice for the merge m_neuron and for each v_neuron; extra m_neurons and
s_neurons draw nothing. Hidden node ids are assigned in the same order, after the output ids.

Informative: the extra m_neurons (`min` over one connected slot) and the v_neurons return
their input exactly. An s_neuron with `k = 1` returns its input minus the input of the
previous step (its input itself on the first step after a reset), so a net built with `P ≥ 1`
starts out seeing differences of its inputs. With `P = 0` the construction is the minimal net
of spec 0.4: one v_neuron per output when `n = 1`, otherwise the merge tree plus one merge
m_neuron per output.
