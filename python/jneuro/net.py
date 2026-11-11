"""Net data model (spec/03)."""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import List, Optional, Union

from .aggregate import aggregate
from .numerics import bitnot, canon, ieee_div
from .rng import Rng
from .types import IDENTITY_WEIGHT, AggOp, DepthType, Inversion, Priority, ValidationError, VOp


@dataclass
class VNeuron:
    input: Optional[int]
    inversion: Inversion
    op: VOp
    weight: float


@dataclass
class SNeuron:
    input: Optional[int]
    op: AggOp
    k: int


@dataclass
class MNeuron:
    op: AggOp
    slots: List[Optional[int]]


Neuron = Union[VNeuron, SNeuron, MNeuron]


@dataclass
class Node:
    birth: int
    born_gen: int
    depth: int
    neuron: Optional[Neuron] = None


class Net:
    def __init__(self, priority: Priority, depth_type: DepthType, in_qty_max: int,
                 n_inputs: int, n_outputs: int, nodes: List[Node], next_birth: int):
        self.priority = Priority(priority)
        self.depth_type = DepthType(depth_type)
        self.in_qty_max = in_qty_max
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.nodes = nodes
        self.next_birth = next_birth
        self._state = None

    @property
    def d_max(self) -> int:
        return self.depth_type.d_max

    def is_input(self, i: int) -> bool:
        return i < self.n_inputs

    def is_output(self, i: int) -> bool:
        return self.n_inputs <= i < self.n_inputs + self.n_outputs

    def is_hidden(self, i: int) -> bool:
        return i >= self.n_inputs + self.n_outputs

    def copy(self) -> "Net":
        """Deep copy of the genome, in the reset state."""
        return Net(self.priority, self.depth_type, self.in_qty_max, self.n_inputs,
                   self.n_outputs, _copy.deepcopy(self.nodes), self.next_birth)

    def get_warmup_steps(self) -> int:
        """Steps after a reset before every s_neuron history on any path is filled (spec/05)."""
        from .analysis import connections
        w = [0] * len(self.nodes)
        for nid in self._order():
            n = self.nodes[nid].neuron
            own = n.k if isinstance(n, SNeuron) else 0
            w[nid] = own + max((w[c] for c in connections(n)), default=0)
        return max((w[o] for o in range(self.n_inputs, self.n_inputs + self.n_outputs)), default=0)

    # ------------------------------------------------------------ evaluation (spec/05)

    def reset(self):
        """Clear all s_neuron state and cached evaluation plans."""
        self._state = None

    def _order(self):
        return sorted(range(self.n_inputs, len(self.nodes)), key=lambda i: (self.nodes[i].depth, i))

    def _eval_node(self, nid, neuron, value_of):
        def inp(r):
            return 0.0 if r is None else value_of(r)

        if isinstance(neuron, VNeuron):
            return eval_v(neuron, inp(neuron.input))
        if isinstance(neuron, SNeuron):
            rings = self._state["rings"]
            k = neuron.k
            ring = rings.get(nid)
            if ring is None or len(ring[0]) != k:
                ring = [[0.0] * k, 0, 0]
                rings[nid] = ring
            buf, head, filled = ring
            x = inp(neuron.input)
            h = [buf[(head + i) % k] for i in range(k)]
            present = [i >= k - filled for i in range(k)]
            out = canon(x - aggregate(neuron.op, h, present))
            buf[head] = x
            ring[1] = (head + 1) % k
            ring[2] = min(filled + 1, k)
            return out
        vals = [inp(s) for s in neuron.slots]
        present = [s is not None for s in neuron.slots]
        return canon(aggregate(neuron.op, vals, present))

    def step(self, inputs):
        """Evaluate one step. Call reset() after editing nodes directly."""
        if len(inputs) != self.n_inputs:
            raise ValueError("step: wrong number of inputs")
        if self._state is None:
            self._state = {"rings": {}, "order": None, "values": None}
        st = self._state
        n_in = self.n_inputs
        outputs = [0.0] * self.n_outputs

        if self.priority == Priority.MEMORY:
            # No cached plan; each value is freed once its last reader has been evaluated.
            from .analysis import connections
            order = self._order()
            last_use = {}
            for p, nid in enumerate(order):
                for c in connections(self.nodes[nid].neuron):
                    last_use[c] = p
            live = {i: canon(float(inputs[i])) for i in range(n_in)}
            for p, nid in enumerate(order):
                neuron = self.nodes[nid].neuron
                v = self._eval_node(nid, neuron, live.__getitem__)
                if self.is_output(nid):
                    outputs[nid - n_in] = v
                live[nid] = v
                for c in connections(neuron):
                    if last_use.get(c) == p and not self.is_output(c):
                        live.pop(c, None)
            return outputs

        if st["order"] is None:
            st["order"] = self._order()
            st["values"] = [0.0] * len(self.nodes)
        values = st["values"]
        for i in range(n_in):
            values[i] = canon(float(inputs[i]))
        for nid in st["order"]:
            values[nid] = self._eval_node(nid, self.nodes[nid].neuron, values.__getitem__)
        return values[n_in:n_in + self.n_outputs]

    # ------------------------------------------------------------ creation (spec/03)

    @classmethod
    def create(cls, n_inputs: int, n_outputs: int, depth_type: DepthType = DepthType.U32,
               in_qty_max: int = 5, priority: Priority = Priority.BALANCED,
               rng: Optional[Rng] = None) -> "Net":
        """Minimal net in which every input reaches every output. Only the counts are required."""
        _check_params(n_inputs, n_outputs, in_qty_max)
        rng = Rng(0) if rng is None else rng
        depth_type = DepthType(depth_type)
        d_max = depth_type.d_max
        q = in_qty_max
        nodes = [Node(i, 0, 0, None) for i in range(n_inputs)]
        nodes += [Node(n_inputs + o, 0, d_max, None) for o in range(n_outputs)]

        def slots(ids):
            return list(ids) + [None] * (q - len(ids))

        if n_inputs == 1:
            for o in range(n_outputs):
                op = VOp(rng.choice(4))
                nodes[1 + o].neuron = VNeuron(0, Inversion.NONE, op, IDENTITY_WEIGHT[op])
        else:
            level = list(range(n_inputs))
            hidden_levels = []
            while len(level) > q:
                nxt = []
                for c in range(0, len(level), q):
                    nid = len(nodes)
                    nodes.append(Node(nid, 0, 0, MNeuron(AggOp(rng.choice(4)), slots(level[c:c + q]))))
                    nxt.append(nid)
                hidden_levels.append(nxt)
                level = nxt
            n_levels = len(hidden_levels)
            for li, ids in enumerate(hidden_levels, start=1):
                for nid in ids:
                    nodes[nid].depth = (d_max // (n_levels + 1)) * li
            for o in range(n_outputs):
                nodes[n_inputs + o].neuron = MNeuron(AggOp(rng.choice(4)), slots(level))
        if len(nodes) > d_max:
            raise ValidationError(f"node cap: {len(nodes)} nodes exceed D_MAX of {depth_type.name}")
        for i, node in enumerate(nodes):
            node.birth = i
        return cls(priority, depth_type, q, n_inputs, n_outputs, nodes, len(nodes))

    # ------------------------------------------------------------ validation (spec/03, spec/09)

    def validate(self):
        """Raise ValidationError unless every structural invariant holds."""
        from .analysis import connections, path_rule_ok, reaching_output
        _check_params(self.n_inputs, self.n_outputs, self.in_qty_max)
        n = len(self.nodes)
        if n < self.n_inputs + self.n_outputs:
            raise ValidationError("fewer nodes than inputs + outputs")
        d_max = self.d_max
        if n > d_max:
            raise ValidationError(f"node cap: {n} nodes exceed D_MAX")
        births = set()
        for i, node in enumerate(self.nodes):
            if not isinstance(node.depth, int) or isinstance(node.depth, bool) or not 0 <= node.depth <= d_max:
                raise ValidationError(f"node {i}: depth out of range for {self.depth_type.name}")
            if self.is_input(i):
                if node.depth != 0 or node.neuron is not None:
                    raise ValidationError(f"input node {i}: depth must be 0 and no neuron")
            elif self.is_output(i):
                if node.depth != d_max:
                    raise ValidationError(f"output node {i}: depth must be D_MAX")
            elif not 1 <= node.depth <= d_max - 1:
                raise ValidationError(f"hidden node {i}: depth must be in [1, D_MAX-1]")
            if not self.is_input(i):
                _check_neuron(node.neuron, self.in_qty_max, i)
                for c in connections(node.neuron):
                    if not isinstance(c, int) or not 0 <= c < n:
                        raise ValidationError(f"node {i}: bad reference {c}")
                    if self.nodes[c].depth >= node.depth:
                        raise ValidationError(f"node {i}: connection depth must be smaller")
            if not isinstance(node.birth, int) or node.birth in births or node.birth >= self.next_birth \
                    or node.birth < 0:
                raise ValidationError(f"node {i}: birth must be unique and < next_birth")
            if not isinstance(node.born_gen, int) or node.born_gen < 0:
                raise ValidationError(f"node {i}: bad born_gen")
            births.add(node.birth)
        if not path_rule_ok(self):
            raise ValidationError("path rule: some input has no path to some output")
        reach = reaching_output(self)
        for i in range(self.n_inputs + self.n_outputs, n):
            if i not in reach:
                raise ValidationError(f"dead node {i}: does not reach any output")


def eval_v(n: VNeuron, x: float) -> float:
    if n.inversion == Inversion.NONE:
        y = x
    elif n.inversion == Inversion.BITNOT:
        y = bitnot(x)
    else:
        y = ieee_div(1.0, x)
    w = n.weight
    if n.op == VOp.MUL:
        z = y * w
    elif n.op == VOp.DIV:
        z = ieee_div(y, w)
    elif n.op == VOp.ADD:
        z = y + w
    else:
        z = y - w
    return canon(z)


def _check_params(n_inputs, n_outputs, q):
    if not isinstance(q, int) or q < 3 or q % 2 == 0:
        raise ValidationError("in_qty_max must be an odd integer >= 3")
    if not isinstance(n_inputs, int) or n_inputs < 1 or not isinstance(n_outputs, int) or n_outputs < 1:
        raise ValidationError("n_inputs and n_outputs must be >= 1")


def _check_neuron(neuron, q, i):
    import math
    if isinstance(neuron, VNeuron):
        if not isinstance(neuron.inversion, Inversion) or not isinstance(neuron.op, VOp):
            raise ValidationError(f"node {i}: bad v_neuron op")
        if not isinstance(neuron.weight, float) or not math.isfinite(neuron.weight):
            raise ValidationError(f"node {i}: weight must be finite")
    elif isinstance(neuron, SNeuron):
        if not isinstance(neuron.op, AggOp):
            raise ValidationError(f"node {i}: bad s_neuron op")
        if not isinstance(neuron.k, int) or not 1 <= neuron.k <= q:
            raise ValidationError(f"node {i}: k range")
        if neuron.op == AggOp.MEDIAN and neuron.k % 2 == 0:
            raise ValidationError(f"node {i}: median k must be odd")
    elif isinstance(neuron, MNeuron):
        if not isinstance(neuron.op, AggOp):
            raise ValidationError(f"node {i}: bad m_neuron op")
        if len(neuron.slots) != q:
            raise ValidationError(f"node {i}: slot count must equal in_qty_max")
    else:
        raise ValidationError(f"node {i}: missing neuron")
