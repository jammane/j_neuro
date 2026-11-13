"""Adding and removing inputs and outputs of an existing net (spec/12).

Every function validates the given net, leaves it unchanged and returns a new, valid net in the
reset state.
"""
from __future__ import annotations

import copy as _copy
from typing import List, Optional

from .analysis import connections, depth_floor, readers, struct_depth
from .net import MNeuron, Net, Node, SNeuron, VNeuron
from .removal import compact, garbage_collect
from .rng import Rng
from .types import AggOp, Inversion, ValidationError, VOp

__all__ = ["remove_input", "remove_output", "add_input", "add_output_clone", "add_output_fresh"]


# ------------------------------------------------------------------ helpers

def _remap(neuron, f):
    """Apply f to every connection of a neuron, in place."""
    if neuron is None:
        return
    if isinstance(neuron, MNeuron):          # a connection mapped to None becomes empty, position 0
        for i, s in enumerate(neuron.slots):
            if s is not None:
                to = f(s)
                neuron.set(i, to, neuron.slot_pos(i) if to is not None else 0)
    elif neuron.input is not None:
        neuron.input = f(neuron.input)
        if neuron.input is None:
            neuron.input_pos = 0


def _insert_node(net: Net, at: int, node: Node):
    """Insert a node at id `at`; every id >= at moves up by 1."""
    for n in net.nodes:
        _remap(n.neuron, lambda c: c + 1 if c >= at else c)
    _remap(node.neuron, lambda c: c + 1 if c >= at else c)
    net.nodes.insert(at, node)


def _new_node(net: Net, generation: int, depth: int, neuron) -> Node:
    node = Node(net.next_birth, generation, depth, neuron)
    net.next_birth += 1
    return node


def _check_cap(net: Net):
    if len(net.nodes) > net.d_max:
        raise ValidationError(f"node cap: {len(net.nodes)} nodes exceed D_MAX")


def _position(position: Optional[int], count: int) -> int:
    p = count if position is None else position
    if not 0 <= p <= count:
        raise ValidationError(f"position {p} out of range [0, {count}]")
    return p


def _identity_v(src: int) -> VNeuron:
    return VNeuron(src, Inversion.NONE, VOp.MUL, 1.0)


def _forward(rd, starts) -> set:
    seen = set(starts)
    stack = list(starts)
    while stack:
        x = stack.pop()
        for r in rd[x]:
            if r not in seen:
                seen.add(r)
                stack.append(r)
    return seen


def _type_index(neuron) -> int:
    return 0 if isinstance(neuron, VNeuron) else 1 if isinstance(neuron, SNeuron) else 2


# ------------------------------------------------------------------ removal

def remove_output(net: Net, index: int) -> Net:
    """Delete output `index` and every hidden node that only fed it (spec/12)."""
    net.validate()
    if not 0 <= index < net.n_outputs:
        raise ValidationError(f"output index {index} out of range")
    if net.n_outputs < 2:
        raise ValidationError("cannot remove the only output")
    out = net.copy()
    removed = {out.n_inputs + index}
    garbage_collect(out, removed)
    compact(out, removed)
    out.n_outputs -= 1
    out.validate()
    return out


def remove_input(net: Net, index: int) -> Net:
    """Delete input `index`, the neurons only it reaches, and what only fed them (spec/12)."""
    net.validate()
    if not 0 <= index < net.n_inputs:
        raise ValidationError(f"input index {index} out of range")
    if net.n_inputs < 2:
        raise ValidationError("cannot remove the only input")
    out = net.copy()
    rd = readers(out)
    a = _forward(rd, [index])
    b = _forward(rd, [i for i in range(out.n_inputs) if i != index])
    doomed = {index} | (a - b)
    for i, node in enumerate(out.nodes):
        if i in doomed or node.neuron is None:
            continue
        _remap(node.neuron, lambda c: None if c in doomed else c)
    removed = set(doomed)
    garbage_collect(out, removed)
    compact(out, removed)
    out.n_inputs -= 1
    out.validate()
    return out


# ------------------------------------------------------------------ outputs

def add_output_clone(net: Net, source: int, position: Optional[int] = None, generation: int = 0) -> Net:
    """New output whose neuron copies output `source`'s; it starts with the same values (spec/12)."""
    net.validate()
    if not 0 <= source < net.n_outputs:
        raise ValidationError(f"source output {source} out of range")
    p = _position(position, net.n_outputs)
    out = net.copy()
    neuron = _copy.deepcopy(out.nodes[out.n_inputs + source].neuron)
    _insert_node(out, out.n_inputs + p, _new_node(out, generation, out.d_max, neuron))
    out.n_outputs += 1
    _check_cap(out)
    out.validate()
    return out


def add_output_fresh(net: Net, rng: Rng, position: Optional[int] = None, generation: int = 0) -> Net:
    """New output fed by its own minimal chain from the inputs, built as `create` would (spec/12)."""
    net.validate()
    p = _position(position, net.n_outputs)
    g = Net.create(net.n_inputs, 1, net.depth_type, net.in_qty_max, net.priority, rng,
                   net.path_mix, net.max_run)
    out = net.copy()
    n_in = out.n_inputs
    o = n_in + p
    _insert_node(out, o, _new_node(out, generation, out.d_max, None))
    out.n_outputs += 1
    base = len(out.nodes)

    def mapping(c):
        if c < n_in:
            return c
        if c == n_in:
            return o
        return base + (c - n_in - 1)

    neuron = _copy.deepcopy(g.nodes[n_in].neuron)
    _remap(neuron, mapping)
    out.nodes[o].neuron = neuron
    for node in g.nodes[n_in + 1:]:
        neuron = _copy.deepcopy(node.neuron)
        _remap(neuron, mapping)
        out.nodes.append(_new_node(out, generation, node.depth, neuron))
    _check_cap(out)
    out.validate()
    return out


# ------------------------------------------------------------------ inputs

def add_input(net: Net, position: Optional[int] = None, active: bool = False,
              generation: int = 0) -> Net:
    """New input connected to every output through one chain ending in W (spec/12).

    With active=False, W is a `mul 0` v_neuron and every existing output stays bit-for-bit
    unchanged; with active=True, W is an identity v_neuron.
    """
    net.validate()
    p = _position(position, net.n_inputs)
    out = net.copy()
    q, d_max, k_run = out.in_qty_max, out.d_max, out.max_run
    _insert_node(out, p, _new_node(out, generation, 0, None))
    out.n_inputs += 1
    n_in, n_out = out.n_inputs, out.n_outputs
    new_input = p

    # 1. receivers: greedy cover of the outputs
    outs: List[set] = [set() for _ in out.nodes]
    for o in range(n_out):
        for x in _reverse(out, n_in + o):
            outs[x].add(o)
    cands = [i for i in range(n_in, len(out.nodes))
             if isinstance(out.nodes[i].neuron, MNeuron)
             and any(s is None for s in out.nodes[i].neuron.slots)
             and (active or out.nodes[i].neuron.op != AggOp.MIN)]
    uncovered = set(range(n_out))
    receivers: List[int] = []
    while uncovered:
        best, best_key = None, None
        for c in cands:
            gain = len(outs[c] & uncovered)
            key = (-gain, out.nodes[c].depth, c)
            if gain and (best_key is None or key < best_key):
                best, best_key = c, key
        if best is None:
            break
        receivers.append(best)
        uncovered -= outs[best]

    # 2. a neutral median in front of every output no receiver reaches
    for o in sorted(uncovered):
        oid = n_in + o
        onode = out.nodes[oid]
        n = onode.neuron
        lo = depth_floor(out, oid)                    # N's own s gap included (spec/12)
        hi = d_max - 1
        if lo > hi:
            raise ValidationError(f"output {o}: no depth room for a receiver")
        b = len(out.nodes)
        out.nodes.append(Node(onode.birth, onode.born_gen, lo + (hi - lo) // 2, n))
        src = b
        need_sep = isinstance(n, MNeuron) and k_run > 0
        median = MNeuron(AggOp.MEDIAN, [None] * q)
        out.nodes[oid] = _new_node(out, generation, d_max, median)
        if need_sep:
            db = out.nodes[b].depth
            if db >= hi:
                raise ValidationError(f"output {o}: no depth room for a separator")
            out.nodes.append(_new_node(out, generation, db + 1 + (hi - db - 1) // 2, _identity_v(b)))
            src = len(out.nodes) - 1
        median.slots = [src] * ((q + 1) // 2) + [None] * ((q - 1) // 2)
        receivers.append(oid)

    # 3. chain contents
    fewest = _fewest_to_output(out)
    pm = out.path_mix
    need = [max(max(0, pm - fewest[r][t]) for r in receivers) for t in range(3)]
    left = {"m": need[2], "s": need[1], "v": max(0, need[0] - 1)}
    parts = []
    while any(left.values()):
        for kind in ("m", "s", "v"):
            if left[kind]:
                parts.append(kind)
                left[kind] -= 1
    parts.append("w")
    if k_run:
        walked, run_1, run_m = [], 0, 0
        for kind in parts:
            if kind == "m":
                if run_m == k_run:
                    walked.append("v")
                    run_1, run_m = 1, 0
                run_m, run_1 = run_m + 1, 0
            else:
                if run_1 == k_run:
                    walked.append("m")
                    run_m, run_1 = 1, 0
                run_1, run_m = run_1 + 1, 0
            walked.append(kind)
        parts = walked

    # 4. build
    d = min(out.nodes[r].depth for r in receivers)
    step = d // (len(parts) + 1)
    if step == 0:
        raise ValidationError("no depth room for the new input's chain")
    if "s" in parts and step <= q + 1:
        raise ValidationError("no depth room for the new input's chain (s gap)")
    prev = new_input
    for j, kind in enumerate(parts, start=1):
        if kind == "m":
            neuron = MNeuron(AggOp.MIN, [prev] + [None] * (q - 1))
        elif kind == "s":
            neuron = SNeuron(prev, AggOp.MEAN, 1)
        elif kind == "v":
            neuron = _identity_v(prev)
        else:
            neuron = VNeuron(prev, Inversion.NONE, VOp.MUL, 1.0 if active else 0.0)
        out.nodes.append(_new_node(out, generation, step * j, neuron))
        prev = len(out.nodes) - 1

    # 5. connect
    for r in receivers:
        rm = out.nodes[r].neuron
        rm.set(rm.slots.index(None), prev)
    _check_cap(out)
    out.reset()
    out.validate()
    return out


def _reverse(net: Net, start: int) -> set:
    """Nodes from which `start` is reachable (start included)."""
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for c in connections(net.nodes[x].neuron):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def _fewest_to_output(net: Net):
    """For each node: fewest (v, s, m) neurons on any path from it to an output, its own included."""
    rd = readers(net)
    order = sorted(range(net.n_inputs, len(net.nodes)), key=lambda i: (-struct_depth(net, i), i))
    fewest = [None] * len(net.nodes)
    for i in order:
        own = [0, 0, 0]
        own[_type_index(net.nodes[i].neuron)] = 1
        if net.is_output(i):
            fewest[i] = tuple(own)
            continue
        best = None
        for r in rd[i]:
            f = fewest[r]
            if f is not None:
                best = f if best is None else tuple(min(x, y) for x, y in zip(best, f))
        if best is not None:
            fewest[i] = tuple(x + y for x, y in zip(best, own))
    return fewest
