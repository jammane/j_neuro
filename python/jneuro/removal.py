"""Removal candidates, bypass, cascades, garbage collection and compaction (spec/07).

Functions mark node ids in `removed` (tombstones); `compact` deletes them.
"""
from __future__ import annotations

from typing import Set

import math

from .aggregate import aggregate
from .analysis import connections, conns, history_floors, path_rule_ok, reaching_output, readers
from .net import MNeuron, Net, SNeuron, VNeuron, eval_v
from .numerics import canon
from .types import IDENTITY_WEIGHT, AggOp, Inversion, VOp


def is_identity(net: Net, nid: int) -> bool:
    n = net.nodes[nid].neuron
    return (isinstance(n, VNeuron) and n.input is not None and n.inversion == Inversion.NONE
            and n.weight == IDENTITY_WEIGHT[n.op])


def is_silenced(net: Net, nid: int, removed: Set[int] = None) -> bool:
    """Output is always +0.0 (and so is everything a live reader reads from it: an s_neuron whose
    length node is read is not silenced, spec/07)."""
    n = net.nodes[nid].neuron
    if isinstance(n, VNeuron) and n.op == VOp.MUL and n.weight == 0.0:
        return True
    if connections(n):
        return False
    if isinstance(n, VNeuron):
        return eval_v(n, 0.0) == 0.0
    if isinstance(n, SNeuron):
        for r, node in enumerate(net.nodes):
            if (removed and r in removed) or node.neuron is None:
                continue
            if (nid, 1) in conns(node.neuron):
                return False
    return isinstance(n, (SNeuron, MNeuron))


def has_min_reader(net: Net, nid: int, removed: Set[int]) -> bool:
    for r in readers(net, removed)[nid]:
        n = net.nodes[r].neuron
        if isinstance(n, MNeuron) and n.op == AggOp.MIN:
            return True
    return False


def _redirect(neuron, old: int, new, new_pos: int = 0):
    """Every connection to node `old` (any position) becomes (new, new_pos), or empty / missing."""
    if new is None:
        new_pos = 0
    if isinstance(neuron, MNeuron):
        for i, s in enumerate(neuron.slots):
            if s == old:
                neuron.set(i, new, new_pos)
    elif neuron.input == old:
        neuron.input, neuron.input_pos = new, new_pos


def bypass(net: Net, nid: int, removed: Set[int]):
    """Remove an identity neuron; its readers read its input directly."""
    x = net.nodes[nid].neuron
    src, src_pos = x.input, x.input_pos          # the input may itself be a history node
    for r in readers(net, removed)[nid]:
        _redirect(net.nodes[r].neuron, nid, src, src_pos)
    removed.add(nid)


def remove_silenced(net: Net, nid: int, removed: Set[int]):
    """Remove a silenced neuron and cascade downstream (spec/07)."""
    queue = [nid]
    while queue:
        y = queue.pop(0)
        rd = readers(net, removed)[y]
        for r in rd:
            _redirect(net.nodes[r].neuron, y, None)
        removed.add(y)
        for r in rd:
            if (net.is_hidden(r) and not connections(net.nodes[r].neuron) and is_silenced(net, r, removed)
                    and not has_min_reader(net, r, removed)):
                queue.append(r)


def constant_values(net: Net, removed: Set[int]):
    """For each live node: its output if provably the same at every step from reset, else None."""
    order = sorted((i for i in range(len(net.nodes)) if i not in removed), key=lambda i: net.nodes[i].depth)
    const = [None] * len(net.nodes)

    def cconn(r, pos):
        """A connection's constant: the node's at position 0; never for a length node; for a history
        node +0.0 when S's input is missing or its constant is +0.0 (the history is always 0)."""
        if pos == 0:
            return const[r]
        if pos == 1:
            return None
        s = net.nodes[r].neuron
        if s.input is None:
            return 0.0
        c = cconn(s.input, s.input_pos)
        return 0.0 if c is not None and c == 0.0 else None

    for i in order:
        n = net.nodes[i].neuron
        if n is None:
            continue
        if isinstance(n, VNeuron):
            c = cconn(n.input, n.input_pos) if n.input is not None else None
            if n.input is None:
                const[i] = eval_v(n, 0.0)
            elif c is not None:
                const[i] = eval_v(n, c)
            elif n.op == VOp.MUL and n.weight == 0.0:
                const[i] = 0.0
        elif isinstance(n, SNeuron):
            if n.input is None or _is_pos_zero(cconn(n.input, n.input_pos)):
                const[i] = 0.0
        else:
            cs = [None if s is None else cconn(s, n.slot_pos(k)) for k, s in enumerate(n.slots)]
            if all(s is None or c is not None for s, c in zip(n.slots, cs)):
                vals = [0.0 if s is None else c for s, c in zip(n.slots, cs)]
                const[i] = canon(aggregate(n.op, vals, [s is not None for s in n.slots]))
    return const


def _is_pos_zero(x) -> bool:
    return x is not None and x == 0.0 and math.copysign(1.0, x) > 0


def fold_constants(net: Net, removed: Set[int], generation: int, min_age):
    """Replace constant hidden neurons by constant v_neurons (spec/07); no RNG, no quota."""
    const = constant_values(net, removed)
    cands = [i for i in range(len(net.nodes))
             if i not in removed and net.is_hidden(i) and const[i] is not None
             and connections(net.nodes[i].neuron)
             and generation - net.nodes[i].born_gen >= min_age(_type(net.nodes[i].neuron))]
    cands.sort(key=lambda i: (-net.nodes[i].depth, i))
    folded = 0
    for i in cands:
        old = net.nodes[i].neuron
        net.nodes[i].neuron = VNeuron(None, Inversion.NONE, VOp.ADD, const[i])
        # restore if the path rule breaks, or a depth rule (an s_neuron whose history is read)
        if not path_rule_ok(net, removed) or history_floors(net, removed)[i] > 0:
            net.nodes[i].neuron = old
        else:
            folded += 1
    return folded


def _type(neuron) -> str:
    return "v" if isinstance(neuron, VNeuron) else "s" if isinstance(neuron, SNeuron) else "m"


def garbage_collect(net: Net, removed: Set[int]):
    """Mark every live hidden node that reaches no output as removed."""
    reach = reaching_output(net, removed)
    for i in range(net.n_inputs + net.n_outputs, len(net.nodes)):
        if i not in removed and i not in reach:
            removed.add(i)


def compact(net: Net, removed: Set[int]):
    """Delete removed nodes, keep order, renumber ids and connections."""
    mapping = {}
    keep = []
    for i, node in enumerate(net.nodes):
        if i not in removed:
            mapping[i] = len(keep)
            keep.append(node)
    for node in keep:
        n = node.neuron
        if isinstance(n, MNeuron):
            n.slots = [None if s is None else mapping[s] for s in n.slots]
        elif n is not None and n.input is not None:
            n.input = mapping[n.input]
    net.nodes = keep
    removed.clear()
    net.reset()
