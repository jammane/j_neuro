"""Removal candidates, bypass, cascades, garbage collection and compaction (spec/07).

Functions mark node ids in `removed` (tombstones); `compact` deletes them.
"""
from __future__ import annotations

from typing import Set

from .analysis import connections, reaching_output, readers
from .net import MNeuron, Net, SNeuron, VNeuron, eval_v
from .types import IDENTITY_WEIGHT, AggOp, Inversion, VOp


def is_identity(net: Net, nid: int) -> bool:
    n = net.nodes[nid].neuron
    return (isinstance(n, VNeuron) and n.input is not None and n.inversion == Inversion.NONE
            and n.weight == IDENTITY_WEIGHT[n.op])


def is_silenced(net: Net, nid: int) -> bool:
    """Output is always +0.0."""
    n = net.nodes[nid].neuron
    if isinstance(n, VNeuron) and n.op == VOp.MUL and n.weight == 0.0:
        return True
    if connections(n):
        return False
    if isinstance(n, VNeuron):
        return eval_v(n, 0.0) == 0.0
    return isinstance(n, (SNeuron, MNeuron))


def has_min_reader(net: Net, nid: int, removed: Set[int]) -> bool:
    for r in readers(net, removed)[nid]:
        n = net.nodes[r].neuron
        if isinstance(n, MNeuron) and n.op == AggOp.MIN:
            return True
    return False


def _redirect(neuron, old: int, new):
    if isinstance(neuron, MNeuron):
        neuron.slots = [new if s == old else s for s in neuron.slots]
    elif neuron.input == old:
        neuron.input = new


def bypass(net: Net, nid: int, removed: Set[int]):
    """Remove an identity neuron; its readers read its input directly."""
    src = net.nodes[nid].neuron.input
    for r in readers(net, removed)[nid]:
        _redirect(net.nodes[r].neuron, nid, src)
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
            if (net.is_hidden(r) and not connections(net.nodes[r].neuron) and is_silenced(net, r)
                    and not has_min_reader(net, r, removed)):
                queue.append(r)


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
