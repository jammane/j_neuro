"""Structural analysis (spec/03 path rule, spec/07 immunity, spec/06 legal depth ranges).

`removed` is an optional set of node ids treated as absent (tombstoned during mutation).
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .net import MNeuron, Net


def connections(neuron) -> List[int]:
    """Connected node ids of a neuron, in slot order (duplicates kept)."""
    if neuron is None:
        return []
    if isinstance(neuron, MNeuron):
        return [s for s in neuron.slots if s is not None]
    return [] if neuron.input is None else [neuron.input]


def _live(net: Net, removed) -> List[bool]:
    alive = [True] * len(net.nodes)
    if removed:
        for r in removed:
            alive[r] = False
    return alive


def readers(net: Net, removed: Optional[Set[int]] = None) -> List[List[int]]:
    """For each node: ascending ids of live neurons connected to it (each once)."""
    alive = _live(net, removed)
    out: List[List[int]] = [[] for _ in net.nodes]
    for i, node in enumerate(net.nodes):
        if not alive[i]:
            continue
        for c in connections(node.neuron):
            lst = out[c]
            if not lst or lst[-1] != i:
                lst.append(i)
    return out


def reaching_output(net: Net, removed: Optional[Set[int]] = None) -> Set[int]:
    """Live nodes from which some output node is reachable (outputs included)."""
    alive = _live(net, removed)
    marked = set()
    stack = [o for o in range(net.n_inputs, net.n_inputs + net.n_outputs) if alive[o]]
    marked.update(stack)
    while stack:
        i = stack.pop()
        for c in connections(net.nodes[i].neuron):
            if alive[c] and c not in marked:
                marked.add(c)
                stack.append(c)
    return marked


def path_rule_ok(net: Net, removed: Optional[Set[int]] = None, cut: Optional[int] = None) -> bool:
    """Every input reaches every output. `cut` is a node no path may pass through."""
    alive = _live(net, removed)
    if cut is not None:
        alive[cut] = False
    rd = readers(net, removed)
    outputs = range(net.n_inputs, net.n_inputs + net.n_outputs)
    for i in range(net.n_inputs):
        if not alive[i]:
            return False
        seen = {i}
        stack = [i]
        while stack:
            x = stack.pop()
            for r in rd[x]:
                if alive[r] and r not in seen:
                    seen.add(r)
                    stack.append(r)
        if any(o not in seen for o in outputs):
            return False
    return True


def is_immune(net: Net, node_id: int, removed: Optional[Set[int]] = None) -> bool:
    """True if removing node_id would break the path rule (spec/07)."""
    return not path_rule_ok(net, removed, cut=node_id)


def legal_depth_range(net: Net, node_id: int, removed: Optional[Set[int]] = None) -> Tuple[int, int]:
    """[lo, hi] depths node_id may take without breaking any neuron (spec/06)."""
    alive = _live(net, removed)
    conns = [c for c in connections(net.nodes[node_id].neuron) if alive[c]]
    lo = max(net.nodes[c].depth for c in conns) + 1 if conns else 1
    rd = [r for r in readers(net, removed)[node_id]]
    hi = min(net.nodes[r].depth for r in rd) - 1 if rd else net.d_max - 1
    return lo, hi
