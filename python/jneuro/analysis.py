"""Structural analysis (spec/03 path rule, spec/07 immunity, spec/06 legal depth ranges).

`removed` is an optional set of node ids treated as absent (tombstoned during mutation).
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .net import MNeuron, Net, SNeuron, VNeuron


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


def mix_ok(net: Net, removed: Optional[Set[int]] = None) -> bool:
    """Mix rule (spec/03): every input-to-output path has >= path_mix v, s and m neurons."""
    p = net.path_mix
    if p == 0:
        return True
    alive = _live(net, removed)
    order = sorted((i for i in range(len(net.nodes)) if alive[i]), key=lambda i: net.nodes[i].depth)
    fewest: List[Optional[Tuple[int, int, int]]] = [None] * len(net.nodes)  # min (v, s, m) from any input
    for i in order:
        neuron = net.nodes[i].neuron
        if neuron is None:
            fewest[i] = (0, 0, 0)
            continue
        best = None
        for c in connections(neuron):
            f = fewest[c] if alive[c] else None
            if f is not None:
                best = f if best is None else (min(best[0], f[0]), min(best[1], f[1]), min(best[2], f[2]))
        if best is None:
            continue
        t = 0 if isinstance(neuron, VNeuron) else 1 if isinstance(neuron, SNeuron) else 2
        fewest[i] = tuple(x + (k == t) for k, x in enumerate(best))
    for o in range(net.n_inputs, net.n_inputs + net.n_outputs):
        if alive[o] and fewest[o] is not None and min(fewest[o]) < p:
            return False
    return True


def run_ok(net: Net, removed: Optional[Set[int]] = None) -> bool:
    """Run rule (spec/03): no input-to-output path has more than max_run 1:1 (v, s) or many:1 (m)
    neurons in a row."""
    k = net.max_run
    if k == 0:
        return True
    alive = _live(net, removed)
    reach = reaching_output(net, removed)
    order = sorted((i for i in range(len(net.nodes)) if alive[i]), key=lambda i: net.nodes[i].depth)
    run_1: List[Optional[int]] = [None] * len(net.nodes)    # longest run ending here, from an input
    run_m: List[Optional[int]] = [None] * len(net.nodes)
    for i in order:
        neuron = net.nodes[i].neuron
        if neuron is None:
            run_1[i] = run_m[i] = 0
            continue
        preds = [c for c in connections(neuron) if alive[c] and run_1[c] is not None]
        if not preds:
            continue
        if isinstance(neuron, MNeuron):
            run_m[i], run_1[i] = 1 + max(run_m[c] for c in preds), 0
        else:
            run_1[i], run_m[i] = 1 + max(run_1[c] for c in preds), 0
        if i in reach and (run_1[i] > k or run_m[i] > k):
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
