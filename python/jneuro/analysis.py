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


def conns(neuron) -> List[Tuple[int, int]]:
    """Connections (node, position) of a neuron, in slot order (spec/03)."""
    if neuron is None:
        return []
    if isinstance(neuron, MNeuron):
        return [(s, neuron.slot_pos(i)) for i, s in enumerate(neuron.slots) if s is not None]
    return [] if neuron.input is None else [(neuron.input, neuron.input_pos)]


def conn_depth(net: Net, node: int, pos: int) -> int:
    """Depth of a connection: depth(node) - position (spec/03)."""
    return net.nodes[node].depth - pos


def depth_rule_error(net: Net, removed: Optional[Set[int]] = None) -> Optional[str]:
    """The first broken depth rule of spec/03 (connections shallower than their reader, the s gap,
    history positions within k of a hidden s_neuron), or None. Live nodes only."""
    q = net.in_qty_max
    n = len(net.nodes)
    for i in range(net.n_inputs, n):
        if (removed and i in removed) or net.nodes[i].neuron is None:
            continue
        d = net.nodes[i].depth
        neuron = net.nodes[i].neuron
        for c, p in conns(neuron):
            if c >= n:
                return "bad reference"
            if p > 0:
                src = net.nodes[c]
                if not net.is_hidden(c) or not isinstance(src.neuron, SNeuron) or p > src.neuron.k + 1 \
                        or p > src.depth:
                    return "history connection: no such history node"
            if conn_depth(net, c, p) >= d:
                return "connection depth must be smaller"
        if isinstance(neuron, SNeuron):
            inp = conn_depth(net, neuron.input, neuron.input_pos) if neuron.input is not None else 0
            if inp + q + 1 >= d:
                return "s gap: an s_neuron's input must be shallower than its depth - (in_qty_max + 1)"
    return None


def history_floors(net: Net, removed: Optional[Set[int]] = None) -> List[int]:
    """Largest history position read from each node (spec/03 history floor), by node id."""
    hf = [0] * len(net.nodes)
    for i, node in enumerate(net.nodes):
        if (removed and i in removed) or node.neuron is None:
            continue
        for c, p in conns(node.neuron):
            if c < len(hf):
                hf[c] = max(hf[c], p)
    return hf


def s_gap(net: Net, neuron) -> int:
    """gap(X) of spec/06: in_qty_max + 1 for an s_neuron (length node and longest history), else 0."""
    return net.in_qty_max + 1 if isinstance(neuron, SNeuron) else 0


def depth_floor(net: Net, node_id: int, removed: Optional[Set[int]] = None) -> int:
    """floor(X) of spec/06: the shallowest depth node_id may have (live connections; its own s gap)."""
    neuron = net.nodes[node_id].neuron
    f = 1
    for c, p in conns(neuron):
        if not removed or c not in removed:
            f = max(f, conn_depth(net, c, p) + 1)
    return f + s_gap(net, neuron)


def depth_ceil(net: Net, node_id: int, removed: Optional[Set[int]] = None) -> int:
    """ceil(X) of spec/06: min over live connections (node_id, j) of depth(r) - gap(r) + j - 1."""
    hi = net.d_max - 1
    for r, node in enumerate(net.nodes):
        if (removed and r in removed) or node.neuron is None:
            continue
        for c, j in conns(node.neuron):
            if c == node_id:
                hi = min(hi, node.depth - s_gap(net, node.neuron) + j - 1)
    return hi


def struct_depth(net: Net, i: int) -> int:
    """Structural depth (spec/03 structural order): the depth, or for an s_neuron the bottom of its
    gap, depth - (in_qty_max + 1); each node then comes after everything it reads."""
    d = net.nodes[i].depth
    if isinstance(net.nodes[i].neuron, SNeuron):
        return max(0, d - (net.in_qty_max + 1))
    return d


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
    order = sorted((i for i in range(len(net.nodes)) if alive[i]), key=lambda i: struct_depth(net, i))
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
    order = sorted((i for i in range(len(net.nodes)) if alive[i]), key=lambda i: struct_depth(net, i))
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
    """[lo, hi] depths node_id may take without breaking any neuron (spec/06 floor and ceil)."""
    return depth_floor(net, node_id, removed), depth_ceil(net, node_id, removed)
