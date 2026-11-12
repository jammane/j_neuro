"""Shared test helpers: compact constructors for hand-built nets."""
import json
import math
import os
import struct

from jneuro import (AggOp, DepthType, Inversion, MNeuron, Net, Node, Priority, SNeuron,
                    VNeuron, VOp)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLDEN = os.path.join(REPO, "tests", "golden")


def bits(x: float) -> str:
    return struct.pack(">d", x).hex()


def from_hex(h: str) -> float:
    return struct.unpack(">d", bytes.fromhex(h))[0]


def same(a: float, b: float) -> bool:
    """Bit-exact equality."""
    return bits(a) == bits(b)


def V(inp, op="mul", w=1.0, inv="none"):
    return VNeuron(inp, Inversion[inv.upper()], VOp[op.upper()], w)


def S(inp, op="mean", k=1):
    return SNeuron(inp, AggOp[op.upper()], k)


def M(op, slots):
    return MNeuron(AggOp[op.upper()], list(slots))


def make_net(n_in, n_out, q, noninput, depth_type="U16", priority="SPEED", born_gen=0, path_mix=0):
    """noninput: list of (depth, neuron) for ids n_in.. (outputs first, then hidden)."""
    nodes = [Node(i, 0, 0, None) for i in range(n_in)]
    for j, (depth, neuron) in enumerate(noninput):
        nodes.append(Node(n_in + j, born_gen, depth, neuron))
    return Net(Priority[priority], DepthType[depth_type], q, n_in, n_out, nodes, len(nodes), path_mix)


def outputs_for(net, input_rows):
    net.reset()
    return [[bits(y) for y in net.step(list(row))] for row in input_rows]


def sample_inputs(n_in, rows=6):
    base = [0.0, 1.0, -2.5, 3.25, 1e-3, -7.0, 0.5, 100.0, -0.125, 42.0]
    return [[base[(r * 3 + i * 7) % len(base)] * (1 + r) for i in range(n_in)] for r in range(rows)]


def load_golden(kind_dir):
    d = os.path.join(GOLDEN, kind_dir)
    if not os.path.isdir(d):
        return []
    return [(f, json.load(open(os.path.join(d, f)))) for f in sorted(os.listdir(d))
            if f.endswith(".json")]


INF = math.inf
NAN = math.nan
