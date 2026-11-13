"""Binary .jnet files (spec/09). Byte-identical to the C++ implementation."""
from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Union

from .net import MNeuron, Net, Node, SNeuron, VNeuron
from .types import AggOp, DepthType, Inversion, Priority, ValidationError, VOp

MAGIC = b"JNET"
NET_FORMAT_VERSION = 2     # .jnet; 1 = spec 0.5 (read and converted, spec/09)
POP_FORMAT_VERSION = 1     # .jpnet
SPEC_VERSION = (0, 6, 0)
WIDTH = {DepthType.U16: 2, DepthType.U32: 4, DepthType.U64: 8}


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def is_legacy_05(major: int, minor: int) -> bool:
    """Spec 0.5 genomes are still read; nets are converted on load (spec/09)."""
    return major == 0 and minor == 5


def spec_version_ok(major: int, minor: int) -> bool:
    """Same MAJOR; while MAJOR is 0, also the same MINOR (spec/README)."""
    return major == SPEC_VERSION[0] and (major != 0 or minor == SPEC_VERSION[1])


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def u(self, value: int, width: int):
        self.buf += int(value).to_bytes(width, "little")

    def f64(self, x: float):
        self.buf += struct.pack("<d", x)

    def raw(self, b: bytes):
        self.buf += b

    def finish(self) -> bytes:
        """Append the SHA-256 of everything written so far."""
        return bytes(self.buf) + sha256(bytes(self.buf))


class Reader:
    def __init__(self, data: bytes, what: str):
        if len(data) < 32:
            raise ValidationError(f"{what}: file too short")
        body, digest = data[:-32], data[-32:]
        if sha256(body) != digest:
            raise ValidationError(f"{what}: checksum mismatch")
        self.data = body
        self.pos = 0
        self.what = what

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValidationError(f"{self.what}: truncated")
        b = self.data[self.pos:self.pos + n]
        self.pos += n
        return b

    def u(self, width: int) -> int:
        return int.from_bytes(self.take(width), "little")

    def f64(self) -> float:
        return struct.unpack("<d", self.take(8))[0]

    def enum(self, cls, what: str):
        v = self.u(1)
        if v >= len(cls):
            raise ValidationError(f"{self.what}: {what} byte {v} out of range")
        return cls(v)

    def header(self, magic: bytes) -> bool:
        """Check magic, format and spec version; True for a spec 0.5 file (to convert)."""
        if self.take(len(magic)) != magic:
            raise ValidationError(f"{self.what}: wrong magic")
        fmt = self.u(2)
        major, minor, _patch = self.u(2), self.u(2), self.u(2)
        legacy = is_legacy_05(major, minor)
        if not legacy and not spec_version_ok(major, minor):
            raise ValidationError(f"{self.what}: unsupported spec version {major}.{minor}")
        want = (1 if legacy else NET_FORMAT_VERSION) if magic == MAGIC else POP_FORMAT_VERSION
        if fmt != want:
            raise ValidationError(f"{self.what}: unknown format version")
        return legacy

    def done(self):
        if self.pos != len(self.data):
            raise ValidationError(f"{self.what}: trailing bytes")


def write_header(w: Writer, magic: bytes):
    w.raw(magic)
    w.u(NET_FORMAT_VERSION if magic == MAGIC else POP_FORMAT_VERSION, 2)
    for v in SPEC_VERSION:
        w.u(v, 2)


# ------------------------------------------------------------------ nets

def net_to_bytes(net: Net) -> bytes:
    """The canonical .jnet bytes of a net."""
    wd = WIDTH[net.depth_type]
    none = (1 << (8 * wd)) - 1
    w = Writer()
    write_header(w, MAGIC)
    w.u(net.priority, 1)
    w.u(net.depth_type, 1)
    w.u(net.in_qty_max, 4)
    w.u(net.n_inputs, 4)
    w.u(net.n_outputs, 4)
    w.u(net.path_mix, 1)
    w.u(net.max_run, 1)
    w.u(net.next_birth, 8)
    w.u(len(net.nodes), wd)

    def ref(r, pos):                     # a conn: id, position
        w.u(none if r is None else r, wd)
        w.u(0 if r is None else pos, 4)

    for node in net.nodes:
        w.u(node.birth, 8)
        w.u(node.born_gen, 8)
        w.u(node.depth, wd)
        n = node.neuron
        if n is None:
            w.u(0, 1)
        elif isinstance(n, VNeuron):
            w.u(1, 1)
            ref(n.input, n.input_pos)
            w.u(n.inversion, 1)
            w.u(n.op, 1)
            w.f64(n.weight)
        elif isinstance(n, SNeuron):
            w.u(2, 1)
            ref(n.input, n.input_pos)
            w.u(n.op, 1)
            w.u(n.k, 4)
        else:
            w.u(3, 1)
            w.u(n.op, 1)
            for i, s in enumerate(n.slots):
                ref(s, n.slot_pos(i))
    return w.finish()


def net_from_bytes(data: bytes) -> Net:
    """Parse and fully validate .jnet bytes."""
    r = Reader(bytes(data), "jnet")
    legacy = r.header(MAGIC)
    priority = r.enum(Priority, "priority")
    depth_type = r.enum(DepthType, "depth_type")
    wd = WIDTH[depth_type]
    none = (1 << (8 * wd)) - 1
    q = r.u(4)
    n_in = r.u(4)
    n_out = r.u(4)
    path_mix = r.u(1)
    max_run = r.u(1)
    next_birth = r.u(8)
    count = r.u(wd)
    if count > depth_type.d_max:
        raise ValidationError("jnet: more than D_MAX nodes")
    if q < 3 or q % 2 == 0:
        raise ValidationError("jnet: in_qty_max must be odd and >= 3")

    def ref():                           # a conn: id, then (format 2) its position
        v = r.u(wd)
        pos = 0 if legacy else r.u(4)
        if v == none:
            if pos != 0:
                raise ValidationError("jnet: empty connection with a position")
            return None, 0
        return v, pos

    nodes = []
    for _ in range(count):
        birth, born_gen, depth = r.u(8), r.u(8), r.u(wd)
        kind = r.u(1)
        if kind == 0:
            neuron = None
        elif kind == 1:
            inp, ip = ref()
            inv = r.enum(Inversion, "inversion")
            op = r.enum(VOp, "v op")
            neuron = VNeuron(inp, inv, op, r.f64(), ip)
        elif kind == 2:
            inp, ip = ref()
            op = r.enum(AggOp, "op")
            neuron = SNeuron(inp, op, r.u(4), ip)
        elif kind == 3:
            op = r.enum(AggOp, "op")
            if op == AggOp.LAG:
                raise ValidationError("jnet: lag is an s_neuron op, not an m_neuron op")
            neuron = MNeuron(op, [None] * q)
            for k in range(q):
                c, cp = ref()
                neuron.set(k, c, cp)
        else:
            raise ValidationError(f"jnet: neuron type byte {kind} out of range")
        nodes.append(Node(birth, born_gen, depth, neuron))
    r.done()
    net = Net(priority, depth_type, q, n_in, n_out, nodes, next_birth, path_mix, max_run)
    if legacy:
        respace_05(net)
    net.validate()
    return net


def respace_05(net: Net):
    """Spec 0.5 nets: re-space depths so the s gap holds (spec/09 reading 0.5 nets). Every
    connection stays strictly shallower than its reader, so outputs are unchanged."""
    from .analysis import connections
    n = len(net.nodes)
    order = sorted(range(n), key=lambda i: (net.nodes[i].depth, i))
    nd = [0] * n
    d_max = net.d_max
    for i in order:
        node = net.nodes[i]
        if net.is_input(i) or node.neuron is None:
            nd[i] = 0
            continue
        floor = max((nd[c] + 1 for c in connections(node.neuron) if c < n), default=1)
        if isinstance(node.neuron, SNeuron):
            floor += net.in_qty_max + 1
        if net.is_output(i):
            if floor > d_max:
                raise ValidationError("0.5 conversion: no depth room for the s gap")
            nd[i] = d_max
        else:
            nd[i] = max(node.depth, floor)
            if nd[i] > d_max - 1:
                raise ValidationError("0.5 conversion: no depth room for the s gap")
    for i in range(n):
        net.nodes[i].depth = nd[i]


def save_net(net: Net, path: Union[str, Path]):
    Path(path).write_bytes(net_to_bytes(net))


def load_net(path: Union[str, Path]) -> Net:
    return net_from_bytes(Path(path).read_bytes())
