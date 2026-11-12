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
FORMAT_VERSION = 1
SPEC_VERSION = (0, 5, 0)
WIDTH = {DepthType.U16: 2, DepthType.U32: 4, DepthType.U64: 8}


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


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

    def header(self, magic: bytes):
        if self.take(len(magic)) != magic:
            raise ValidationError(f"{self.what}: wrong magic")
        if self.u(2) != FORMAT_VERSION:
            raise ValidationError(f"{self.what}: unknown format version")
        major, minor, _patch = self.u(2), self.u(2), self.u(2)
        if not spec_version_ok(major, minor):
            raise ValidationError(f"{self.what}: unsupported spec version {major}.{minor}")

    def done(self):
        if self.pos != len(self.data):
            raise ValidationError(f"{self.what}: trailing bytes")


def write_header(w: Writer, magic: bytes):
    w.raw(magic)
    w.u(FORMAT_VERSION, 2)
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
    w.u(net.next_birth, 8)
    w.u(len(net.nodes), wd)

    def ref(r):
        w.u(none if r is None else r, wd)

    for node in net.nodes:
        w.u(node.birth, 8)
        w.u(node.born_gen, 8)
        w.u(node.depth, wd)
        n = node.neuron
        if n is None:
            w.u(0, 1)
        elif isinstance(n, VNeuron):
            w.u(1, 1)
            ref(n.input)
            w.u(n.inversion, 1)
            w.u(n.op, 1)
            w.f64(n.weight)
        elif isinstance(n, SNeuron):
            w.u(2, 1)
            ref(n.input)
            w.u(n.op, 1)
            w.u(n.k, 4)
        else:
            w.u(3, 1)
            w.u(n.op, 1)
            for s in n.slots:
                ref(s)
    return w.finish()


def net_from_bytes(data: bytes) -> Net:
    """Parse and fully validate .jnet bytes."""
    r = Reader(bytes(data), "jnet")
    r.header(MAGIC)
    priority = r.enum(Priority, "priority")
    depth_type = r.enum(DepthType, "depth_type")
    wd = WIDTH[depth_type]
    none = (1 << (8 * wd)) - 1
    q = r.u(4)
    n_in = r.u(4)
    n_out = r.u(4)
    path_mix = r.u(1)
    next_birth = r.u(8)
    count = r.u(wd)
    if count > depth_type.d_max:
        raise ValidationError("jnet: more than D_MAX nodes")
    if q < 3 or q % 2 == 0:
        raise ValidationError("jnet: in_qty_max must be odd and >= 3")

    def ref():
        v = r.u(wd)
        return None if v == none else v

    nodes = []
    for _ in range(count):
        birth, born_gen, depth = r.u(8), r.u(8), r.u(wd)
        kind = r.u(1)
        if kind == 0:
            neuron = None
        elif kind == 1:
            inp = ref()
            inv = r.enum(Inversion, "inversion")
            op = r.enum(VOp, "v op")
            neuron = VNeuron(inp, inv, op, r.f64())
        elif kind == 2:
            inp = ref()
            op = r.enum(AggOp, "op")
            neuron = SNeuron(inp, op, r.u(4))
        elif kind == 3:
            op = r.enum(AggOp, "op")
            neuron = MNeuron(op, [ref() for _ in range(q)])
        else:
            raise ValidationError(f"jnet: neuron type byte {kind} out of range")
        nodes.append(Node(birth, born_gen, depth, neuron))
    r.done()
    net = Net(priority, depth_type, q, n_in, n_out, nodes, next_birth, path_mix)
    net.validate()
    return net


def save_net(net: Net, path: Union[str, Path]):
    Path(path).write_bytes(net_to_bytes(net))


def load_net(path: Union[str, Path]) -> Net:
    return net_from_bytes(Path(path).read_bytes())
