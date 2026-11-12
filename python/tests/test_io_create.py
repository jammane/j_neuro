import hashlib
import json
import struct

import pytest
from helpers import M, S, V, load_golden, make_net

from jneuro import AggOp, DepthType, MNeuron, Net, Priority, ValidationError, VNeuron
from jneuro.analysis import path_rule_ok
from jneuro.jnet import load_net, net_from_bytes, net_to_bytes, save_net
from jneuro.rng import Rng

INVALID = load_golden("invalid")
D16 = 65535


@pytest.mark.parametrize("name,case", INVALID, ids=[c[0] for c in INVALID])
def test_invalid_golden(name, case):
    with pytest.raises(ValidationError):
        net_from_bytes(bytes.fromhex(case["jnet"]))


def test_invalid_present():
    assert len(INVALID) >= 25


def test_exact_layout():
    """Bytes built by hand from spec/09, independent of the library's writer."""
    net = make_net(1, 1, 3, [(D16, V(0, "add", 0.5))])
    body = b"JNET" + struct.pack("<HHHH", 1, 0, 4, 0)
    body += struct.pack("<BBIIIQ", 0, 0, 3, 1, 1, 2)          # SPEED, U16, q, n_in, n_out, next_birth
    body += struct.pack("<H", 2)                               # node_count (W = 2)
    body += struct.pack("<QQHB", 0, 0, 0, 0)                   # input node
    body += struct.pack("<QQHB", 1, 0, 0xFFFF, 1)              # output node, v neuron
    body += struct.pack("<HBBd", 0, 0, 2, 0.5)                 # input 0, none, add, 0.5
    assert net_to_bytes(net) == body + hashlib.sha256(body).digest()


def test_none_sentinel_and_widths():
    for dt, w, none in (("U16", 2, b"\xff" * 2), ("U32", 4, b"\xff" * 4), ("U64", 8, b"\xff" * 8)):
        dmax = (1 << (8 * w)) - 1
        net = make_net(2, 1, 3, [(dmax, M("max", [0, 1, None])), ], depth_type=dt)
        b = net_to_bytes(net)
        assert b[-32 - w:-32] == none                          # last slot of the m neuron is empty
        header = 4 + 8 + 2 + 12 + 8
        nodes = 3 * (16 + w + 1) + (1 + 3 * w)
        assert len(b) == header + w + nodes + 32


def test_roundtrip_all_types():
    net = make_net(2, 1, 5, [(D16, M("median", [3, 4, 1, None, None])),
                             (10, V(0, "div", -1e-07, "bitnot")),
                             (20, S(0, "max", 4))])
    data = net_to_bytes(net)
    assert net_to_bytes(net_from_bytes(data)) == data
    assert Net.from_bytes(net.to_bytes()).to_bytes() == data


def test_missing_input_roundtrip():
    net = make_net(1, 1, 3, [(D16, M("mean", [0, 2, None])), (5, V(None, "add", 3.0))])
    assert net_from_bytes(net_to_bytes(net)).nodes[2].neuron.input is None


def test_u64_depth_roundtrip():
    net = make_net(1, 1, 3, [(2**64 - 1, V(2, "add", 1.0)), (2**53 + 1, V(0, "mul", 3.0))], "U64")
    assert net_from_bytes(net_to_bytes(net)).nodes[2].depth == 2**53 + 1


def test_weights_exact():
    for w in (0.1, -0.0, 5e-324, 1.7976931348623157e308, 1 / 3):
        net = make_net(1, 1, 3, [(D16, V(0, "mul", w))])
        got = net_from_bytes(net_to_bytes(net)).nodes[1].neuron.weight
        assert struct.pack("<d", got) == struct.pack("<d", w)


def test_save_load_file(tmp_path):
    net = Net.create(4, 2, DepthType.U32, 5, Priority.MEMORY, Rng(3))
    p = tmp_path / "a.jnet"
    save_net(net, p)
    assert p.read_bytes() == net_to_bytes(net)
    assert load_net(p).to_bytes() == net.to_bytes()
    net.save(tmp_path / "b.jnet")
    assert Net.load(tmp_path / "b.jnet").to_bytes() == net.to_bytes()


def test_damage_detected():
    data = bytearray(net_to_bytes(Net.create(3, 1)))
    for bad in (data[:-1], data + b"\x00", b"XNET" + data[4:]):
        with pytest.raises(ValidationError):
            net_from_bytes(bytes(bad))
    flipped = bytearray(data)
    flipped[40] ^= 1
    with pytest.raises(ValidationError):
        net_from_bytes(bytes(flipped))


@pytest.mark.parametrize("n_in,n_out,q", [(1, 3, 3), (3, 2, 3), (2, 1, 5), (7, 2, 3), (30, 2, 3), (5, 1, 3)])
def test_create_minimal(n_in, n_out, q):
    for dt in DepthType:
        net = Net.create(n_in, n_out, dt, q, Priority.MEMORY, Rng(11))
        net.validate()
        assert path_rule_ok(net)
        assert net.next_birth == len(net.nodes)
        for i, node in enumerate(net.nodes):
            assert node.birth == i and node.born_gen == 0
        outs = net.nodes[n_in:n_in + n_out]
        if n_in == 1:
            assert all(isinstance(o.neuron, VNeuron) for o in outs)
            assert len(net.nodes) == n_in + n_out
        elif n_in <= q:
            assert all(isinstance(o.neuron, MNeuron) for o in outs)
            assert len(net.nodes) == n_in + n_out
            assert outs[0].neuron.slots[:n_in] == list(range(n_in))
        else:
            assert len(net.nodes) > n_in + n_out


def test_create_tree_shape():
    net = Net.create(7, 2, DepthType.U16, 3, Priority.SPEED, Rng(1))
    hidden = net.nodes[9:]
    assert len(hidden) == 3
    assert [h.neuron.slots for h in hidden] == [[0, 1, 2], [3, 4, 5], [6, None, None]]
    assert all(h.depth == (D16 // 2) for h in hidden)
    assert net.nodes[7].neuron.slots == [9, 10, 11]


def test_create_from_counts_only():
    net = Net.create(4, 2)
    net.validate()
    assert net.depth_type == DepthType.U32 and net.in_qty_max == 5
    assert net.priority == Priority.BALANCED
    assert net_to_bytes(net) == net_to_bytes(
        Net.create(4, 2, DepthType.U32, 5, Priority.BALANCED, Rng(0)))
    assert len(net.step([1.0, 2.0, 3.0, 4.0])) == 2


def test_create_deterministic():
    a = net_to_bytes(Net.create(9, 3, DepthType.U32, 3, Priority.SPEED, Rng(5)))
    b = net_to_bytes(Net.create(9, 3, DepthType.U32, 3, Priority.SPEED, Rng(5)))
    assert a == b


def test_create_rejects_bad_params():
    with pytest.raises(ValidationError):
        Net.create(2, 1, DepthType.U16, 4, Priority.SPEED, Rng(1))
    with pytest.raises(ValidationError):
        Net.create(0, 1, DepthType.U16, 3, Priority.SPEED, Rng(1))
