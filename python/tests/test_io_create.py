import json

import pytest
from helpers import M, V, load_golden, make_net

from jneuro import AggOp, DepthType, MNeuron, Net, Priority, ValidationError, VNeuron
from jneuro.analysis import path_rule_ok
from jneuro.io import format_double, genome_from_json, genome_to_json
from jneuro.rng import Rng

INVALID = load_golden("invalid")
D16 = 65535


@pytest.mark.parametrize("name,case", INVALID, ids=[c[0] for c in INVALID])
def test_invalid_golden(name, case):
    with pytest.raises(ValidationError):
        genome_from_json(json.dumps(case["genome"]))


def test_invalid_present():
    assert len(INVALID) >= 19


@pytest.mark.parametrize("x,s", [
    (1.0, "1.0"), (0.1, "0.1"), (-0.0, "-0.0"), (0.0, "0.0"), (1e16, "1e+16"),
    (1.5e-05, "1.5e-05"), (0.0001, "0.0001"), (123456789.123, "123456789.123"),
    (1e22, "1e+22"), (5e-324, "5e-324"), (-2.5, "-2.5"), (9999999999999998.0, "9999999999999998.0"),
    (1.7976931348623157e308, "1.7976931348623157e+308"), (100.0, "100.0"), (3e-05, "3e-05"),
])
def test_format_double(x, s):
    assert format_double(x) == s
    assert repr(x) == s


def test_canonical_text():
    net = make_net(1, 1, 3, [(D16, V(0, "add", 0.5))])
    assert genome_to_json(net) == (
        '{"format":"j_neuro.genome","spec_version":"0.2.0","params":{"priority":"SPEED",'
        '"depth_type":"U16","in_qty_max":3,"n_inputs":1,"n_outputs":1},"next_birth":2,'
        '"nodes":[{"birth":0,"born_gen":0,"depth":0},{"birth":1,"born_gen":0,"depth":65535,'
        '"neuron":{"type":"v","input":0,"inversion":"none","op":"add","weight":0.5}}]}')


def test_roundtrip_all_types():
    net = make_net(2, 1, 5, [(D16, M("median", [3, 4, 1, None, None])),
                             (10, V(0, "div", -1e-07, "bitnot")),
                             (20, M("min", [0, None, None, None, None]))])
    from helpers import S
    net.nodes[4].neuron = S(0, "max", 4)
    text = genome_to_json(net)
    again = genome_to_json(genome_from_json(text))
    assert text == again


def test_u64_depth_roundtrip():
    net = make_net(1, 1, 3, [(2**64 - 1, V(2, "add", 1.0)), (2**53 + 1, V(0, "mul", 3.0))], "U64")
    net2 = genome_from_json(genome_to_json(net))
    assert net2.nodes[2].depth == 2**53 + 1


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
    assert genome_to_json(net) == genome_to_json(
        Net.create(4, 2, DepthType.U32, 5, Priority.BALANCED, Rng(0)))
    assert len(net.step([1.0, 2.0, 3.0, 4.0])) == 2


def test_create_deterministic():
    a = genome_to_json(Net.create(9, 3, DepthType.U32, 3, Priority.SPEED, Rng(5)))
    b = genome_to_json(Net.create(9, 3, DepthType.U32, 3, Priority.SPEED, Rng(5)))
    assert a == b


def test_create_rejects_bad_params():
    with pytest.raises(ValidationError):
        Net.create(2, 1, DepthType.U16, 4, Priority.SPEED, Rng(1))
    with pytest.raises(ValidationError):
        Net.create(0, 1, DepthType.U16, 3, Priority.SPEED, Rng(1))
