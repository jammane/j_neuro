from helpers import M, S, V, make_net, outputs_for, sample_inputs

from jneuro import MNeuron
from jneuro.analysis import path_rule_ok
from jneuro.removal import (bypass, compact, garbage_collect, is_identity, is_silenced,
                            remove_silenced)

D16 = 65535


def check_same_outputs(before_net, after_net, n_in):
    rows = sample_inputs(n_in)
    assert outputs_for(before_net, rows) == outputs_for(after_net, rows)


def test_identity_and_silenced_predicates():
    net = make_net(1, 1, 3, [(D16, M("mean", [2, 3, 4])),
                             (10, V(0, "mul", 1.0)), (10, V(0, "add", 0.0)),
                             (10, V(0, "mul", 0.0, "recip"))])
    assert is_identity(net, 2) and is_identity(net, 3)
    assert not is_identity(net, 4)
    assert is_silenced(net, 4)
    assert not is_silenced(net, 2)
    assert is_identity(net, 3)
    net.nodes[3].neuron = V(0, "sub", -0.0)
    assert is_identity(net, 3)
    net.nodes[3].neuron = V(0, "add", 0.0, "bitnot")
    assert not is_identity(net, 3)


def test_connectionless_constants():
    net = make_net(1, 1, 3, [(D16, M("mean", [0, 2, 3])), (5, V(None, "add", 2.0)),
                             (5, V(None, "div", 3.0))])
    assert not is_silenced(net, 2)       # constant 2
    assert is_silenced(net, 3)           # 0 / 3 = 0
    assert is_silenced(net, 3)
    net.nodes[2].neuron = S(None, "max", 3)
    assert is_silenced(net, 2)
    net.nodes[2].neuron = M("max", [None, None, None])
    assert is_silenced(net, 2)
    net.nodes[2].neuron = V(None, "add", 5.0, "bitnot")   # NaN + 5 -> 0
    assert is_silenced(net, 2)


def test_bypass_exact():
    net = make_net(2, 1, 3, [(D16, M("mean", [3, 1, None])), (10, V(4, "div", 1.0)), (5, V(0, "mul", 2.0))])
    before = net.copy()
    removed = set()
    bypass(net, 3, removed)
    assert removed == {3}
    assert net.nodes[2].neuron.slots == [4, 1, None]
    compact(net, removed)
    assert len(net.nodes) == 4
    net.validate()
    check_same_outputs(before, net, 2)


def test_silenced_cascade_downstream():
    # 3 = v(in0 mul 0) silenced; 4 = v(3 mul 5) -> becomes connection-less 0; out mean [4, 1, 0]
    net = make_net(2, 1, 3, [(D16, M("mean", [4, 1, 0])), (10, V(0, "mul", 0.0)), (20, V(3, "mul", 5.0))])
    before = net.copy()
    removed = set()
    remove_silenced(net, 3, removed)
    assert removed == {3, 4}
    assert net.nodes[2].neuron.slots == [None, 1, 0]
    compact(net, removed)
    net.validate()
    check_same_outputs(before, net, 2)


def test_cascade_stops_at_min_reader():
    net = make_net(2, 1, 3, [(D16, M("min", [4, 1, 0])), (10, V(0, "mul", 0.0)), (20, V(3, "mul", 5.0))])
    before = net.copy()
    removed = set()
    remove_silenced(net, 3, removed)
    assert removed == {3}
    assert net.nodes[4].neuron.input is None
    compact(net, removed)
    check_same_outputs(before, net, 2)


def test_cascade_stops_at_nonzero_constant():
    net = make_net(2, 1, 3, [(D16, M("mean", [4, 1, 0])), (10, V(0, "mul", 0.0)), (20, V(3, "add", 2.0))])
    before = net.copy()
    removed = set()
    remove_silenced(net, 3, removed)
    assert removed == {3}
    compact(net, removed)
    check_same_outputs(before, net, 2)


def test_garbage_collect():
    net = make_net(1, 1, 3, [(D16, V(0)), (5, V(0)), (6, S(2))])
    removed = set()
    garbage_collect(net, removed)
    assert removed == {2, 3}
    compact(net, removed)
    assert len(net.nodes) == 2
    net.validate()


def test_compact_keeps_order_and_births():
    net = make_net(1, 1, 3, [(D16, M("max", [3, 2, None])), (5, V(0)), (6, V(0))])
    removed = {2}
    net.nodes[1].neuron.slots = [3, None, None]
    compact(net, removed)
    assert [n.birth for n in net.nodes] == [0, 1, 3]
    assert net.nodes[1].neuron.slots == [2, None, None]
    assert isinstance(net.nodes[1].neuron, MNeuron)
    assert path_rule_ok(net)
