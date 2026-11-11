from helpers import M, S, V, make_net

from jneuro.analysis import (connections, is_immune, legal_depth_range, path_rule_ok,
                             reaching_output, readers)

D16 = 65535


def diamond():
    # in0, in1 -> out(2) = m mean [3, 4, 5]
    # 3 (d10) = v(in0); 4 (d10) = v(in0); 5 (d20) = v(in1)
    return make_net(2, 1, 3, [(D16, M("mean", [3, 4, 5])), (10, V(0)), (10, V(0)), (20, V(1))])


def test_connections_and_readers():
    net = diamond()
    assert connections(net.nodes[2].neuron) == [3, 4, 5]
    assert connections(V(None)) == []
    r = readers(net)
    assert r[0] == [3, 4] and r[1] == [5] and r[3] == [2] and r[2] == []


def test_path_rule():
    net = diamond()
    assert path_rule_ok(net)
    net.nodes[5].neuron = V(0)
    assert not path_rule_ok(net)  # in1 has no path


def test_path_rule_with_removed_and_cut():
    net = diamond()
    assert path_rule_ok(net, removed={3})
    assert not path_rule_ok(net, removed={3, 4})
    assert path_rule_ok(net, cut=3)
    assert not path_rule_ok(net, cut=5)


def test_immunity():
    net = diamond()
    assert not is_immune(net, 3)
    assert not is_immune(net, 4)
    assert is_immune(net, 5)
    assert is_immune(net, 2)


def test_immunity_lifted_by_new_path():
    net = diamond()
    net.nodes[4].neuron = V(1)  # now in1 reaches out via 4 as well
    assert not is_immune(net, 5)
    assert is_immune(net, 3)


def test_legal_depth_range():
    net = diamond()
    assert legal_depth_range(net, 3) == (1, D16 - 1)
    net2 = make_net(1, 1, 3, [(D16, V(2)), (50, V(3)), (10, V(0))])
    assert legal_depth_range(net2, 3) == (1, 49)
    assert legal_depth_range(net2, 2) == (11, D16 - 1)


def test_reaching_output():
    net = make_net(1, 1, 3, [(D16, V(0)), (5, V(0)), (6, S(2))])
    assert reaching_output(net) == {0, 1}
