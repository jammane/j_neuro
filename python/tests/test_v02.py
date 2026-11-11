"""Spec 0.2.0 features: finite clamping, warm-up, node cap, rewiring, inversion on insert,
diversity injection, run-log diversity entries."""
import json

import pytest
from helpers import INF, M, S, V, load_golden, make_net, outputs_for, sample_inputs

from jneuro import DepthType, Inversion, Net, Node, Priority, SNeuron, ValidationError, VNeuron, VOp
from jneuro.analysis import path_rule_ok
from jneuro.io import genome_to_json
from jneuro.mutate import Profile, inject_diversity, mutate
from jneuro.rng import Rng

D16 = 65535
MAX = 1.7976931348623157e308
PHASES = ["op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
          "depth_shift_max_pct"]


def zero_profile():
    p = Profile.default()
    for f in PHASES:
        p.overall[f] = 0.0
    return p


# ------------------------------------------------------------------ finite values

def test_outputs_always_finite():
    net = make_net(1, 2, 3, [(D16, V(0, "div", 0.0)), (D16, V(0, "mul", 1e308))])
    assert net.step([5.0]) == [MAX, MAX]
    assert net.step([INF]) == [MAX, MAX]
    assert net.step([-INF]) == [-MAX, -MAX]


# ------------------------------------------------------------------ warm-up

def test_warmup_steps():
    net = make_net(1, 1, 5, [(D16, M("mean", [2, 3, None, None, None])),
                             (30, S(4, "mean", 1)), (10, S(0, "mean", 2)), (20, S(3, "max", 3))])
    assert net.get_warmup_steps() == 6          # out <- s(k1) <- s(k3) <- s(k2) <- in0
    assert make_net(1, 1, 3, [(D16, V(0))]).get_warmup_steps() == 0
    assert make_net(1, 1, 3, [(D16, S(0, "mean", 3))]).get_warmup_steps() == 3


# ------------------------------------------------------------------ node cap

def test_create_respects_node_cap():
    with pytest.raises(ValidationError):
        Net.create(70000, 1, DepthType.U16, 3)


def test_growth_stops_at_node_cap():
    # 1 input, 1 output, 65533 hidden: exactly D_MAX = 65535 nodes for U16
    nodes = [Node(0, 0, 0, None), Node(1, 0, D16, VNeuron(D16 - 1, Inversion.NONE, VOp.MUL, 1.0))]
    for i in range(2, D16):
        nodes.append(Node(i, 0, i - 1, VNeuron(0 if i == 2 else i - 1, Inversion.NONE, VOp.MUL, 1.0)))
    net = Net(Priority.SPEED, DepthType.U16, 3, 1, 1, nodes, len(nodes))
    net.validate()
    p = zero_profile()
    p.overall["growth_max_pct"] = 0.05
    child = mutate(net, p, 1, Rng(3))
    assert len(child.nodes) == D16
    nodes.append(Node(D16, 0, 5, VNeuron(0, Inversion.NONE, VOp.MUL, 1.0)))
    with pytest.raises(ValidationError):
        Net(Priority.SPEED, DepthType.U16, 3, 1, 1, nodes, len(nodes)).validate()


# ------------------------------------------------------------------ rewiring

def rewire_net():
    # out = mean[3, 4, 5]; 3 and 4 read in0, 5 reads in1
    return make_net(2, 1, 3, [(D16, M("mean", [3, 4, 5])), (10, V(0)), (10, V(0)), (20, V(1))])


def rewire_profile():
    p = zero_profile()
    p.overall["edit_max_pct"] = 100.0
    p.overall["rewire_prob"] = 1.0
    p.per_type["m"]["edit_max_pct"] = 0.0
    return p


def test_rewire_keeps_every_node_and_the_path_rule():
    net = rewire_net()
    p = rewire_profile()
    rewired = False
    for seed in range(60):
        child = mutate(net, p, 1, Rng(seed))
        child.validate()
        assert path_rule_ok(child)
        assert len(child.nodes) == len(net.nodes)                    # nothing orphaned / collected
        ins = [child.nodes[i].neuron.input for i in (3, 4, 5)]
        rewired = rewired or ins != [0, 0, 1]
    assert rewired


def test_rewire_never_orphans():
    # every v_neuron is the only reader of its input, so no rewire is allowed
    net = make_net(2, 1, 3, [(D16, M("mean", [3, 4, None])), (10, V(0)), (20, V(1))])
    p = rewire_profile()
    for seed in range(30):
        assert genome_to_json(mutate(net, p, 1, Rng(seed))) == genome_to_json(net)


def test_rewire_prob_zero_never_rewires():
    net = rewire_net()
    p = rewire_profile()
    p.overall["rewire_prob"] = 0.0
    for seed in range(20):
        child = mutate(net, p, 1, Rng(seed))
        assert [child.nodes[i].neuron.input for i in (3, 4, 5)] == [0, 0, 1]


# ------------------------------------------------------------------ inversion on insert

def test_insert_inversion():
    net = make_net(1, 1, 3, [(D16, V(0, "mul", 2.0))])
    p = zero_profile()
    p.overall["growth_max_pct"] = 100.0
    p.overall["add_method_weights"] = {"slot_fill": 0, "v_insert": 1, "s_insert": 0, "m_after_m": 0}
    p.overall["insert_inversion_prob"] = 1.0
    for seed in range(40):
        child = mutate(net, p, 1, Rng(seed))
        if len(child.nodes) == 3:
            n = child.nodes[1].neuron
            assert n.inversion in (Inversion.BITNOT, Inversion.RECIP)
            assert n.weight == (1.0 if n.op in (VOp.MUL, VOp.DIV) else 0.0)
            return
    raise AssertionError("no insertion happened")


def test_default_profile_has_new_fields():
    p = Profile.default()
    assert p.overall["rewire_prob"] == 0.1
    assert p.overall["insert_inversion_prob"] == 0.05


# ------------------------------------------------------------------ diversity injection

def diversity_net():
    return make_net(1, 1, 5, [(D16, M("mean", [2, 3, 4, 5, None])), (10, V(0, "mul", 1.0)),
                              (10, V(0, "add", 0.0)), (10, V(0, "div", 1.0, "bitnot")),
                              (10, S(0, "mean", 1))])


def test_diversity_changes_only_weights():
    net = diversity_net()
    before = genome_to_json(net)
    child = inject_diversity(net, 100.0, Rng(4))
    assert genome_to_json(net) == before
    for a, b in zip(net.nodes, child.nodes):
        if isinstance(a.neuron, VNeuron):
            assert (a.neuron.input, a.neuron.inversion, a.neuron.op) == \
                   (b.neuron.input, b.neuron.inversion, b.neuron.op)
            assert a.neuron.weight != b.neuron.weight
        else:
            assert a == b


def test_diversity_counts():
    net = diversity_net()
    assert genome_to_json(inject_diversity(net, 0.0, Rng(1))) == genome_to_json(net)
    for seed in range(20):
        child = inject_diversity(net, 50.0, Rng(seed))          # prob_round(1.5) -> 1 or 2
        changed = sum(1 for a, b in zip(net.nodes, child.nodes)
                      if isinstance(a.neuron, VNeuron) and a.neuron.weight != b.neuron.weight)
        assert changed in (1, 2)
    a = genome_to_json(inject_diversity(net, 70.0, Rng(9), sigma_reset=3.0))
    assert a == genome_to_json(inject_diversity(net, 70.0, Rng(9), sigma_reset=3.0))


DIV = load_golden("diversity")


@pytest.mark.parametrize("name,case", DIV, ids=[c[0] for c in DIV])
def test_diversity_golden(name, case):
    from jneuro.io import genome_from_json
    net = genome_from_json(json.dumps(case["genome"]))
    child = inject_diversity(net, case["pct"], Rng(case["seed"]), case["sigma_reset"])
    assert genome_to_json(child) == genome_to_json(genome_from_json(json.dumps(case["expected_genome"])))
