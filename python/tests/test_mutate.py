import json

import pytest
from helpers import M, S, V, load_golden, make_net, outputs_for, sample_inputs

from jneuro import AggOp, DepthType, MNeuron, Net, Priority, SNeuron, VNeuron
from jneuro.analysis import path_rule_ok
from jneuro.io import genome_from_json, genome_to_json, profile_from_json
from jneuro.mutate import Profile, mutate
from jneuro.rng import Rng

D16 = 65535
PHASE_FIELDS = ["op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
                "depth_shift_max_pct"]


def profile(**overall):
    p = Profile.default()
    for f in PHASE_FIELDS:
        p.overall[f] = 0.0
    p.overall.update(overall)
    return p


def only_method(name):
    return {k: (1.0 if k == name else 0.0) for k in ("slot_fill", "v_insert", "s_insert", "m_after_m")}


def base_net(seed=3, n_in=4, n_out=2, q=5, dt=DepthType.U32):
    return Net.create(n_in, n_out, dt, q, Priority.SPEED, Rng(seed))


def n_neurons(net):
    return len(net.nodes) - net.n_inputs


def first_changed(net, p, gen, pred=None):
    """Budgets are "up to", so a single draw may change nothing; find a seed that does."""
    for seed in range(300):
        child = mutate(net, p, gen, Rng(seed))
        ok = pred(child) if pred else genome_to_json(child) != genome_to_json(net)
        if ok:
            return child
    raise AssertionError("no seed produced the expected change")


# ---------------------------------------------------------------- profile

def test_profile_defaults():
    p = Profile.default()
    assert p.overall["edit_max_pct"] == 5.0
    assert p.overall["growth_max_pct"] == 1.5
    assert p.overall["add_method_weights"] == {"slot_fill": 40, "v_insert": 30, "s_insert": 20,
                                               "m_after_m": 10}
    assert p.effective("sigma_weight", "v") == 0.1


def test_profile_overrides_and_patch():
    p = Profile.default()
    p.apply_patch({"m": {"edit_max_pct": 2.0}, "overall": {"sigma_weight": 0.05}})
    assert p.effective("edit_max_pct", "m") == 2.0
    assert p.has_override("edit_max_pct", "m")
    assert not p.has_override("edit_max_pct", "v")
    assert p.effective("edit_max_pct", "v") == 5.0
    assert p.effective("sigma_weight", "v") == 0.05
    p.apply_patch({"m": {"edit_max_pct": None}})
    assert not p.has_override("edit_max_pct", "m")


def test_profile_json_roundtrip():
    p = Profile.default()
    p.apply_patch({"s": {"sigma_k": 2.0}})
    q = profile_from_json(json.dumps(p.to_dict()))
    assert q.to_dict() == p.to_dict()


# ---------------------------------------------------------------- general

def test_parent_untouched_and_deterministic():
    net = base_net()
    text = genome_to_json(net)
    p = Profile.default()
    for f in PHASE_FIELDS:
        p.overall[f] = 50.0
    a = mutate(net, p, 1, Rng(9))
    b = mutate(net, p, 1, Rng(9))
    assert genome_to_json(net) == text
    assert genome_to_json(a) == genome_to_json(b)


def test_zero_profile_is_identity():
    net = base_net()
    out = mutate(net, profile(), 4, Rng(1))
    assert genome_to_json(out) == genome_to_json(net)


def test_offspring_starts_reset():
    net = make_net(1, 1, 3, [(D16, S(0, "mean", 1))])
    net.step([5.0])
    child = mutate(net, profile(), 1, Rng(1))
    assert child.step([5.0]) == [5.0]


# ---------------------------------------------------------------- neutral growth

@pytest.mark.parametrize("method", ["v_insert", "slot_fill"])
def test_neutral_growth(method):
    net = base_net()
    rows = sample_inputs(net.n_inputs)
    p = profile(growth_max_pct=100.0, add_method_weights=only_method(method))
    grown = 0
    for seed in range(20):
        child = mutate(net, p, 1, Rng(seed))
        child.validate()
        assert outputs_for(child, rows) == outputs_for(net, rows)
        grown += n_neurons(child) - n_neurons(net)
    assert grown > 0


def test_v_insert_identity_and_birth():
    net = make_net(1, 1, 3, [(D16, V(0, "mul", 2.0))])
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("v_insert"))
    child = first_changed(net, p, 7)
    assert len(child.nodes) == 3
    out, b = child.nodes[1], child.nodes[2]
    assert isinstance(out.neuron, VNeuron) and out.neuron.input == 2
    assert out.born_gen == 7 and out.birth == 2
    assert b.birth == 1 and b.born_gen == 0 and b.neuron.weight == 2.0
    assert 1 <= b.depth <= D16 - 1


def test_s_insert_creates_k1():
    net = make_net(1, 1, 3, [(D16, V(0, "mul", 2.0))])
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("s_insert"))
    child = first_changed(net, p, 1)
    assert isinstance(child.nodes[1].neuron, SNeuron) and child.nodes[1].neuron.k == 1


def test_m_after_m_starts_max_and_neutral_after_max_with_empty_slot():
    net = make_net(2, 1, 3, [(D16, M("max", [0, 1, None]))])
    rows = sample_inputs(2)
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("m_after_m"))
    child = first_changed(net, p, 1)
    new = child.nodes[2].neuron
    assert isinstance(new, MNeuron) and new.op == AggOp.MAX and new.slots == [3, None, None]
    assert outputs_for(child, rows) == outputs_for(net, rows)


def test_slot_fill_skips_min_receivers():
    net = make_net(2, 1, 3, [(D16, M("min", [0, 1, None]))])
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("slot_fill"))
    for seed in range(10):
        assert genome_to_json(mutate(net, p, 1, Rng(seed))) == genome_to_json(net)


def test_growth_cap_per_type():
    net = base_net()
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("v_insert"))
    p.per_type["v"]["growth_max_pct"] = 0.0
    assert genome_to_json(mutate(net, p, 1, Rng(1))) == genome_to_json(net)


def test_no_gap_skips():
    net = make_net(1, 1, 3, [(D16, V(2)), (1, V(0))], "U16")
    net.nodes[1].depth = D16
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("v_insert"))
    for seed in range(10):
        mutate(net, p, 1, Rng(seed)).validate()


# ---------------------------------------------------------------- edits / op changes / depth

def test_op_changes_valid():
    net = base_net()
    p = profile(op_change_max_pct=100.0)
    child = first_changed(net, p, 1)
    child.validate()


def test_op_change_resets_weight():
    net = make_net(1, 1, 3, [(D16, V(0, "mul", 3.0))])
    child = first_changed(net, profile(op_change_max_pct=100.0), 1)
    n = child.nodes[1].neuron
    assert (n.inversion, n.op) != (net.nodes[1].neuron.inversion, net.nodes[1].neuron.op)
    assert n.weight == (1.0 if n.op.name in ("MUL", "DIV") else 0.0)


def test_edits_valid():
    net = make_net(2, 1, 5, [(D16, M("mean", [3, 4, 5, None, None])), (10, V(0, "mul", 2.0)),
                             (20, V(1, "add", 1.0)), (30, S(4, "median", 3))])
    p = profile(edit_max_pct=100.0)
    for seed in range(30):
        child = mutate(net, p, 1, Rng(seed))
        child.validate()
        assert all(n.neuron.k % 2 == 1 for n in child.nodes if isinstance(n.neuron, SNeuron))


def test_depth_shift_neutral():
    net = base_net(n_in=9, q=3)
    rows = sample_inputs(net.n_inputs)
    child = first_changed(net, profile(depth_shift_max_pct=100.0), 1)
    assert outputs_for(child, rows) == outputs_for(net, rows)
    assert [n.birth for n in child.nodes] == [n.birth for n in net.nodes]


# ---------------------------------------------------------------- removal

def test_removal_of_inserted_identities_after_grace():
    net = base_net(n_in=9, q=3)                    # has hidden nodes
    rows = sample_inputs(net.n_inputs)
    grow = profile(growth_max_pct=100.0, add_method_weights=only_method("v_insert"))
    big = net
    for i in range(4):
        big = mutate(big, grow, 0, Rng(i))
    assert n_neurons(big) > n_neurons(net)
    prune = profile(removal_max_pct=100.0)
    for seed in range(20):                         # inside grace period (min age 5)
        assert genome_to_json(mutate(big, prune, 2, Rng(seed))) == genome_to_json(big)
    later = first_changed(big, prune, 10, lambda c: n_neurons(c) < n_neurons(big))
    later.validate()
    assert outputs_for(later, rows) == outputs_for(net, rows)


def test_immune_silenced_not_removed():
    net = make_net(2, 1, 3, [(D16, M("mean", [0, 3, None])), (10, V(1, "mul", 0.0))])
    child = mutate(net, profile(removal_max_pct=100.0), 50, Rng(1))
    assert genome_to_json(child) == genome_to_json(net)


# ---------------------------------------------------------------- golden

GOLD = load_golden("mutate")


@pytest.mark.parametrize("name,case", GOLD, ids=[c[0] for c in GOLD])
def test_mutate_golden(name, case):
    net = genome_from_json(json.dumps(case["genome"]))
    p = profile_from_json(json.dumps(case["profile"]))
    child = mutate(net, p, case["generation"], Rng(case["seed"]))
    assert genome_to_json(child) == genome_to_json(genome_from_json(json.dumps(case["expected_genome"])))


CREATE = load_golden("create")


@pytest.mark.parametrize("name,case", CREATE, ids=[c[0] for c in CREATE])
def test_create_golden(name, case):
    net = Net.create(case["n_inputs"], case["n_outputs"], DepthType[case["depth_type"]],
                     case["in_qty_max"], Priority[case["priority"]], Rng(case["seed"]))
    assert genome_to_json(net) == genome_to_json(genome_from_json(json.dumps(case["expected_genome"])))


# ---------------------------------------------------------------- properties

@pytest.mark.parametrize("seed", range(12))
def test_property_random_chains_stay_valid(seed):
    r = Rng(1000 + seed)
    net = Net.create(1 + r.below(8), 1 + r.below(3), list(DepthType)[r.below(3)], 3 + 2 * r.below(3),
                     Priority.BALANCED, r)
    p = Profile.default()
    for f in PHASE_FIELDS:
        p.overall[f] = 30.0
    p.overall["min_age_for_removal"] = 1
    for gen in range(1, 25):
        net = mutate(net, p, gen, r)
        net.validate()
        assert path_rule_ok(net)
        net.step([0.5] * net.n_inputs)


@pytest.mark.parametrize("seed", range(12))
def test_property_neutral_chains_preserve_outputs(seed):
    r = Rng(2000 + seed)
    net = Net.create(2 + r.below(6), 1 + r.below(2), DepthType.U32, 3 + 2 * r.below(2),
                     Priority.SPEED, r)
    rows = sample_inputs(net.n_inputs)
    ref = outputs_for(net, rows)
    p = profile(growth_max_pct=40.0, removal_max_pct=40.0, depth_shift_max_pct=40.0,
                add_method_weights={"slot_fill": 1.0, "v_insert": 1.0, "s_insert": 0.0, "m_after_m": 0.0})
    p.overall["min_age_for_removal"] = 0
    for gen in range(1, 25):
        net = mutate(net, p, gen, r)
        net.validate()
        assert outputs_for(net, rows) == ref
