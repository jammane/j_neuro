import json
import math
import pathlib

import pytest
from helpers import M, S, V, load_golden, make_net, outputs_for, sample_inputs

from jneuro import AggOp, DepthType, MNeuron, Net, Priority, SNeuron, VNeuron
from jneuro.analysis import connections, path_rule_ok, run_ok
from jneuro.io import profile_from_json
from jneuro.jnet import net_from_bytes, net_to_bytes
from jneuro.mutate import Profile, mutate
from jneuro.rng import Rng

D16 = 65535
PHASE_FIELDS = ["op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
                "depth_shift_max_pct"]


def profile(**overall):
    """Only the given phases enabled; new v_neurons never inverted, so insertions stay neutral."""
    p = Profile.default()
    for f in PHASE_FIELDS:
        p.overall[f] = 0.0
    p.overall["insert_inversion_prob"] = 0.0
    p.overall.update(overall)
    return p


def only_method(name):
    return {k: (1.0 if k == name else 0.0) for k in ("slot_fill", "v_insert", "s_insert", "m_after_m",
                                                     "v_tap", "s_tap", "m_tap", "m_clone", "m_insert",
                                                     "v_link", "s_link")}


def base_net(seed=3, n_in=4, n_out=2, q=5, dt=DepthType.U32):
    """A minimal net without the run rule (these tests predate it; the rule has its own tests)."""
    return Net.create(n_in, n_out, dt, q, Priority.SPEED, Rng(seed), max_run=0)


def n_neurons(net):
    return len(net.nodes) - net.n_inputs


def first_changed(net, p, gen, pred=None):
    """Budgets are "up to", so a single draw may change nothing; find a seed that does."""
    for seed in range(300):
        child = mutate(net, p, gen, Rng(seed))
        ok = pred(child) if pred else net_to_bytes(child) != net_to_bytes(net)
        if ok:
            return child
    raise AssertionError("no seed produced the expected change")


# ---------------------------------------------------------------- profile

def test_profile_defaults():
    p = Profile.default()
    assert p.overall["edit_max_pct"] == 5.0
    assert p.overall["growth_max_pct"] == 1.5
    assert p.overall["add_method_weights"] == {"slot_fill": 0, "v_insert": 10, "s_insert": 5,
                                               "m_after_m": 5, "v_tap": 10, "s_tap": 10, "m_tap": 5,
                                               "m_clone": 60, "m_insert": 20,
                                               "v_link": 24, "s_link": 24}
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
    text = net_to_bytes(net)
    p = Profile.default()
    for f in PHASE_FIELDS:
        p.overall[f] = 50.0
    a = mutate(net, p, 1, Rng(9))
    b = mutate(net, p, 1, Rng(9))
    assert net_to_bytes(net) == text
    assert net_to_bytes(a) == net_to_bytes(b)


def test_zero_profile_is_identity():
    net = base_net()
    out = mutate(net, profile(), 4, Rng(1))
    assert net_to_bytes(out) == net_to_bytes(net)


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
        assert net_to_bytes(mutate(net, p, 1, Rng(seed))) == net_to_bytes(net)


def test_growth_cap_per_type():
    net = base_net()
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("v_insert"))
    p.per_type["v"]["growth_max_pct"] = 0.0
    assert net_to_bytes(mutate(net, p, 1, Rng(1))) == net_to_bytes(net)


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
        assert net_to_bytes(mutate(big, prune, 2, Rng(seed))) == net_to_bytes(big)
    later = first_changed(big, prune, 10, lambda c: n_neurons(c) < n_neurons(big))
    later.validate()
    assert outputs_for(later, rows) == outputs_for(net, rows)


def test_immune_silenced_not_removed():
    net = make_net(2, 1, 3, [(D16, M("mean", [0, 3, None])), (10, V(1, "mul", 0.0))])
    child = mutate(net, profile(removal_max_pct=100.0), 50, Rng(1))
    assert net_to_bytes(child) == net_to_bytes(net)


# ---------------------------------------------------------------- golden

GOLD = load_golden("mutate")


@pytest.mark.parametrize("name,case", GOLD, ids=[c[0] for c in GOLD])
def test_mutate_golden(name, case):
    net = net_from_bytes(bytes.fromhex(case["jnet"]))
    p = profile_from_json(json.dumps(case["profile"]))
    child = mutate(net, p, case["generation"], Rng(case["seed"]))
    assert net_to_bytes(child) == bytes.fromhex(case["expected_jnet"])


CREATE = load_golden("create")


@pytest.mark.parametrize("name,case", CREATE, ids=[c[0] for c in CREATE])
def test_create_golden(name, case):
    net = Net.create(case["n_inputs"], case["n_outputs"], DepthType[case["depth_type"]],
                     case["in_qty_max"], Priority[case["priority"]], Rng(case["seed"]), case["path_mix"],
                     case["max_run"])
    assert net_to_bytes(net) == bytes.fromhex(case["expected_jnet"])


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
                add_method_weights={**only_method("slot_fill"), "v_insert": 1.0})
    p.overall["min_age_for_removal"] = 0
    for gen in range(1, 25):
        net = mutate(net, p, gen, r)
        net.validate()
        assert outputs_for(net, rows) == ref


def test_nonzero_cap_is_at_least_one():
    net = make_net(1, 1, 3, [(D16, V(0, "mul", 2.0))])          # a single neuron
    p = profile(edit_max_pct=1.0)                               # 1% of 1 neuron -> cap 1
    changed = sum(net_to_bytes(mutate(net, p, 1, Rng(s))) != net_to_bytes(net) for s in range(60))
    assert changed >= 10
    assert all(net_to_bytes(mutate(net, profile(), 1, Rng(s))) == net_to_bytes(net) for s in range(20))


# ---------------------------------------------------------------- mix rule (spec/03, spec/06)

def _mix_net(path_mix):
    """input 0 -> v 2 (identity) -> s 3 / s 4 -> output m 1. Every path: 1 v, 1 s, 1 m."""
    return make_net(1, 1, 3, [(D16, M("max", [3, 4, None])), (100, V(0, "mul", 1.0)),
                              (200, S(2, "mean", 1)), (200, S(2, "mean", 1))], path_mix=path_mix)


@pytest.mark.parametrize("name,prof", [
    ("bypass", dict(removal_max_pct=100.0, min_age_for_removal=0)),
    ("rewire", dict(edit_max_pct=100.0, rewire_prob=1.0)),
    ("slot_connect", dict(edit_max_pct=100.0, rewire_prob=0.0, sigma_slots=3.0)),
    ("slot_fill", dict(growth_max_pct=100.0,
                       add_method_weights=only_method("slot_fill"))),
])
def test_mutation_keeps_mix_rule(name, prof):
    """Each change that could leave a path short of a type is skipped (or undone) instead."""
    p = profile(**prof)
    changed = {}
    for mix in (0, 1):
        net = _mix_net(mix)
        net.validate()
        n = 0
        for s in range(80):
            child = mutate(net, p, 10, Rng(s))
            child.validate()                                   # includes the mix rule
            n += structure(child) != structure(net)
        changed[mix] = n
    assert changed[0] > changed[1]                             # the rule blocked something


def structure(net):
    return [(type(x.neuron).__name__, connections(x.neuron)) for x in net.nodes]


# ---------------------------------------------------------------- growth_budget, constant_folding

def test_exact_growth_budget_adds_exactly_one():
    net = Net.create(3, 1, DepthType.U32, 3, Priority.SPEED, Rng(1), max_run=0)
    only_v = only_method("v_insert")
    sizes = {}
    for mode in ("up_to", "exact"):
        p = profile(growth_max_pct=1e-9, add_method_weights=only_v, growth_budget=mode)
        sizes[mode] = [len(mutate(net, p, 1, Rng(s)).nodes) - len(net.nodes) for s in range(40)]
    assert set(sizes["exact"]) == {1}                          # the whole cap: max(1, ~0) = 1 attempt
    assert set(sizes["up_to"]) == {0, 1}                       # a random fraction of it


def test_fill_growth_budget_retries_skipped_attempts():
    net = Net.create(5, 2, DepthType.U32, 3, Priority.SPEED, Rng(1), 2, 0)
    w = dict(only_method("v_tap"), v_insert=1.0)               # taps near the inputs break the mix rule
    sizes = {}
    for mode in ("exact", "fill"):
        p = profile(growth_max_pct=1e-9, add_method_weights=w, growth_budget=mode)
        sizes[mode] = [len(mutate(net, p, 1, Rng(s)).nodes) - len(net.nodes) for s in range(60)]
    assert set(sizes["exact"]) == {0, 1}                       # one attempt, sometimes skipped
    assert set(sizes["fill"]) == {1}                           # retried until one neuron is added


def _fold_net():
    """output m max [3, 4]; 3 = identity of input 0; 4 = (5 + 2) where 5 = input * 0 (constant)."""
    return make_net(1, 1, 3, [(D16, M("max", [2, 3, None])), (200, V(0, "mul", 1.0)),
                              (200, V(4, "add", 2.0)), (100, V(0, "mul", 0.0))])


def test_constant_folding():
    net = _fold_net()
    net.validate()
    rows = sample_inputs(1)
    ref = outputs_for(net, rows)
    p = profile(constant_folding="on", min_age_for_removal=0)
    child = mutate(net, p, 1, Rng(0))
    assert len(child.nodes) == 4                               # the constant source is gone
    folded = child.nodes[3].neuron
    assert folded == V(None, "add", 2.0)
    assert outputs_for(child, rows) == ref
    off = mutate(net, profile(min_age_for_removal=0), 1, Rng(0))
    assert net_to_bytes(off) == net_to_bytes(net)              # default: no folding
    young = mutate(net, profile(constant_folding="on"), 1, Rng(0))
    assert net_to_bytes(young) == net_to_bytes(net)            # younger than min_age_for_removal


def test_constant_folding_keeps_path_rule_and_s_zero():
    # input -> v*0 -> s -> output: folding s or v would cut the only path, so nothing changes
    net = make_net(1, 1, 3, [(D16, V(3, "add", 0.0)), (100, V(0, "mul", 0.0)), (200, S(2, "mean", 1))])
    p = profile(constant_folding="on", min_age_for_removal=0)
    assert net_to_bytes(mutate(net, p, 1, Rng(0))) == net_to_bytes(net)
    # with a second path, the s of a constant 0 folds to a constant 0
    net = make_net(1, 1, 3, [(D16, M("max", [0, 3, None])), (100, V(0, "mul", 0.0)), (200, S(2, "mean", 1))])
    child = mutate(net, p, 1, Rng(0))
    assert len(child.nodes) == 3 and child.nodes[2].neuron == V(None, "add", 0.0)


# ---------------------------------------------------------------- input taps (spec/06)

@pytest.mark.parametrize("method", ["v_tap", "s_tap", "m_tap"])
@pytest.mark.parametrize("mix", [0, 2])
def test_input_taps(method, mix):
    q = 5
    grew = 0
    for seed in range(40):
        net = Net.create(4, 2, DepthType.U32, q, Priority.SPEED, Rng(seed), path_mix=mix)
        for gen in range(1, 4):                                # grow m chains, so m_neurons read hidden nodes
            net = mutate(net, profile(growth_max_pct=60.0, add_method_weights=only_method("m_after_m")), gen,
                         Rng(seed + gen))
        p = profile(growth_max_pct=1e-9, growth_budget="exact", add_method_weights=only_method(method))
        child = mutate(net, p, 9, Rng(seed))
        child.validate()
        new = [i for i, x in enumerate(child.nodes) if x.birth >= net.next_birth]
        assert len(new) <= 1
        if not new:
            continue
        grew += 1
        x = new[0]
        n = child.nodes[x].neuron
        kind = {"v_tap": VNeuron, "s_tap": SNeuron, "m_tap": MNeuron}[method]
        assert isinstance(n, kind)
        if kind is MNeuron:
            filled = [s for s in n.slots if s is not None]
            assert len(filled) == (q + 1) // 2                 # half the slots, rounded up
            assert all(s < child.n_inputs for s in filled[:2])  # half of those (rounded up) are inputs
            assert all(child.nodes[s].depth < child.nodes[x].depth for s in filled)
        else:
            assert n.input < child.n_inputs
        readers_of_x = [i for i, y in enumerate(child.nodes) if isinstance(y.neuron, MNeuron) and x in y.neuron.slots]
        assert len(readers_of_x) == 1
        r = child.nodes[readers_of_x[0]].neuron
        others = [s for s in r.slots if s is not None and s != x]
        assert child.nodes[x].depth <= max(child.nodes[s].depth for s in others)
    assert grew >= 10


def test_input_tap_needs_a_deeper_reader():
    # the only m_neuron reads inputs only (depth 0): nowhere to put the tap
    net = make_net(2, 1, 3, [(D16, M("max", [0, 1, None]))])
    p = profile(growth_max_pct=100.0, add_method_weights=only_method("v_tap"))
    assert all(net_to_bytes(mutate(net, p, 1, Rng(s))) == net_to_bytes(net) for s in range(10))


# ---------------------------------------------------------------- insert_depth, m_clone (spec/06)

def test_midpoint_insert_depth():
    """With insert_depth = midpoint, a path insertion's new node B sits halfway in its gap."""
    checked = 0
    for seed in range(30):
        net = base_net(seed)
        p = profile(growth_max_pct=1e-9, growth_budget="exact", add_method_weights=only_method("v_insert"),
                    insert_depth="midpoint")                   # exactly one insertion
        child = mutate(net, p, 1, Rng(seed))
        child.validate()
        for b, node in enumerate(child.nodes):
            if node.birth >= net.next_birth:
                continue                                       # B keeps N's birth; X is the new birth
            readers_b = [i for i, y in enumerate(child.nodes) if b in connections(y.neuron)]
            if len(readers_b) != 1 or child.nodes[readers_b[0]].birth < net.next_birth:
                continue
            x = readers_b[0]                                   # X: the inserted neuron, writes A
            conns = connections(node.neuron)
            lo = max(child.nodes[c].depth for c in conns) + 1 if conns else 1
            if isinstance(node.neuron, SNeuron):
                lo += child.in_qty_max + 1                     # N keeps its s gap (spec/03)
            hi = child.nodes[x].depth - 1
            assert node.depth == lo + (hi - lo) // 2
            checked += 1
    assert checked >= 20


def test_m_clone():
    grew = 0
    for seed in range(40):
        net = Net.create(7, 2, DepthType.U32, 5, Priority.SPEED, Rng(seed), path_mix=0)   # merge tree: m reads m
        rows = sample_inputs(7)
        p = profile(growth_max_pct=1e-9, growth_budget="exact", add_method_weights=only_method("m_clone"))
        child = mutate(net, p, 3, Rng(seed))
        child.validate()
        new = [i for i, x in enumerate(child.nodes) if x.birth >= net.next_birth]
        if not new:
            continue
        grew += 1
        x = child.nodes[new[0]]
        twins = [i for i, y in enumerate(child.nodes[:new[0]])
                 if isinstance(y.neuron, MNeuron) and y.neuron == x.neuron and y.depth == x.depth]
        assert twins                                           # an exact copy of an existing m, same depth
        r = [i for i, y in enumerate(child.nodes) if isinstance(y.neuron, MNeuron) and new[0] in y.neuron.slots]
        assert len(r) == 1 and twins[0] in child.nodes[r[0]].neuron.slots   # attached beside its original
    assert grew >= 20


def test_slot_retarget():
    # output m max [3, 4]; 3 and 4 are s_neurons of v 2 (depth 200); slot 0 or 1 alone is not critical
    net = make_net(1, 1, 3, [(D16, M("max", [3, 4, None])), (100, V(0, "mul", 1.0)),
                             (200, S(2, "mean", 1)), (200, S(2, "mean", 1))])
    p = profile(edit_max_pct=100.0, slot_retarget_prob=1.0, rewire_prob=0.0, prune="any", link_max_span=1.0)
    moved = 0
    for s in range(60):
        child = mutate(net, p, 1, Rng(s))
        child.validate()
        slots = [x for x in child.nodes[1].neuron.slots if x is not None]
        assert child.nodes[1].neuron.slots[2] is None          # a retarget never fills an empty slot
        if slots != [3, 4]:
            moved += 1
            assert all(child.nodes[x].depth <= 200 for x in slots)
    assert moved >= 5
    # a slot on the only path is critical: never retargeted
    only = make_net(1, 1, 3, [(D16, M("max", [2, None, None])), (100, V(0, "mul", 1.0))])
    assert all(mutate(only, p, 1, Rng(s)).nodes[1].neuron.slots == [2, None, None] for s in range(20))


def test_m_insert_is_neutral():
    for seed in range(15):
        net = Net.create(5, 2, DepthType.U32, 3, Priority.SPEED, Rng(seed))
        rows = sample_inputs(5)
        ref = outputs_for(net, rows)
        p = profile(growth_max_pct=60.0, add_method_weights=only_method("m_insert"))
        for gen in range(1, 6):
            child = mutate(net, p, gen, Rng(100 * seed + gen))
            child.validate()
            assert outputs_for(child, rows) == ref               # min over one slot passes its input
            net = child
        assert sum(isinstance(n.neuron, MNeuron) and n.neuron.op == AggOp.MIN for n in net.nodes) > 2



# ---------------------------------------------------------------- run rule (spec/03)

@pytest.mark.parametrize("seed", range(10))
def test_mutation_keeps_run_rule(seed):
    """Heavy growth of every kind never leaves a path with more than max_run of a kind in a row."""
    r = Rng(3000 + seed)
    net = Net.create(1 + r.below(8), 1 + r.below(2), DepthType.U32, 3 + 2 * r.below(3), Priority.SPEED, r,
                     path_mix=r.below(3), max_run=1 + r.below(3))
    p = Profile.default()
    for f in PHASE_FIELDS:
        p.overall[f] = 30.0
    p.overall["min_age_for_removal"] = 1
    p.overall["constant_folding"] = "on"
    for gen in range(1, 20):
        net = mutate(net, p, gen, r)
        net.validate()                                         # includes the run rule
        assert run_ok(net)


def test_run_ok_counts_consecutive_kinds():
    # 0 -> v -> v -> v -> output m: three 1:1 in a row
    net = make_net(1, 1, 3, [(D16, M("max", [4, None, None])), (10, V(0, "mul", 1.0)), (20, V(2, "mul", 1.0)),
                             (30, V(3, "mul", 1.0))])
    for k, ok in ((0, True), (2, False), (3, True)):
        net.max_run = k
        assert run_ok(net) == ok


@pytest.mark.parametrize("method", ["v_link", "s_link"])
def test_links_fill_open_m_slots(method):
    """After m_inserts, links connect existing nodes into the open slots (spec/06)."""
    linked = 0
    for seed in range(30):
        net = Net.create(4, 2, DepthType.U32, 5, Priority.SPEED, Rng(seed))
        for gen in range(1, 4):
            net = mutate(net, profile(growth_max_pct=60.0, add_method_weights=only_method("m_insert")), gen,
                         Rng(seed + 50 * gen))
        p = profile(growth_max_pct=1e-9, growth_budget="exact", add_method_weights=only_method(method))
        child = mutate(net, p, 9, Rng(seed))
        child.validate()
        new = [i for i, x in enumerate(child.nodes) if x.birth >= net.next_birth]
        if not new:
            continue
        linked += 1
        x = new[0]
        n = child.nodes[x].neuron
        assert isinstance(n, VNeuron if method == "v_link" else SNeuron)
        r = [i for i, y in enumerate(child.nodes) if isinstance(y.neuron, MNeuron) and x in y.neuron.slots]
        assert len(r) == 1
        others = [s for s in child.nodes[r[0]].neuron.slots if s is not None and s != x]
        assert child.nodes[n.input].depth <= max(child.nodes[s].depth for s in others)
        assert child.nodes[n.input].depth < child.nodes[x].depth < child.nodes[r[0]].depth
        assert child.nodes[r[0]].depth - child.nodes[n.input].depth <= int(0.25 * float(child.d_max))
    assert linked >= 5



def test_prune_static_keeps_live_branches():
    """With prune = static, slot edits never cut a slot whose source is still live (non-constant)."""
    from jneuro.removal import constant_values
    for seed in range(20):
        net = Net.create(4, 2, DepthType.U32, 5, Priority.SPEED, Rng(seed))
        for gen in range(1, 6):
            net = mutate(net, profile(growth_max_pct=60.0, add_method_weights={**only_method("m_insert"),
                                                                               "v_link": 1.0}), gen, Rng(seed + gen))
        before = {(i, s) for i, x in enumerate(net.nodes) if isinstance(x.neuron, MNeuron)
                  for s in x.neuron.slots if s is not None}
        const = constant_values(net, set())
        p = profile(edit_max_pct=100.0, slot_retarget_prob=0.0, sigma_slots=3.0)
        child = mutate(net, p, 9, Rng(seed))
        child.validate()
        assert len(child.nodes) >= len(net.nodes)              # nothing stranded and collected
        for i, x in enumerate(child.nodes[:len(net.nodes)]):
            if isinstance(x.neuron, MNeuron):
                for s in net.nodes[i].neuron.slots:
                    if s is not None and const[s] is None:
                        assert s in x.neuron.slots             # live sources are never cut



def test_growth_net_prob():
    net = base_net(5)
    only_v = only_method("v_insert")
    grew = {}
    for prob in (0.0, 0.5, 1.0):
        p = profile(growth_max_pct=1e-9, growth_budget="exact", add_method_weights=only_v, growth_net_prob=prob)
        grew[prob] = sum(len(mutate(net, p, 1, Rng(s)).nodes) > len(net.nodes) for s in range(60))
    assert grew[0.0] == 0 and grew[1.0] == 60 and 15 < grew[0.5] < 45
    p = profile(growth_max_pct=1e-9, growth_budget="exact", add_method_weights=only_v)
    assert all(len(mutate(net, p, 1, Rng(s), growth_generation=False).nodes) == len(net.nodes) for s in range(10))


def test_growth_generation_is_stateless():
    from jneuro.mutate import is_growth_generation
    p = profile(growth_gen_prob=0.5)
    days = [is_growth_generation(42, g, p) for g in range(200)]
    assert days == [is_growth_generation(42, g, p) for g in range(200)]   # same every time
    assert 60 < sum(days) < 140
    assert all(is_growth_generation(42, g, profile()) for g in range(20))  # 1.0: always



def test_type_swap_when_this_net_does_not_grow():
    """In a growth generation, an offspring whose growth draw fails swaps one 1:1 neuron s <-> v."""
    net = base_net(7)
    for gen in range(1, 5):                                    # extra v and s, so swaps keep the mix rule
        net = mutate(net, profile(growth_max_pct=60.0, add_method_weights={**only_method("v_insert"),
                                                                           "s_insert": 1.0}), gen, Rng(gen))
    swapped = 0
    for s in range(40):
        counts = {}
        p = profile(growth_max_pct=1e-9, growth_budget="exact", growth_net_prob=0.0)
        child = mutate(net, p, 1, Rng(s), counts)
        child.validate()
        assert len(child.nodes) == len(net.nodes)              # nothing added
        kinds = lambda n: [type(x.neuron).__name__ for x in n.nodes]
        diff = [i for i, (a, b) in enumerate(zip(kinds(net), kinds(child))) if a != b]
        assert len(diff) <= 1
        if diff:
            swapped += 1
            assert {kinds(net)[diff[0]], kinds(child)[diff[0]]} == {"VNeuron", "SNeuron"}
            assert counts.get("type_swap") == 1
    assert swapped >= 20
    off = profile(growth_max_pct=1e-9, growth_budget="exact", growth_net_prob=0.0, no_growth_swap="off")
    assert all(net_to_bytes(mutate(net, off, 1, Rng(s))) == net_to_bytes(net) for s in range(10))


def test_biased_slot_edits():
    """slot_edit = biased: slot_add_prob 1 only adds connections, 0 only cuts them."""
    def slots_count(n):
        return sum(1 for x in n.nodes if isinstance(x.neuron, MNeuron) for s in x.neuron.slots if s is not None)
    net = base_net(9)
    for gen in range(1, 4):
        net = mutate(net, profile(growth_max_pct=60.0, add_method_weights=only_method("m_insert")), gen, Rng(gen))
    for prob, sign in ((1.0, 1), (0.0, -1)):
        p = profile(edit_max_pct=100.0, slot_edit="biased", slot_add_prob=prob, prune="any", sigma_slots=2.0,
                    slot_retarget_prob=0.0)
        moved = 0
        for s in range(30):
            child = mutate(net, p, 9, Rng(s))
            d = slots_count(child) - slots_count(net)
            assert d * sign >= 0                               # never the other direction
            moved += d != 0
        if sign > 0:
            assert moved > 5                                   # (cuts here are mostly blocked by the path rule)


def _grown_net(seed, n_in=12, n_out=3, q=5):
    grow = Profile.default()
    grow.overall["growth_max_pct"] = 3.0
    r = Rng(seed)
    net = Net.create(n_in, n_out, DepthType.U32, q, Priority.SPEED, r, 2, 3)
    for g in range(1, 80):
        net = mutate(net, grow, g, r)
    return net


def _neurons(n):
    return sum(1 for x in n.nodes if x.neuron is not None)


def _bulk_profile(pct, open_pct):
    p = profile_from_json((pathlib.Path(__file__).parents[2] / "profiles" / "bulk.json").read_text())
    p.overall["growth_max_pct"] = pct
    p.overall["open_slot_fill_pct"] = open_pct
    return p


def test_bulk_profile_adds_the_requested_percentage():
    for seed in range(3):
        net = _grown_net(seed)
        for pct in (10.0, 40.0):
            big = mutate(net, _bulk_profile(pct, 80.0), 100, Rng(seed + 7))
            target = max(1.0, pct * _neurons(net) / 100.0)
            assert math.floor(target) <= _neurons(big) - _neurons(net) <= math.ceil(target)


def test_bulk_growth_without_links_leaves_outputs_unchanged():
    for seed in range(3):
        net = _grown_net(seed)
        big = mutate(net, _bulk_profile(10.0, 0.0), 100, Rng(seed + 7))
        assert _neurons(big) > _neurons(net)
        ri = Rng(seed + 99)
        net.reset()
        big.reset()
        for _ in range(40):
            x = [(ri.u() - 0.5) * 10 for _ in range(12)]
            assert [v.hex() for v in net.step(x)] == [v.hex() for v in big.step(x)]


def test_open_slot_fill_needs_the_fill_budget():
    net = _grown_net(1)
    p = _bulk_profile(10.0, 80.0)
    p.overall["growth_budget"] = "exact"
    q = _bulk_profile(10.0, 0.0)
    q.overall["growth_budget"] = "exact"
    assert net_to_bytes(mutate(net, p, 100, Rng(3))) == net_to_bytes(mutate(net, q, 100, Rng(3)))
    with pytest.raises(Exception):
        profile_from_json(json.dumps({"format": "j_neuro.profile", "spec_version": "0.5.0",
                                      "overall": {"open_slot_fill_pct": 101}}))
