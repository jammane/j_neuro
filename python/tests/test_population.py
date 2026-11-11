"""Spec 0.3.0: population container (spec/11)."""
import json

import pytest
from helpers import M, S, V, bits, from_hex, load_golden, make_net, sample_inputs

from jneuro import AggOp, DepthType, Net, Priority, ValidationError, VNeuron
from jneuro.io import genome_from_json, genome_to_json
from jneuro.mutate import Profile, mutate
from jneuro.population import Population
from jneuro.rng import Rng

D32 = 2**32 - 1
PHASES = ["op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
          "depth_shift_max_pct"]


def pop(seed=11, prio=Priority.BALANCED):
    return Population(DepthType.U32, 5, 3, 2, prio, seed=seed)


def heavy(mode="fresh"):
    p = Profile.default()
    for f in PHASES:
        p.overall[f] = 50.0
    p.overall["rng_mode"] = mode
    return p


def genomes(p):
    return [genome_to_json(p.get(i)) for i in p.ids()]


def warm_net():
    # out2 = mean(in0..in2); out3 = s(k=2) over hidden mean(in0..in2): warm-up 2
    return make_net(3, 2, 5, [(D32, M("mean", [0, 1, 2, None, None])), (D32, S(5, "mean", 2)),
                              (10, M("mean", [0, 1, 2, None, None]))], depth_type="U32")


def test_default_rng_mode_is_fresh():
    assert Profile.default().overall["rng_mode"] == "fresh"


def test_create_and_ids():
    p = pop()
    assert p.create(4) == [0, 1, 2, 3]
    assert p.ids() == [0, 1, 2, 3] and p.size() == 4 and p.generation == 0
    for i in p.ids():
        p.get(i).validate()
        assert (p.get(i).n_inputs, p.get(i).n_outputs) == (3, 2)
    assert p.create(2) == [4, 5]


def test_add_rejects_mismatch_and_copies():
    p = pop()
    with pytest.raises(ValidationError):
        p.add(Net.create(3, 2, DepthType.U16, 5))
    with pytest.raises(ValidationError):
        p.add(Net.create(2, 2, DepthType.U32, 5))
    with pytest.raises(ValidationError):
        p.add(Net.create(3, 2, DepthType.U32, 3))
    net = Net.create(3, 2, DepthType.U32, 5)
    i = p.add(net)
    net.nodes[3].neuron.op = AggOp((net.nodes[3].neuron.op + 1) % 4)
    assert genome_to_json(p.get(i)) != genome_to_json(net)


def test_ids_never_reused_and_unknown_ids():
    p = pop()
    p.create(3)
    p.remove(1)
    assert p.ids() == [0, 2]
    assert p.clone(0) == 3
    assert genome_to_json(p.get(3)) == genome_to_json(p.get(0))
    for call in (lambda: p.get(1), lambda: p.remove(1), lambda: p.clone(1),
                 lambda: p.mutate(1, heavy()), lambda: p.inject_diversity(1, 10.0)):
        with pytest.raises(KeyError):
            call()


def test_evaluate_matches_step_and_skips_warmup():
    p = pop()
    p.create(2)
    w = p.add(warm_net())
    rows = sample_inputs(3, 6)
    res = p.evaluate(rows)
    assert list(res) == p.ids()
    for i in p.ids():
        net = p.get(i).copy()
        full = [[bits(y) for y in net.step(r)] for r in rows]
        skip = net.get_warmup_steps()
        assert [[bits(y) for y in row] for row in res[i]] == full[skip:]
    assert len(res[w]) == 4
    assert all(len(v) == 6 for v in p.evaluate(rows, skip_warmup=False).values())


def test_evaluate_priority_invariance():
    out = []
    for prio in Priority:
        p = pop(seed=5, prio=prio)
        p.create(6, rng_mode="shared")
        p.add(warm_net())
        out.append({i: [[bits(y) for y in r] for r in rows]
                    for i, rows in p.evaluate(sample_inputs(3, 8)).items()})
    assert out[0] == out[1] == out[2]


def test_mutate_offspring_keeps_parent():
    p = pop()
    [a] = p.create(1)
    before = genome_to_json(p.get(a))
    b = p.mutate(a, heavy(), times=3)
    assert b == 1 and genome_to_json(p.get(a)) == before
    p.get(b).validate()


def test_shared_mutate_equals_manual():
    prof = heavy("shared")
    p = Population(DepthType.U32, 5, 3, 2, Priority.BALANCED, seed=7)
    [a] = p.create(1, rng_mode="shared")
    b = p.mutate(a, prof, times=2)
    r = Rng(7)
    net = Net.create(3, 2, DepthType.U32, 5, Priority.BALANCED, r)
    manual = mutate(mutate(net, prof, 0, r), prof, 0, r)
    assert genome_to_json(p.get(b)) == genome_to_json(manual)


def test_shared_reproducible_fresh_varies():
    def session(mode):
        p = Population(DepthType.U32, 5, 3, 2, seed=99)
        for i in p.create(4, rng_mode=mode):
            p.mutate(i, heavy(mode), times=2)
        return genomes(p)
    assert session("shared") == session("shared")
    assert session("fresh") != session("fresh")


def test_inject_diversity_in_place():
    p = pop()
    net = make_net(3, 2, 5, [(D32, M("mean", [5, 1, 2, None, None])), (D32, M("max", [5, 1, 2, None, None])),
                             (10, V(0, "mul", 1.0))], depth_type="U32")
    i = p.add(net)
    p.inject_diversity(i, 100.0)
    assert p.ids() == [i]
    assert isinstance(p.get(i).nodes[5].neuron, VNeuron) and p.get(i).nodes[5].neuron.weight != 1.0


def test_generation_used_for_new_neurons():
    p = pop()
    [a] = p.create(1)
    p.next_generation()
    p.next_generation()
    prof = Profile.default()
    for f in PHASES:
        prof.overall[f] = 0.0
    prof.overall["growth_max_pct"] = 100.0
    for _ in range(30):
        b = p.mutate(a, prof)
        new = [n for n in p.get(b).nodes if n.birth >= p.get(a).next_birth]
        if new:
            assert all(n.born_gen == 2 for n in new)
            return
    raise AssertionError("no growth happened")


def session_with_everything(seed=None):
    p = Population(DepthType.U32, 5, 3, 2, Priority.SPEED, seed=seed)
    ids = p.create(3)
    p.create(2, rng_mode="shared")
    a = p.mutate(ids[0], heavy("fresh"), times=2)
    b = p.mutate(ids[1], heavy("shared"))
    p.next_generation()
    c = p.clone(a)
    p.inject_diversity(c, 50.0, sigma_reset=2.0)
    p.inject_diversity(b, 30.0, rng_mode="shared")
    p.add(warm_net())
    p.remove(ids[2])
    prof = heavy("shared")
    prof.overall["sigma_weight"] = 0.3
    p.mutate(c, prof, times=3)
    return p


def test_replay_reproduces_exactly():
    p = session_with_everything()          # fresh seeds from the OS
    log = p.run_log()
    q = Population.replay(log)
    assert q.ids() == p.ids() and q.generation == p.generation
    assert genomes(q) == genomes(p)
    assert q.run_log() == log
    # the shared stream continues identically after replay
    assert genome_to_json(p.get(p.mutate(0, heavy("shared")))) == \
           genome_to_json(q.get(q.mutate(0, heavy("shared"))))


def test_profiles_deduplicated_in_log():
    p = pop()
    [a] = p.create(1)
    for _ in range(3):
        p.mutate(a, heavy())
    p.mutate(a, heavy("shared"))
    log = json.loads(p.run_log())
    assert len(log["profiles"]) == 2
    assert [o["profile"] for o in log["ops"] if o["op"] == "mutate"] == [0, 0, 0, 1]
    assert [o["seed"] is None for o in log["ops"] if o["op"] == "mutate"] == [False, False, False, True]


GOLD = load_golden("population")


@pytest.mark.parametrize("name,case", GOLD, ids=[c[0] for c in GOLD])
def test_population_golden(name, case):
    q = Population.replay(json.dumps(case["log"]))
    exp = case["expected"]
    assert q.ids() == exp["ids"] and q.generation == exp["generation"]
    for i in q.ids():
        assert genome_to_json(q.get(i)) == genome_to_json(genome_from_json(json.dumps(exp["genomes"][str(i)])))
    ev = case["evaluate"]
    res = q.evaluate([[from_hex(h) for h in row] for row in ev["inputs"]])
    assert {str(i): [[bits(y) for y in r] for r in rows] for i, rows in res.items()} == ev["outputs"]


def test_returned_ids_do_not_alias_the_log():
    p = pop()
    ids = p.create(2)
    ids += [99]
    assert json.loads(p.run_log())["ops"][0]["ids"] == [0, 1]
