"""Spec 0.3.0: population container (spec/11)."""
import json

import pytest
from helpers import M, S, V, bits, from_hex, load_golden, make_net, sample_inputs

from jneuro import AggOp, DepthType, Net, Priority, ValidationError, VNeuron
from jneuro.jnet import net_from_bytes, net_to_bytes
from jneuro.mutate import Profile, mutate
from jneuro.population import Population
from jneuro.rng import Rng

D32 = 2**32 - 1
PHASES = ["op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
          "depth_shift_max_pct"]


def pop(seed=11, prio=Priority.BALANCED):
    """path_mix 0, so the hand-built nets below can be added."""
    return Population(DepthType.U32, 5, 3, 2, prio, seed=seed, path_mix=0, max_run=0)


def heavy(mode="fresh"):
    p = Profile.default()
    for f in PHASES:
        p.overall[f] = 50.0
    p.overall["rng_mode"] = mode
    return p


def genomes(p):
    return [net_to_bytes(p.get(i)) for i in p.ids()]


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
    with pytest.raises(ValidationError):
        p.add(Net.create(3, 2, DepthType.U32, 5))              # path_mix 2, population has 0
    net = Net.create(3, 2, DepthType.U32, 5, path_mix=0, max_run=0)
    i = p.add(net)
    net.nodes[3].neuron.op = AggOp((net.nodes[3].neuron.op + 1) % 4)
    assert net_to_bytes(p.get(i)) != net_to_bytes(net)


def test_ids_never_reused_and_unknown_ids():
    p = pop()
    p.create(3)
    p.remove(1)
    assert p.ids() == [0, 2]
    assert p.clone(0) == 3
    assert net_to_bytes(p.get(3)) == net_to_bytes(p.get(0))
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
    before = net_to_bytes(p.get(a))
    b = p.mutate(a, heavy(), times=3)
    assert b == 1 and net_to_bytes(p.get(a)) == before
    p.get(b).validate()


def test_shared_mutate_equals_manual():
    prof = heavy("shared")
    p = Population(DepthType.U32, 5, 3, 2, Priority.BALANCED, seed=7)
    [a] = p.create(1, rng_mode="shared")
    b = p.mutate(a, prof, times=2)
    r = Rng(7)
    net = Net.create(3, 2, DepthType.U32, 5, Priority.BALANCED, r)
    manual = mutate(mutate(net, prof, 0, r), prof, 0, r)
    assert net_to_bytes(p.get(b)) == net_to_bytes(manual)


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
    p = Population(DepthType.U32, 5, 3, 2, Priority.SPEED, seed=seed, path_mix=0, max_run=0)  # adds a hand-built net
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
    assert net_to_bytes(p.get(p.mutate(0, heavy("shared")))) == \
           net_to_bytes(q.get(q.mutate(0, heavy("shared"))))


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
def test_population_golden(name, case, tmp_path):
    q = Population.replay(json.dumps(case["log"]))
    exp = case["expected"]
    assert q.ids() == exp["ids"] and q.generation == exp["generation"]
    for i in q.ids():
        assert net_to_bytes(q.get(i)) == bytes.fromhex(exp["jnets"][str(i)])
    ev = case["evaluate"]
    res = q.evaluate([[from_hex(h) for h in row] for row in ev["inputs"]])
    assert {str(i): [[bits(y) for y in r] for r in rows] for i, rows in res.items()} == ev["outputs"]
    q.save(tmp_path / "pop.jpnet")
    assert (tmp_path / "pop.jpnet").read_bytes().hex() == case["save"]["jpnet"]
    for i in q.ids():
        assert (tmp_path / "pop_nets" / f"{i}.jnet").read_bytes().hex() == exp["jnets"][str(i)]


def test_returned_ids_do_not_alias_the_log():
    p = pop()
    ids = p.create(2)
    ids += [99]
    assert json.loads(p.run_log())["ops"][0]["ids"] == [0, 1]


# ------------------------------------------------------------------ .jpnet files

def test_save_load_roundtrip(tmp_path):
    p = session_with_everything(seed=21)
    p.save(tmp_path / "run.jpnet")
    assert sorted(f.name for f in (tmp_path / "run_nets").iterdir()) == sorted(f"{i}.jnet" for i in p.ids())
    q = Population.load(tmp_path / "run.jpnet")
    assert q.ids() == p.ids() and q.generation == p.generation and genomes(q) == genomes(p)
    # shared stream and id counter continue exactly
    assert net_to_bytes(p.get(p.mutate(0, heavy("shared")))) == net_to_bytes(q.get(q.mutate(0, heavy("shared"))))
    assert p.create(1) == q.create(1)


def test_save_is_byte_identical(tmp_path):
    p = session_with_everything(seed=4)
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    p.save(tmp_path / "x" / "pop.jpnet")
    p.save(tmp_path / "y" / "pop.jpnet")
    assert (tmp_path / "x" / "pop.jpnet").read_bytes() == (tmp_path / "y" / "pop.jpnet").read_bytes()


def test_load_rejects_tampering(tmp_path):
    p = session_with_everything(seed=5)
    p.save(tmp_path / "run.jpnet")
    victim = tmp_path / "run_nets" / f"{p.ids()[0]}.jnet"
    good = victim.read_bytes()
    victim.write_bytes(good[:-1] + bytes([good[-1] ^ 1]))
    with pytest.raises(ValidationError):
        Population.load(tmp_path / "run.jpnet")
    victim.unlink()
    with pytest.raises(ValidationError):
        Population.load(tmp_path / "run.jpnet")
    victim.write_bytes(good)
    jp = (tmp_path / "run.jpnet").read_bytes()
    (tmp_path / "run.jpnet").write_bytes(jp[:-1] + bytes([jp[-1] ^ 1]))
    with pytest.raises(ValidationError):
        Population.load(tmp_path / "run.jpnet")


def test_embedded_log_replays_from_scratch(tmp_path):
    p = session_with_everything()
    p.save(tmp_path / "run.jpnet", include_log=True)
    q = Population.load(tmp_path / "run.jpnet")
    assert q.run_log() == p.run_log()
    q.mutate(0, heavy())
    r = Population.replay(q.run_log())
    assert genomes(r) == genomes(q)


def test_resumed_log_needs_base(tmp_path):
    p = session_with_everything()
    p.save(tmp_path / "run.jpnet")
    q = Population.load(tmp_path / "run.jpnet")
    log0 = json.loads(q.run_log())
    import hashlib
    assert log0["resumed_from"] == hashlib.sha256((tmp_path / "run.jpnet").read_bytes()).hexdigest()
    assert log0["ops"] == []
    q.mutate(0, heavy())
    q.remove(1)
    q.next_generation()
    with pytest.raises(ValidationError):
        Population.replay(q.run_log())
    base = Population.load(tmp_path / "run.jpnet")
    r = Population.replay(q.run_log(), base=base)
    assert r.ids() == q.ids() and genomes(r) == genomes(q) and r.run_log() == q.run_log()
    other = session_with_everything(seed=3)
    other.save(tmp_path / "other.jpnet")
    with pytest.raises(ValidationError):
        Population.replay(q.run_log(), base=Population.load(tmp_path / "other.jpnet"))


def test_default_path_mix_is_two():
    p = Population(DepthType.U32, 3, 9, 1, Priority.SPEED, seed=5)
    assert p.path_mix == 2
    for i in p.create(3):
        assert p.get(i).path_mix == 2
    child = p.mutate(0, heavy("fresh"), times=5)
    p.get(child).validate()                                    # mix rule preserved
    assert Population.replay(p.run_log()).path_mix == 2


def test_mutation_counts_cover_every_operation():
    from jneuro.mutate import OPERATIONS
    p = Population(DepthType.U32, 5, 4, 2, Priority.SPEED, seed=3)
    ids = p.create(6)
    prof = heavy("fresh")
    prof.overall["constant_folding"] = "on"
    prof.overall["add_method_weights"].update({"v_tap": 40, "s_tap": 40, "m_tap": 40, "m_clone": 40,
                                               "slot_fill": 40})   # slot fill feeds silenced removal and folding
    prof.overall["slot_retarget_prob"] = 0.5
    for gen in range(18):
        ids = [p.mutate(i, prof) for i in ids]
        p.next_generation()
    counts = p.mutation_counts()
    assert set(counts) == set(OPERATIONS)
    unused = [op for op, n in counts.items() if n == 0]
    assert not unused, unused
    p.reset_mutation_counts()
    assert sum(p.mutation_counts().values()) == 0
