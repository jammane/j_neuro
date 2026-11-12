"""spec/12: adding and removing inputs and outputs."""
import pytest
from helpers import M, V, load_golden, make_net, outputs_for, sample_inputs

from jneuro import AggOp, DepthType, MNeuron, Net, Priority, ValidationError, VNeuron, VOp
from jneuro.jnet import net_from_bytes, net_to_bytes
from jneuro.modify import add_input, add_output_clone, add_output_fresh, remove_input, remove_output
from jneuro.mutate import Profile, mutate
from jneuro.rng import Rng

D16 = 65535


def grown(seed, n_in, n_out, dt=DepthType.U32, q=5, mix=2, run=3, gens=14):
    grow = Profile.default()
    for f in ("op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
              "depth_shift_max_pct"):
        grow.overall[f] = 35.0
    r = Rng(seed)
    net = Net.create(n_in, n_out, dt, q, Priority.SPEED, r, mix, run)
    for gen in range(1, gens + 1):
        net = mutate(net, grow, gen, r)
    return net


NETS = [grown(s, 1 + s % 5, 1 + s % 4, list(DepthType)[s % 3], [3, 5, 7][s % 3], s % 3, s % 4)
        for s in range(12)]


@pytest.mark.parametrize("net", NETS)
def test_add_input_neutral_keeps_outputs(net):
    rows = sample_inputs(net.n_inputs)
    before = outputs_for(net, rows)
    for pos in range(net.n_inputs + 1):
        child = add_input(net, pos, generation=7)
        assert child.n_inputs == net.n_inputs + 1
        rows2 = [r[:pos] + [3.5 * (t + 1) - 9.0] + r[pos:] for t, r in enumerate(rows)]
        assert outputs_for(child, rows2) == before
        assert child.nodes[pos].birth == net.next_birth and child.nodes[pos].born_gen == 7


@pytest.mark.parametrize("net", NETS)
def test_add_input_active_reaches_outputs(net):
    child = add_input(net, active=True)
    child.validate()
    ws = [n.neuron for n in child.nodes[len(net.nodes):] if isinstance(n.neuron, VNeuron)]
    assert ws and ws[-1].op == VOp.MUL and ws[-1].weight == 1.0


def test_add_input_min_receivers_use_median():
    net = Net.create(1, 3, DepthType.U16, 3, Priority.SPEED, Rng(3))   # chains use only min m's
    child = add_input(net)
    for o in range(3):
        n = child.nodes[child.n_inputs + o].neuron
        assert isinstance(n, MNeuron) and n.op == AggOp.MEDIAN
    rows = sample_inputs(1)
    assert outputs_for(child, [r + [7.0] for r in rows]) == outputs_for(net, rows)


def test_add_input_parent_untouched():
    net = NETS[3]
    before = net_to_bytes(net)
    add_input(net, 0)
    assert net_to_bytes(net) == before


@pytest.mark.parametrize("net", NETS)
def test_add_output_clone(net):
    rows = sample_inputs(net.n_inputs)
    before = outputs_for(net, rows)
    child = add_output_clone(net, net.n_outputs - 1, 0, generation=4)
    out = outputs_for(child, rows)
    assert [o[1:] for o in out] == before
    assert all(o[0] == o[-1] for o in out)


@pytest.mark.parametrize("net", NETS)
def test_add_output_fresh(net):
    rows = sample_inputs(net.n_inputs)
    before = outputs_for(net, rows)
    child = add_output_fresh(net, Rng(5))
    assert [o[:-1] for o in outputs_for(child, rows)] == before
    assert child.n_outputs == net.n_outputs + 1


@pytest.mark.parametrize("net", [n for n in NETS if n.n_outputs > 1])
def test_remove_output_keeps_others(net):
    rows = sample_inputs(net.n_inputs)
    before = outputs_for(net, rows)
    for idx in range(net.n_outputs):
        child = remove_output(net, idx)
        assert outputs_for(child, rows) == [o[:idx] + o[idx + 1:] for o in before]


def test_remove_input_cascades_both_ways():
    # input 1 feeds only v 4 (with a bias constant 5 behind m 3); input 0 feeds v 6
    n = make_net(2, 1, 3, [
        (D16, M("max", [6, 3, None])),            # 2: output
        (200, M("mean", [1, 5, None])),            # 3: only input 1 reaches it
        (100, V(1)),                               # 4: unused here, reads input 1 ... (dead below)
        (50, V(None, "add", 2.0)),                 # 5: bias, feeds only 3
        (100, V(0)),                               # 6
    ])
    n.nodes[4].neuron = V(1)
    n.nodes[3].neuron = M("mean", [4, 5, None])
    n.validate()
    child = remove_input(n, 1)
    assert child.n_inputs == 1 and len(child.nodes) == 3   # input 0, output, v 6
    assert child.nodes[1].neuron.slots == [2, None, None]


def test_remove_input_keeps_shared():
    n = make_net(2, 1, 3, [
        (D16, V(3)),                               # 2: output
        (100, M("mean", [0, 1, None])),            # 3: shared by both inputs
    ])
    child = remove_input(n, 0)
    assert len(child.nodes) == 3 and child.nodes[2].neuron.slots == [None, 0, None]


def test_errors():
    one = Net.create(1, 1)
    with pytest.raises(ValidationError):
        remove_input(one, 0)
    with pytest.raises(ValidationError):
        remove_output(one, 0)
    with pytest.raises(ValidationError):
        add_input(one, 2)
    with pytest.raises(ValidationError):
        add_output_clone(one, 1)
    with pytest.raises(ValidationError):
        add_output_fresh(one, Rng(0), 5)


def test_round_trip_shapes():
    net = NETS[7]
    a = add_output_fresh(add_input(net), Rng(1), 0)
    b = remove_output(remove_input(a, a.n_inputs - 1), 0)
    assert (b.n_inputs, b.n_outputs) == (net.n_inputs, net.n_outputs)
    b.validate()


# ---------------------------------------------------------------- golden

GOLD = load_golden("modify")


@pytest.mark.parametrize("name,case", GOLD, ids=[c[0] for c in GOLD])
def test_modify_golden(name, case):
    net = net_from_bytes(bytes.fromhex(case["jnet"]))
    a = case["args"]
    op = case["op"]
    if op == "add_input":
        child = add_input(net, a["position"], a["active"], a["generation"])
    elif op == "add_output_clone":
        child = add_output_clone(net, a["source"], a["position"], a["generation"])
    elif op == "add_output_fresh":
        child = add_output_fresh(net, Rng(a["seed"]), a["position"], a["generation"])
    elif op == "remove_input":
        child = remove_input(net, a["index"])
    else:
        child = remove_output(net, a["index"])
    assert net_to_bytes(child) == bytes.fromhex(case["expected_jnet"])
