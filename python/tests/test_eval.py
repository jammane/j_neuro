import pytest
from helpers import INF, M, S, V, bits, from_hex, load_golden, make_net, outputs_for, same

from jneuro import Priority
from jneuro.io import genome_from_json

import json

D16 = 65535
EVAL_CASES = load_golden("handmade")


@pytest.mark.parametrize("name,case", EVAL_CASES, ids=[c[0] for c in EVAL_CASES])
def test_eval_golden(name, case):
    for prio in case["priorities"]:
        net = genome_from_json(json.dumps(case["genome"]))
        net.priority = Priority[prio]
        net.reset()
        for st in case["steps"]:
            out = net.step([from_hex(h) for h in st["inputs"]])
            assert [bits(y) for y in out] == st["outputs"], (prio, st)


def test_handmade_present():
    assert len(EVAL_CASES) >= 7


def test_depth_order_and_same_depth_independence():
    # out <- v(h2 add 1); h2 (depth 20) <- v(h1 mul 2); h1 (depth 10) <- v(in0 mul 3)
    net = make_net(1, 1, 3, [(D16, V(3, "add", 1.0)), (20, V(4, "mul", 2.0)), (10, V(0, "mul", 3.0))])
    assert net.step([1.0]) == [7.0]


def test_reset_clears_history():
    net = make_net(1, 1, 3, [(D16, S(0, "mean", 1))])
    assert net.step([5.0]) == [5.0]
    assert net.step([5.0]) == [0.0]
    net.reset()
    assert net.step([5.0]) == [5.0]


def test_missing_input_constants():
    net = make_net(1, 1, 3, [(D16, M("mean", [0, 2, None])), (5, V(None, "add", 3.0))])
    assert net.step([0.0]) == [1.0]  # (0 + 3 + 0) / 3


def test_priority_invariance_and_switching():
    net = make_net(2, 2, 3, [(D16, M("max", [4, 1, None])), (D16, S(4, "median", 3)),
                             (100, V(0, "div", 2.0, "recip"))])
    rows = [[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0], [7.0, -1.0], [INF, 2.0]]
    results = []
    for p in Priority:
        net.priority = p
        results.append(outputs_for(net, rows))
    assert results[0] == results[1] == results[2]


def test_output_values_canonical():
    net = make_net(1, 1, 3, [(D16, V(0, "mul", -1.0))])
    assert same(net.step([0.0])[0], 0.0)  # 0 * -1 = -0 -> +0
