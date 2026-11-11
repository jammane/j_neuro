import pytest
from helpers import from_hex, load_golden, same

from jneuro.rng import Rng, round_half_away

CASES = load_golden("rng")


@pytest.mark.parametrize("name,case", CASES, ids=[c[0] for c in CASES])
def test_rng_golden(name, case):
    seed = case["seed"]
    r = Rng(seed)
    assert [f"{r.next_u64():016x}" for _ in case["next_u64"]] == case["next_u64"]
    r = Rng(seed)
    assert all(same(r.u(), from_hex(h)) for h in case["u"])
    for b in case["below"]:
        r = Rng(seed)
        assert [r.below(b["n"]) for _ in b["values"]] == b["values"]
    r = Rng(seed)
    assert all(same(r.normal(), from_hex(h)) for h in case["normal"])
    for p in case["prob_round"]:
        r = Rng(seed)
        x = from_hex(p["x"])
        assert [r.prob_round(x) for _ in p["values"]] == p["values"]
    for w in case["weighted"]:
        r = Rng(seed)
        ws = [from_hex(h) for h in w["w"]]
        assert [r.weighted(ws) for _ in w["values"]] == w["values"]


def test_golden_present():
    assert len(CASES) == 3


def test_state_save_restore():
    r = Rng(7)
    r.next_u64()
    st = r.get_state()
    a = [r.next_u64() for _ in range(5)]
    r.set_state(st)
    assert [r.next_u64() for _ in range(5)] == a


def test_ranges():
    r = Rng(1)
    for _ in range(2000):
        x = r.u()
        assert 0.0 <= x < 1.0
        assert 0 <= r.below(7) < 7
        assert 0 <= r.choice(4) < 4
        assert -6.0 <= r.normal() <= 6.0


def test_weighted_skips_zero_weights():
    r = Rng(3)
    for _ in range(500):
        assert r.weighted([0.0, 1.0, 0.0, 2.0]) in (1, 3)


def test_prob_round_always_draws():
    a, b = Rng(5), Rng(5)
    a.prob_round(3.0)
    b.u()
    assert a.next_u64() == b.next_u64()


def test_round_half_away():
    assert round_half_away(0.5) == 1
    assert round_half_away(-0.5) == -1
    assert round_half_away(1.49) == 1
    assert round_half_away(-2.5) == -3
    assert round_half_away(0.0) == 0
