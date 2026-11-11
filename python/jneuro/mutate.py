"""Mutation profile and mutation (spec/06, removal phase in spec/07)."""
from __future__ import annotations

import copy
import math

from .analysis import connections, is_immune, legal_depth_range, path_rule_ok, readers
from .net import MNeuron, Net, Node, SNeuron, VNeuron
from .removal import (bypass, compact, garbage_collect, has_min_reader, is_identity,
                      is_silenced, remove_silenced)
from .rng import Rng, round_half_away
from .types import IDENTITY_WEIGHT, AggOp, Inversion, ValidationError, VOp

TYPES = ("v", "s", "m")
METHODS = ("slot_fill", "v_insert", "s_insert", "m_after_m")
PHASE_FIELDS = ("op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
                "depth_shift_max_pct")

DEFAULTS = {
    "op_change_max_pct": 1.0,
    "edit_max_pct": 5.0,
    "growth_max_pct": 1.5,
    "removal_max_pct": 1.0,
    "depth_shift_max_pct": 5.0,
    "depth_shift_mode": "uniform",
    "sigma_weight": 0.1,
    "sign_flip_prob": 0.02,
    "sigma_k": 1.0,
    "sigma_slots": 1.0,
    "min_age_for_removal": 5,
    "add_method_weights": {"slot_fill": 40, "v_insert": 30, "s_insert": 20, "m_after_m": 10},
    "slot_fill_type_weights": {"v": 1, "s": 1, "m": 1},
}
OVERRIDABLE = set(PHASE_FIELDS) | {"sigma_weight", "sign_flip_prob", "sigma_k", "sigma_slots",
                                   "min_age_for_removal"}


def _tkey(t) -> str:
    return t if isinstance(t, str) else TYPES[int(t)]


class Profile:
    """Mutation profile: overall values plus per-type overrides (spec/06)."""

    def __init__(self, overall=None, per_type=None):
        self.overall = copy.deepcopy(DEFAULTS)
        if overall:
            self._merge_overall(overall)
        self.per_type = {t: {} for t in TYPES}
        for t, sec in (per_type or {}).items():
            for k, v in sec.items():
                self._set_override(t, k, v)

    @classmethod
    def default(cls) -> "Profile":
        return cls()

    def copy(self) -> "Profile":
        return Profile(self.overall, self.per_type)

    def _merge_overall(self, d):
        for k, v in d.items():
            if k not in DEFAULTS:
                raise ValidationError(f"unknown profile field {k!r}")
            if v is None:
                self.overall[k] = copy.deepcopy(DEFAULTS[k])
            elif isinstance(DEFAULTS[k], dict):
                for kk, vv in v.items():
                    if kk not in DEFAULTS[k]:
                        raise ValidationError(f"unknown key {kk!r} in {k}")
                    self.overall[k][kk] = DEFAULTS[k][kk] if vv is None else vv
            else:
                self.overall[k] = v

    def _set_override(self, t, k, v):
        if t not in TYPES or k not in OVERRIDABLE:
            raise ValidationError(f"{k!r} cannot be overridden for type {t!r}")
        if v is None:
            self.per_type[t].pop(k, None)
        else:
            self.per_type[t][k] = v

    def apply_patch(self, patch: dict):
        """JSON Merge Patch (RFC 7386); null removes a per-type override."""
        for sec, body in patch.items():
            if sec == "overall":
                self._merge_overall(body)
            elif sec in TYPES:
                for k, v in (body or {}).items():
                    self._set_override(sec, k, v)
            elif sec not in ("format", "spec_version"):
                raise ValidationError(f"unknown profile section {sec!r}")

    def effective(self, field, t):
        return self.per_type[_tkey(t)].get(field, self.overall[field])

    def has_override(self, field, t) -> bool:
        return field in self.per_type[_tkey(t)]

    def to_dict(self) -> dict:
        from .io import PROFILE_FORMAT, SPEC_VERSION
        d = {"format": PROFILE_FORMAT, "spec_version": SPEC_VERSION,
             "overall": copy.deepcopy(self.overall)}
        for t in TYPES:
            d[t] = dict(self.per_type[t])
        return d

    @classmethod
    def from_dict(cls, d) -> "Profile":
        return cls(d.get("overall"), {t: d.get(t) or {} for t in TYPES})


# ---------------------------------------------------------------------- mutation

def _type_of(neuron) -> str:
    return "v" if isinstance(neuron, VNeuron) else "s" if isinstance(neuron, SNeuron) else "m"


class _Mutation:
    def __init__(self, parent: Net, profile: Profile, generation: int, rng: Rng):
        self.net = parent.copy()
        self.p = profile
        self.gen = generation
        self.rng = rng
        self.removed = set()
        self.q = self.net.in_qty_max

    # helpers
    def live_neurons(self):
        return [i for i in range(self.net.n_inputs, len(self.net.nodes)) if i not in self.removed]

    def live_nodes(self):
        return [i for i in range(len(self.net.nodes)) if i not in self.removed]

    def budget(self, field):
        rng = self.rng
        n = float(len(self.live_neurons()))
        b = rng.prob_round((rng.u() * self.p.overall[field]) * n / 100.0)
        caps = {}
        for t in TYPES:
            if self.p.has_override(field, t):
                caps[t] = rng.prob_round((rng.u() * self.p.per_type[t][field]) * n / 100.0)
            else:
                caps[t] = None
        return b, caps, {t: 0 for t in TYPES}

    def run_phase(self, field, attempt):
        b, caps, used = self.budget(field)
        for _ in range(b):
            t = attempt(caps, used)
            if t is not None:
                used[t] += 1

    def pick_neuron(self, caps, used):
        lst = self.live_neurons()
        if not lst:
            return None, None
        nid = lst[self.rng.below(len(lst))]
        t = _type_of(self.net.nodes[nid].neuron)
        if caps[t] is not None and used[t] >= caps[t]:
            return None, None
        return nid, t

    # phase 1
    def op_change(self, caps, used):
        nid, t = self.pick_neuron(caps, used)
        if nid is None:
            return None
        n = self.net.nodes[nid].neuron
        rng = self.rng
        if t == "v":
            cur = int(n.inversion) * 4 + int(n.op)
            j = rng.below(11)
            if j >= cur:
                j += 1
            n.inversion, n.op = Inversion(j // 4), VOp(j % 4)
            n.weight = IDENTITY_WEIGHT[n.op]
        else:
            j = rng.below(3)
            if j >= int(n.op):
                j += 1
            n.op = AggOp(j)
            if t == "s" and n.op == AggOp.MEDIAN and n.k % 2 == 0:
                n.k -= 1
        return t

    # phase 2
    def edit(self, caps, used):
        nid, t = self.pick_neuron(caps, used)
        if nid is None:
            return None
        n = self.net.nodes[nid].neuron
        rng, p = self.rng, self.p
        if t == "v":
            z = rng.normal()
            sw = p.effective("sigma_weight", "v")
            if n.op in (VOp.ADD, VOp.SUB):
                w2 = n.weight + sw * z
            else:
                a = sw * z
                f = 1.0 + a if a >= 0 else 1.0 / (1.0 - a)
                w2 = n.weight * f
            if rng.u() < p.effective("sign_flip_prob", "v"):
                w2 = -w2
            if not math.isfinite(w2):
                return None
            n.weight = w2
            return t
        if t == "s":
            z = rng.normal()
            d = round_half_away(p.effective("sigma_k", "s") * z)
            k2 = min(max(n.k + d, 1), self.q)
            if n.op == AggOp.MEDIAN and k2 % 2 == 0:
                k2 = k2 - 1 if k2 > n.k else k2 + 1
            if k2 == n.k:
                return None
            n.k = k2
            return t
        z = rng.normal()
        c = min(self.q, 1 + math.floor(abs(p.effective("sigma_slots", "m") * z)))
        my_depth = self.net.nodes[nid].depth
        changed = False
        for _ in range(c):
            i = rng.below(self.q)
            if n.slots[i] is None:
                cands = [j for j in self.live_nodes()
                         if self.net.nodes[j].depth < my_depth and j not in n.slots]
                if not cands:
                    continue
                n.slots[i] = cands[rng.below(len(cands))]
                changed = True
            else:
                old = n.slots[i]
                n.slots[i] = None
                if not path_rule_ok(self.net, self.removed):
                    n.slots[i] = old
                    continue
                changed = True
        return t if changed else None

    # phase 3
    def growth(self, caps, used):
        rng = self.rng
        w = self.p.overall["add_method_weights"]
        method = METHODS[rng.weighted([float(w[m]) for m in METHODS])]
        if method == "slot_fill":
            return self.slot_fill(caps, used)
        t = {"v_insert": "v", "s_insert": "s", "m_after_m": "m"}[method]
        if caps[t] is not None and used[t] >= caps[t]:
            return None
        return self.path_insert(method, t)

    def path_insert(self, method, t):
        net, rng = self.net, self.rng
        if method == "m_after_m":
            cands = [i for i in self.live_neurons() if isinstance(net.nodes[i].neuron, MNeuron)]
        else:
            cands = self.live_neurons()
        if not cands:
            return None
        a = cands[rng.below(len(cands))]
        node_a = net.nodes[a]
        conns = [c for c in connections(node_a.neuron) if c not in self.removed]
        lo = max(net.nodes[c].depth for c in conns) + 1 if conns else 1
        hi = node_a.depth - 1
        if lo > hi:
            if not net.is_hidden(a):
                return None
            rd = readers(net, self.removed)[a]
            ub = min(net.nodes[r].depth for r in rd) - 1 if rd else net.d_max - 1
            if lo > ub - 1:
                return None
            node_a.depth = ub
            hi = ub - 1
        d_b = lo + rng.below(hi - lo + 1)
        b = len(net.nodes)
        net.nodes.append(Node(node_a.birth, node_a.born_gen, d_b, node_a.neuron))
        if method == "v_insert":
            op = VOp(rng.choice(4))
            x = VNeuron(b, Inversion.NONE, op, IDENTITY_WEIGHT[op])
        elif method == "s_insert":
            x = SNeuron(b, AggOp(rng.choice(4)), 1)
        else:
            x = MNeuron(AggOp.MAX, [b] + [None] * (self.q - 1))
        node_a.neuron = x
        node_a.birth = net.next_birth
        node_a.born_gen = self.gen
        net.next_birth += 1
        return t

    def slot_fill(self, caps, used):
        net, rng = self.net, self.rng
        recv = [i for i in self.live_neurons()
                if isinstance(net.nodes[i].neuron, MNeuron) and net.nodes[i].neuron.op != AggOp.MIN
                and None in net.nodes[i].neuron.slots]
        if not recv:
            return None
        r = recv[rng.below(len(recv))]
        rn = net.nodes[r].neuron
        empties = [s for s, x in enumerate(rn.slots) if x is None]
        slot = empties[rng.below(len(empties))]
        tw = self.p.overall["slot_fill_type_weights"]
        t = TYPES[rng.weighted([float(tw[x]) for x in TYPES])]
        if caps[t] is not None and used[t] >= caps[t]:
            return None
        d_r = net.nodes[r].depth
        if d_r < 3:
            return None
        d_w = 2 + rng.below(d_r - 2)
        d_q = 1 + rng.below(d_w - 1)
        cands = [j for j in self.live_nodes() if net.nodes[j].depth < d_q]
        if t == "v":
            inp = cands[rng.below(len(cands))]
            op = VOp(rng.choice(4))
            qn = VNeuron(inp, Inversion.NONE, op, IDENTITY_WEIGHT[op])
        elif t == "s":
            inp = cands[rng.below(len(cands))]
            qn = SNeuron(inp, AggOp(rng.choice(4)), 1)
        else:
            count = 1 + rng.below(min(self.q, len(cands)))
            rem = list(cands)
            chosen = [rem.pop(rng.below(len(rem))) for _ in range(count)]
            qn = MNeuron(AggOp(rng.choice(4)), chosen + [None] * (self.q - count))
        q_id = len(net.nodes)
        net.nodes.append(Node(net.next_birth, self.gen, d_q, qn))
        w_id = len(net.nodes)
        net.nodes.append(Node(net.next_birth + 1, self.gen, d_w, VNeuron(q_id, Inversion.NONE, VOp.MUL, 0.0)))
        net.next_birth += 2
        rn.slots[slot] = w_id
        return t

    # phase 4
    def removal(self, caps, used):
        nid, t = self.pick_neuron(caps, used)
        if nid is None:
            return None
        net = self.net
        if not net.is_hidden(nid):
            return None
        if self.gen - net.nodes[nid].born_gen < self.p.effective("min_age_for_removal", t):
            return None
        if is_identity(net, nid):
            bypass(net, nid, self.removed)
            return t
        if not is_silenced(net, nid):
            return None
        if has_min_reader(net, nid, self.removed) or is_immune(net, nid, self.removed):
            return None
        remove_silenced(net, nid, self.removed)
        return t

    # phase 5
    def depth_shift(self, caps, used):
        net, rng = self.net, self.rng
        hidden = [i for i in range(net.n_inputs + net.n_outputs, len(net.nodes)) if i not in self.removed]
        if not hidden:
            return None
        nid = hidden[rng.below(len(hidden))]
        t = _type_of(net.nodes[nid].neuron)
        if caps[t] is not None and used[t] >= caps[t]:
            return None
        lo, hi = legal_depth_range(net, nid, self.removed)
        if lo == hi:
            return None
        if self.p.overall["depth_shift_mode"] == "center":
            d = lo + (hi - lo) // 2
        else:
            d = lo + rng.below(hi - lo + 1)
        if d == net.nodes[nid].depth:
            return None
        net.nodes[nid].depth = d
        return t

    def run(self) -> Net:
        self.run_phase("op_change_max_pct", self.op_change)
        self.run_phase("edit_max_pct", self.edit)
        self.run_phase("growth_max_pct", self.growth)
        self.run_phase("removal_max_pct", self.removal)
        self.run_phase("depth_shift_max_pct", self.depth_shift)
        garbage_collect(self.net, self.removed)
        compact(self.net, self.removed)
        return self.net


def mutate(parent: Net, profile: Profile, generation: int, rng: Rng) -> Net:
    """Return a mutated offspring; the parent is not modified (spec/06)."""
    return _Mutation(parent, profile, generation, rng).run()
