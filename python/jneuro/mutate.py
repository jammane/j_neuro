"""Mutation profile and mutation (spec/06, removal phase in spec/07)."""
from __future__ import annotations

import copy
import math

from .analysis import connections, is_immune, legal_depth_range, mix_ok, path_rule_ok, readers, run_ok
from .net import MNeuron, Net, Node, SNeuron, VNeuron
from .removal import (bypass, compact, constant_values, fold_constants, garbage_collect,
                      has_min_reader, is_identity, is_silenced, remove_silenced)
from .rng import Rng, round_half_away
from .types import IDENTITY_WEIGHT, M_OPS, S_OPS, AggOp, Inversion, ValidationError, VOp

TYPES = ("v", "s", "m")
METHODS = ("slot_fill", "v_insert", "s_insert", "m_after_m", "v_tap", "s_tap", "m_tap", "m_clone", "m_insert",
           "v_link", "s_link")
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
    "rewire_prob": 0.05,
    "insert_inversion_prob": 0.05,
    "min_age_for_removal": 5,
    "add_method_weights": {"slot_fill": 0, "v_insert": 10, "s_insert": 5, "m_after_m": 5,
                           "v_tap": 10, "s_tap": 10, "m_tap": 5, "m_clone": 60, "m_insert": 20,
                           "v_link": 24, "s_link": 24},
    "slot_fill_type_weights": {"v": 1, "s": 1, "m": 1},
    "rng_mode": "fresh",
    "growth_budget": "up_to",
    "constant_folding": "off",
    "insert_depth": "midpoint",
    "slot_retarget_prob": 0.05,
    "width_bias": "deficit",
    "width_peak": 0.5,
    "tap_peak": 0.25,
    "prune": "static",
    "link_max_span": 0.25,
    "growth_gen_prob": 1.0,
    "growth_net_prob": 1.0,
    "no_growth_swap": "on",
    "slot_edit": "toggle",
    "slot_add_prob": 0.75,
    "open_slot_fill_pct": 0.0,
}
CHOICES = {"depth_shift_mode": ("uniform", "center"), "rng_mode": ("fresh", "shared"),
           "growth_budget": ("up_to", "exact", "fill"), "constant_folding": ("off", "on"),
           "insert_depth": ("uniform", "midpoint"), "width_bias": ("none", "center", "deficit"),
           "prune": ("any", "static"), "no_growth_swap": ("off", "on"),
           "slot_edit": ("toggle", "biased")}
OVERRIDABLE = set(PHASE_FIELDS) | {"sigma_weight", "sign_flip_prob", "sigma_k", "sigma_slots",
                                   "rewire_prob", "insert_inversion_prob", "min_age_for_removal",
                                   "slot_retarget_prob"}


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
            if k in ("width_peak", "tap_peak") and v is not None and not 0.0 < float(v) < 1.0:
                raise ValidationError(f"{k} must be in (0, 1)")
            if k == "link_max_span" and v is not None and not 0.0 < float(v) <= 1.0:
                raise ValidationError("link_max_span must be in (0, 1]")
            if k == "open_slot_fill_pct" and v is not None and not 0.0 <= float(v) <= 100.0:
                raise ValidationError("open_slot_fill_pct must be in [0, 100]")
            if k in ("growth_gen_prob", "growth_net_prob", "slot_add_prob") and v is not None and \
                    not 0.0 <= float(v) <= 1.0:
                raise ValidationError(f"{k} must be in [0, 1]")
            if v is None:
                self.overall[k] = copy.deepcopy(DEFAULTS[k])
            elif k in CHOICES:
                if v not in CHOICES[k]:
                    raise ValidationError(f"{k} must be one of {CHOICES[k]}")
                self.overall[k] = v
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
    def __init__(self, parent: Net, profile: Profile, generation: int, rng: Rng,
                 growth_generation: bool = True):
        self.net = parent.copy()
        self.p = profile
        self.gen = generation
        self.rng = rng
        self.removed = set()
        self.q = self.net.in_qty_max
        self.op = None                    # name of the operation being attempted (for counts)
        self.growth_generation = growth_generation
        self.counts = {}                  # operation name -> successful uses (informative)

    # helpers
    def live_neurons(self):
        return [i for i in range(self.net.n_inputs, len(self.net.nodes)) if i not in self.removed]

    def live_nodes(self):
        return [i for i in range(len(self.net.nodes)) if i not in self.removed]

    def budget(self, field):
        rng = self.rng
        n = float(len(self.live_neurons()))
        exact = field == "growth_max_pct" and self.p.overall["growth_budget"] in ("exact", "fill")

        def limit(pct):                       # a nonzero cap is at least 1 (spec/06)
            return 0.0 if pct == 0 else max(1.0, pct * n / 100.0)

        def draw(pct):                        # "up to" draws a fraction; exact uses the whole cap
            return rng.prob_round(limit(pct) if exact else rng.u() * limit(pct))

        b = draw(self.p.overall[field])
        caps = {}
        for t in TYPES:
            if self.p.has_override(field, t):
                caps[t] = draw(self.p.per_type[t][field])
            else:
                caps[t] = None
        return b, caps, {t: 0 for t in TYPES}

    def run_phase(self, field, attempt):
        b, caps, used = self.budget(field)
        # fill: b is a target in neurons added; skipped attempts are retried, up to 10 * b tries
        fill = field == "growth_max_pct" and self.p.overall["growth_budget"] == "fill"
        added = tries = 0
        while (added < b and tries < 10 * b) if fill else tries < b:
            tries += 1
            self.op = None
            t = attempt(caps, used)
            if t is not None:
                used[t] += 1
                self.counts[self.op] = self.counts.get(self.op, 0) + 1
                added += 2 if self.op == "slot_fill" else 1

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
        self.op = "op_change"
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
            n_ops = S_OPS if t == "s" else M_OPS          # s: also lag (spec/04)
            j = rng.below(n_ops - 1)
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
        if t in ("v", "s") and rng.u() < p.effective("rewire_prob", t):
            self.op = "rewire"
            return self.rewire(nid, n, t)
        self.op = {"v": "v_weight", "s": "s_k", "m": "m_slots"}[t]
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
        pr = p.effective("slot_retarget_prob", "m")
        if pr > 0 and rng.u() < pr:                       # no draw when off (spec/06)
            self.op = "m_retarget"
            return self.retarget(nid, n)
        z = rng.normal()
        c = min(self.q, 1 + math.floor(abs(p.effective("sigma_slots", "m") * z)))
        my_depth = self.net.nodes[nid].depth
        changed = False
        biased = self.p.overall["slot_edit"] == "biased"
        for _ in range(c):
            if biased:                                    # add or cut first, then which slot (spec/06)
                add = rng.u() < float(self.p.overall["slot_add_prob"])
                pool = [s for s in range(self.q) if (n.slots[s] is None) == add]
                if not pool:
                    continue
                i = pool[rng.below(len(pool))]
            else:
                i = rng.below(self.q)
            if n.slots[i] is None:
                cands = [j for j in self.live_nodes()
                         if self.net.nodes[j].depth < my_depth and j not in n.slots]
                if not cands:
                    continue
                n.slots[i] = cands[rng.below(len(cands))]
                if not self.rules_ok():
                    n.slots[i] = None
                    continue
                changed = True
            else:
                old = n.slots[i]
                if self.p.overall["prune"] == "static":   # only static (constant) sources may be cut
                    if constant_values(self.net, self.removed)[old] is None:
                        continue
                n.slots[i] = None
                if not path_rule_ok(self.net, self.removed):
                    n.slots[i] = old
                    continue
                changed = True
        return t if changed else None

    def retarget(self, nid, n):
        """Move one non-critical m slot to another node no deeper than the one it reads (spec/06)."""
        net, rng = self.net, self.rng
        conn = [i for i, s in enumerate(n.slots) if s is not None]
        if not conn:
            return None
        i = conn[rng.below(len(conn))]
        old = n.slots[i]
        if self.p.overall["prune"] == "static" and n.slots.count(old) == 1 and \
                readers(net, self.removed)[old] == [nid]:
            return None                                   # would strand old's branch
        n.slots[i] = None
        if not path_rule_ok(net, self.removed):           # critical: the only path for some input
            n.slots[i] = old
            return None
        lo = self.span_floor(net.nodes[nid].depth)
        cands = [j for j in self.live_nodes()
                 if lo <= net.nodes[j].depth <= net.nodes[old].depth and j != old and j not in n.slots]
        if not cands:
            n.slots[i] = old
            return None
        n.slots[i] = cands[rng.below(len(cands))]
        if not self.rules_ok():
            n.slots[i] = old
            return None
        return "m"

    def rewire(self, nid, n, t):
        net, rng = self.net, self.rng
        old = n.input
        if old is not None and readers(net, self.removed)[old] == [nid]:
            return None                                   # old input would be orphaned
        my_depth = net.nodes[nid].depth
        lo = self.span_floor(my_depth)
        cands = [j for j in self.live_nodes() if lo <= net.nodes[j].depth < my_depth and j != old]
        if not cands:
            return None
        n.input = cands[rng.below(len(cands))]
        if not path_rule_ok(net, self.removed) or not self.rules_ok():
            n.input = old
            return None
        return t

    def grows(self):
        """Phase 3 runs only in a growth generation, and then with growth_net_prob (spec/06).
        Returns True (grow), False (not a growth generation) or None (a growth generation, but
        this offspring's draw failed)."""
        if not self.growth_generation:
            return False
        p = float(self.p.overall["growth_net_prob"])
        if p >= 1.0 or self.rng.u() < p:                   # no draw when 1.0
            return True
        return None

    def type_swap(self):
        """Turn one 1:1 neuron into the other kind, same input: v -> s or s -> v (spec/06)."""
        net, rng = self.net, self.rng
        cands = [i for i in self.live_neurons() if isinstance(net.nodes[i].neuron, (VNeuron, SNeuron))]
        if not cands:
            return False
        i = cands[rng.below(len(cands))]
        old = net.nodes[i].neuron
        if isinstance(old, VNeuron):
            net.nodes[i].neuron = SNeuron(old.input, AggOp(rng.choice(S_OPS)), 1)
        else:
            op = VOp(rng.choice(4))
            net.nodes[i].neuron = VNeuron(old.input, Inversion.NONE, op, IDENTITY_WEIGHT[op])
        if not self.rules_ok():                            # a path would be short of v or s
            net.nodes[i].neuron = old
            return False
        return True

    def span_floor(self, depth):
        """Shallowest source depth a link, rewire or retarget may reach from a node at `depth`
        (spec/06 link_max_span)."""
        span = int(float(self.p.overall["link_max_span"]) * float(self.net.d_max))
        return depth - span if depth > span else 0

    def rules_ok(self):
        """The structural rules a mutation may not break besides the path rule (spec/03)."""
        return mix_ok(self.net, self.removed) and run_ok(self.net, self.removed)

    def pick_receiver(self, recv, peak_field="width_peak", by_empty=False):
        """An m_neuron to receive new width: uniform, or weighted toward the field's peak depth
        (spec/06 width_bias; taps use tap_peak, the others width_peak). Links (by_empty) also
        weight each receiver by its number of empty slots."""
        bias = self.p.overall["width_bias"]
        if by_empty and bias == "none":
            w = [float(net_empty) for net_empty in (self.net.nodes[i].neuron.slots.count(None) for i in recv)]
            return recv[self.rng.weighted(w)]
        if bias in ("center", "deficit"):
            net = self.net
            peak = float(self.p.overall[peak_field])
            d_max = float(net.d_max)
            edges = []                                    # (depth of source, depth of reader), live only
            if bias == "deficit":
                for i in range(len(net.nodes)):
                    if i not in self.removed:
                        for c in connections(net.nodes[i].neuron):
                            edges.append((net.nodes[c].depth, net.nodes[i].depth))
            w = []
            for i in recv:
                d = net.nodes[i].depth
                x = float(d) / d_max
                tri = x / peak if x < peak else (1.0 - x) / (1.0 - peak)
                if bias == "deficit":                     # bell target over current width at R's depth
                    width = sum(1 for a, b in edges if a < d <= b)
                    tri = tri / float(max(width, 1))
                weight = tri + 1e-6                       # > 0, so an output-only net still has a pick
                if by_empty:
                    weight *= float(net.nodes[i].neuron.slots.count(None))
                w.append(weight)
            return recv[self.rng.weighted(w)]
        return recv[self.rng.below(len(recv))]

    def place(self, lo, hi):
        """Depth for a new node in [lo, hi]: uniform draw, or the midpoint (spec/06 insert_depth)."""
        if self.p.overall["insert_depth"] == "midpoint":
            return lo + (hi - lo) // 2
        return lo + self.rng.below(hi - lo + 1)

    def new_inversion(self):
        if self.rng.u() < self.p.effective("insert_inversion_prob", "v"):
            return Inversion(1 + self.rng.below(2))
        return Inversion.NONE

    # phase 3
    def growth(self, caps, used):
        rng = self.rng
        w = self.p.overall["add_method_weights"]
        dflt = DEFAULTS["add_method_weights"]
        method = METHODS[rng.weighted([float(w.get(m, dflt[m])) for m in METHODS])]  # missing = default
        self.op = method
        needed = 2 if method == "slot_fill" else 1
        if len(self.live_nodes()) + needed > self.net.d_max:
            return None                                   # node cap (spec/03)
        if method == "slot_fill":
            return self.slot_fill(caps, used)
        t = {"v_insert": "v", "s_insert": "s", "m_after_m": "m", "v_tap": "v", "s_tap": "s", "m_tap": "m",
             "m_clone": "m", "m_insert": "m", "v_link": "v", "s_link": "s"}[method]
        if caps[t] is not None and used[t] >= caps[t]:
            return None
        if method == "m_clone":
            return self.m_clone()
        if method.endswith("_link"):
            return self.link(t)
        if method.endswith("_tap"):
            return self.input_tap(t)
        return self.path_insert(method, t)

    def m_clone(self):
        """Copy an m_neuron U read by an m_neuron R into an empty slot of R (spec/06)."""
        net, rng = self.net, self.rng

        def m_reads(i):
            return sorted({s for s in net.nodes[i].neuron.slots
                           if s is not None and isinstance(net.nodes[s].neuron, MNeuron)})

        recv = [i for i in self.live_neurons()
                if isinstance(net.nodes[i].neuron, MNeuron) and None in net.nodes[i].neuron.slots and m_reads(i)]
        if not recv:
            return None
        r = self.pick_receiver(recv)
        ups = m_reads(r)
        u = ups[rng.below(len(ups))]
        rn = net.nodes[r].neuron
        empties = [s for s, x in enumerate(rn.slots) if x is None]
        slot = empties[rng.below(len(empties))]
        un = net.nodes[u].neuron
        x = len(net.nodes)
        net.nodes.append(Node(net.next_birth, self.gen, net.nodes[u].depth, MNeuron(un.op, list(un.slots))))
        net.next_birth += 1
        rn.slots[slot] = x
        if not self.rules_ok():                 # cannot happen (paths mirror U's); kept as a guard
            rn.slots[slot] = None
            del net.nodes[x:]
            net.next_birth -= 1
            return None
        return "m"

    def link(self, t):
        """A new 1:1 neuron from an existing node into an empty m slot (spec/06 v_link, s_link)."""
        net, rng = self.net, self.rng
        recv = [i for i in self.live_neurons()
                if isinstance(net.nodes[i].neuron, MNeuron) and None in net.nodes[i].neuron.slots
                and any(s is not None for s in net.nodes[i].neuron.slots)]
        if not recv:
            return None
        r = self.pick_receiver(recv, by_empty=True)
        empties = [s for s, x in enumerate(net.nodes[r].neuron.slots) if x is None]
        return self.link_into(t, r, empties[rng.below(len(empties))])

    def link_into(self, t, r, slot):
        """Links steps 3-6 (spec/06): a new 1:1 neuron from an existing node into empty slot `slot` of R."""
        net, rng = self.net, self.rng
        rn = net.nodes[r].neuron
        hi_src = max(net.nodes[s].depth for s in rn.slots if s is not None)   # no deeper than R's reads
        lo_src = self.span_floor(net.nodes[r].depth)       # longest link allowed
        cands = [j for j in self.live_nodes()
                 if lo_src <= net.nodes[j].depth <= hi_src and j not in rn.slots]
        if not cands:
            return None
        src = cands[rng.below(len(cands))]
        lo, hi = net.nodes[src].depth + 1, net.nodes[r].depth - 1
        if lo > hi:
            return None
        d = self.place(lo, hi)
        if t == "v":
            op = VOp(rng.choice(4))
            neuron = VNeuron(src, self.new_inversion(), op, IDENTITY_WEIGHT[op])
        else:
            neuron = SNeuron(src, AggOp(rng.choice(S_OPS)), 1)
        x = len(net.nodes)
        net.nodes.append(Node(net.next_birth, self.gen, d, neuron))
        net.next_birth += 1
        rn.slots[slot] = x
        if not self.rules_ok():
            rn.slots[slot] = None
            del net.nodes[x:]
            net.next_birth -= 1
            return None
        return t

    def open_slot_fill(self):
        """Open-slot fill (spec/06): after the growth attempts, fill open_slot_fill_pct % of every
        m_neuron's empty slots with links."""
        pct = float(self.p.overall["open_slot_fill_pct"])
        if not pct > 0:
            return
        net, rng = self.net, self.rng
        recv = [i for i in self.live_neurons() if isinstance(net.nodes[i].neuron, MNeuron)
                and any(s is not None for s in net.nodes[i].neuron.slots)]
        for r in recv:
            e = sum(1 for s in net.nodes[r].neuron.slots if s is None)
            if e == 0:
                continue
            k = rng.prob_round(pct * float(e) / 100.0)
            filled = tries = 0
            while filled < k and tries < 10 * k:
                tries += 1
                t = "v" if rng.below(2) == 0 else "s"
                if len(self.live_nodes()) + 1 > net.d_max:
                    continue                              # node cap (spec/03)
                empties = [s for s, x in enumerate(net.nodes[r].neuron.slots) if x is None]
                if self.link_into(t, r, empties[rng.below(len(empties))]) is not None:
                    filled += 1
                    op = "v_link" if t == "v" else "s_link"
                    self.counts[op] = self.counts.get(op, 0) + 1

    def input_tap(self, t):
        """A new neuron reading network inputs, feeding an empty slot of an m_neuron (spec/06)."""
        net, rng = self.net, self.rng
        recv = [i for i in self.live_neurons()
                if isinstance(net.nodes[i].neuron, MNeuron) and None in net.nodes[i].neuron.slots
                and any(s is not None for s in net.nodes[i].neuron.slots)]
        if not recv:
            return None
        r = self.pick_receiver(recv, "tap_peak")
        rn = net.nodes[r].neuron
        empties = [s for s, x in enumerate(rn.slots) if x is None]
        slot = empties[rng.below(len(empties))]
        hi = max(net.nodes[s].depth for s in rn.slots if s is not None)  # deepest node R reads
        if hi < 1:
            return None                                   # R reads only inputs: no room
        d = self.place(1, hi)
        n_in = net.n_inputs
        if t == "v":
            op = VOp(rng.choice(4))
            neuron = VNeuron(rng.below(n_in), self.new_inversion(), op, IDENTITY_WEIGHT[op])
        elif t == "s":
            inp = rng.below(n_in)
            neuron = SNeuron(inp, AggOp(rng.choice(S_OPS)), 1)
        else:
            filled = (self.q + 1) // 2                    # half the slots, rounded up
            from_inputs = min((filled + 1) // 2, n_in)    # half of those from inputs, rounded up
            ins = list(range(n_in))
            chosen = [ins.pop(rng.below(len(ins))) for _ in range(from_inputs)]
            cands = [j for j in self.live_nodes() if net.nodes[j].depth < d and j not in chosen]
            for _ in range(min(filled - from_inputs, len(cands))):
                chosen.append(cands.pop(rng.below(len(cands))))
            neuron = MNeuron(AggOp(rng.choice(M_OPS)), chosen + [None] * (self.q - len(chosen)))
        x = len(net.nodes)
        net.nodes.append(Node(net.next_birth, self.gen, d, neuron))
        net.next_birth += 1
        rn.slots[slot] = x
        if not self.rules_ok():                 # undo: the new paths break the mix rule
            rn.slots[slot] = None
            del net.nodes[x:]
            net.next_birth -= 1
            return None
        return t

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
        old_depth = node_a.depth
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
        d_b = self.place(lo, hi)
        b = len(net.nodes)
        net.nodes.append(Node(node_a.birth, node_a.born_gen, d_b, node_a.neuron))
        if method == "v_insert":
            op = VOp(rng.choice(4))
            x = VNeuron(b, self.new_inversion(), op, IDENTITY_WEIGHT[op])
        elif method == "s_insert":
            x = SNeuron(b, AggOp(rng.choice(S_OPS)), 1)
        elif method == "m_insert":
            x = MNeuron(AggOp.MIN, [b] + [None] * (self.q - 1))   # min of one slot: exactly neutral
        else:
            x = MNeuron(AggOp.MAX, [b] + [None] * (self.q - 1))
        saved = (node_a.neuron, node_a.birth, node_a.born_gen, old_depth)
        node_a.neuron = x
        node_a.birth = net.next_birth
        node_a.born_gen = self.gen
        net.next_birth += 1
        if not self.rules_ok():                           # undo: a run would become too long
            node_a.neuron, node_a.birth, node_a.born_gen, node_a.depth = saved
            del net.nodes[b:]
            net.next_birth -= 1
            return None
        return t

    def slot_fill(self, caps, used):
        net, rng = self.net, self.rng
        recv = [i for i in self.live_neurons()
                if isinstance(net.nodes[i].neuron, MNeuron) and net.nodes[i].neuron.op != AggOp.MIN
                and None in net.nodes[i].neuron.slots]
        if not recv:
            return None
        r = self.pick_receiver(recv)
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
        d_w = self.place(2, d_r - 1)
        d_q = self.place(1, d_w - 1)
        cands = [j for j in self.live_nodes() if net.nodes[j].depth < d_q]
        if t == "v":
            inp = cands[rng.below(len(cands))]
            op = VOp(rng.choice(4))
            qn = VNeuron(inp, self.new_inversion(), op, IDENTITY_WEIGHT[op])
        elif t == "s":
            inp = cands[rng.below(len(cands))]
            qn = SNeuron(inp, AggOp(rng.choice(S_OPS)), 1)
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
        if not self.rules_ok():                 # undo: the new paths break the mix rule
            rn.slots[slot] = None
            del net.nodes[q_id:]
            net.next_birth -= 2
            return None
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
            self.op = "bypass"
            saved = [(r, copy.deepcopy(net.nodes[r].neuron)) for r in readers(net, self.removed)[nid]]
            bypass(net, nid, self.removed)
            if not self.rules_ok():             # undo: paths through it would lose a v
                for r, neuron in saved:
                    net.nodes[r].neuron = neuron
                self.removed.discard(nid)
                return None
            return t
        if not is_silenced(net, nid):
            return None
        if has_min_reader(net, nid, self.removed) or is_immune(net, nid, self.removed):
            return None
        self.op = "remove_silenced"
        remove_silenced(net, nid, self.removed)
        return t

    # phase 5
    def depth_shift(self, caps, used):
        self.op = "depth_shift"
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
        g = self.grows()
        if g:
            self.run_phase("growth_max_pct", self.growth)
            self.open_slot_fill()
        elif g is None and self.p.overall["no_growth_swap"] == "on":
            if self.type_swap():
                self.counts["type_swap"] = self.counts.get("type_swap", 0) + 1
        self.run_phase("removal_max_pct", self.removal)
        self.run_phase("depth_shift_max_pct", self.depth_shift)
        if self.p.overall["constant_folding"] == "on":
            folded = fold_constants(self.net, self.removed, self.gen,
                                    lambda t: self.p.effective("min_age_for_removal", t))
            if folded:
                self.counts["fold_constant"] = self.counts.get("fold_constant", 0) + folded
        garbage_collect(self.net, self.removed)
        compact(self.net, self.removed)
        return self.net


# Every operation name that mutation counts (informative; not part of the spec's results).
OPERATIONS = ("op_change", "v_weight", "s_k", "m_slots", "rewire", "m_retarget") + METHODS + \
             ("type_swap", "bypass", "remove_silenced", "depth_shift", "fold_constant")


def is_growth_generation(seed: int, generation: int, profile: Profile) -> bool:
    """Whether `generation` is a growth generation for a population seeded `seed` (spec/06):
    a stateless draw, so it is the same on replay and after a save and load."""
    p = float(profile.overall["growth_gen_prob"])
    if p >= 1.0:
        return True
    return Rng((seed ^ (generation * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF).u() < p


def mutate(parent: Net, profile: Profile, generation: int, rng: Rng, counts: dict = None,
           growth_generation: bool = True) -> Net:
    """Return a mutated offspring; the parent is not modified (spec/06).

    If `counts` is given, the number of successful uses of each operation is added to it.
    `growth_generation` False skips phase 3 (see is_growth_generation)."""
    m = _Mutation(parent, profile, generation, rng, growth_generation)
    child = m.run()
    if counts is not None:
        for k, v in m.counts.items():
            counts[k] = counts.get(k, 0) + v
    return child


def inject_diversity(parent: Net, pct: float, rng: Rng, sigma_reset: float = 1.0) -> Net:
    """Give pct % of the v_neurons a new random weight; only weights change (spec/06)."""
    net = parent.copy()
    vs = [i for i, node in enumerate(net.nodes) if isinstance(node.neuron, VNeuron)]
    count = min(len(vs), rng.prob_round(pct * float(len(vs)) / 100.0))
    for _ in range(count):
        i = vs.pop(rng.below(len(vs)))
        net.nodes[i].neuron.weight = rng.normal() * sigma_reset
    return net
