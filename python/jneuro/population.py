"""Population: nets by stable id, evaluation, operations, RNG modes, replay (spec/11)."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .io import (RUNLOG_FORMAT, SPEC_VERSION, _check_version, _dumps, genome_from_obj,
                 genome_to_obj)
from .mutate import Profile, inject_diversity, mutate
from .net import Net, _check_params
from .rng import Rng
from .types import DepthType, Priority, ValidationError

_M64 = (1 << 64) - 1


def fresh_seed() -> int:
    """A new 64-bit seed from the operating system's entropy source."""
    return int.from_bytes(os.urandom(8), "little")


class Population:
    """Container of nets with stable ids. The caller's trainer scores, selects and culls."""

    def __init__(self, depth_type: DepthType, in_qty_max: int, n_inputs: int, n_outputs: int,
                 priority: Priority = Priority.BALANCED, seed: Optional[int] = None):
        _check_params(n_inputs, n_outputs, in_qty_max)
        self.depth_type = DepthType(depth_type)
        self.in_qty_max = in_qty_max
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.priority = Priority(priority)
        self.seed = fresh_seed() if seed is None else int(seed) & _M64
        self._rng = Rng(self.seed)
        self._nets: Dict[int, Net] = {}
        self._next_id = 0
        self._generation = 0
        self._profiles: List[dict] = []
        self._profile_keys: List[str] = []
        self._ops: List[dict] = []

    # ------------------------------------------------------------ inspection

    @property
    def generation(self) -> int:
        return self._generation

    def ids(self) -> List[int]:
        return sorted(self._nets)

    def size(self) -> int:
        return len(self._nets)

    def get(self, net_id: int) -> Net:
        return self._nets[net_id]

    # ------------------------------------------------------------ helpers

    def _op_rng(self, mode: str, seed: Optional[int]):
        if mode == "shared":
            return self._rng, None
        if mode != "fresh":
            raise ValidationError(f"unknown rng_mode {mode!r}")
        s = fresh_seed() if seed is None else seed
        return Rng(s), s

    def _store(self, net: Net) -> int:
        nid = self._next_id
        self._next_id += 1
        self._nets[nid] = net
        return nid

    def _profile_index(self, profile: Profile) -> int:
        d = profile.to_dict()
        key = _dumps(d)
        if key not in self._profile_keys:
            self._profile_keys.append(key)
            self._profiles.append(d)
        return self._profile_keys.index(key)

    # ------------------------------------------------------------ operations

    def create(self, count: int, rng_mode: str = "fresh", _seed: Optional[int] = None) -> List[int]:
        """Add `count` minimal nets, all drawn from one operation RNG."""
        rng, seed = self._op_rng(rng_mode, _seed)
        ids = [self._store(Net.create(self.n_inputs, self.n_outputs, self.depth_type,
                                      self.in_qty_max, self.priority, rng)) for _ in range(count)]
        self._ops.append({"op": "create", "count": count, "ids": list(ids), "seed": seed})  # copy: caller may modify ids
        return ids

    def add(self, net: Net) -> int:
        """Add a copy of an existing net with the population's parameters."""
        if (net.depth_type, net.in_qty_max, net.n_inputs, net.n_outputs) != \
                (self.depth_type, self.in_qty_max, self.n_inputs, self.n_outputs):
            raise ValidationError("net parameters differ from the population's")
        net.validate()
        nid = self._store(net.copy())
        self._ops.append({"op": "add", "new_id": nid, "genome": genome_to_obj(net)})
        return nid

    def mutate(self, net_id: int, profile: Profile, times: int = 1, _seed: Optional[int] = None) -> int:
        """Offspring of `net_id` after `times` mutations; the parent stays."""
        parent = self._nets[net_id]
        if times < 1:
            raise ValueError("times must be >= 1")
        rng, seed = self._op_rng(profile.overall["rng_mode"], _seed)
        child = parent
        for _ in range(times):
            child = mutate(child, profile, self._generation, rng)
        nid = self._store(child)
        self._ops.append({"op": "mutate", "id": net_id, "times": times,
                          "profile": self._profile_index(profile), "new_id": nid, "seed": seed})
        return nid

    def clone(self, net_id: int) -> int:
        nid = self._store(self._nets[net_id].copy())
        self._ops.append({"op": "clone", "id": net_id, "new_id": nid})
        return nid

    def inject_diversity(self, net_id: int, pct: float, sigma_reset: float = 1.0,
                         rng_mode: str = "fresh", _seed: Optional[int] = None):
        """Replace the net at `net_id` with a diversity-injected version (id kept)."""
        net = self._nets[net_id]
        rng, seed = self._op_rng(rng_mode, _seed)
        self._nets[net_id] = inject_diversity(net, pct, rng, sigma_reset)
        self._ops.append({"op": "inject_diversity", "id": net_id, "pct": float(pct),
                          "sigma_reset": float(sigma_reset), "seed": seed})

    def remove(self, net_id: int):
        del self._nets[net_id]
        self._ops.append({"op": "remove", "id": net_id})

    def next_generation(self):
        self._generation += 1
        self._ops.append({"op": "next_generation"})

    # ------------------------------------------------------------ evaluation

    def evaluate(self, inputs, skip_warmup: bool = True) -> Dict[int, List[List[float]]]:
        """Outputs of every net on the same input rows, keyed by id (ascending).

        Evaluated sequentially: CPython's global interpreter lock gives no benefit from
        threads here, which spec/11 allows. Results are identical under every priority."""
        out = {}
        for nid in self.ids():
            net = self._nets[nid]
            net.reset()
            rows = [net.step(list(r)) for r in inputs]
            out[nid] = rows[net.get_warmup_steps():] if skip_warmup else rows
        return out

    # ------------------------------------------------------------ run log

    def run_log(self) -> str:
        return _dumps({
            "format": RUNLOG_FORMAT, "spec_version": SPEC_VERSION,
            "population": {"depth_type": self.depth_type.name, "in_qty_max": self.in_qty_max,
                           "n_inputs": self.n_inputs, "n_outputs": self.n_outputs,
                           "priority": self.priority.name, "seed": self.seed},
            "profiles": self._profiles, "ops": self._ops})

    @classmethod
    def replay(cls, text: str) -> "Population":
        """Re-execute a run log; reproduces ids, genomes, generation and shared-stream state."""
        d = json.loads(text)
        _check_version(d, RUNLOG_FORMAT)
        pp = d["population"]
        pop = cls(DepthType[pp["depth_type"]], pp["in_qty_max"], pp["n_inputs"], pp["n_outputs"],
                  Priority[pp["priority"]], seed=pp["seed"])
        profiles = [Profile.from_dict(x) for x in d["profiles"]]

        def mode(op):
            return "shared" if op["seed"] is None else "fresh"

        for op in d["ops"]:
            kind = op["op"]
            if kind == "create":
                got = pop.create(op["count"], mode(op), _seed=op["seed"])
                expect = op["ids"]
            elif kind == "add":
                got, expect = pop.add(genome_from_obj(op["genome"])), op["new_id"]
            elif kind == "mutate":
                prof = profiles[op["profile"]]
                if mode(op) != prof.overall["rng_mode"]:
                    raise ValidationError("run log: mutate seed does not match the profile's rng_mode")
                got, expect = pop.mutate(op["id"], prof, op["times"], _seed=op["seed"]), op["new_id"]
            elif kind == "clone":
                got, expect = pop.clone(op["id"]), op["new_id"]
            elif kind == "inject_diversity":
                pop.inject_diversity(op["id"], op["pct"], op["sigma_reset"], mode(op), _seed=op["seed"])
                got = expect = None
            elif kind == "remove":
                pop.remove(op["id"])
                got = expect = None
            elif kind == "next_generation":
                pop.next_generation()
                got = expect = None
            else:
                raise ValidationError(f"run log: unknown op {kind!r}")
            if got != expect:
                raise ValidationError(f"run log: {kind} produced id {got}, log says {expect}")
        return pop
