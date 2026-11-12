"""Population: nets by stable id, evaluation, operations, RNG modes, replay (spec/11)."""
from __future__ import annotations

import copy as _copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from .io import RUNLOG_FORMAT, SPEC_VERSION, _check_version, _dumps
from .jnet import Reader, Writer, net_from_bytes, net_to_bytes, sha256, write_header
from .mutate import Profile, inject_diversity, mutate
from .net import Net, _check_max_run, _check_params, _check_path_mix
from .rng import Rng
from .types import DepthType, Priority, ValidationError

_M64 = (1 << 64) - 1


def fresh_seed() -> int:
    """A new 64-bit seed from the operating system's entropy source."""
    return int.from_bytes(os.urandom(8), "little")


class Population:
    """Container of nets with stable ids. The caller's trainer scores, selects and culls."""

    def __init__(self, depth_type: DepthType, in_qty_max: int, n_inputs: int, n_outputs: int,
                 priority: Priority = Priority.BALANCED, seed: Optional[int] = None,
                 path_mix: int = 2, max_run: int = 3):
        _check_params(n_inputs, n_outputs, in_qty_max)
        _check_path_mix(path_mix)
        _check_max_run(max_run)
        self.path_mix = path_mix
        self.max_run = max_run
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
        self._resumed_from: Optional[str] = None   # hex SHA-256 of the .jpnet this run resumed from
        self._snapshot_hash: Optional[str] = None  # hex SHA-256 of the .jpnet this object was loaded from
        self._mutation_counts: Dict[str, int] = {}  # operation -> successful uses (informative, not saved)

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

    def mutation_counts(self) -> Dict[str, int]:
        """Successful uses of each mutation operation since creation or the last reset, every
        operation in jneuro.mutate.OPERATIONS listed (0 if unused). Informative; not saved."""
        from .mutate import OPERATIONS
        return {op: self._mutation_counts.get(op, 0) for op in OPERATIONS}

    def reset_mutation_counts(self):
        self._mutation_counts = {}

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
                                      self.in_qty_max, self.priority, rng, self.path_mix, self.max_run))
               for _ in range(count)]
        self._ops.append({"op": "create", "count": count, "ids": list(ids), "seed": seed})  # copy: caller may modify ids
        return ids

    def add(self, net: Net) -> int:
        """Add a copy of an existing net with the population's parameters."""
        if (net.depth_type, net.in_qty_max, net.n_inputs, net.n_outputs, net.path_mix, net.max_run) != \
                (self.depth_type, self.in_qty_max, self.n_inputs, self.n_outputs, self.path_mix, self.max_run):
            raise ValidationError("net parameters differ from the population's")
        net.validate()
        nid = self._store(net.copy())
        self._ops.append({"op": "add", "new_id": nid, "jnet": net_to_bytes(net).hex()})
        return nid

    def mutate(self, net_id: int, profile: Profile, times: int = 1, _seed: Optional[int] = None) -> int:
        """Offspring of `net_id` after `times` mutations; the parent stays."""
        parent = self._nets[net_id]
        if times < 1:
            raise ValueError("times must be >= 1")
        rng, seed = self._op_rng(profile.overall["rng_mode"], _seed)
        child = parent
        for _ in range(times):
            child = mutate(child, profile, self._generation, rng, self._mutation_counts)
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
                           "path_mix": self.path_mix, "max_run": self.max_run,
                           "priority": self.priority.name, "seed": self.seed},
            "resumed_from": self._resumed_from,
            "profiles": self._profiles, "ops": self._ops})

    @classmethod
    def replay(cls, text: str, base: Optional["Population"] = None) -> "Population":
        """Re-execute a run log; reproduces ids, nets, generation and shared-stream state.

        A log that resumed from a .jpnet needs `base`: the population loaded from that file."""
        d = json.loads(text)
        _check_version(d, RUNLOG_FORMAT)
        pp = d["population"]
        resumed = d.get("resumed_from")
        if resumed is None:
            pop = cls(DepthType[pp["depth_type"]], pp["in_qty_max"], pp["n_inputs"], pp["n_outputs"],
                      Priority[pp["priority"]], seed=pp["seed"], path_mix=pp["path_mix"],
                      max_run=pp["max_run"])
        else:
            if base is None or base._snapshot_hash != resumed:
                raise ValidationError("run log resumes from a .jpnet: pass the population loaded "
                                      "from that file as base")
            pop = base._copy_state()
            pop._resumed_from = resumed
        profiles = [Profile.from_dict(x) for x in d["profiles"]]

        def mode(op):
            return "shared" if op["seed"] is None else "fresh"

        for op in d["ops"]:
            kind = op["op"]
            if kind == "create":
                got = pop.create(op["count"], mode(op), _seed=op["seed"])
                expect = op["ids"]
            elif kind == "add":
                got, expect = pop.add(net_from_bytes(bytes.fromhex(op["jnet"]))), op["new_id"]
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

    # ------------------------------------------------------------ .jpnet files (spec/09)

    def _copy_state(self) -> "Population":
        """Same nets, ids, generation and shared-stream state; an empty journal."""
        p = Population(self.depth_type, self.in_qty_max, self.n_inputs, self.n_outputs,
                       self.priority, seed=self.seed, path_mix=self.path_mix, max_run=self.max_run)
        p._rng.set_state(self._rng.get_state())
        p._nets = {k: v.copy() for k, v in self._nets.items()}
        p._next_id = self._next_id
        p._generation = self._generation
        p._snapshot_hash = self._snapshot_hash
        return p

    def save(self, path: Union[str, Path], include_log: bool = False):
        """Write <name>.jpnet plus <name>_nets/<id>.jnet beside it (spec/09)."""
        path = Path(path)
        name = path.stem
        net_dir = path.parent / f"{name}_nets"
        net_dir.mkdir(parents=True, exist_ok=True)
        w = Writer()
        write_header(w, JPNET_MAGIC)
        w.u(self.priority, 1)
        w.u(self.depth_type, 1)
        w.u(self.in_qty_max, 4)
        w.u(self.n_inputs, 4)
        w.u(self.n_outputs, 4)
        w.u(self.path_mix, 1)
        w.u(self.max_run, 1)
        w.u(self.seed, 8)
        for s in self._rng.get_state():
            w.u(s, 8)
        w.u(self._generation, 8)
        w.u(self._next_id, 8)
        w.u(len(self._nets), 8)
        for nid in self.ids():
            data = net_to_bytes(self._nets[nid])
            (net_dir / f"{nid}.jnet").write_bytes(data)
            rel = f"{name}_nets/{nid}.jnet".encode("utf-8")
            w.u(nid, 8)
            w.u(len(rel), 2)
            w.raw(rel)
            w.raw(sha256(data))
        if include_log:
            log = self.run_log().encode("utf-8")
            w.u(1, 1)
            w.u(len(log), 8)
            w.raw(log)
        else:
            w.u(0, 1)
        path.write_bytes(w.finish())

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Population":
        """Load a .jpnet, verifying every .jnet hash; continues exactly where it was saved."""
        path = Path(path)
        data = path.read_bytes()
        r = Reader(data, "jpnet")
        r.header(JPNET_MAGIC)
        priority = r.enum(Priority, "priority")
        depth_type = r.enum(DepthType, "depth_type")
        q, n_in, n_out = r.u(4), r.u(4), r.u(4)
        path_mix = r.u(1)
        max_run = r.u(1)
        seed = r.u(8)
        state = [r.u(8) for _ in range(4)]
        generation, next_id, count = r.u(8), r.u(8), r.u(8)
        pop = cls(depth_type, q, n_in, n_out, priority, seed=seed, path_mix=path_mix, max_run=max_run)
        pop._rng.set_state(state)
        pop._generation = generation
        pop._next_id = next_id
        last = -1
        for _ in range(count):
            nid = r.u(8)
            rel = r.take(r.u(2)).decode("utf-8")
            digest = r.take(32)
            if nid <= last or nid >= next_id:
                raise ValidationError("jpnet: net ids must ascend and be below next_id")
            last = nid
            file = path.parent / rel
            if not file.is_file():
                raise ValidationError(f"jpnet: missing net file {rel}")
            raw = file.read_bytes()
            if sha256(raw) != digest:
                raise ValidationError(f"jpnet: hash mismatch for {rel}")
            net = net_from_bytes(raw)
            if (net.depth_type, net.in_qty_max, net.n_inputs, net.n_outputs, net.path_mix, net.max_run) != \
                    (depth_type, q, n_in, n_out, path_mix, max_run):
                raise ValidationError(f"jpnet: {rel} does not match the population's parameters")
            pop._nets[nid] = net
        flag = r.u(1)
        log = None
        if flag == 1:
            log = json.loads(r.take(r.u(8)).decode("utf-8"))
        elif flag != 0:
            raise ValidationError("jpnet: log flag out of range")
        r.done()
        pop._snapshot_hash = sha256(data).hex()
        if log is not None:
            _check_version(log, RUNLOG_FORMAT)
            pop._resumed_from = log.get("resumed_from")
            pop._profiles = log["profiles"]
            pop._profile_keys = [_dumps(x) for x in pop._profiles]
            pop._ops = log["ops"]
        else:
            pop._resumed_from = pop._snapshot_hash
        return pop


JPNET_MAGIC = b"JPOP"
