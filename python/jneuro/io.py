"""Genome, profile and run-log JSON (spec/09)."""
from __future__ import annotations

import json
import math

from .net import MNeuron, Net, Node, SNeuron, VNeuron
from .types import AggOp, DepthType, Inversion, Priority, ValidationError, VOp

SPEC_VERSION = "0.1.0"
GENOME_FORMAT = "j_neuro.genome"
PROFILE_FORMAT = "j_neuro.profile"
RUNLOG_FORMAT = "j_neuro.runlog"

_INV = {e.name.lower(): e for e in Inversion}
_VOP = {e.name.lower(): e for e in VOp}
_AGG = {e.name.lower(): e for e in AggOp}


def format_double(x: float) -> str:
    """Canonical text of a finite binary64 value (Python repr rules, spec/09)."""
    if not math.isfinite(x):
        raise ValueError("only finite values are serialized")
    return repr(float(x))


def _dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), allow_nan=False)


# ------------------------------------------------------------------ genome

def _neuron_to_obj(n):
    if isinstance(n, VNeuron):
        return {"type": "v", "input": n.input, "inversion": n.inversion.name.lower(),
                "op": n.op.name.lower(), "weight": float(n.weight)}
    if isinstance(n, SNeuron):
        return {"type": "s", "input": n.input, "op": n.op.name.lower(), "k": n.k}
    return {"type": "m", "op": n.op.name.lower(), "slots": list(n.slots)}


def genome_to_obj(net: Net) -> dict:
    nodes = []
    for node in net.nodes:
        d = {"birth": node.birth, "born_gen": node.born_gen, "depth": node.depth}
        if node.neuron is not None:
            d["neuron"] = _neuron_to_obj(node.neuron)
        nodes.append(d)
    return {
        "format": GENOME_FORMAT,
        "spec_version": SPEC_VERSION,
        "params": {"priority": net.priority.name, "depth_type": net.depth_type.name,
                   "in_qty_max": net.in_qty_max, "n_inputs": net.n_inputs,
                   "n_outputs": net.n_outputs},
        "next_birth": net.next_birth,
        "nodes": nodes,
    }


def genome_to_json(net: Net) -> str:
    """Canonical (byte-comparable) genome text."""
    return _dumps(genome_to_obj(net))


def _int(v, what):
    if not isinstance(v, int) or isinstance(v, bool):
        raise ValidationError(f"{what} must be an integer")
    return v


def _ref(v, what):
    return None if v is None else _int(v, what)


def _enum(table, v, what):
    if v not in table:
        raise ValidationError(f"unknown {what}: {v!r}")
    return table[v]


def _check_version(d, fmt):
    if not isinstance(d, dict) or d.get("format") != fmt:
        raise ValidationError(f"format must be {fmt!r}")
    ver = d.get("spec_version")
    if not isinstance(ver, str) or ver.split(".")[0] != SPEC_VERSION.split(".")[0]:
        raise ValidationError(f"unsupported spec_version {ver!r}")


def _neuron_from_obj(o):
    if not isinstance(o, dict):
        raise ValidationError("neuron must be an object")
    t = o.get("type")
    if t == "v":
        w = o.get("weight")
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            raise ValidationError("weight must be a number")
        return VNeuron(_ref(o.get("input"), "input"), _enum(_INV, o.get("inversion"), "inversion"),
                       _enum(_VOP, o.get("op"), "op"), float(w))
    if t == "s":
        return SNeuron(_ref(o.get("input"), "input"), _enum(_AGG, o.get("op"), "op"), _int(o.get("k"), "k"))
    if t == "m":
        slots = o.get("slots")
        if not isinstance(slots, list):
            raise ValidationError("slots must be a list")
        return MNeuron(_enum(_AGG, o.get("op"), "op"), [_ref(s, "slot") for s in slots])
    raise ValidationError(f"unknown neuron type {t!r}")


def genome_from_obj(d) -> Net:
    _check_version(d, GENOME_FORMAT)
    try:
        p = d["params"]
        nodes = []
        for nd in d["nodes"]:
            neuron = _neuron_from_obj(nd["neuron"]) if "neuron" in nd else None
            nodes.append(Node(_int(nd["birth"], "birth"), _int(nd["born_gen"], "born_gen"),
                              _int(nd["depth"], "depth"), neuron))
        net = Net(_enum(Priority.__members__, p["priority"], "priority"),
                  _enum(DepthType.__members__, p["depth_type"], "depth_type"),
                  _int(p["in_qty_max"], "in_qty_max"), _int(p["n_inputs"], "n_inputs"),
                  _int(p["n_outputs"], "n_outputs"), nodes, _int(d["next_birth"], "next_birth"))
    except (KeyError, TypeError) as e:
        raise ValidationError(f"malformed genome: {e}") from e
    net.validate()
    return net


def genome_from_json(text: str) -> Net:
    """Parse and fully validate a genome."""
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValidationError(f"invalid JSON: {e}") from e
    return genome_from_obj(d)


# ------------------------------------------------------------------ profile

def profile_to_json(profile) -> str:
    return _dumps(profile.to_dict())


def profile_from_json(text: str):
    from .mutate import Profile
    d = json.loads(text)
    _check_version(d, PROFILE_FORMAT)
    return Profile.from_dict(d)


# ------------------------------------------------------------------ run log

class RunLog:
    """Seed, starting profile and every profile change (spec/09)."""

    def __init__(self, seed: int, initial_profile, changes=None):
        self.seed = seed
        self.initial_profile = initial_profile
        self.changes = list(changes or [])   # [(generation, patch)]

    def record(self, generation: int, patch: dict):
        self.changes.append((generation, patch))

    def profile_at(self, generation: int):
        """Profile in effect for mutations of `generation`."""
        p = self.initial_profile.copy()
        for g, patch in self.changes:
            if g <= generation:
                p.apply_patch(patch)
        return p

    def to_json(self) -> str:
        return _dumps({"format": RUNLOG_FORMAT, "spec_version": SPEC_VERSION, "seed": self.seed,
                       "initial_profile": self.initial_profile.to_dict(),
                       "changes": [{"generation": g, "patch": p} for g, p in self.changes]})

    @classmethod
    def from_json(cls, text: str) -> "RunLog":
        from .mutate import Profile
        d = json.loads(text)
        _check_version(d, RUNLOG_FORMAT)
        return cls(d["seed"], Profile.from_dict(d["initial_profile"]),
                   [(c["generation"], c["patch"]) for c in d["changes"]])
