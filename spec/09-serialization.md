# 09 — Serialization

All files are UTF-8 JSON. Writers MUST emit keys in the order shown so that equal genomes
produce **byte-identical** files (used by conformance, [10](10-conformance.md)).

## Number encoding

- Integers (ids, depths, counts) are JSON integers. Readers MUST parse them exactly, including
  `U64` depths above 2⁵³.
- binary64 values (weights) are JSON numbers, parsed exactly, written exactly as Python's
  `repr(float)` writes them:
  - digits: the shortest decimal digit string that round-trips exactly;
  - if `1e-4 ≤ |x| < 1e16` (or `x` is zero), positional notation with at least one digit after
    the point (`1.0`, `0.0001`, `123.25`, `-0.0`);
  - otherwise scientific notation `d[.ddd]e±XX` with at least two exponent digits
    (`1e+16`, `1.5e-05`, `5e-324`).

  Weights are always finite.
- **Canonical form** (what writers emit): compact JSON with no whitespace, separators `,` and
  `:`, keys in the order shown. The examples below are pretty-printed for reading only.

## Genome

```json
{
  "format": "j_neuro.genome",
  "spec_version": "0.1.0",
  "params": {
    "priority": "SPEED",
    "depth_type": "U16",
    "in_qty_max": 5,
    "n_inputs": 2,
    "n_outputs": 1
  },
  "next_birth": 4,
  "nodes": [
    {"birth": 0, "born_gen": 0, "depth": 0},
    {"birth": 1, "born_gen": 0, "depth": 0},
    {"birth": 2, "born_gen": 0, "depth": 65535,
     "neuron": {"type": "m", "op": "max", "slots": [3, null, null, null, null]}},
    {"birth": 3, "born_gen": 7, "depth": 30000,
     "neuron": {"type": "m", "op": "mean", "slots": [0, 1, null, null, null]}}
  ]
}
```

A node's `id` is its index in `nodes`. For non-input nodes, `birth` and `born_gen` describe
the neuron.

| Neuron type | Fields (in order) |
|---|---|
| `v` | `"type": "v"`, `"input": id or null`, `"inversion": "none"/"bitnot"/"recip"`, `"op": "mul"/"div"/"add"/"sub"`, `"weight": number` |
| `s` | `"type": "s"`, `"input": id or null`, `"op": "mean"/"median"/"min"/"max"`, `"k": integer` |
| `m` | `"type": "m"`, `"op": …`, `"slots": [id or null, …]` (exactly `in_qty_max` entries) |

s_neuron history is state and is **not** serialized.

### Validation on load

A reader MUST reject a genome unless all of these hold:

- Known `format`; `spec_version` with the same MAJOR version.
- `in_qty_max` odd and ≥ 3; `n_inputs`, `n_outputs` ≥ 1.
- Node order and depths as in [03](03-net-structure.md#nodes); depths within `depth_type`.
- Exactly the non-input nodes have a neuron; all fields valid; `k` within range and odd for
  `median`; weights finite.
- Every connection refers to an existing node of strictly smaller depth.
- The path rule holds; no node fails to reach an output.
- `birth` values unique and all `< next_birth`.

## Mutation profile

```json
{
  "format": "j_neuro.profile",
  "spec_version": "0.1.0",
  "overall": {
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
    "slot_fill_type_weights": {"v": 1, "s": 1, "m": 1}
  },
  "v": {},
  "s": {},
  "m": {"edit_max_pct": 2.0}
}
```

The overall section MUST contain every field. Per-type sections contain only overrides, and
only of fields marked overridable in [06](06-mutation.md#mutation-profile).

## Run log

Reproduces a whole run: the seed, the starting profile, and every profile change.

```json
{
  "format": "j_neuro.runlog",
  "spec_version": "0.1.0",
  "seed": 12345,
  "initial_profile": { "...": "profile object as above" },
  "changes": [
    {"generation": 40, "patch": {"overall": {"sigma_weight": 0.05}}},
    {"generation": 90, "patch": {"m": {"edit_max_pct": null}}}
  ]
}
```

- A change takes effect **before** mutations of the given generation.
- `patch` is a JSON Merge Patch (RFC 7386) applied to the current profile; `null` removes a
  per-type override.
- Changes are only made between generations, so replaying the log reproduces the run exactly.
