# 09 — Serialization

| File | Contents | Encoding |
|---|---|---|
| `.jnet` | One net (genome) | Binary, below |
| `.jpnet` | One population: its state plus a table of `.jnet` files and their hashes | Binary, below |
| profile | A mutation profile | JSON |
| run log | A population's operation journal | JSON |

Nets have **no** text form: nobody edits them by hand, and the binary form is exact and
compact. Both implementations MUST write **byte-identical** `.jnet` and `.jpnet` files for
equal content, so a file exported from one language imports unchanged into the other.

## Binary conventions

- All integers are **unsigned, little-endian**, written field by field (never by copying
  in-memory structures).
- `u8`, `u16`, `u32`, `u64`: 1, 2, 4, 8 bytes.
- `W`: the net's **depth width**: 2 bytes for `U16`, 4 for `U32`, 8 for `U64`. Depths, node
  counts and node ids use this width, so `depth_type` also sets file size.
- `NONE` = all bits set in `W` bytes (`0xFFFF`, `0xFFFFFFFF`, `0xFFFFFFFFFFFFFFFF`): an empty
  slot or a missing input. It is never a valid id, because a net has at most `D_MAX` nodes and
  ids run `0 … D_MAX − 1`.
- `f64`: the 8 raw bytes of the binary64 value, little-endian. Exact; no text conversion.
- `sha256`: 32 bytes, SHA-256 (FIPS 180-4).
- Enumerations are `u8` indices in the enumeration orders of [04](04-neurons.md) and
  [03](03-net-structure.md): priority `SPEED 0, MEMORY 1, BALANCED 2`; depth type
  `U16 0, U32 1, U64 2`; inversion `none 0, bitnot 1, recip 2`; v op `mul 0, div 1, add 2,
  sub 3`; aggregation op `mean 0, median 1, min 2, max 3`.

## `.jnet`

| Field | Type | Notes |
|---|---|---|
| magic | 4 bytes | ASCII `JNET` |
| format version | `u16` | `1` |
| spec version | 3 × `u16` | major, minor, patch, e.g. `0, 5, 0` |
| priority | `u8` | |
| depth_type | `u8` | fixes `W` for the rest of the file |
| in_qty_max | `u32` | |
| n_inputs | `u32` | |
| n_outputs | `u32` | |
| path_mix | `u8` | |
| next_birth | `u64` | |
| node_count | `W` | |
| nodes | node_count × node | in id order |
| checksum | `sha256` | of every preceding byte |

Each node:

| Field | Type | Notes |
|---|---|---|
| birth | `u64` | |
| born_gen | `u64` | |
| depth | `W` | |
| neuron type | `u8` | `0` none (input nodes), `1` v, `2` s, `3` m |
| v | input `W`, inversion `u8`, op `u8`, weight `f64` | |
| s | input `W`, op `u8`, k `u32` | |
| m | op `u8`, then `in_qty_max` × slot `W` | |

s_neuron history is state and is not stored.

### Loading

A reader MUST reject a `.jnet` unless all of these hold:

- Magic `JNET`, a known format version, and a spec version with the same MAJOR (and, while
  MAJOR is 0, the same MINOR) version.
- The file is exactly as long as its contents require (no truncation, no trailing bytes) and
  the checksum matches.
- Every enumeration byte is in range.
- At most `D_MAX` nodes; `in_qty_max` odd and ≥ 3; `n_inputs`, `n_outputs` ≥ 1.
- Node order and depths as in [03](03-net-structure.md#nodes).
- Exactly the non-input nodes have a neuron; `k` within range and odd for `median`; weights
  finite.
- Every connection refers to an existing node of strictly smaller depth.
- The path rule and the mix rule hold; no node fails to reach an output.
- `birth` values unique and all `< next_birth`.

## `.jpnet`

A population file holds the population's state and a table of its nets. The nets themselves
are separate `.jnet` files, each verified by its SHA-256 hash.

| Field | Type | Notes |
|---|---|---|
| magic | 4 bytes | ASCII `JPOP` |
| format version | `u16` | `1` |
| spec version | 3 × `u16` | |
| priority | `u8` | population priority |
| depth_type | `u8` | |
| in_qty_max | `u32` | |
| n_inputs | `u32` | |
| n_outputs | `u32` | |
| path_mix | `u8` | |
| seed | `u64` | shared-stream seed |
| rng state | 4 × `u64` | shared-stream xoshiro256** state, so training resumes exactly |
| generation | `u64` | |
| next_id | `u64` | |
| net_count | `u64` | |
| net table | net_count × entry | ascending id |
| log flag | `u8` | `0` no run log, `1` run log follows |
| run log | `u64` byte length + UTF-8 JSON | only if the flag is `1` |
| checksum | `sha256` | of every preceding byte |

Each net table entry:

| Field | Type | Notes |
|---|---|---|
| id | `u64` | |
| path length | `u16` | |
| path | UTF-8 | relative to the `.jpnet` file's directory, `/` as separator |
| hash | `sha256` | of the complete `.jnet` file |

### Saving

`save(path, include_log = false)` writes:

- `<name>.jpnet` at `path`;
- one `.jnet` per net in the directory `<name>_nets/` beside it, named `<id>.jnet`; the table
  stores the path `<name>_nets/<id>.jnet`.

Existing files with those names are overwritten. Other files in the directory are left alone;
the table, not the directory contents, defines the population.

### Loading

A reader MUST reject a `.jpnet` unless the magic, versions, length and checksum are valid and
**every** listed `.jnet` file exists, has exactly the listed hash, loads as a valid net, and
has the population's `depth_type`, `in_qty_max`, `n_inputs`, `n_outputs` and `path_mix`.

After loading, the population continues exactly: same ids, nets, generation, `next_id` and
shared-stream state. If a run log was embedded it becomes the population's journal and keeps
growing; otherwise the journal restarts, marked as resuming from this file
([run log](#run-log-json)).

## Mutation profile (JSON)

```json
{
  "format": "j_neuro.profile",
  "spec_version": "0.5.0",
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
    "rewire_prob": 0.1,
    "insert_inversion_prob": 0.05,
    "min_age_for_removal": 5,
    "add_method_weights": {"slot_fill": 40, "v_insert": 30, "s_insert": 20, "m_after_m": 10},
    "slot_fill_type_weights": {"v": 1, "s": 1, "m": 1},
    "rng_mode": "fresh"
  },
  "v": {},
  "s": {},
  "m": {"edit_max_pct": 2.0}
}
```

Missing overall fields take their defaults. Per-type sections contain only overrides, and only
of fields marked overridable in [06](06-mutation.md#mutation-profile).

## Run log (JSON)

The journal of a population ([11](11-population.md#run-log)).

```json
{
  "format": "j_neuro.runlog",
  "spec_version": "0.5.0",
  "population": {"depth_type": "U32", "in_qty_max": 5, "n_inputs": 4, "n_outputs": 2,
                 "path_mix": 2, "priority": "BALANCED", "seed": 12345},
  "resumed_from": null,
  "profiles": [ { "...": "profile object" } ],
  "ops": [
    {"op": "create", "count": 10, "ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "seed": 998877},
    {"op": "mutate", "id": 3, "times": 2, "profile": 0, "new_id": 10, "seed": 5566},
    {"op": "mutate", "id": 4, "times": 1, "profile": 0, "new_id": 11, "seed": null},
    {"op": "clone", "id": 10, "new_id": 12},
    {"op": "inject_diversity", "id": 12, "pct": 25.0, "sigma_reset": 1.0, "seed": 4433},
    {"op": "add", "new_id": 13, "jnet": "4a4e4554…"},
    {"op": "remove", "id": 0},
    {"op": "next_generation"}
  ]
}
```

- `seed` on an operation is the fresh seed it used, or `null` when it drew from the shared
  stream.
- `profile` is an index into `profiles`; identical profiles are stored once.
- `add` stores the added net as the lowercase hex of its `.jnet` bytes.
- `resumed_from` is `null` for a population built from scratch. For a population loaded from a
  `.jpnet` without an embedded run log, it is the lowercase hex SHA-256 of that `.jpnet` file,
  and replaying requires that population as the starting point
  ([11](11-population.md#run-log)).
- `evaluate`, `get`, `ids`, `size`, `save` and `load` change nothing and are not journaled.
