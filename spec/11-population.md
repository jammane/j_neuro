# 11 — Population

A **population** is a container of nets with stable ids. j_neuro does not train: the caller
writes the trainer (scoring, selection, culling, how often to mutate). The population gives
the trainer the outputs of every net, keyed by id, and operations to apply to nets by id.

## Construction

```
Population(depth_type, in_qty_max, n_inputs, n_outputs, priority = BALANCED, seed = none,
           path_mix = 2)
```

- Every net in the population has exactly these five parameters (`depth_type`, `in_qty_max`,
  `n_inputs`, `n_outputs`, `path_mix`); mixed nets are rejected.
- `priority` is the population's speed/memory choice for `evaluate` (below). Each net keeps
  its own priority for its own evaluation.
- `seed` seeds the population's **shared RNG stream**. If omitted, a seed is taken from the
  operating system's entropy source. Either way the seed is recorded in the run log.
- The population starts with no nets and `generation = 0`.

## Ids

Ids are unsigned 64-bit integers assigned in increasing order starting at 0. An id is never
reused, even after its net is removed. `ids()` returns the live ids in ascending order.

## RNG modes

Operations that draw random numbers (`create`, `mutate`, `inject_diversity`) take an RNG
mode:

| Mode | The operation draws from |
|---|---|
| `fresh` (default) | A new `Rng(s)` where `s` is a fresh 64-bit seed from the operating system's entropy source (Python `os.urandom`, C++ `std::random_device`). `s` is recorded in the run log. |
| `shared` | The population's shared stream, continuing from where the previous shared operation stopped. |

For `mutate` the mode is the profile's `rng_mode` field ([06](06-mutation.md#mutation-profile)).
`create` and `inject_diversity` take it as an argument (default `fresh`). Either way every run
is exactly replayable from its run log (below).

## Operations

| Operation | Effect |
|---|---|
| `create(count, rng_mode)` → ids | Adds `count` minimal nets with the population's `path_mix` ([03](03-net-structure.md#minimal-net)), all drawn from one operation RNG, in id order. |
| `add(net)` → id | Adds a copy of an existing net. Rejected if its parameters differ from the population's. |
| `get(id)` | The net itself (for direct inspection or stepping). |
| `ids()`, `size()`, `generation` | Inspection. |
| `evaluate(inputs, skip_warmup = true)` → `{id: outputs}` | For every net, in ascending id order: `reset()`, then `step` each input row in order. Returns one output row per step; with `skip_warmup`, the first `get_warmup_steps()` rows of that net are dropped. |
| `mutate(id, profile, times = 1)` → new id | Offspring: `mutate` applied `times` times in a row to a copy of the net, all with the same operation RNG and the current `generation`. The parent stays. |
| `clone(id)` → new id | Exact copy of the genome. |
| `inject_diversity(id, pct, sigma_reset = 1.0, rng_mode)` | Replaces the net at `id` with `inject_diversity(net, pct, rng, sigma_reset)` ([06](06-mutation.md#diversity-injection)). The id is kept. |
| `remove(id)` | Culls the net. Its id is never reused. |
| `next_generation()` | `generation += 1`. Neurons created by later mutations record the new generation. |
| `save(path, include_log = false)` | Writes the `.jpnet` and its `.jnet` files ([09](09-serialization.md#jpnet)). Not journaled. |
| `Population.load(path)` | Loads a `.jpnet`, verifying every `.jnet` hash; continues exactly where the saved population stopped. |

An unknown id is an error. Operations never change other nets.

## Evaluation priority

Nets are independent, so `evaluate` MAY process them concurrently; results MUST be identical
in every case:

| Population priority | Evaluation |
|---|---|
| `MEMORY` | One net at a time. |
| `BALANCED` | Concurrently with at most half the available hardware threads. |
| `SPEED` | Concurrently with all available hardware threads. |

Implementations MAY evaluate sequentially where concurrency brings no benefit (for example
CPython with the global interpreter lock).

## Run log

Every population keeps a run log ([09](09-serialization.md#run-log-json)) journaling its
parameters, shared-stream seed, every profile used, and every operation with its arguments,
resulting ids and RNG seed (for `fresh` mode). `Population.replay(log, base = none)`
re-executes the journal and MUST reproduce the population exactly: the same ids, nets,
generation and shared-stream state, in either language.

- A population built from scratch replays from an empty start (`base` omitted).
- A population loaded from a `.jpnet` **with** an embedded log continues that log, so it also
  replays from an empty start.
- A population loaded **without** an embedded log has a log with `resumed_from` set to the
  `.jpnet`'s SHA-256. Replaying it requires `base`: the population loaded from that same file
  (checked by hash); the operations are then applied to a copy of it.
