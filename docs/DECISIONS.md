# Design decisions

Each entry: the decision and why. **Provisional** entries were chosen during spec writing
without an explicit decision by the maintainer and may be revisited.

## Decided

| # | Decision | Why |
|---|---|---|
| D1 | Two mirrored implementations (C++, Python) of one language-neutral spec. | Spec-first prevents the two from drifting. |
| D2 | No neural-network libraries; only math dependencies. | It is our own library. |
| D3 | Accuracy is the first priority; per-net secondary priority `SPEED` / `MEMORY` / `BALANCED`. | Performance choices must never change results. |
| D4 | Results are bit-identical across priorities and languages. | Makes accuracy testable and the mirror verifiable. |
| D5 | Per-net `DEPTH` as fixed-width `U16/U32/U64`. | `unsigned long` differs between platforms; width bounds net layering. |
| D6 | Nodes hold values; inputs at depth 0, outputs at max depth; a neuron's inputs are strictly shallower than its output. | Guarantees acyclic nets and a trivial evaluation order. |
| D7 | Three neurons: v (single in, weighted op with optional inversion), s (single in, history), m (multi in). | User design. |
| D8 | v inversion is either bitwise NOT of the 64-bit pattern or `1/x`. | User design; bitwise NOT ≈ −4/x, a distinct nonlinearity. |
| D9 | NaN → 0; empty slots and empty/unfilled history read 0. | One simple rule for undefined values. |
| D10 | s output is `x − agg(previous k values)`; history excludes the current value; ring buffer zero-filled. | Change-from-recent-past detector. |
| D11 | s and m share the aggregation ops mean, median, min, max. Mode dropped. | Mode on continuous values degenerates to min/max. |
| D12 | Empty slots participate as 0, except `min` skips them. | Max then acts as built-in ReLU; min stays a true minimum. |
| D13 | `IN_QTY_MAX` must be odd; s history length must be odd for median. | Median always has a middle element without averaging. |
| D14 | Mutation: s history length, m slot add/remove, v weights, any operation, node depth shifts. | User design. |
| D15 | Additions: v/s path insertion, slot fill, m after m; new neurons get a random op with least disruptive values. | Grow without hurting existing fitness. |
| D16 | Addition method default odds: slot fill 40%, v insert 30%, s insert 20%, m after m 10%. | User choice; disruptive methods weighted down. |
| D17 | New downstream m (m after m) starts as `max`. | Neutral when its input is never negative; later latching is always neutral. |
| D18 | Mutation profile: net-wide "up to %" caps, settable overall and per neuron type, editable between generations, changes logged. | Flexible control with reproducible runs. |
| D19 | Unavailable or invalid mutation attempts are skipped, not redistributed. | Caps are maxima, not targets. |
| D20 | Path rule: every input has a structural path to every output; a `*0` link still counts. | Minimum meaningful net; cheap structural check. |
| D21 | Removal candidates: identity and silenced neurons; cascades downstream (constant-0 readers) and upstream (dead nodes). | Shrinks nets with exactly unchanged outputs. |
| D22 | Neurons whose removal would break the path rule are immune; the cascade stops there. Immunity is recomputed as the graph changes. | Preserves the path rule; new paths can lift immunity. |
| D23 | Shared custom RNG (xoshiro256** + splitmix64) and hand-written distributions. | Platform RNGs/distributions differ. |
| D24 | License: PolyForm Noncommercial 1.0.0 plus paid commercial licenses. | Free for personal use, paid for business. |
| D25 | Dependencies only under attribution-only licenses (MIT, BSD, ISC, Zlib, BSL-1.0, public domain). No Eigen (MPL-2.0), no Apache-2.0. | Avoid obligations beyond recognition. |
| D26 | CLA required for outside contributions. | Needed to sell commercial licenses covering contributed code. |

## Decided in the v0.2 review (formerly provisional)

| # | Decision | Why |
|---|---|---|
| P1 | −0.0 becomes +0.0 at every neuron output. | Exact neutrality of `*0` silencing and of ties in min/max/median. |
| P2 | Mean uses Neumaier-compensated summation in slot order. | Accuracy first. Costs ~4 flops per value over ≤ `IN_QTY_MAX` values; SIMD/matrix speedups remain possible across neurons (one sum per lane). |
| P3 | `normal()` is Irwin–Hall; multiplicative step `1+a` / `1/(1−a)`. Large jumps come only from the explicit `inject_diversity(net, pct, rng, sigma_reset)`, which resets weights of `pct`% of v_neurons and is recorded in the run log. | No transcendental functions; diversity injection stays under the user's control. |
| P4 | Node values are always finite: ±inf is clamped to ±max double; NaN → 0. Binary64 only for now (128-bit would need a software float in Python). | Values must fit the 64-bit value format as numbers. |
| P5 | Op change resets the weight to the new op's identity weight. | Least disruptive. |
| P6 | Slot fill places the new neuron behind a v `*0` wrapper and never fills `min` receivers. | Exactly neutral additions. |
| P7 | New s_neurons start with `k = 1`; new v_neurons get an inversion with probability `insert_inversion_prob` (default 0.05). | Inversions can appear at insertion, but rarely, since they are not neutral. |
| P8 | Total node count ≤ `D_MAX` of the depth type; additions beyond it are skipped. No gap → try shifting once, else skip. | DEPTH bounds net size directly. |
| P9 | Depth shift default mode is uniform within the legal range (`center` available). | Exact integer arithmetic on every depth type. |
| P10 | Removal grace period `min_age_for_removal = 5` (profile-adjustable). | Neutral insertions get time to be tuned. |
| P11 | Default rates as listed in spec/06; to be revisited on real problems. | Starting points; every profile can override them. |
| P12 | m slots may hold duplicates; v/s inputs can be rewired by edits (`rewire_prob`, default 0.1) unless the old input would be left without readers or the path rule would break. | Exact bypass; structural search for single-input neurons without orphaning nodes. |
| P13 | Removed nodes are tombstoned during mutation and compacted at the end, keeping order. | Stable ids during a mutation; dense arrays after. |
| P14 | s history is not serialized; nets start reset. `get_warmup_steps()` reports the longest chain of s_neuron history along any path; fitness should skip that many outputs after a reset. | State is not part of the genome. |
| P16 | nlohmann/json (MIT) for C++ parsing. | Exact uint64 support; attribution-only license. |
| P17 | Commercial license terms are drafted outside this repository. | Legal, not code. |

## Population (v0.3)

| # | Decision | Why |
|---|---|---|
| P15 | The library supplies a `Population` (nets by stable id, `evaluate`, `mutate`/`clone`/`inject_diversity`/`remove`, generation counter); scoring, selection and culling belong to the caller's trainer. | Training strategy is the user's job; the library supplies outputs and operations. |
| P15a | Ids are never reused; `mutate` returns an offspring id and keeps the parent; `inject_diversity` works in place. | Stored ids stay valid after culling; elitism is trivial. |
| P15b | RNG mode is a profile setting, `rng_mode` = `fresh` (default, OS entropy per operation) or `shared` (population stream). Fresh seeds are recorded in the run log. | Exploration by default, yet every run stays exactly replayable. |
| P15c | Population priority controls evaluation concurrency (MEMORY 1 thread, BALANCED half, SPEED all); Python evaluates sequentially (GIL). | Concurrency is a speed/memory trade-off and cannot change results. |
| P15d | No mixed nets in a population; no crossover. | Arbitrary nets have no meaningful combination. |
| P15e | The run log is the population's operation journal (replaces the 0.2 profile/diversity log). | One mechanism for reproducibility. |

## Found during implementation (v0.1)

| # | Decision | Why |
|---|---|---|
| I1 | Canonical float text follows Python `repr` exactly; C++ reformats `std::to_chars` output to match (spec/09). | Byte-identical genome files across languages. |
| I2 | Path insertion moves the target node only when that move actually opens a gap; otherwise nothing changes (spec/06). | A skipped attempt must leave the net unchanged. |
| I3 | `create` needs only input/output counts; defaults `U32`, `in_qty_max 5`, `BALANCED`, seed 0 (spec/03). | User request: no hand-specified layers or starting neurons. |
| I4 | Immunity is computed by a path-rule check with the node cut out, not by dominator trees. | Same result as the spec definition and simpler; dominators remain an optimization option. |
| I5 | `BALANCED` currently uses the same execution plan as `SPEED`; `MEMORY` recomputes the order each step and frees values after their last reader. | Results are identical either way; finer tuning can come later. |
