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

## Provisional

| # | Decision | Why |
|---|---|---|
| P1 | Canonicalize −0.0 to +0.0 at every neuron output (in addition to NaN → 0). Additive identity weight is therefore `0.0`. | Equal values get equal bits, so min/max/median ties and `*0` silencing are exactly neutral. Supersedes the earlier `+(−0.0)` identity idea. |
| P2 | Mean uses Neumaier-compensated summation in slot order; plain sum when non-finite. | Accuracy first; deterministic. |
| P3 | `normal()` is Irwin–Hall (sum of 12 uniforms − 6); multiplicative weight step uses `1+a` / `1/(1−a)`. | Avoids transcendental functions, which differ between math libraries. |
| P4 | ±inf passes through as a valid value. | Mathematically honest; fitness can penalize it. |
| P5 | Op change resets the weight to the new op's identity weight. | Least disruptive. |
| P6 | Slot fill places the new neuron behind a v `*0` wrapper and never fills `min` receivers. | Exactly neutral addition with random type, op and inputs. |
| P7 | s path insertion and s slot fill start with `k = 1`; new v neurons have no inversion. | Simplest least-disruptive start. |
| P8 | When no depth gap exists, try moving the target node to the top of its legal range once; otherwise skip. No DEPTH promotion; no node-count cap. | Simple and deterministic; depth-shift phase opens gaps over time. |
| P9 | Depth shift default mode is uniform within the legal range (`center` available). | Avoids binary64 arithmetic on U64 depths. |
| P10 | Grace period `min_age_for_removal = 5` generations; removal applies only to hidden nodes. | Lets neutral insertions be tuned before pruning. |
| P11 | Default rates: op change 1%, edit 5%, growth 1.5%, removal 1%, depth shift 5%; `sigma_weight 0.1`, `sign_flip_prob 0.02`, `sigma_k 1`, `sigma_slots 1`. | Starting points only. |
| P12 | m slots may hold duplicates (from bypass); v/s inputs are not rewired by edits. | Bypass must be exact; rewiring was never specified. |
| P13 | Removed nodes are tombstoned during mutation and compacted at the end, keeping order; `birth` ids give lineage. | Stable ids during a mutation; dense arrays after. |
| P14 | s history is not serialized; nets always start reset. | State is not part of the genome. |
| P15 | Population, selection, fitness and crossover are out of scope for v0.1 (mutation only). | Not yet designed. |
| P16 | nlohmann/json (MIT) for C++ serialization. | Exact uint64 support; attribution-only license. |
| P17 | Commercial license contract terms not drafted. | Recommend a lawyer when the first buyer appears. |
