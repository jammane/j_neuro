# j_neuro

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

An **evolutionary neural-network library**, written twice — once in **C++** and once in
**Python** — as exact mirrors of a single language-neutral specification. Networks are not
trained by gradients; they are **grown, tuned and pruned by mutation**.

> **Status:** spec v0.6.0-draft is implemented in both Python and C++. The two produce
> byte-identical results on every shared golden case, and a run logged in either language
> replays exactly in the other. Three example programs train and analyze nets on card games and
> measure how mutation alone shapes a net.

## Quick start

```python
from jneuro import Net
from jneuro.mutate import Profile, inject_diversity, mutate
from jneuro.rng import Rng

net = Net.create(4, 2)              # minimal net from input/output counts only
print(net.step([0.1, 0.2, 0.3, 0.4]))
child = mutate(net, Profile.default(), generation=1, rng=Rng(42))
skip = net.get_warmup_steps()        # outputs to ignore after reset() (s_neuron history filling)
fresh = inject_diversity(child, 20.0, Rng(7))   # explicit: re-randomize 20% of v weights
```

```cpp
#include <jneuro/jneuro.hpp>
auto any = jneuro::create(4, 2);    // AnyNet; default depth type U32
auto& net = std::get<jneuro::Net<std::uint32_t>>(any);
auto out = net.step({0.1, 0.2, 0.3, 0.4});
jneuro::Rng rng(42);
auto child = jneuro::mutate(net, jneuro::Profile::defaults(), 1, rng);
```

### Writing a trainer

j_neuro does not decide which nets are good; your trainer does. A `Population` gives you every
net's outputs by id and operations to apply by id:

```python
from jneuro import DepthType
from jneuro.population import Population

pop = Population(DepthType.U32, 5, n_inputs=4, n_outputs=2)
pop.create(50)
for gen in range(100):
    results = pop.evaluate(inputs)                     # {id: output rows}, warm-up skipped
    ranked = sorted(results, key=lambda i: my_score(results[i]))   # your fitness
    for i in ranked[:25]:
        pop.remove(i)                                  # cull
    for i in ranked[25:]:
        pop.mutate(i, profile, times=2)                # offspring get new ids
    pop.next_generation()
pop.save("run.jpnet")                                  # run.jpnet + run_nets/<id>.jnet, hash-verified
pop = Population.load("run.jpnet")                     # resumes exactly, in Python or C++
```

Ready-made mutation profiles live in [`profiles/`](profiles/README.md): `expansion.json` (grows
wide nets fast), `training.json` (slow growth for training: 1% of a net's neurons, min 1, in 10% of
generations for 10% of nets) and `bulk.json` (growth only: add a set percentage of neurons before a
training pass). Load one with `profile_from_json` in either language.

Nets are stored as binary `.jnet` files and populations as `.jpnet` files
([spec/09](spec/09-serialization.md)). Both languages write byte-identical files, so a run
can be exported from one and continued in the other.

Build and test instructions: [cpp/README.md](cpp/README.md), [python/README.md](python/README.md).

## Goals

1. **Accuracy first.** Every computation is defined bit-for-bit (IEEE-754 binary64, fixed
   evaluation and summation order, explicit NaN rules). A net gives identical outputs in C++
   and Python, on every platform, under every performance setting.
2. **Selectable secondary priority per net:** `SPEED`, `MEMORY` or `BALANCED`. The priority
   changes *how* a net is stored and evaluated, never *what* it computes.
3. **Mirrored implementations.** Same seed + same genome + same mutation profile ⇒ same
   result in both languages, verified by shared golden tests.
4. **No neural-network libraries.** Only math/numeric dependencies whose licenses require
   nothing beyond attribution (see [CONTRIBUTING.md](CONTRIBUTING.md#dependency-license-rule)).

## Core concepts (short version)

| Concept | Meaning |
|---|---|
| **Node** | Holds one value. Has a **depth**. Inputs are at depth 0, outputs at the maximum depth. |
| **Neuron** | Writes exactly one node. Every node it reads must be strictly shallower than the node it writes, so nets are always acyclic. |
| `v_neuron` | 1 in → 1 out: optional inversion (bitwise `~` or `1/x`), then `* / + −` by a weight. |
| `s_neuron` | 1 in → 1 out, stateful: `x − agg(previous k inputs)`, or with `lag` the raw input from k steps ago. |
| **History nodes** | Every s_neuron's memory is readable: history node `j` (its input `j` steps ago) and a length node (how many entries are filled), at the consecutive depths just below it. A connection is `(node, position)`. Every s_neuron keeps a depth gap of `IN_QTY_MAX + 1` below it for them, so they can be wired in later without moving anything. Spec 0.5 nets load and are re-spaced, with identical outputs. |
| `m_neuron` | `IN_QTY_MAX` slots in → 1 out: `mean`, `median`, `min` or `max`. Empty slots read 0. |
| `DEPTH` | Per-net integer type of depth values: `U16`, `U32` or `U64`. Bounds how finely structure can be layered. |
| `IN_QTY_MAX` | Per-net odd number: m_neuron slot count and maximum s_neuron history. |
| **Path rule** | Every input always has at least one path to every output. |
| **Mix rule** | Every input-to-output path passes through at least `path_mix` v, s and m neurons each (default 2, set when a net is created). |
| **Run rule** | No path has more than `max_run` 1:1 (v, s) or many:1 (m) neurons in a row (default 3), so the kinds alternate. |
| **Mutation** | A profile sets "up to %" budgets for op changes, edits (including rewiring and slot retargeting), growth, removal and depth shifts. Growth has eleven methods: neutral path insertions (`v_insert`, `s_insert`, `m_after_m`, `m_insert`), branches into open m slots (`slot_fill`, input taps, `v_link`/`s_link`, `m_clone`), with receivers chosen to spread width across the net. Removal is exact neutral pruning with cascades, plus optional constant folding. Growth can be limited to a share of generations and nets, with an s↔v swap for the nets that miss out. A `fill` growth budget adds an exact share of the net's neurons (retrying skipped attempts), optionally filling open m slots with links inside the same total (bulking). With `history_read_prob`, new connections can read length and history nodes. Every mutation operation is counted. |

The full, normative definition lives in [`spec/`](spec/README.md).

## Repository layout

```
spec/        language-neutral specification (normative)
docs/        architecture notes and the design decision log
cpp/         C++ implementation (C++20, header-mostly)
python/      Python implementation (NumPy)
tests/golden shared cross-language conformance cases
tools/       golden-case generators and the RNG reference oracle
profiles/    ready-made mutation profiles (JSON)
examples/    C++ example programs (built by CMake with the library)
```

## Examples

| Program | What it does |
|---|---|
| [`blackjack_trainer`](examples/blackjack/README.md) | Evolves nets that play blackjack (5-deck shoe, real payout) against the same deals, logging score and win rate beside two fixed reference strategies. |
| [`blackjack_analyze`](examples/blackjack/README.md#analysis) | Plays a saved population on fresh deals: each net's result, its mix of actions, and the best net's structure and decision table. |
| [`highlow_trainer`](examples/highlow/README.md) | Evolves nets that guess the next card of a shoe (suit, and higher/lower/equal), where remembering earlier cards can help. |
| [`categorize_trainer`](examples/categorize/README.md) | Evolves nets that sort 64×64 black-and-white images of 188 symbols (ASCII and Latin-1, in 30 fonts, at several sizes and positions) into categories the nets define themselves. The score asks only for consistent, usable categories, not particular ones. |
| [`categorize_analyze`](examples/categorize/README.md#analysis) | Shows what the categories hold: per category its size, most common symbols, how many fonts, sizes and positions it spans, and a `.bmp` sheet of its members. |
| [`expansion_test`](examples/expansion/README.md) | Mutation only, no scoring: how a profile grows a net — method use, paths, width by depth and open m slots per generation. |

All of them run until interrupted or `--stop-at N`, save their population, and resume with
`--resume`.

## Documentation

- [Specification](spec/README.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Design decisions](docs/DECISIONS.md)
- [Contributing](CONTRIBUTING.md)
- [Credits](CREDITS.md)

## License

Free for **noncommercial** use under the [PolyForm Noncommercial License 1.0.0](LICENSE)
(personal, hobby, research, education, charities, government).
**Any business or commercial use requires a paid commercial license** — see
[COMMERCIAL.md](COMMERCIAL.md).

Required Notice: Copyright Jeremiah Mane (https://github.com/jammane/j_neuro)
