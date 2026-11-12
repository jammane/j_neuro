# j_neuro

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-blue)](LICENSE)

An **evolutionary neural-network library**, written twice — once in **C++** and once in
**Python** — as exact mirrors of a single language-neutral specification. Networks are not
trained by gradients; they are **grown, tuned and pruned by mutation**.

> **Status:** spec v0.4.0-draft is implemented in both Python and C++. The two produce
> byte-identical results on every shared golden case, and a run logged in either language
> replays exactly in the other.

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
| `s_neuron` | 1 in → 1 out, stateful: `x − agg(previous k inputs)`. |
| `m_neuron` | `IN_QTY_MAX` slots in → 1 out: `mean`, `median`, `min` or `max`. Empty slots read 0. |
| `DEPTH` | Per-net integer type of depth values: `U16`, `U32` or `U64`. Bounds how finely structure can be layered. |
| `IN_QTY_MAX` | Per-net odd number: m_neuron slot count and maximum s_neuron history. |
| **Path rule** | Every input always has at least one path to every output. |
| **Mutation** | Configurable "up to %" profiles for edits, op changes, growth and removal; neutral insertions; exact neutral pruning with cascades. |

The full, normative definition lives in [`spec/`](spec/README.md).

## Repository layout

```
spec/        language-neutral specification (normative)
docs/        architecture notes and the design decision log
cpp/         C++ implementation (C++20, header-mostly)
python/      Python implementation (NumPy)
tests/golden shared cross-language conformance cases
tools/       golden-case generators and the RNG reference oracle
```

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
