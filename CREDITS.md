# Credits

Recognition for the people, projects and research j_neuro builds on.

## Contributors

| Name | Role |
|---|---|
| Jeremiah Mane ([@jammane](https://github.com/jammane)) | Creator, designer and maintainer |

Design discussion and documentation drafting were assisted by Claude (Anthropic).

New contributors are added here when their first contribution is merged
(see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Third-party software

j_neuro only depends on software whose license requires nothing beyond attribution
(see [CONTRIBUTING.md](CONTRIBUTING.md#dependency-license-rule)). Dependencies are installed
by the user's package manager or build system and are **not vendored** into this repository.

When a dependency is actually adopted, its full copyright notice and license text are added
to this section in the same change.

### Planned

| Package | Used by | Purpose | License |
|---|---|---|---|
| [NumPy](https://numpy.org) | Python | float64 arrays, bit-level views of doubles | BSD-3-Clause |
| [pytest](https://pytest.org) | Python (dev only) | tests | MIT |
| [Catch2](https://github.com/catchorg/Catch2) | C++ (dev only) | tests | BSL-1.0 |
| [nlohmann/json](https://github.com/nlohmann/json) | C++ | genome / profile serialization | MIT |

### Adopted

_None yet._

## Algorithms and research

| Work | Authors | Used for |
|---|---|---|
| xoshiro256\*\* and splitmix64 ([prng.di.unimi.it](https://prng.di.unimi.it/)) | David Blackman, Sebastiano Vigna (splitmix64 after Guy L. Steele Jr., Doug Lea, Christine H. Flood) | Shared cross-language PRNG ([spec/08](spec/08-rng.md)). Reference code is public domain. |
| Improved Kahan–Babuška summation (1974) | Arnold Neumaier, after William Kahan | Compensated mean ([spec/04](spec/04-neurons.md)) |
| Irwin–Hall distribution | Joseph Oscar Irwin, Philip Hall | Transcendental-free Gaussian approximation ([spec/08](spec/08-rng.md)) |
| Dominator analysis (e.g. "A Simple, Fast Dominance Algorithm", 2001; Lengauer–Tarjan, 1979) | Keith D. Cooper, Timothy J. Harvey, Ken Kennedy; Thomas Lengauer, Robert E. Tarjan | Computing deletion immunity ([spec/07](spec/07-removal.md)) |
| NEAT — NeuroEvolution of Augmenting Topologies (2002) | Kenneth O. Stanley, Risto Miikkulainen | Inspiration for neutral structural insertion |

## Licenses and standards

- [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — The PolyForm Project.
- IEEE 754-2008/2019 floating-point arithmetic.
