# Architecture (planned)

Informative. The [spec](../spec/README.md) defines behavior; this document describes how the
two implementations are intended to be organized.

## Repository layout

```
spec/                     normative specification
docs/                     this file, decision log
cpp/
  CMakeLists.txt
  include/jneuro/         public headers
  src/
  tests/                  Catch2 unit tests + golden runner
python/
  pyproject.toml
  jneuro/                 package
  tests/                  pytest unit tests + golden runner
tests/golden/             shared conformance cases (spec/10)
examples/                 the same examples in both languages
```

Names: C++ namespace `jneuro`, Python package `jneuro`.

## Module map (same in both languages)

| Module | Responsibility | Spec |
|---|---|---|
| `numerics` | `canon`, `bitnot`, IEEE division helper, compensated mean | 02 |
| `rng` | splitmix64, xoshiro256**, `u`, `below`, `weighted`, `normal`, `prob_round` | 08 |
| `aggregate` | `mean/median/min/max` over values + presence flags | 04 |
| `net` | parameters, nodes, neurons, invariants, `create` | 03 |
| `eval` | `step`, `reset`, priority-specific execution plans | 05 |
| `analysis` | readers, reachability, path-rule check, immunity, legal depth ranges | 03, 07 |
| `mutate` | profile, budgets, phases 1–5 | 06 |
| `removal` | candidates, bypass, cascade, GC, compaction | 07 |
| `jnet` / `io` | binary `.jnet` files, SHA-256; profile and run-log JSON | 09 |
| `population` | nets by id, evaluation, operations, `.jpnet` files, replay | 11 |

Keeping the module boundaries identical makes the two code bases easy to compare line by line.

## C++

- C++20 (`std::bit_cast`, concepts). CMake. No runtime dependencies except nlohmann/json
  for `io`.
- Depth type as a template parameter: `Net<std::uint16_t>`, `Net<std::uint32_t>`,
  `Net<std::uint64_t>`.
- Priority as a runtime strategy object inside the net (it can change at any time without
  affecting results), selecting the execution plan and caches.
- `AnyNet` (a `std::variant` over the three depth widths) for code that handles mixed nets.
- The CMake build refuses fast-math flags and sets `-ffp-contract=off` (spec/02).

## Python

- Python ≥ 3.11, NumPy only.
- Values are Python `float` or `numpy.float64`; all divisions go through the IEEE helper.
- Depths are Python `int` with explicit range checks against `depth_type`
  (`numpy.uint16/32/64` arrays where a vectorized layout is used).
- `bitnot` via `numpy.float64(x).view(numpy.uint64)`.

## How priority maps to implementation

| Aspect | `SPEED` | `MEMORY` | `BALANCED` |
|---|---|---|---|
| Evaluation order | Cached, grouped by depth level | Recomputed (sort by depth) per structural change, not stored | Cached |
| Node values | One contiguous array | Freed after last reader's depth is passed | Contiguous array |
| m_neuron slots | Dense `in_qty_max` arrays | Sparse: connected slots + indices; empties implied | Dense |
| Reader lists / reachability | Cached, updated incrementally | Recomputed on demand | Cached |
| Immunity | Dominator trees cached | Computed per removal attempt | Computed per attempt |
| s history | Pooled ring buffers | Per-neuron buffers sized `k` | Pooled |

Every combination MUST produce identical results; golden `eval` cases run under all three.

**Vectorization.** The bit-exact rules forbid reordering arithmetic *within* one neuron (for
example the compensated mean), not computing many neurons at once. All neurons at one depth
are independent, so a future SIMD or matrix layout can evaluate them side by side, one neuron
per lane, each in its own fixed order. Independent episodes or nets can also run in parallel.

## Testing strategy

1. Unit tests per module in each language.
2. Golden cases (`tests/golden/`) run by both suites, compared bit-exactly.
3. Property tests: random genomes/mutations checked for invariants (path rule, mix rule, depth rules,
   no dead nodes) and for exact output preservation by neutral operations (v insertion,
   bypass, silenced removal, depth shifts, GC).
