# j_neuro — C++ implementation

C++20, CMake. Mirrors the Python implementation bit for bit (see the
[specification](../spec/README.md)).

## Requirements

- GCC 12+ or Clang 15+ on a 64-bit target.
- nlohmann/json ≥ 3.11 (MIT). Fedora: `dnf install json-devel`.
- Catch2 v3 (BSL-1.0), tests only. Fedora: `dnf install catch-devel`.

## Build and test

```
cmake -S cpp -B build
cmake --build build -j
ctest --test-dir build
```

The build refuses fast-math flags and always uses `-ffp-contract=off`
([spec/02](../spec/02-numerics.md#build-rules-c)).

## Layout

| Header | Contents |
|---|---|
| `jneuro/jneuro.hpp` | umbrella header |
| `types.hpp`, `numerics.hpp`, `rng.hpp`, `aggregate.hpp` | enums, exact float helpers, shared RNG, aggregation |
| `net.hpp` | `Net<D>` (D = `uint16_t`/`uint32_t`/`uint64_t`), neurons, evaluation |
| `analysis.hpp`, `net_build.hpp` | path rule, immunity, depth ranges; `create`, `validate` |
| `io.hpp`, `sha256.hpp` | `AnyNet`; `.jnet` files: `to_bytes`, `from_bytes`, `save_net`, `load_net`; `create(n_in, n_out, …)`; SHA-256 |
| `population.hpp` | `Population<D>`: nets by id, `evaluate`, operations, `save`/`load` (`.jpnet`), `replay` |
| `profile.hpp`, `mutate.hpp`, `removal.hpp` | mutation profile, `mutate`, removal and compaction |
