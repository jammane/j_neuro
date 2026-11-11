# j_neuro — C++ implementation

**Status:** not yet implemented. See the [specification](../spec/README.md) and
[architecture notes](../docs/ARCHITECTURE.md#c).

Planned:

- C++20, CMake, 64-bit targets with SSE2 (no x87).
- Mandatory flags: `-ffp-contract=off`; fast-math flags rejected by the build
  ([spec/02](../spec/02-numerics.md#build-rules-c)).
- Dependencies: nlohmann/json (MIT); tests with Catch2 (BSL-1.0). See
  [CONTRIBUTING.md](../CONTRIBUTING.md#dependency-license-rule).
- Runs the shared golden cases in `../tests/golden/`.
