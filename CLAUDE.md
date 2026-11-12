# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

j_neuro is an evolutionary neural-network library (nets grow and change by mutation, not gradients),
implemented **twice**, in C++ (`cpp/`) and Python (`python/`). The two must behave **bit for bit
identically**. `spec/` is normative: where code and spec disagree, the code is wrong.

## Commands

C++ (C++20, CMake; needs nlohmann/json and Catch2 v3). The machine has little RAM, so always build
with `-j2`, never a bare `-j`:

```
cmake -S cpp -B build
cmake --build build -j2
ctest --test-dir build
build/jneuro_tests "bypass exact"          # one Catch2 TEST_CASE by name
build/jneuro_tests -# "[#test_removal]"   # every test in one file
cmake --build build -j2 --target jneuro_tests      # tests only; skip the examples
```

`-DJNEURO_BUILD_EXAMPLES=OFF` skips the example programs. `JNEURO_TEST_SOURCES` is a cache variable
listing the test files to compile, so you can narrow it to save memory. `build-dbg/` is a debug
build directory.

Python (≥ 3.11, NumPy only):

```
cd python && python3 -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/pytest
.venv/bin/pytest tests/test_removal.py::test_name
```

Golden cases: `python/.venv/bin/python tools/make_impl_golden.py [create|mutate|all]` (run from the
repo root) regenerates the implementation-produced cases from Python. C++ must then reproduce them
byte for byte. `tools/make_handmade_golden.py` writes the hand-derived and invalid cases.

## Change rules (from CONTRIBUTING.md)

Every behavioral change lands in one change set, in this order:

1. Update the spec and bump its version (`spec/README.md`, Versioning).
2. Implement it in **both** languages.
3. Add or update golden cases in `tests/golden/`.
4. Make both suites pass.

Keep the spec version in step with `kSpecVersion` and friends in `cpp/include/jneuro/io.hpp` and its
Python counterpart. Record non-obvious design choices in `docs/DECISIONS.md` (tables grouped by
version). A change to only one language is acceptable only when it does not alter observable
behavior.

Numeric rules:

- binary64 only.
- Never use fast-math. CMake refuses it and forces `-ffp-contract=off`.
- Never reorder arithmetic: `x/w` is not `x*(1/w)`, and sums follow the specified order.
- Never use `std::*_distribution`, `numpy.random` or `random`. Every random draw goes through the
  spec's RNG (`rng.hpp` / `rng.py`), in exactly the order the spec lists them. An extra or
  reordered draw breaks cross-language replay and the golden cases.

Dependencies: only attribution-only licenses (MIT, BSD, ISC, Zlib, BSL-1.0, …). No Apache-2.0, MPL
(e.g. Eigen) or GPL-family licenses. No ML libraries of any kind. Every dependency needs a CREDITS.md
entry.

## Architecture

Both languages use the same module map, so the code compares line by line (`docs/ARCHITECTURE.md`).
Each C++ header in `cpp/include/jneuro/` has a matching Python module in `python/jneuro/`. C++ is
header-mostly; `cpp/src/` holds only the non-template parts.

- **Net model** (`net.hpp`/`net.py`, spec/03–05): a net is a list of nodes, each with a depth. Ids
  `0..n_inputs-1` are the inputs (depth 0), the next `n_outputs` ids are the outputs (depth `D_MAX`),
  and hidden nodes follow. Every non-input node has exactly one neuron (`VNeuron`, `SNeuron` or
  `MNeuron`, a `std::variant` in C++). A neuron reads only strictly shallower nodes, so evaluation is
  a stable sort by depth.
- **Depth type**: C++ templates everything on the depth type `D` (`Net<std::uint16_t|32|64>`).
  `AnyNet` in `io.hpp` is the variant over all three. Python uses ints plus range checks.
- **Priority** (`SPEED`, `MEMORY` or `BALANCED`) changes only caching and storage. It must never
  change results.
- **Structural invariants** (`analysis.hpp`/`analysis.py`) are re-checked after every structural
  edit:
  - path rule: every input reaches every output;
  - mix rule: every path has at least `path_mix` v, s and m neurons;
  - run rule: no path has more than `max_run` 1:1 or many:1 neurons in a row;
  - no dead nodes.

  `Net::validate` (`net_build.hpp`) enforces all of them on load. Mutation operations follow the
  same pattern each time: change the net, check the rules, undo and skip the attempt if one broke.
- **Mutation** (`mutate.hpp`/`mutate.py`, spec/06): phases run in a fixed order: op changes, edits,
  growth (eleven methods), removal, depth shifts, then folding, GC and compaction. Removal
  (`removal.hpp`, spec/07) marks nodes as tombstones in a `Removed` mask. Ids stay stable until
  `compact` renumbers them in phase 6.
- **Profiles** (`profile.hpp`, `io.py`): JSON mutation profiles. Ready-made ones live in `profiles/`.
- **Population** (`population.hpp`/`population.py`, spec/11): nets by stable id, plus a journal of
  operations, so a run can be replayed exactly in either language.
- **Modification** (`modify.hpp`/`modify.py`, spec/12): adds and removes inputs and outputs of an
  existing net, keeping every rule.
- **Files**: `.jnet` holds one net and `.jpnet` holds a population plus `<name>_nets/`. Both are
  binary and end with a SHA-256 checksum (spec/09). Both languages must write byte-identical files.
- **Tests**: each language has per-module unit tests. Both suites also run every JSON case in
  `tests/golden/` (format in spec/10) and require an exact match.

## Examples

The programs in `examples/` are C++ only and are built with the library: `blackjack_trainer`,
`blackjack_analyze`, `highlow_trainer` and `expansion_test`. They write to `runs/`, run until Ctrl+C
or `--stop-at N`, and resume with `--resume FILE.jpnet`. Training runs are long: don't start one
unless the user asks.
