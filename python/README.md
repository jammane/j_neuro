# j_neuro — Python implementation

**Status:** not yet implemented. See the [specification](../spec/README.md) and
[architecture notes](../docs/ARCHITECTURE.md#python).

Planned:

- Python ≥ 3.11, package `jneuro`, NumPy (BSD-3-Clause) as the only runtime dependency;
  pytest (MIT) for tests.
- IEEE-754 division everywhere (no `ZeroDivisionError`), explicit `DEPTH` range checks
  ([spec/02](../spec/02-numerics.md)).
- Runs the shared golden cases in `../tests/golden/`.
