# Contributing

## The mirroring rule

j_neuro has two implementations, C++ and Python, that must behave **identically, bit for
bit**. The [specification](spec/README.md) is the single source of truth.

Every behavioral change follows this order, **in the same change set**:

1. Update the spec (and bump its version, see [spec/README.md](spec/README.md#versioning)).
2. Implement it in **both** `cpp/` and `python/`.
3. Add or update golden cases in `tests/golden/` ([spec/10](spec/10-conformance.md)).
4. Both test suites pass, including all golden cases.

A change that touches only one language is accepted only if it does not alter observable
behavior (refactoring, performance work within the rules of the active priority, docs).

Record any non-obvious design choice in [docs/DECISIONS.md](docs/DECISIONS.md).

## Numeric rules for implementers (summary)

- IEEE-754 binary64 only. No `float`/`float32`, no `long double`.
- C++: never `-ffast-math`, `-Ofast` or `-funsafe-math-optimizations`; always
  `-ffp-contract=off`; 64-bit targets with SSE2 (no x87). The build fails otherwise.
- Never use `std::*_distribution`, `numpy.random`, or `random` — only the spec's RNG.
- Never reorder arithmetic (`x/w` is not `x*(1/w)`; sums go in the specified order).
- Python: never let `ZeroDivisionError` happen; division follows IEEE (see
  [spec/02](spec/02-numerics.md)).

## Dependency license rule

j_neuro is licensed for noncommercial use and sold commercially. To keep that simple,
**a dependency is allowed only if its license requires nothing beyond attribution.**

| Allowed | Not allowed |
|---|---|
| MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, BSL-1.0 (Boost), CC0, Unlicense, public domain | Apache-2.0 (NOTICE and change-marking duties), MPL-2.0 (source-availability duty — e.g. **Eigen**), LGPL, GPL, AGPL, EPL, CDDL, any noncommercial or source-available license, anything without a license |

Also:

- No neural-network or machine-learning libraries of any license.
- Check the license of the dependency's **own** dependencies that would ship with ours.
- Dependencies are never copied into this repository.
- Adding a dependency requires, in the same change, an entry with its full copyright notice
  and license text in [CREDITS.md](CREDITS.md).

## Contributor License Agreement

Because j_neuro is also sold under commercial licenses, the maintainer must hold the rights
to every contribution. **Before a contribution can be merged, the contributor must agree to a
Contributor License Agreement (CLA)** granting the maintainer a perpetual, irrevocable,
worldwide right to use, modify, sublicense and relicense the contribution, including under
commercial terms. Contributors keep their own copyright.

The CLA text and signing process will be published before outside contributions are
accepted. Until then, please open an issue to discuss rather than a pull request.

## Recognition

Every merged contributor is listed in [CREDITS.md](CREDITS.md#contributors).
