# j_neuro Specification

**Version:** 0.1.0-draft

This specification is **normative** for both the C++ and the Python implementation. Where an
implementation and this document disagree, the implementation is wrong.

## Contents

| # | Document | Covers |
|---|---|---|
| 01 | [Concepts](01-concepts.md) | Glossary and the overall model |
| 02 | [Numerics](02-numerics.md) | binary64, canonical values, operation rules, build rules |
| 03 | [Net structure](03-net-structure.md) | Per-net parameters, nodes, depth, invariants, minimal net |
| 04 | [Neurons](04-neurons.md) | v / s / m neurons and the shared `aggregate` |
| 05 | [Evaluation](05-evaluation.md) | Order, state, reset, priority invariance |
| 06 | [Mutation](06-mutation.md) | Mutation profile, phases, every operator |
| 07 | [Removal](07-removal.md) | Candidates, cascades, immunity, garbage collection, IDs |
| 08 | [RNG](08-rng.md) | PRNG and every random distribution, exactly |
| 09 | [Serialization](09-serialization.md) | Genome, profile and run-log JSON formats |
| 10 | [Conformance](10-conformance.md) | Shared golden-test format |

## Normative language

The words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT** and **MAY** are used as in
RFC 2119.

## Conformance

An implementation conforms to a spec version when:

1. Given the same genome and input sequence, it produces **bit-identical** outputs
   ([05](05-evaluation.md)).
2. Given the same genome, mutation profile, generation number and RNG seed, mutation produces
   a **byte-identical** serialized genome ([06](06-mutation.md), [09](09-serialization.md)).
3. It passes every golden case of that version ([10](10-conformance.md)).
4. Results do not depend on the net's `priority` ([03](03-net-structure.md)).

## Versioning

`MAJOR.MINOR.PATCH`:

- **PATCH**: wording/clarification, no behavior change.
- **MINOR**: new behavior that leaves existing genomes, profiles and golden results valid.
- **MAJOR**: any change that alters results for an existing genome, profile or seed.

Genome and profile files record the spec version they were written under
([09](09-serialization.md)).

## Out of scope for v0.1

Population management, fitness and selection, and crossover are not specified yet. Mutation
is defined as a function from one parent genome to one offspring genome.
