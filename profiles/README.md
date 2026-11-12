# Mutation profiles

Ready-made profiles (spec/09 profile JSON). Load one with `jneuro.io.profile_from_json` (Python)
or `jneuro::profile_from_json` (C++), or pass it to a trainer or the expansion test with
`--profile FILE`.

| File | What it is |
|---|---|
| `expansion.json` | The settings that grew nets with width spreading from the inputs through the middle (expansion tests, Nov 2026): parameter edits on up to 20% of neurons (m: 5%), op changes up to 1%, exactly one growth attempt in 50% of generations for 80% of nets, no removal phase, constant folding on, and the library's default growth weights. |
