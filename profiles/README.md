# Mutation profiles

Ready-made profiles (spec/09 profile JSON). Load one with `jneuro.io.profile_from_json` (Python)
or `jneuro::profile_from_json` (C++), or pass it to a trainer or the expansion test with
`--profile FILE`.

| File | What it is |
|---|---|
| `expansion.json` | The settings that grew nets with width spreading from the inputs through the middle (expansion tests, Nov 2026): parameter edits on up to 20% of neurons (m: 5%), op changes up to 1%, exactly one growth attempt in 50% of generations for 80% of nets, no removal phase, constant folding on, and the library's default growth weights. |
| `training.json` | A training profile (high/low runs, Nov 2026): edits up to 20% (m: 5%), op changes up to 1%, one exact growth attempt in 10% of generations for 10% of nets (type swap for the others), growth methods weighted 50/50 between the original set (slot fill 40, v_insert 30, s_insert 20, m_after_m 10) and the current defaults, random new-node depths, m slot edits biased 75% add / 25% cut with live branches allowed to be cut, constant folding. |
