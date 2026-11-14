# Mutation profiles

Ready-made profiles (spec/09 profile JSON). Load one with `jneuro.io.profile_from_json` (Python)
or `jneuro::profile_from_json` (C++), or pass it to a trainer or the expansion test with
`--profile FILE`.

| File | What it is |
|---|---|
| `expansion.json` | The settings that grew nets with width spreading from the inputs through the middle (expansion tests, Nov 2026): parameter edits on up to 20% of neurons (m: 5%), op changes up to 1%, exactly one growth attempt in 50% of generations for 80% of nets, no removal phase, constant folding on, and the library's default growth weights. |
| `training.json` | A training profile (high/low runs, Nov 2026): edits up to 20% (m: 5%), op changes up to 1%, growth of max(1, 1% of live neurons) neurons (`fill`: skipped attempts are retried) in 10% of generations for 10% of nets (type swap for the others), growth methods weighted 50/50 between the original set (slot fill 40, v_insert 30, s_insert 20, m_after_m 10) and the current defaults, random new-node depths, m slot edits biased 75% add / 25% cut with live branches allowed to be cut, constant folding. |
| `core_finder.json` | For finding a net's core size (stage 1 of two-stage training): `training.json`'s settings with the ones that make the size settle spelled out — edits may cut live branches (`prune: any`, slot edits 75% add / 25% cut), constant folding on, no removal phase, `gc_replace` off — against slow growth (max(1, 1%) in 10% of generations for 10% of nets). Trained with selection, nets settle where cuts balance growth, from above or below. |
| `train-10-10-gc.json` | `training.json` with `gc_replace` on: growth of max(1, 1% of live neurons) in 10% of generations for 10% of nets, no removal phase, and every neuron a mutation loses (cut branches collected, folding's leftovers) grown back elsewhere, so nets keep their size between growth events. |
| `train-15-15-gc.json` | The same with growth in 15% of generations for 15% of nets. |
| `bulk.json` | Growth only, for bulking a net up before a training pass: every other phase off, constant folding off, and `growth_max_pct` is the total to add: the net gains max(1, 10% of its live neurons) neurons, within rounding (`fill` budget). Within that total, links from live sources first fill up to 80% of each m_neuron's empty slots (`open_slot_fill_pct`, m_neurons near the inputs are skipped when the mix rule rejects them); the rest is growth by v_insert (no inversion) and m_insert, whose new m_neurons are filled the same way. A small bulk can therefore be all links. Links change the net's outputs; with `open_slot_fill_pct` 0 the bulk leaves them unchanged. Set `growth_max_pct` to the percentage you want (40 for +40%). Pass it to `mutate` (or `Population::mutate`) in place of the training profile. |

Any profile can keep its nets' size with `"gc_replace": "on"`: every neuron a mutation removes
(removal, its cascades, and branches garbage collection deletes after cuts) is grown back elsewhere
with the profile's own growth methods, so only phase 3 growth changes the size (spec/06
replacement). The `-gc` profiles have it on; the others have it off.

**Two-stage training.** A workflow that has worked well:

1. **Find the core** with `core_finder.json` (the same settings as `training.json`, `gc_replace`
   off), trained with selection. Its live cuts
   (`prune: any`, slot edits biased 25% to cuts) let garbage collection drop whole branches, and
   constant folding trims constant ones; against its slow growth the net size settles where the two
   balance. The level is an attractor: chess nets climbed from 23 to about 38 neurons, and after a
   200% bulk fell back from about 120 to about 45 within 3,000 generations. (`expansion.json` cannot
   find a core: with `prune: static` nothing is ever cut, so nets only grow.)
2. **Train** from that core with slow, gated growth and replacement: `train-15-15-gc.json` (or
   `train-10-10-gc.json`), whose `gc_replace` grows back what cuts remove, so selection reshapes the
   net while its size holds.
