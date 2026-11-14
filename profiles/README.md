# Mutation profiles

Ready-made profiles (spec/09 profile JSON). Load one with `jneuro.io.profile_from_json` (Python)
or `jneuro::profile_from_json` (C++), or pass it to a trainer or the expansion test with
`--profile FILE`.

| File | What it is |
|---|---|
| `expansion.json` | The settings that grew nets with width spreading from the inputs through the middle (expansion tests, Nov 2026): parameter edits on up to 20% of neurons (m: 5%), op changes up to 1%, exactly one growth attempt in 50% of generations for 80% of nets, no removal phase, constant folding on, and the library's default growth weights. |
| `training.json` | A training profile (high/low runs, Nov 2026): edits up to 20% (m: 5%), op changes up to 1%, growth of max(1, 1% of live neurons) neurons (`fill`: skipped attempts are retried) in 10% of generations for 10% of nets (type swap for the others), growth methods weighted 50/50 between the original set (slot fill 40, v_insert 30, s_insert 20, m_after_m 10) and the current defaults, random new-node depths, m slot edits biased 75% add / 25% cut with live branches allowed to be cut, constant folding. |
| `train-10-10-gc.json` | `training.json` with `gc_replace` on: growth of max(1, 1% of live neurons) in 10% of generations for 10% of nets, no removal phase, and every neuron a mutation loses (cut branches collected, folding's leftovers) grown back elsewhere, so nets keep their size between growth events. |
| `train-15-15-gc.json` | The same with growth in 15% of generations for 15% of nets. |
| `bulk.json` | Growth only, for bulking a net up before a training pass: every other phase off, constant folding off, and `growth_max_pct` is the total to add: the net gains max(1, 10% of its live neurons) neurons, within rounding (`fill` budget). Within that total, links from live sources first fill up to 80% of each m_neuron's empty slots (`open_slot_fill_pct`, m_neurons near the inputs are skipped when the mix rule rejects them); the rest is growth by v_insert (no inversion) and m_insert, whose new m_neurons are filled the same way. A small bulk can therefore be all links. Links change the net's outputs; with `open_slot_fill_pct` 0 the bulk leaves them unchanged. Set `growth_max_pct` to the percentage you want (40 for +40%). Pass it to `mutate` (or `Population::mutate`) in place of the training profile. |

Any profile can keep its nets' size with `"gc_replace": "on"`: every neuron a mutation removes
(removal, its cascades, and branches garbage collection deletes after cuts) is grown back elsewhere
with the profile's own growth methods, so only phase 3 growth changes the size (spec/06
replacement). The `-gc` profiles have it on; the others have it off.

**Two-stage training.** A workflow that has worked well:

1. **Build a core** with `expansion.json` (`gc_replace` off), trained with selection. Its fast
   growth levels off under training: cuts, garbage collection and selection trim the nets back,
   and the neuron count settles at a core worth training.
2. **Train** from that core with slow, gated growth: `training.json`-style 10%/10% gating with +1%
   growth and no removal phase, or `train-15-15-gc.json` (or `train-10-10-gc.json`), whose
   replacement keeps cuts from shrinking the nets while selection reshapes them.
