# Expansion test (C++ example)

How does a mutation profile grow a net on its own? `expansion_test` starts several independent
copies (lineages) of a minimal net and mutates each one generation after generation. There is
no game, no scoring and no selection: it measures only what mutation does.

```
cmake --build build -j2 --target expansion_test
build/expansion_test --profile profiles/expansion.json --generations 5000
```

Each generation it logs, averaged over the lineages (`log.csv`):

- how often each mutation operation was used (every operation in `mutation_operations()`);
- neurons, input-to-output paths (log10), and **width**: connections crossing 10 evenly spaced
  depth cuts, from the inputs to the outputs;
- **open slots**: empty m_neuron slots, in total and by depth band.

At the end, `effects.txt` lists for each operation the average change in paths, neurons and
mid-depth width in lineage-generations where it was used versus where it was not.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--profile FILE` | built-in settings | A saved profile (e.g. `profiles/expansion.json`); the flags below still apply on top of it |
| `--patch JSON` | | A profile merge patch, e.g. `'{"overall":{"width_peak":0.4}}'` |
| `--lineages N`, `--generations N`, `--seed N` | 20, 5000, 1 | |
| `--inputs N`, `--outputs N`, `--in-qty-max N`, `--path-mix N` | 9, 1, 3, 2 | The starting net (default: the blackjack shape) |
| `--edit`, `--op`, `--removal`, `--depth-shift` (%) | 20, 1, 0, 5 | Built-in settings when no `--profile` is given |
| `--growth PCT` / `--growth-exact PCT` | one exact attempt | "Up to" or exact growth budget |
| `--grow-gen PCT`, `--grow-net PCT` | | Set the profile's `growth_gen_prob` / `growth_net_prob` |
| `--tap-weight W`, `--clone-weight W` | | Override those growth weights |
