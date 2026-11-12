# High/low trainer (C++ example)

Evolves nets that guess the next card of a shoe, using `jneuro::Population`. Unlike
blackjack, the cards already seen matter: the shoe is dealt without replacement, so a net
that keeps track of history (s_neurons) can shift the odds in its favor.

```
cmake --build build -j2 --target highlow_trainer
build/highlow_trainer                      # runs until Ctrl+C, then saves
build/highlow_trainer --resume runs/highlow-<timestamp>/population.jpnet
```

Options: `--out DIR`, `--seed N` (shoe stream), `--resume FILE.jpnet`, `--shoes N` (default 4),
`--path-mix N` (default 2), `--in-qty-max N` (odd, ≥ 3, default 3: m_neuron slots and the
longest s_neuron history, so how far back a `lag` s_neuron can look), `--stop-at N` (stop and
save after N generations), `--edit PCT`, `--op PCT`, `--grow-gen PCT`, `--grow-net PCT`,
`--tap-weight W` (optional overrides of the profile), `--profile FILE` (default
`profiles/expansion.json`), `--console-every SECONDS`.

## Game

- A fully shuffled 5-deck shoe: 260 cards, ranks 2–10, J, Q, K, A (ace high), 4 suits.
- Dealt from the first card to the last. Before each next card (259 guesses per shoe) the net
  sees the current card and guesses the next one:
  - **Inputs (2):** rank 2–14 and suit 0–3 of the current card.
  - **Output 1, suit:** clamped to 0–3 and rounded up. Correct: **3 points**.
  - **Output 2, higher/lower/equal:** the nearest of −1 (lower), 0 (equal), 1 (higher), i.e.
    ≤ −0.5 lower, ≥ 0.5 higher, otherwise equal. Correct higher or lower: **1 point**;
    correct equal: **6 points**; a wrong guess scores 0.
- The net is reset at the start of each shoe only, so its s_neuron history spans the shoe.
- Why 6: ignoring cards already dealt, equal is right 19/259 = 7.3% of the time, so always
  guessing equal pays 0.44 points per guess: less than a random higher/lower guess (0.46) and
  far less than the better of the two (0.71 on average). Equal pays only when counting shows
  the current rank is likely to repeat (6 × p > 1 × the better higher/lower chance).

## Training loop

- 40 nets; every net plays the same `--shoes` shoes each generation; top 10 kept, each
  mutated 3 times; ties at the cut broken by sudden-death shoes, then at random after 10.
- Mutation: the profile file (default `profiles/expansion.json`: edits up to 20%, op changes up
  to 1%, one exact growth attempt in 50% of generations for 80% of nets, constant folding, the
  library's default growth weights); `path_mix` 2, `max_run` 3.
- **Baseline:** the best no-memory strategy: always the same suit, and for each current rank
  the relation guess with the highest expected points. Beating it means the net is using the
  cards already seen.

## Output (in `runs/highlow-<timestamp>/`)

| File | Contents |
|---|---|
| `log.csv` | Per generation, points per shoe: top-10 average, best, population average, baseline; suit and higher/lower/equal accuracy %, how often "equal" is guessed and how often those guesses hit (best net and top 10); top-10 neuron counts; sudden-death shoes; neurons added by growth; elapsed seconds |
| `best.jnet` | Best net of the most recent console update |
| `population.jpnet` + `population_nets/` | Saved every 1000 generations and on exit |
| `run_info.txt` | Seeds and settings of each start |
