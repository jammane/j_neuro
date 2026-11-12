# Blackjack trainer (C++ example)

Evolves nets that play blackjack, using `jneuro::Population`. The library only supplies nets
and operations; this program is the trainer: it plays the games, scores the nets, and
decides which to keep and mutate.

## Build and run

```
cmake -S cpp -B build && cmake --build build --target blackjack_trainer
build/blackjack_trainer                      # runs until Ctrl+C, then saves
build/blackjack_trainer --resume runs/blackjack-…/population.jpnet --out runs/blackjack-…
```

Options: `--out DIR`, `--seed N` (deal stream), `--resume FILE.jpnet`, `--alter PCT`
(default 22), `--grow PCT` (default 12), `--console-every SECONDS` (default 2).

## Game

- Cards `1 2 3 4 5 6 7 8 9 10 J Q K A`; J, Q, K count 10; A counts 1 or 11. Suits ignored.
- Each deal shuffles a fresh 5-deck shoe (280 cards) and draws 7 cards for the player and 7
  for the dealer up front.
- **Net inputs (9):** the dealer's 2 cards, then 7 player slots: revealed cards by value
  (A = 11), unrevealed slots 0. Two player cards start revealed.
- **Net output (1):** clamped to 0–3 and rounded up: 0 stand, 1 hit, 2 double down, 3 split.
- **Hit:** reveal the next card and decide again; a bust or 7 revealed cards ends the hand.
- **Double down:** reveal one card, then stand; a win scores 2.
- **Split:** only with exactly 2 revealed cards of the same rank (and at most 4 hands);
  otherwise it counts as stand. The first card moves to a new hand; the existing hand gets
  one new card at the end (back to 7) and the new hand is filled to 7 from the shoe. Hands
  are played one after the other, each starting with 2 revealed cards.
- **Dealer:** after all player hands, reveals cards while below 17 (stands on 17, including
  soft 17), up to 7 cards.
- **Score:** each winning hand scores 1 (2 if doubled); pushes and losses score 0.

## Training loop

- 40 nets; every net plays the same 50 deals each generation.
- The top 10 are kept, the other 30 removed, and each survivor is mutated 3 times.
- Mutation caps: 22% of neurons altered (edits and operation changes), 12% added/removed.
  Caps are percentages of the net's neurons, drawn "up to" each generation.
- Each net's s_neuron history is reset at the start of every deal.

## Output (in `runs/blackjack-<timestamp>/`)

| File | Contents |
|---|---|
| `log.csv` | One row per generation: top-10 average score, best score, population average, and top-10 averages of nodes, neurons, v/s/m neurons, elapsed seconds |
| `best.jnet` | Best net of the most recent console update |
| `population.jpnet` + `population_nets/` | Saved every 1000 generations and on exit; resume with `--resume` |
| `run_info.txt` | Seeds and settings of each start |
