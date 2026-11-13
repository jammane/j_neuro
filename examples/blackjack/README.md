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

Options: `--deals N` (deals per generation, default 200), `--out DIR`, `--seed N` (deal stream), `--pop-seed N` (population seed; with `--seed` it repeats a run exactly only under a profile with `"rng_mode": "shared"`), `--resume FILE.jpnet`, `--rules new|old`
(default new), `--alter PCT`
(default 22), `--grow PCT` (default 12), `--big-alter PCT` (default 5), `--big-grow PCT`
(default 1.5) for `--rules old`; `--stop-at N` (stop and save after N generations); `--profile FILE` and the overrides `--edit`, `--op`, `--grow-gen`,
`--grow-net`, `--tap-weight` for `--rules new`; `--console-every SECONDS` (default 2).

## Game

- Cards `2 3 4 5 6 7 8 9 10 J Q K A`; J, Q, K count 10; A counts 1 or 11, whichever is better
  for the hand. Suits ignored.
- Each deal shuffles a fresh 5-deck shoe (260 cards) and draws every card up front, in a
  fixed order: 7 for the player, 7 for the dealer, then a block of 7 for each possible split.
  Split *k* always uses block *k*, so every net that splits sees the same cards.
- **Net inputs (9):** the dealer's face-up card, the hole card (always 0: it is face down), then 7
  player slots: revealed cards by value
  (A = 11), unrevealed slots 0. Two player cards start revealed.
- **Net output (1):** clamped to 0–3 and rounded up: 0 stand, 1 hit, 2 double down, 3 split.
- **Hit:** reveal the next card and decide again; a bust or 7 revealed cards ends the hand.
- **Double down:** reveal one card, then stand; the hand's stake is doubled (wins +2, losses −2).
- **Split:** only with exactly 2 revealed cards of the same rank (and at most 4 hands);
  otherwise it counts as stand. The first card moves to a new hand; the existing hand gets
  one new card at the end (back to 7) and the new hand is filled to 7, all from the split's
  block. Hands
  are played one after the other, each starting with 2 revealed cards.
- **Dealer:** after all player hands, reveals cards while below 17 (stands on 17, including
  soft 17), up to 7 cards.
- **Score:** real blackjack payout per hand: win +1, loss −1, push 0; a doubled hand pays ±2. A
  player bust loses even if the dealer busts too. A **blackjack** (the original, unsplit two cards
  totalling 21: an ace and a ten-value card) pays 1.5, and so does a **5-card charlie** (five
  cards without busting, which wins against any dealer hand). A deal's score is the sum over its
  hands, so it can be negative.

## Training loop

- 40 nets; every net plays the same 200 deals each generation.
- The top 10 are kept, the other 30 removed, and each survivor is mutated 3 times.
- **Ties at the cut:** if nets tied on score straddle 10th place, they play sudden-death
  deals (single elimination): each round all of them play one new deal; nets scoring below
  the rest are out, a group that fits in the open places goes through, and the group still
  straddling the line plays again. Sudden-death deals don't count toward scores. Nets that
  play identically never separate, so after 10 sudden-death deals the remaining places are
  picked at random.
- Nets start from the library's minimal net with `path_mix` 2: every input-to-output path
  passes through at least 2 v, 2 s and 2 m neurons, and mutation keeps it that way.
- **Mutation, new rules (default, `--rules new`):** the profile file `--profile` (default
  `profiles/expansion.json`: parameter edits on up to 20% of neurons, op changes up to 1%,
  exactly one growth attempt in 50% of generations for 80% of nets, no removal phase, constant
  folding, the library's default growth weights). `--edit`, `--op`, `--grow-gen`, `--grow-net`
  (in %) and `--tap-weight` override its values when given.
- **Mutation, old rules (`--rules old`):** up to 22% of neurons altered (edits and operation
  changes), up to 12% added and removed; once a net has more than 50 neurons, its offspring
  use 5% and 1.5% instead (`--alter`, `--grow`, `--big-alter`, `--big-grow`).
- "Up to" caps are percentages of the net's neurons, drawn at random each time; a nonzero cap
  is never below 1 neuron, so small nets can always mutate.
- Each net's s_neuron history is reset at the start of every deal.

## Output (in `runs/blackjack-<timestamp>/`)

| File | Contents |
|---|---|
| `log.csv` | One row per generation: top-10 average score, best score, population average, and top-10 averages of nodes, neurons, v/s/m neurons; `unwinnable` (deals where no line of play ends with a positive score, found by trying every decision sequence); `best_win_pct_winnable` and `top10_win_pct_winnable` (deals with a positive score ÷ winnable deals, for the best net and the top-10 average); `hitrule_score`, `hitrule_win_pct_winnable`, `doublerule_score`, `doublerule_win_pct_winnable` (the two fixed rules on the same deals); `sudden_death_deals` (tiebreak deals played that generation); `growth_added` (neurons added by growth); elapsed seconds |
| `best.jnet` | Best net of the most recent console update |
| `population.jpnet` + `population_nets/` | Saved every 1000 generations and on exit; resume with `--resume` |
| `run_info.txt` | Seeds and settings of each start |

## Analysis

`blackjack_analyze POPULATION.jpnet [--deals N] [--seed N]` plays every net of a saved population,
the simple hit rule (hit below 12, or below 17 vs a dealer 7+) and the simple double rule (the hit
rule, plus double on a two-card 10 or 11 when the dealer's card is lower) on the same fresh deals
(default 100,000). It prints each net's points per deal, win % of winnable deals and mix of actions,
and for the best net its structure and its decision table (most common action per player total and
dealer card) beside the double rule's.

`--bulk PROFILE` (with `--resume`) mutates every resumed net once with that profile (e.g.
`profiles/bulk.json`) before training continues, and writes `removals.csv`: each generation, the
hidden neurons removed by mutation (counted in every offspring, before selection) and the neurons in
the top 10 and the whole population, split by birth into originals (in the resumed nets), bulk
(added by `--bulk`) and new (added by later evolution). A profile with `growth_max_pct` 0 gives a
control run with the same accounting.
