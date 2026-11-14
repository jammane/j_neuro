# Chess (C++ example)

64 nets learn chess by playing each other. No opening books, no evaluation function, no search:
each net only scores positions, and whoever loses changes.

## Build and run

```
cmake -S cpp -B build && cmake --build build --target chess_trainer chess_play
build/chess_trainer                          # until Ctrl+C (stops after the current match, saves)
build/chess_trainer --resume runs/chess-…    # continue a run
build/chess_play runs/chess-…                # play the latest round's best net yourself
build/chess_trainer --perft                  # check the rules engine against known move counts
```

The trainer runs one game at a time at the lowest CPU priority (nice 19), so the machine stays
usable.

## Tournament

- **64 players, round robin.** Each round every pair meets once (2,016 matches), in a shuffled order.
- **A match** is up to 3 games. White always moves first; colors switch after each game. A
  decisive game ends the match and **the loser is mutated from itself** (its mutated self takes its
  slot). After **3 draws both are mutated**. Players never copy each other, so each slot keeps its own
  lineage and can develop its own style.
- Each round's numbers go to `rounds.csv`: games, decisive games, how games ended, mean length,
  mutations, mean net size. `state.txt` keeps the round and each slot's current net,
  `population.jpnet` the nets, `best.jnet` the round's player with the most wins.
- `--baseline N` (default 0, off) also plays the round's best player 2N games against a random mover
  and 2N against a greedy capturer each round, as a fixed yardstick; off by default so that no cycles
  go to fixed opponents until the players are worth measuring.

Options: `--out DIR`, `--seed N` (pairing order), `--pop-seed N`, `--resume DIR`, `--profile FILE`
(default `profiles/expansion.json`), `--rounds N`, `--baseline N`, `--q N` (`in_qty_max` of new nets).

## Rules

`chess.hpp` is a complete rules engine, checked by perft against the standard test positions
(start, "Kiwipete", and others, up to 4.9 million positions deep).

- Only legal moves can be chosen: castling (not out of, through or into check; king and rook unmoved),
  en passant, and promotion to queen, rook, bishop or knight (four separate choices); nothing that
  leaves the mover's king in check.
- **Checkmate** wins. Draws: **stalemate**; **insufficient material**, at once (king against king, a
  lone knight or bishop, or only bishops all on one square color); **21 moves by each side** (42
  plies) without a pawn move or a capture; the **5th occurrence** of a position (same pieces, side
  to move, castling rights, and en passant only when a capture is really possible). No resigning,
  and no other length limit.

## How a net plays

For each legal move the net is shown the position after it and gives one number; the highest is
played (the first on ties). It never sees a move list, so it cannot make an illegal move. Each
position is scored on its own (s_neuron state is reset), so the net is a position evaluator.

**Inputs (66), from the player's seat** (the board rotated 180° for black, as across a table):

| Input | Meaning |
|---|---|
| 0–63 | One per square: own pieces +1 to +6 (pawn, knight, bishop, rook, queen, king), the opponent's −1 to −6, empty 0; a rook that can still castle ±7; a pawn that can be taken en passant ±8 |
| 64 | Plies since the last pawn move or capture (42 is a draw) |
| 65 | How many times this position has now occurred (5 is a draw) |

The last two give the nets draw awareness.

## Playing a net yourself

```
build/chess_play runs/chess-…                     # the latest best.jnet, you play white
build/chess_play runs/chess-… --slot 17 --color black
build/chess_play some_net.jnet
```

Moves are from-square and to-square: `e2e4`, `g1f3`; castle with the king's move (`e1g1`); promote
with the piece letter (`e7e8q`). `moves` lists your legal moves, `quit` ends. The board is shown from
your side. It only reads the saved files, so you can play while training runs.

## First rounds

A short trial (seeds 5 and 6): about 46 s per round on a 4-core laptop, about 5,700 games each.
New nets mostly shuffle pieces: about 91% of games end in fivefold repetition and 4% in checkmate,
so most matches end with both players mutated.
