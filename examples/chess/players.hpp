// Playing chess games for the chess example: game loop, the nets' inputs and move choice, and the
// fixed opponents. Shared by chess_trainer and chess_play, so both play by the same rules.
#pragma once

#include "chess.hpp"

#include <jneuro/jneuro.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <map>
#include <string>
#include <vector>

namespace chess {

using jneuro::Net;
using jneuro::Rng;
using D = std::uint32_t;
constexpr int kInputs = 66;

enum class End { Checkmate, Stalemate, Insufficient, Quiet, Repetition };
inline const char* end_name(End e) {
    switch (e) {
        case End::Checkmate: return "checkmate";
        case End::Stalemate: return "stalemate";
        case End::Insufficient: return "insufficient";
        case End::Quiet: return "21 moves";
        case End::Repetition: return "repetition";
    }
    return "?";
}

struct Game {
    int winner = 0;  // +1 white, -1 black, 0 draw
    End end = End::Checkmate;
    int plies = 0;
};

using Reps = std::map<std::string, int>;
using Chooser = std::function<Move(const Position&, const std::vector<Move>&, const Reps&)>;

using Observer = std::function<void(const Position&, const std::vector<Move>&)>;

/// A game from `from` (the start position by default). `observe`, if given, sees every position
/// that still has a move to make, with its legal moves.
inline Game play(const Chooser& white, const Chooser& black, const Position& from = Position::start(),
                 const Observer& observe = nullptr) {
    Position p = from;
    Reps reps;
    reps[key(p)] = 1;
    Game g;
    for (;;) {
        const auto moves = legal_moves(p);
        if (moves.empty()) {
            if (in_check(p)) { g.winner = -p.side; g.end = End::Checkmate; }
            else g.end = End::Stalemate;
            return g;
        }
        if (insufficient(p)) { g.end = End::Insufficient; return g; }
        if (reps[key(p)] >= kRepetitions) { g.end = End::Repetition; return g; }
        if (p.quiet >= kDrawPlies) { g.end = End::Quiet; return g; }
        if (observe) observe(p, moves);
        p = make(p, (p.side > 0 ? white : black)(p, moves, reps));
        ++reps[key(p)];
        ++g.plies;
    }
}

/// The net's 66 inputs for position q, seen by `me` (+1 white, -1 black) from its seat.
inline std::vector<double> encode(const Position& q, int me, int occurrences) {
    std::vector<double> in(kInputs, 0.0);
    const bool ep = ep_available(q);
    const int ep_pawn = ep ? q.ep - 8 * q.side : -1;  // the pawn that just stepped two squares
    for (int s = 0; s < 64; ++s) {
        const int x = q.sq[static_cast<std::size_t>(s)];
        if (x == Empty) continue;
        int code = kind(x);
        if (code == Rook && ((s == 7 && (q.castle & WK)) || (s == 0 && (q.castle & WQ)) ||
                             (s == 63 && (q.castle & BK)) || (s == 56 && (q.castle & BQ))))
            code = 7;  // can still castle
        if (s == ep_pawn) code = 8;  // can be taken en passant
        const int view = me > 0 ? s : 63 - s;
        in[static_cast<std::size_t>(view)] = static_cast<double>(sign(x) == me ? code : -code);
    }
    in[64] = static_cast<double>(q.quiet);
    in[65] = static_cast<double>(occurrences);
    return in;
}

/// A net plays the legal move whose resulting position it scores highest (first on ties).
inline Chooser net_player(Net<D>& net) {
    return [&net](const Position& p, const std::vector<Move>& moves, const Reps& reps) {
        std::size_t best = 0;
        double best_v = 0;
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const Position q = make(p, moves[i]);
            const auto it = reps.find(key(q));
            net.reset_state();  // each position is scored on its own
            const double v = net.step(encode(q, p.side, (it == reps.end() ? 0 : it->second) + 1))[0];
            if (i == 0 || v > best_v) { best_v = v; best = i; }
        }
        return moves[best];
    };
}

inline Chooser random_player(Rng& r) {
    return [&r](const Position&, const std::vector<Move>& moves, const Reps&) { return moves[r.below(moves.size())]; };
}

/// Takes the most valuable piece it can (promotions count), at random among equals.
inline Chooser greedy_player(Rng& r) {
    return [&r](const Position& p, const std::vector<Move>& moves, const Reps&) {
        static const int value[7] = {0, 1, 3, 3, 5, 9, 0};
        int best = -1;
        std::vector<std::size_t> top;
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const Move& m = moves[i];
            int v = value[kind(p.sq[m.to])] + (m.flags & kEnPassant ? 1 : 0) + (m.promo ? value[m.promo] - 1 : 0);
            if (v > best) { best = v; top.clear(); }
            if (v == best) top.push_back(i);
        }
        return moves[top[r.below(top.size())]];
    };
}

}  // namespace chess

namespace chess {

// ------------------------------------------------------------------ puzzles

/// A puzzle (Lichess format): the position before the opponent's last move, and the moves from
/// there in from-to notation (e2e4, e7e8q): the opponent's move first, then the player's and the
/// opponent's replies in turn.
struct Puzzle {
    int rating = 0;
    std::string fen;
    std::vector<std::string> moves;
};

/// puzzles.csv: rating,fen,moves (moves separated by spaces); a header line is skipped.
inline std::vector<Puzzle> load_puzzles(const std::string& path) {
    std::ifstream in(path);
    std::vector<Puzzle> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto c1 = line.find(','), c2 = line.find(',', c1 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos || line.compare(0, 6, "rating") == 0) continue;
        Puzzle p;
        p.rating = std::stoi(line.substr(0, c1));
        p.fen = line.substr(c1 + 1, c2 - c1 - 1);
        std::istringstream ms(line.substr(c2 + 1));
        for (std::string m; ms >> m;) p.moves.push_back(m);
        out.push_back(std::move(p));
    }
    return out;
}

inline std::string uci(const Move& m) {
    std::string t{static_cast<char>('a' + file_of(m.from)), static_cast<char>('1' + rank_of(m.from)),
                  static_cast<char>('a' + file_of(m.to)), static_cast<char>('1' + rank_of(m.to))};
    if (m.promo) t += "  nbrq"[m.promo];
    return t;
}

/// True if the player finds every one of its moves (any checkmating move counts where the
/// solution mates, as on Lichess).
inline bool solve(const Chooser& player, const Puzzle& pz) {
    Position p = Position::fen(pz.fen);
    Reps reps;
    for (std::size_t i = 0; i < pz.moves.size(); ++i) {
        const auto moves = legal_moves(p);
        const Move* want = nullptr;
        for (const Move& m : moves) if (uci(m) == pz.moves[i]) want = &m;
        if (!want) return false;  // malformed puzzle
        if (i % 2 == 1) {  // the player's turn
            const Move got = player(p, moves, reps);
            if (uci(got) != pz.moves[i]) {
                const Position q = make(p, got);
                return i + 1 == pz.moves.size() && legal_moves(q).empty() && in_check(q);  // another mate
            }
        }
        p = make(p, *want);
        ++reps[key(p)];
    }
    return true;
}

// ------------------------------------------------------------------ mating drills

/// The mating material of a drill: pieces the player has besides its king, against a bare king.
struct MateSet {
    const char* name;
    std::vector<int> pieces;  // kinds
};
inline const std::vector<MateSet>& mate_sets() {
    static const std::vector<MateSet> s = {{"KQ", {Queen}}, {"KR", {Rook}}, {"KRR", {Rook, Rook}},
                                           {"KQR", {Queen, Rook}}, {"KBB", {Bishop, Bishop}}};
    return s;
}

/// A random legal drill position: the player (`me`, to move) has its king and the set's pieces,
/// the defender a bare king; kings not adjacent, the defender not in check, bishops on opposite
/// colors. No castling rights, no en passant.
inline Position mate_position(const MateSet& set, int me, Rng& r) {
    for (;;) {
        Position p;
        p.side = me;
        std::vector<int> used;
        auto free_sq = [&](auto ok) {
            for (;;) {
                const int s = static_cast<int>(r.below(64));
                if (std::find(used.begin(), used.end(), s) == used.end() && ok(s)) { used.push_back(s); return s; }
            }
        };
        const int mk = free_sq([](int) { return true; });
        const int ok_ = free_sq([&](int s) { return std::abs(rank_of(s) - rank_of(mk)) > 1 || std::abs(file_of(s) - file_of(mk)) > 1; });
        p.sq[static_cast<std::size_t>(mk)] = King * me;
        p.sq[static_cast<std::size_t>(ok_)] = -King * me;
        int bishop_color = -1;
        for (int k : set.pieces) {
            const int s = free_sq([&](int x) { return k != Bishop || bishop_color < 0 || (rank_of(x) + file_of(x)) % 2 != bishop_color; });
            if (k == Bishop) bishop_color = (rank_of(s) + file_of(s)) % 2;
            p.sq[static_cast<std::size_t>(s)] = k * me;
        }
        if (!attacked(p, ok_, me)) return p;  // the defender must not start in check
    }
}

/// One drill: the player against a bare king making random legal moves. Plies to mate, or -1.
inline int mate_drill(const Chooser& player, const MateSet& set, int me, Rng& r) {
    const Position p = mate_position(set, me, r);
    Rng defender_rng(r.next_u64());
    const Chooser defender = random_player(defender_rng);
    const Game g = me > 0 ? play(player, defender, p) : play(defender, player, p);
    return g.winner == me ? g.plies : -1;
}

/// Drill scores, in thousandths of a point.
constexpr int kMateScore = 1000;       // checkmate: 1 point
constexpr int kStalemateScore = 480;   // stalemate: 0.48 (all three rings blocked, 0.44, plus 0.04)
constexpr int kRingScore[3] = {20, 10, 5};  // per blocked square at king distance 1, 2, 3

/// How boxed in the king of the side to move is, in thousandths of a point (at most 440). For each
/// square at king distance r = 1, 2, 3 from it (8, 16 and 24 squares, counting squares off the
/// board): kRingScore[r-1] if the king has no legal path there in r moves, the board otherwise
/// unchanged. A step is legal onto a square on the board that holds no piece of the king's own
/// side nor the other king and that the other side does not attack (with the king itself lifted
/// off the board, so it cannot hide behind itself on a line). Capturing a piece does not remove it.
inline int restriction_score(const Position& p) {
    const int me = p.side;
    const int k = p.king(me);
    Position q = p;
    q.sq[static_cast<std::size_t>(k)] = Empty;
    std::array<bool, 64> safe{};
    for (int s = 0; s < 64; ++s)
        safe[static_cast<std::size_t>(s)] = sign(q.sq[static_cast<std::size_t>(s)]) != me && q.sq[static_cast<std::size_t>(s)] != -me * King
                                            && !attacked(q, s, -me);
    // reach[s]: the fewest king moves to s through safe squares (from k), up to 3; 99 if more
    std::array<int, 64> reach{};
    reach.fill(99);
    reach[static_cast<std::size_t>(k)] = 0;
    for (int step = 1; step <= 3; ++step)
        for (int s = 0; s < 64; ++s) {
            if (reach[static_cast<std::size_t>(s)] != step - 1) continue;
            for (int dr = -1; dr <= 1; ++dr)
                for (int df = -1; df <= 1; ++df) {
                    const int r = rank_of(s) + dr, f = file_of(s) + df;
                    if (r < 0 || r > 7 || f < 0 || f > 7) continue;
                    const auto t = static_cast<std::size_t>(r * 8 + f);
                    if (safe[t] && reach[t] > step) reach[t] = step;
                }
        }
    int score = 0;
    for (int dr = -3; dr <= 3; ++dr)
        for (int df = -3; df <= 3; ++df) {
            const int ring = std::max(std::abs(dr), std::abs(df));
            if (ring == 0) continue;
            const int r = rank_of(k) + dr, f = file_of(k) + df;
            const bool off = r < 0 || r > 7 || f < 0 || f > 7;
            if (off || reach[static_cast<std::size_t>(r * 8 + f)] != ring) score += kRingScore[ring - 1];
        }
    return score;
}

/// A drill with partial credit. The position comes from `pos_rng`; the defender moves at random
/// from `defender_seed`. Score (thousandths of a point): a mate is 1000, a stalemate 480; otherwise
/// the defending king's tightest moment, the most restriction_score it had at any of its turns.
struct DrillResult {
    bool mate = false;
    int plies = 0;
    int score = 0;
};
inline DrillResult scored_drill(const Chooser& player, const MateSet& set, int me, Rng& pos_rng, std::uint64_t defender_seed) {
    const Position start = mate_position(set, me, pos_rng);
    Rng defender_rng(defender_seed);
    const Chooser defender = random_player(defender_rng);
    int best = 0;
    const Observer watch = [&](const Position& p, const std::vector<Move>&) {
        if (p.side != me) best = std::max(best, restriction_score(p));
    };
    const Game g = me > 0 ? play(player, defender, start, watch) : play(defender, player, start, watch);
    DrillResult d;
    d.plies = g.plies;
    d.mate = g.winner == me;
    d.score = d.mate ? kMateScore : g.end == End::Stalemate ? kStalemateScore : best;
    return d;
}

}  // namespace chess

