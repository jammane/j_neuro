// Playing chess games for the chess example: game loop, the nets' inputs and move choice, and the
// fixed opponents. Shared by chess_trainer and chess_play, so both play by the same rules.
#pragma once

#include "chess.hpp"

#include <jneuro/jneuro.hpp>

#include <functional>
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

inline Game play(const Chooser& white, const Chooser& black) {
    Position p = Position::start();
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
