// A small, complete chess rules engine for the chess example (examples/chess).
//
// Legal moves only (castling, en passant, promotion to any of Q R B N; nothing that leaves the
// mover's king in check), and the example's endings: checkmate; stalemate, insufficient material,
// 21 moves by each side without a pawn move or capture, and the 5th occurrence of a position are
// draws. Correctness is checked with perft (move counts from known positions), `chess_trainer
// --perft`.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace chess {

// Pieces: + white, - black. Squares: rank * 8 + file, a1 = 0, h8 = 63.
enum Piece : int { Empty = 0, Pawn = 1, Knight = 2, Bishop = 3, Rook = 4, Queen = 5, King = 6 };
enum Castle : int { WK = 1, WQ = 2, BK = 4, BQ = 8 };
constexpr int kDrawPlies = 42;   // 21 moves by each side without a pawn move or capture
constexpr int kRepetitions = 5;  // the 5th occurrence of a position

inline int rank_of(int s) { return s >> 3; }
inline int file_of(int s) { return s & 7; }
inline int sign(int p) { return p > 0 ? 1 : p < 0 ? -1 : 0; }
inline int kind(int p) { return p < 0 ? -p : p; }

struct Move {
    std::uint8_t from = 0, to = 0, promo = 0;  // promo: piece kind, 0 for none
    std::uint8_t flags = 0;                     // kCastle, kEnPassant, kDouble
};
constexpr std::uint8_t kCastleMove = 1, kEnPassant = 2, kDouble = 4;

struct Position {
    std::array<int, 64> sq{};
    int side = 1;      // +1 white to move, -1 black
    int castle = 0;    // Castle bits
    int ep = -1;       // the square a pawn passed over on its double step, or -1
    int quiet = 0;     // plies since the last pawn move or capture

    static Position start() {
        Position p;
        const int back[8] = {Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook};
        for (int f = 0; f < 8; ++f) {
            p.sq[f] = back[f];
            p.sq[8 + f] = Pawn;
            p.sq[48 + f] = -Pawn;
            p.sq[56 + f] = -back[f];
        }
        p.castle = WK | WQ | BK | BQ;
        return p;
    }

    /// From FEN (board, side, castling, en passant, halfmove clock): for tests.
    static Position fen(const std::string& f) {
        Position p;
        std::size_t i = 0;
        int r = 7, c = 0;
        for (; i < f.size() && f[i] != ' '; ++i) {
            const char ch = f[i];
            if (ch == '/') { --r; c = 0; continue; }
            if (ch >= '1' && ch <= '8') { c += ch - '0'; continue; }
            const std::string kinds = "pnbrqk";
            const int k = static_cast<int>(kinds.find(static_cast<char>(ch | 0x20))) + 1;
            p.sq[r * 8 + c] = (ch >= 'a') ? -k : k;
            ++c;
        }
        ++i;
        p.side = f[i] == 'w' ? 1 : -1;
        i += 2;
        for (; i < f.size() && f[i] != ' '; ++i) {
            if (f[i] == 'K') p.castle |= WK;
            if (f[i] == 'Q') p.castle |= WQ;
            if (f[i] == 'k') p.castle |= BK;
            if (f[i] == 'q') p.castle |= BQ;
        }
        ++i;
        if (i < f.size() && f[i] != '-') { p.ep = (f[i + 1] - '1') * 8 + (f[i] - 'a'); i += 2; } else { ++i; }
        if (i + 1 < f.size()) p.quiet = std::stoi(f.substr(i + 1));
        return p;
    }

    int king(int color) const {
        for (int s = 0; s < 64; ++s) if (sq[s] == King * color) return s;
        return -1;
    }
};

/// True if square s is attacked by any piece of `by` (+1 white, -1 black).
inline bool attacked(const Position& p, int s, int by) {
    const int r = rank_of(s), f = file_of(s);
    auto at = [&](int rr, int ff) { return (rr >= 0 && rr < 8 && ff >= 0 && ff < 8) ? p.sq[rr * 8 + ff] : 99; };
    const int pr = r - by;  // a pawn of `by` attacking s stands one rank behind it (from its side)
    if (at(pr, f - 1) == Pawn * by || at(pr, f + 1) == Pawn * by) return true;
    static const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
    for (auto& d : kn) if (at(r + d[0], f + d[1]) == Knight * by) return true;
    for (int dr = -1; dr <= 1; ++dr)
        for (int df = -1; df <= 1; ++df)
            if ((dr || df) && at(r + dr, f + df) == King * by) return true;
    static const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    for (int d = 0; d < 8; ++d) {
        for (int k = 1; k < 8; ++k) {
            const int x = at(r + dirs[d][0] * k, f + dirs[d][1] * k);
            if (x == 99) break;
            if (x == Empty) continue;
            const bool straight = d < 4;
            if (x == Queen * by || x == (straight ? Rook : Bishop) * by) return true;
            break;
        }
    }
    return false;
}

inline bool in_check(const Position& p) {
    const int k = p.king(p.side);
    return k >= 0 && attacked(p, k, -p.side);
}

/// The position after a (pseudo-legal) move.
inline Position make(const Position& p, const Move& m) {
    Position q = p;
    const int piece = q.sq[m.from], captured = q.sq[m.to];
    q.sq[m.to] = piece;
    q.sq[m.from] = Empty;
    if (m.flags & kEnPassant) q.sq[m.to - 8 * p.side] = Empty;
    if (m.promo) q.sq[m.to] = m.promo * p.side;
    if (m.flags & kCastleMove) {  // move the rook too
        const int r = rank_of(m.from);
        if (file_of(m.to) == 6) { q.sq[r * 8 + 5] = q.sq[r * 8 + 7]; q.sq[r * 8 + 7] = Empty; }
        else { q.sq[r * 8 + 3] = q.sq[r * 8 + 0]; q.sq[r * 8 + 0] = Empty; }
    }
    // castling rights end when the king or a rook moves, or a rook is captured on its square
    auto clear = [&](int s) {
        if (s == 4) q.castle &= ~(WK | WQ);
        if (s == 60) q.castle &= ~(BK | BQ);
        if (s == 7) q.castle &= ~WK;
        if (s == 0) q.castle &= ~WQ;
        if (s == 63) q.castle &= ~BK;
        if (s == 56) q.castle &= ~BQ;
    };
    clear(m.from);
    clear(m.to);
    q.ep = (m.flags & kDouble) ? (m.from + m.to) / 2 : -1;
    q.quiet = (kind(piece) == Pawn || captured != Empty || (m.flags & kEnPassant)) ? 0 : p.quiet + 1;
    q.side = -p.side;
    return q;
}

/// Every legal move of the side to move, in a fixed order (by from-square, then generation order).
inline std::vector<Move> legal_moves(const Position& p) {
    std::vector<Move> out, pseudo;
    const int me = p.side;
    auto add = [&](int from, int to, std::uint8_t flags = 0) {
        const bool promo = kind(p.sq[from]) == Pawn && (rank_of(to) == 7 || rank_of(to) == 0);
        if (promo)
            for (int k : {Queen, Rook, Bishop, Knight}) pseudo.push_back({static_cast<std::uint8_t>(from), static_cast<std::uint8_t>(to), static_cast<std::uint8_t>(k), flags});
        else
            pseudo.push_back({static_cast<std::uint8_t>(from), static_cast<std::uint8_t>(to), 0, flags});
    };
    for (int s = 0; s < 64; ++s) {
        const int x = p.sq[s];
        if (sign(x) != me) continue;
        const int r = rank_of(s), f = file_of(s);
        auto target = [&](int rr, int ff) { return rr >= 0 && rr < 8 && ff >= 0 && ff < 8; };
        switch (kind(x)) {
            case Pawn: {
                const int fwd = r + me;
                if (target(fwd, f) && p.sq[fwd * 8 + f] == Empty) {
                    add(s, fwd * 8 + f);
                    const int home = me > 0 ? 1 : 6;
                    if (r == home && p.sq[(r + 2 * me) * 8 + f] == Empty) add(s, (r + 2 * me) * 8 + f, kDouble);
                }
                for (int df : {-1, 1}) {
                    if (!target(fwd, f + df)) continue;
                    const int t = fwd * 8 + f + df;
                    if (sign(p.sq[t]) == -me) add(s, t);
                    else if (t == p.ep) add(s, t, kEnPassant);
                }
                break;
            }
            case Knight: {
                static const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
                for (auto& d : kn)
                    if (target(r + d[0], f + d[1]) && sign(p.sq[(r + d[0]) * 8 + f + d[1]]) != me) add(s, (r + d[0]) * 8 + f + d[1]);
                break;
            }
            case King: {
                for (int dr = -1; dr <= 1; ++dr)
                    for (int df = -1; df <= 1; ++df)
                        if ((dr || df) && target(r + dr, f + df) && sign(p.sq[(r + dr) * 8 + f + df]) != me) add(s, (r + dr) * 8 + f + df);
                // castling: rights, empty squares between, not out of / through / into check
                const int home = me > 0 ? 0 : 7;
                if (s == home * 8 + 4 && !attacked(p, s, -me)) {
                    const int ks = me > 0 ? WK : BK, qs = me > 0 ? WQ : BQ;
                    if ((p.castle & ks) && p.sq[home * 8 + 5] == Empty && p.sq[home * 8 + 6] == Empty &&
                        p.sq[home * 8 + 7] == Rook * me && !attacked(p, home * 8 + 5, -me) && !attacked(p, home * 8 + 6, -me))
                        add(s, home * 8 + 6, kCastleMove);
                    if ((p.castle & qs) && p.sq[home * 8 + 3] == Empty && p.sq[home * 8 + 2] == Empty &&
                        p.sq[home * 8 + 1] == Empty && p.sq[home * 8 + 0] == Rook * me && !attacked(p, home * 8 + 3, -me) &&
                        !attacked(p, home * 8 + 2, -me))
                        add(s, home * 8 + 2, kCastleMove);
                }
                break;
            }
            default: {  // sliders
                static const int dirs[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
                const int d0 = kind(x) == Bishop ? 4 : 0, d1 = kind(x) == Rook ? 4 : 8;
                for (int d = d0; d < d1; ++d)
                    for (int k = 1; k < 8; ++k) {
                        const int rr = r + dirs[d][0] * k, ff = f + dirs[d][1] * k;
                        if (!target(rr, ff)) break;
                        const int t = rr * 8 + ff;
                        if (sign(p.sq[t]) == me) break;
                        add(s, t);
                        if (p.sq[t] != Empty) break;
                    }
            }
        }
    }
    for (const Move& m : pseudo) {
        const Position q = make(p, m);
        const int k = q.king(me);
        if (k >= 0 && !attacked(q, k, -me)) out.push_back(m);
    }
    return out;
}

/// True if the side to move has a legal en passant capture (what makes the right count).
inline bool ep_available(const Position& p) {
    if (p.ep < 0) return false;
    for (const Move& m : legal_moves(p)) if (m.flags & kEnPassant) return true;
    return false;
}

/// The key of a position for repetition: pieces, side to move, castling rights, and the en passant
/// square only when an en passant capture is actually legal.
inline std::string key(const Position& p) {
    std::string k(68, '\0');
    for (int s = 0; s < 64; ++s) k[static_cast<std::size_t>(s)] = static_cast<char>(p.sq[static_cast<std::size_t>(s)] + 8);
    k[64] = static_cast<char>(p.side + 2);
    k[65] = static_cast<char>(p.castle);
    k[66] = static_cast<char>(ep_available(p) ? p.ep : 100);
    return k;
}

/// No sequence of legal moves can checkmate: king vs king, a lone knight or bishop, or only bishops
/// all on squares of one color.
inline bool insufficient(const Position& p) {
    int knights = 0, bishops = 0, light = 0, dark = 0;
    for (int s = 0; s < 64; ++s) {
        const int k = kind(p.sq[static_cast<std::size_t>(s)]);
        if (k == Pawn || k == Rook || k == Queen) return false;
        if (k == Knight) ++knights;
        if (k == Bishop) { ++bishops; ((rank_of(s) + file_of(s)) % 2 ? light : dark)++; }
    }
    if (knights == 0 && (light == 0 || dark == 0)) return true;  // no minors, or bishops of one color
    return knights == 1 && bishops == 0;
}

/// Number of leaf positions at a depth (for testing the move generator).
inline std::uint64_t perft(const Position& p, int depth) {
    if (depth == 0) return 1;
    std::uint64_t n = 0;
    for (const Move& m : legal_moves(p)) n += depth == 1 ? 1 : perft(make(p, m), depth - 1);
    return n;
}

}  // namespace chess
