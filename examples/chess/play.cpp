// Play a game of chess against one of the trained nets (examples/chess).
//
// Usage: chess_play RUN_DIR|NET.jnet [--slot N] [--color white|black]
//   RUN_DIR      a chess_trainer output directory: plays its best.jnet, or with --slot N the
//                current net of player slot N (0..63)
//   NET.jnet     any saved chess net
//   --color      your color (default white; white moves first)
//
// Enter moves as from-square and to-square: e2e4, g1f3; castling is the king's move (e1g1);
// promotion adds the piece: e7e8q (q r b n). "moves" lists your legal moves, "quit" ends.
// The same rules as training: checkmate wins; stalemate, insufficient material, 21 moves by each
// side without a pawn move or capture, and the 5th occurrence of a position are draws. Only reads
// the saved files, so it can run while the trainer does.

#include "players.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

using namespace chess;
using namespace jneuro;
namespace fs = std::filesystem;

namespace {

std::string square(int s) { return std::string{static_cast<char>('a' + file_of(s)), static_cast<char>('1' + rank_of(s))}; }

std::string move_text(const Move& m) {
    std::string t = square(m.from) + square(m.to);
    if (m.promo) t += "  nbrq"[m.promo];
    return t;
}

/// The board as seen from `me`'s side of the table: uppercase white, lowercase black.
void show(const Position& p, int me) {
    const char* names = ".PNBRQK";
    std::cout << "\n";
    for (int row = 0; row < 8; ++row) {
        const int r = me > 0 ? 7 - row : row;
        std::cout << "  " << r + 1 << "  ";
        for (int col = 0; col < 8; ++col) {
            const int f = me > 0 ? col : 7 - col;
            const int x = p.sq[static_cast<std::size_t>(r * 8 + f)];
            char c = names[kind(x)];
            if (x < 0) c = static_cast<char>(c | 0x20);
            std::cout << c << ' ';
        }
        std::cout << "\n";
    }
    std::cout << "\n     ";
    for (int col = 0; col < 8; ++col) std::cout << static_cast<char>('a' + (me > 0 ? col : 7 - col)) << ' ';
    std::cout << "\n  (moves since the last pawn move or capture: " << p.quiet / 2 << " of 21)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: chess_play RUN_DIR|NET.jnet [--slot N] [--color white|black]\n";
        return 2;
    }
    const fs::path src = argv[1];
    std::optional<int> slot_no;
    int me = 1;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--slot") slot_no = std::stoi(v);
        else if (a == "--color") me = v == "black" ? -1 : 1;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    Net<D> net;
    std::string who;
    if (fs::is_directory(src) && slot_no) {
        auto pop = Population<D>::load(src / "population.jpnet");
        std::ifstream st(src / "state.txt");
        std::uint64_t round = 0, seed = 0, id = 0;
        st >> round >> seed;
        for (int k = 0; k <= *slot_no && st; ++k) st >> id;
        if (!st) { std::cerr << "cannot read slot " << *slot_no << " from " << (src / "state.txt").string() << "\n"; return 2; }
        net = pop.get(id).copy_genome();
        who = "slot " + std::to_string(*slot_no) + " (round " + std::to_string(round) + ")";
    } else {
        const fs::path file = fs::is_directory(src) ? src / "best.jnet" : src;
        net = std::get<Net<D>>(load_net(file));
        who = file.string();
    }
    if (net.n_inputs != static_cast<std::uint32_t>(kInputs)) { std::cerr << "not a chess net (" << net.n_inputs << " inputs)\n"; return 2; }
    std::cout << "You play " << (me > 0 ? "white" : "black") << " against " << who << " (" << net.nodes.size() - kInputs
              << " neurons).\n";

    const Chooser computer = net_player(net);
    Position p = Position::start();
    Reps reps;
    reps[key(p)] = 1;
    for (;;) {
        const auto moves = legal_moves(p);
        std::string result;
        if (moves.empty()) result = in_check(p) ? (p.side == me ? "Checkmate: the net wins." : "Checkmate: you win.") : "Stalemate: draw.";
        else if (insufficient(p)) result = "Insufficient material: draw.";
        else if (reps[key(p)] >= kRepetitions) result = "Fifth repetition of the position: draw.";
        else if (p.quiet >= kDrawPlies) result = "21 moves without a pawn move or capture: draw.";
        if (!result.empty()) { show(p, me); std::cout << result << "\n"; return 0; }
        Move m;
        if (p.side == me) {
            show(p, me);
            if (in_check(p)) std::cout << "  You are in check.\n";
            for (;;) {
                std::cout << "your move: " << std::flush;
                std::string in;
                if (!(std::cin >> in) || in == "quit") return 0;
                if (in == "moves") {
                    for (const Move& x : moves) std::cout << move_text(x) << ' ';
                    std::cout << "\n";
                    continue;
                }
                bool found = false;
                for (const Move& x : moves) if (move_text(x) == in) { m = x; found = true; }
                if (found) break;
                std::cout << "not a legal move (\"moves\" lists them)\n";
            }
        } else {
            m = computer(p, moves, reps);
            std::cout << "net plays " << move_text(m) << "\n";
        }
        p = make(p, m);
        ++reps[key(p)];
    }
}
