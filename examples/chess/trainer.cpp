// Example trainer: 64 nets learn chess by playing each other, round robin.
//
// Each round every pair of players meets once. A match is up to 3 games, colors switching after
// each game (white always moves first): a decisive game ends the match and the loser is mutated
// from itself; after 3 draws both are mutated. Players only ever mutate from themselves, so each
// slot keeps its own lineage.
//
// A net plays by scoring the position after each legal move (seen from its own seat, rotated 180
// degrees for black) and playing the best one; it can never choose an illegal move. Inputs (66):
// the 64 squares (own pieces +1..+6 for P N B R Q K, the opponent's -1..-6; a rook that can still
// castle +-7; a pawn that can be taken en passant +-8), then plies since the last pawn move or
// capture, then how often this position has occurred (draw awareness).
//
// With --baseline N, after each round the round's best player also plays 2N games against a random
// mover and 2N against a greedy capturer (off by default, so every cycle goes to the round robin).
// Play a saved net yourself with chess_play.
//
// Every --puzzle-every rounds (default 50, and round 0) every player attempts every puzzle in
// --puzzles (default examples/chess/puzzles.csv: 200 Lichess puzzles, ratings 500 to 2999); the
// solve rates go to puzzles.csv (a running log), and the players x puzzles matrix of the latest check
// (puzzles easiest to hardest) to puzzle_matrix.csv and puzzle_matrix.txt. --puzzles-now 1 also checks
// before the first round.
// One game at a time, at the lowest CPU priority.
//
// A new run starts with mating drills (--mate-max N rounds at most, 0 to skip): each player has a
// mating set (KQ, KR, KRR, KQR, KBB) against a bare king making random legal moves, from random
// legal positions, and must mate under the usual draw rules. Each drill round every player and a
// mutation of itself play the same --mate-drills drills; the one with more mates (then fewer plies)
// keeps the slot (ties: the mutation). When the players' mean success reaches --mate-until % the
// round robin starts. Drill results go to mate.csv.
//
// Usage: chess_trainer [--out DIR] [--seed N] [--pop-seed N] [--resume DIR] [--profile FILE]
//                      [--rounds N] [--baseline N] [--q N] [--init NET.jnet ...] [--perft]
//        chess_trainer --ladder MAX_GENERATIONS [--out DIR] [--seed N] [--profile FILE] [--q N]
//        chess_trainer --mate-core MAX_GENERATIONS --init NET.jnet [--out DIR] [--seed N] [--profile FILE]
//                      [--mate-until PCT]
//
// --mate-core: core training on mating drills, one lineage. The elite holds a record from when it
// won its place, the total score on a fixed training set of 10 drills per material set (a mate
// scores 1 point, a stalemate 0.48; otherwise how boxed in the defending king got at its tightest
// moment, up to 0.44: see restriction_score in players.hpp); it
// does not play again. Each generation one challenger, a mutation of the elite, plays the training
// set (on 3 threads) and takes over if it matches or beats the record (ties go to the challenger);
// its score is then the record. The training set is a fixed pool of 500 drills (100 per material
// set), all of them every generation; the defender's moves change seed every 5,000 generations, and
// the elite is then re-scored.
// Every 100 generations the elite plays a fixed test of 20 drills per set; stops when it mates in
// --mate-until % (default 80) or at MAX_GENERATIONS. Saves mate_core.jnet and mate_core.csv.
//
// --ladder: two players under the tournament's rules, to find where net size levels off under the
// profile: a net and a mutation of it play matches (up to 3 games, colors switching); the loser
// of a decisive game is mutated from itself, after 3 draws both are. Stops when the players' mean
// size grows less than 1% over 12 windows of 500 generations in a row, or at MAX_GENERATIONS;
// saves ladder_a.jnet, ladder_b.jnet and ladder.csv.
// --init NET.jnet (repeatable): the players start from these seed nets, each filling an equal share
// of the 64 slots with itself and mutations of itself (two seeds: each seed and 31 mutations).

#include "players.hpp"

#include <jneuro/jneuro.hpp>

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <thread>

using namespace jneuro;
using namespace chess;
namespace fs = std::filesystem;
using D = std::uint32_t;
using Pop = Population<D>;

namespace {

constexpr int kPlayers = 64, kMaxDraws = 3;
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
    return buf;
}

Profile load_profile(const fs::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot read profile " + file.string());
    std::stringstream text;
    text << in.rdbuf();
    return profile_from_json(text.str());
}

}  // namespace

int main(int argc, char** argv) {
    fs::path out = fs::path("runs") / ("chess-" + timestamp());
    std::optional<std::uint64_t> seed, pop_seed, rounds;
    std::optional<fs::path> resume;
    fs::path profile_file = "profiles/expansion.json";
    int baseline = 0;  // off by default: no cycles on fixed opponents until the players are decent
    fs::path puzzle_file = "examples/chess/puzzles.csv";
    std::uint64_t puzzle_every = 50;  // every player attempts every puzzle each N rounds (0: never)
    bool puzzles_now = false;         // --puzzles-now 1: also check before the first round (e.g. on resume)
    double mate_until = 80.0;         // mating drills until the players' mean success reaches this %
    std::uint64_t mate_max = 2000;    // ... or this many drill rounds (0: no drills)
    int mate_drills = 20;             // drills per player per round
    std::optional<std::uint64_t> mate_core;  // --mate-core GENERATIONS: one lineage on mating drills
    std::uint32_t q = 5;
    bool perft_only = false;
    std::optional<std::uint64_t> ladder;
    std::optional<fs::path> ladder_from;  // continue from DIR/ladder_a.jnet and ladder_b.jnet
    double ladder_bulk = 0;               // first grow both players by this % (profiles/bulk.json)
    std::vector<fs::path> init_nets;  // --init, repeatable
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--perft") { perft_only = true; continue; }
        if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; return 2; }
        const std::string v = argv[++i];
        if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--pop-seed") pop_seed = std::stoull(v);
        else if (a == "--resume") resume = v;
        else if (a == "--profile") profile_file = v;
        else if (a == "--rounds") rounds = std::stoull(v);
        else if (a == "--baseline") baseline = std::stoi(v);
        else if (a == "--q") q = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--ladder") ladder = std::stoull(v);
        else if (a == "--puzzles") puzzle_file = v;
        else if (a == "--puzzle-every") puzzle_every = std::stoull(v);
        else if (a == "--puzzles-now") puzzles_now = v == "1" || v == "on" || v == "yes";
        else if (a == "--mate-until") mate_until = std::stod(v);
        else if (a == "--mate-max") mate_max = std::stoull(v);
        else if (a == "--mate-drills") mate_drills = std::stoi(v);
        else if (a == "--mate-core") mate_core = std::stoull(v);
        else if (a == "--ladder-from") ladder_from = v;
        else if (a == "--ladder-bulk" || a == "--bulk-first") ladder_bulk = std::stod(v);
        else if (a == "--init") init_nets.push_back(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (perft_only) {  // the rules engine against known move counts
        struct T { const char* fen; std::vector<std::uint64_t> want; };
        const T tests[] = {
            {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", {20, 400, 8902, 197281}},
            {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", {48, 2039, 97862}},
            {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", {14, 191, 2812, 43238}},
            {"r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", {6, 264, 9467}},
            {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", {44, 1486, 62379}}};
        bool ok = true;
        for (const T& t : tests)
            for (std::size_t d = 0; d < t.want.size(); ++d) ok = ok && perft(Position::fen(t.fen), static_cast<int>(d + 1)) == t.want[d];
        std::cout << (ok ? "perft: all positions match\n" : "perft: MISMATCH\n");
        return ok ? 0 : 1;
    }
    setpriority(PRIO_PROCESS, 0, 19);  // lowest CPU priority: the laptop stays responsive
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    if (resume) out = *resume;
    fs::create_directories(out);
    const Profile profile = load_profile(profile_file);

    if (mate_core) {  // core training on mating drills: one elite and two mutations, 3 games at once
        if (init_nets.size() != 1) { std::cerr << "--mate-core needs one --init NET.jnet\n"; return 2; }
        const std::uint64_t mseed = seed ? *seed : fresh_seed();
        Rng r(mseed);
        Net<D> elite = std::get<Net<D>>(load_net(init_nets[0]));
        if (ladder_bulk > 0) {  // --bulk-first PCT: grow the starting net first (profiles/bulk.json)
            Profile bulk = load_profile("profiles/bulk.json");
            bulk.set_overall("growth_max_pct", ladder_bulk);
            const std::size_t before = elite.nodes.size() - kInputs;
            elite = mutate(elite, bulk, 0, r, nullptr, true);
            std::cout << "bulk +" << ladder_bulk << "%: " << before << " -> " << elite.nodes.size() - kInputs << " neurons\n";
        }
        std::ofstream ccsv(out / "mate_core.csv");
        ccsv << "generation,test_mates_pct";
        for (const MateSet& m : mate_sets()) ccsv << ',' << m.name << "_pct";
        ccsv << ",test_plies_per_mate,record_score,record_mates,new_best_records,challenger_took,elite_kept,challenger_no_mates,neurons,seconds\n";
        {
            std::ofstream info(out / "run_info.txt", std::ios::app);
            info << "mate_core " << timestamp() << " seed " << mseed << " profile " << profile_file.string() << " init "
                 << init_nets[0].string() << "\n";
        }
        // the fixed test: 20 drills per set, half each color, the same every time
        auto test = [&](Net<D>& net, std::vector<int>& by, long& plies) {
            by.assign(mate_sets().size(), 0);
            plies = 0;
            Rng tr(0x7E57ULL);
            const Chooser me = net_player(net);
            int mates = 0;
            for (std::size_t s = 0; s < mate_sets().size(); ++s)
                for (int i = 0; i < 20; ++i) {
                    const int p = mate_drill(me, mate_sets()[s], i % 2 == 0 ? 1 : -1, tr);
                    if (p >= 0) { ++mates; ++by[s]; plies += p; }
                }
            return mates;
        };
        // the training set: the whole fixed pool of 500 positions (100 per material set) every
        // generation; the defender's random moves change seed every 5,000 generations, and the elite
        // is then re-scored
        struct Record { long score = 0; int mates = 0; long plies = 0; };  // score: thousandths of a point
        auto points = [](long thousandths) {  // "12.345": exact, whatever the stream's precision
            std::ostringstream o;
            o << thousandths / 1000 << '.' << std::setw(3) << std::setfill('0') << thousandths % 1000;
            return o.str();
        };
        constexpr std::uint64_t kPool = 500;
        const int train_drills = static_cast<int>(kPool);  // the full pool every generation
        constexpr std::uint64_t kEpoch = 5000;
        std::uint64_t epoch = 0;
        auto train_score = [&](Net<D>& net) {
            std::array<Record, 3> part{};  // three threads, each a third of the drills
            std::vector<std::thread> th;
            for (int t = 0; t < 3; ++t)
                th.emplace_back([&, t] {
                    Net<D> local = net.copy_genome();  // each thread its own copy (nets keep evaluation state)
                    const Chooser player = net_player(local);
                    for (int i = t; i < train_drills; i += 3) {
                        const std::uint64_t k = (epoch * static_cast<std::uint64_t>(train_drills) + static_cast<std::uint64_t>(i)) % kPool;
                        Rng pos(0x7A1Aull + k * 0x9E3779B97F4A7C15ULL);  // pool drill k: a fixed position
                        const std::uint64_t dseed = (0xDEFull + epoch * 0x2545F4914F6CDD1DULL) ^ (k * 0xBF58476D1CE4E5B9ULL);
                        const MateSet& set = mate_sets()[k % mate_sets().size()];
                        const int color = (k / mate_sets().size()) % 2 == 0 ? 1 : -1;
                        const DrillResult d = scored_drill(player, set, color, pos, dseed);
                        auto& x = part[static_cast<std::size_t>(t)];
                        x.score += d.score;
                        if (d.mate) { ++x.mates; x.plies += d.plies; }
                    }
                });
            for (auto& x : th) x.join();
            Record rec;
            for (const Record& x : part) { rec.score += x.score; rec.mates += x.mates; rec.plies += x.plies; }
            return rec;
        };
        Record record = train_score(elite);
        std::cout << "elite record to beat: score " << points(record.score) << " (" << record.mates << " of " << train_drills << " mated)\n"
                  << std::flush;
        constexpr std::uint64_t kEvery = 100;
        long mated = 0, elite_kept = 0, took = 0, none = 0;
        const auto t0 = std::chrono::steady_clock::now();
        std::uint64_t gen = 1;
        for (; gen <= *mate_core && !g_stop; ++gen) {
            // one challenger, a mutation of the elite; only the challenger plays (the fixed training set),
            // and it takes over if it matches or beats the elite's record (ties go to the challenger)
            if (gen / kEpoch != epoch) {  // the next 50 pool drills and new defender moves: the elite is re-scored
                epoch = gen / kEpoch;
                record = train_score(elite);
                std::cout << "defender moves " << epoch << ": elite re-scored " << points(record.score) << " (" << record.mates
                          << " mated)\n"
                          << std::flush;
            }
            Net<D> challenger = mutate(elite, profile, gen, r, nullptr, is_growth_generation(mseed, gen, profile));
            const Record rc = train_score(challenger);
            if (rc.score >= record.score) {  // matches or beats the record: ties go to the challenger
                elite = std::move(challenger);
                const bool better = rc.score > record.score;
                record = rc;
                ++took;
                if (better) ++mated;  // a new best record
            } else {
                ++elite_kept;
            }
            if (rc.mates == 0) ++none;
            if (gen % kEvery == 0) {
                std::vector<int> by;
                long tp = 0;
                const int tm = test(elite, by, tp);
                const double pct = 100.0 * tm / (20.0 * static_cast<double>(mate_sets().size()));
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                const double neurons = static_cast<double>(elite.nodes.size() - kInputs);
                ccsv << gen << ',' << pct;
                for (int b : by) ccsv << ',' << 5.0 * b;
                ccsv << ',' << (tm ? static_cast<double>(tp) / tm : 0.0) << ',' << points(record.score) << ',' << record.mates << ',' << mated
                     << ',' << took << ',' << elite_kept << ',' << none << ',' << neurons << ',' << s << '\n';
                ccsv.flush();
                std::cout << std::fixed << std::setprecision(1) << "generation " << gen << ": test " << pct << "% mates (";
                for (std::size_t b = 0; b < by.size(); ++b) std::cout << (b ? " " : "") << mate_sets()[b].name << ' ' << 5.0 * by[b] << '%';
                std::cout << "); record " << points(record.score) << " (" << record.mates << "/" << train_drills << " mated); "
                          << mated << " new best, challenger took " << took << ", elite kept " << elite_kept << ", challenger no mates "
                          << none << "; neurons " << neurons << "  " << std::setprecision(0) << s << "s\n" << std::flush;
                save_net(out / "mate_core.jnet", elite);
                mated = elite_kept = took = none = 0;
                if (pct >= mate_until) { std::cout << "test success reached " << mate_until << "%\n"; break; }
            }
        }
        save_net(out / "mate_core.jnet", elite);
        std::cout << "mate core training stopped at generation " << std::min(gen, *mate_core) << " ("
                  << elite.nodes.size() - kInputs << " neurons); saved " << (out / "mate_core.jnet").string() << "\n";
        return 0;
    }

    if (ladder) {  // two players, the tournament's rules: where does their size level off?
        const std::uint64_t lseed = seed ? *seed : fresh_seed();
        Rng r(lseed);
        std::array<Net<D>, 2> pl;
        if (ladder_from) {
            pl[0] = std::get<Net<D>>(load_net(*ladder_from / "ladder_a.jnet"));
            pl[1] = std::get<Net<D>>(load_net(*ladder_from / "ladder_b.jnet"));
        } else {
            pl[0] = Net<D>::create(static_cast<std::uint32_t>(kInputs), 1, q, Priority::Speed, r, 2, 3);
            pl[1] = mutate(pl[0], profile, 0, r, nullptr, is_growth_generation(lseed, 0, profile));
        }
        if (ladder_bulk > 0) {  // a growth-only bulk of both players first
            Profile bulk = load_profile("profiles/bulk.json");
            bulk.set_overall("growth_max_pct", ladder_bulk);
            for (auto& n : pl) {
                const std::size_t before = n.nodes.size() - kInputs;
                n = mutate(n, bulk, 0, r, nullptr, true);
                std::cout << "bulk +" << ladder_bulk << "%: " << before << " -> " << n.nodes.size() - kInputs << " neurons\n";
            }
        }
        std::ofstream lcsv(out / "ladder.csv");
        lcsv << "generation,neurons_a,neurons_b,result,games\n";  // result: winner 0/1, or -1 for three draws
        {
            std::ofstream info(out / "run_info.txt", std::ios::app);
            info << "ladder " << timestamp() << " seed " << lseed << " profile " << profile_file.string() << " q " << q << "\n";
        }
        constexpr std::uint64_t kWindow = 500;
        double window_sum = 0, prev_mean = -1;
        int flat = 0;
        std::array<long, 2> wins{0, 0};
        long draws = 0;
        const auto t0 = std::chrono::steady_clock::now();
        auto neurons = [](const Net<D>& n) { return static_cast<double>(n.nodes.size() - kInputs); };
        auto mut = [&](int k, std::uint64_t gen) { pl[static_cast<std::size_t>(k)] = mutate(pl[static_cast<std::size_t>(k)], profile, gen, r, nullptr, is_growth_generation(lseed, gen, profile)); };
        std::uint64_t gen = 1;
        for (; gen <= *ladder && !g_stop; ++gen) {
            int result = -1, games = 0;
            int white = static_cast<int>(r.below(2));
            for (int g = 0; g < kMaxDraws && result < 0; ++g, white = 1 - white) {
                const Game gm = play(net_player(pl[static_cast<std::size_t>(white)]), net_player(pl[static_cast<std::size_t>(1 - white)]));
                ++games;
                if (gm.winner != 0) result = gm.winner > 0 ? white : 1 - white;
            }
            lcsv << gen << ',' << neurons(pl[0]) << ',' << neurons(pl[1]) << ',' << result << ',' << games << '\n';
            if (result >= 0) { ++wins[static_cast<std::size_t>(result)]; mut(1 - result, gen); }  // the loser changes
            else { ++draws; mut(0, gen); mut(1, gen); }                                              // three draws: both
            window_sum += 0.5 * (neurons(pl[0]) + neurons(pl[1]));
            if (gen % kWindow == 0) {
                const double mean = window_sum / kWindow;
                const double change = prev_mean > 0 ? (mean - prev_mean) / prev_mean : 1.0;
                flat = change < 0.01 ? flat + 1 : 0;
                const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::cout << std::fixed << std::setprecision(1) << "generation " << gen << ": mean neurons " << mean << " ("
                          << std::showpos << 100 * change << std::noshowpos << "%), players " << neurons(pl[0]) << " and "
                          << neurons(pl[1]) << "; wins " << wins[0] << " / " << wins[1] << ", 3 draws " << draws << "  "
                          << std::setprecision(0) << s << "s\n" << std::flush;
                lcsv.flush();
                save_net(out / "ladder_a.jnet", pl[0]);
                save_net(out / "ladder_b.jnet", pl[1]);
                prev_mean = mean;
                window_sum = 0;
                wins = {0, 0};
                draws = 0;
                if (flat >= 12) { std::cout << "plateau: under 1% growth for 12 windows of 500 in a row\n"; break; }
            }
        }
        save_net(out / "ladder_a.jnet", pl[0]);
        save_net(out / "ladder_b.jnet", pl[1]);
        std::cout << "ladder stopped at generation " << std::min(gen, *ladder) << " with players of " << neurons(pl[0])
                  << " and " << neurons(pl[1]) << " neurons; saved ladder_a.jnet and ladder_b.jnet in " << out.string() << "\n";
        return 0;
    }

    // players: slot -> current population id; a loser's mutated self takes its slot
    std::vector<std::uint64_t> slot(kPlayers);
    std::uint64_t round = 0, run_seed = 0;
    int phase_mate = 0;                 // 1: mating drills, 0: round robin
    std::uint64_t mate_round = 0;       // drill rounds done
    std::optional<Pop> pop_holder;
    if (resume) {
        pop_holder.emplace(Pop::load(out / "population.jpnet"));
        std::ifstream st(out / "state.txt");
        st >> round >> run_seed;
        for (auto& s : slot) st >> s;
        if (!st) { std::cerr << "cannot read " << (out / "state.txt").string() << "\n"; return 2; }
        if (!(st >> phase_mate >> mate_round)) { phase_mate = 0; mate_round = 0; }  // older state files: round robin
    } else {
        pop_holder.emplace(q, kInputs, 1, Priority::Speed, pop_seed);
        if (!init_nets.empty()) {  // each seed net fills an equal share of the slots: itself, then mutations of it
            if (kPlayers % static_cast<int>(init_nets.size()) != 0) {
                std::cerr << "--init: " << kPlayers << " players do not divide among " << init_nets.size() << " seed nets\n";
                return 2;
            }
            const int share = kPlayers / static_cast<int>(init_nets.size());
            for (std::size_t i = 0; i < init_nets.size(); ++i) {
                const Net<D> base = std::get<Net<D>>(load_net(init_nets[i]));
                if (base.in_qty_max != q || base.n_inputs != static_cast<std::uint32_t>(kInputs) || base.n_outputs != 1) {
                    std::cerr << "--init: " << init_nets[i].string() << " is not a chess net with in_qty_max " << q << "\n";
                    return 2;
                }
                const std::uint64_t id = pop_holder->add(base);
                for (int k = 0; k < share; ++k)
                    slot[i * static_cast<std::size_t>(share) + static_cast<std::size_t>(k)] = k == 0 ? id : pop_holder->mutate(id, profile);
            }
        } else {
            const auto ids = pop_holder->create(kPlayers, profile.rng_mode());
            for (int k = 0; k < kPlayers; ++k) slot[static_cast<std::size_t>(k)] = ids[static_cast<std::size_t>(k)];
        }
        run_seed = seed ? *seed : fresh_seed();
        phase_mate = mate_max > 0 ? 1 : 0;
    }
    Pop& pop = *pop_holder;
    {
        std::ofstream info(out / "run_info.txt", std::ios::app);
        info << "start " << timestamp() << " round " << round << " seed " << run_seed << " population_seed " << pop.seed()
             << " profile " << profile_file.string() << (resume ? " resumed" : "") << "\n";
    }
    const std::vector<Puzzle> puzzles = puzzle_every ? load_puzzles(puzzle_file.string()) : std::vector<Puzzle>{};
    if (puzzle_every && puzzles.empty()) std::cerr << "no puzzles in " << puzzle_file.string() << " (puzzle checks off)\n";
    const int bands[] = {500, 1000, 1500, 2000, 2500, 3000};  // puzzle rating bands, very easy to very hard
    const bool new_pcsv = !fs::exists(out / "puzzles.csv");
    std::ofstream pcsv(out / "puzzles.csv", std::ios::app);
    if (new_pcsv)
        pcsv << "round,puzzles,mean_solved_pct,best_solved_pct,best_slot,worst_solved_pct,band_500_999_pct,band_1000_1499_pct,"
                "band_1500_1999_pct,band_2000_2499_pct,band_2500_2999_pct,seconds\n";
    const bool new_csv = !fs::exists(out / "rounds.csv");
    std::ofstream csv(out / "rounds.csv", std::ios::app);
    if (new_csv)
        csv << "round,games,decisive,checkmate,stalemate,insufficient,quiet_21,repetition,mean_plies,mutations,"
               "mean_neurons,best_slot,best_wins,vs_random_w,vs_random_d,vs_random_l,vs_greedy_w,vs_greedy_d,vs_greedy_l,seconds\n";
    std::cout << "chess trainer: " << kPlayers << " players round robin, output in " << out.string()
              << ". Ctrl+C to stop after the current match and save.\n";

    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto mutate_slot = [&](int s) {
        const std::uint64_t old = slot[static_cast<std::size_t>(s)];
        slot[static_cast<std::size_t>(s)] = pop.mutate(old, profile);
        pop.remove(old);
    };
    auto save = [&] {
        pop.save(out / "population.jpnet");
        std::ofstream st(out / "state.txt");
        st << round << ' ' << run_seed;
        for (auto s : slot) st << ' ' << s;
        st << ' ' << phase_mate << ' ' << mate_round << '\n';
    };

    // Every player attempts every puzzle: the summary goes to puzzles.csv (a running log), the full
    // players x puzzles matrix (puzzles easiest to hardest) to puzzle_matrix.csv and .txt (the latest
    // check only).
    auto puzzle_check = [&](std::uint64_t at) {
        std::vector<std::vector<char>> hit(kPlayers, std::vector<char>(puzzles.size(), 0));
        std::vector<int> solved(kPlayers, 0);
        double band_solved[5] = {}, band_n[5] = {};
        for (int k = 0; k < kPlayers; ++k) {
            Net<D>& net = pop.get(slot[static_cast<std::size_t>(k)]);
            const Chooser me = net_player(net);
            for (std::size_t i = 0; i < puzzles.size(); ++i) {
                const bool ok = solve(me, puzzles[i]);
                hit[static_cast<std::size_t>(k)][i] = ok;
                solved[static_cast<std::size_t>(k)] += ok;
                for (int b = 0; b < 5; ++b)
                    if (puzzles[i].rating >= bands[b] && puzzles[i].rating < bands[b + 1]) { band_solved[b] += ok; band_n[b] += 1; }
            }
        }
        const auto [lo, hi] = std::minmax_element(solved.begin(), solved.end());
        double mean = 0;
        for (int v : solved) mean += v;
        mean /= kPlayers;
        const double n = static_cast<double>(puzzles.size());
        pcsv << at << ',' << puzzles.size() << ',' << 100 * mean / n << ',' << 100 * *hi / n << ',' << (hi - solved.begin()) << ','
             << 100 * *lo / n;
        for (int b = 0; b < 5; ++b) pcsv << ',' << (band_n[b] ? 100 * band_solved[b] / band_n[b] : 0.0);
        pcsv << ',' << secs() << '\n';
        pcsv.flush();
        // the matrix: puzzles sorted by rating (stable), one row per player
        std::vector<std::size_t> order(puzzles.size());
        for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) { return puzzles[x].rating < puzzles[y].rating; });
        std::ofstream mc(out / "puzzle_matrix.csv"), mt(out / "puzzle_matrix.txt");
        mc << "slot,solved";
        for (std::size_t j = 0; j < order.size(); ++j) mc << ",p" << j + 1 << '_' << puzzles[order[j]].rating;
        mc << '\n';
        mt << "round " << at << ": players (rows) x " << order.size() << " puzzles (columns, easiest " << puzzles[order.front()].rating
           << " to hardest " << puzzles[order.back()].rating << "); # solved, . missed\n"
           << "rating bands start at columns:";
        for (int b = 0; b < 5; ++b)
            for (std::size_t j = 0; j < order.size(); ++j)
                if (puzzles[order[j]].rating >= bands[b]) { mt << ' ' << bands[b] << "@" << j + 1; break; }
        mt << "\n";
        for (int k = 0; k < kPlayers; ++k) {
            mc << k << ',' << solved[static_cast<std::size_t>(k)];
            mt << "slot " << std::setw(2) << k << " " << std::setw(3) << solved[static_cast<std::size_t>(k)] << " ";
            for (std::size_t j : order) {
                mc << ',' << static_cast<int>(hit[static_cast<std::size_t>(k)][j]);
                mt << (hit[static_cast<std::size_t>(k)][j] ? '#' : '.');
            }
            mc << '\n';
            mt << '\n';
        }
        std::cout << std::fixed << std::setprecision(1) << "  puzzles at round " << at << ": mean " << 100 * mean / n
                  << "% solved, best " << 100 * *hi / n << "% (slot " << (hi - solved.begin()) << "), worst " << 100 * *lo / n
                  << "%; by rating 500-999 / 1000-1499 / 1500-1999 / 2000-2499 / 2500-2999:";
        for (int b = 0; b < 5; ++b) std::cout << ' ' << (band_n[b] ? 100 * band_solved[b] / band_n[b] : 0.0) << '%';
        std::cout << "\n" << std::flush;
    };
    if (!puzzles.empty() && puzzle_every && (puzzles_now || !resume) && round % puzzle_every != 0) puzzle_check(round);

    const bool new_mcsv = !fs::exists(out / "mate.csv");
    std::ofstream mcsv(out / "mate.csv", std::ios::app);
    if (new_mcsv) {
        mcsv << "round,mate_round,mean_success_pct";
        for (const MateSet& m : mate_sets()) mcsv << ',' << m.name << "_pct";
        mcsv << ",mean_plies_to_mate,mutations_kept,mean_neurons,seconds\n";
    }
    // a player's drills: the same positions and defender moves for a player and its mutation
    struct Drills { int mates = 0; long plies = 0; std::vector<int> by; };
    auto drills_of = [&](Net<D>& net) {
        Drills d;
        d.by.assign(mate_sets().size(), 0);
        Rng r(run_seed ^ 0xD5111ULL ^ (mate_round * 0x9E3779B97F4A7C15ULL));
        const Chooser me = net_player(net);
        for (int i = 0; i < mate_drills; ++i) {
            const std::size_t set = static_cast<std::size_t>(i) % mate_sets().size();
            const int color = (i / static_cast<int>(mate_sets().size())) % 2 == 0 ? 1 : -1;
            const int plies = mate_drill(me, mate_sets()[set], color, r);
            if (plies >= 0) { ++d.mates; d.plies += plies; ++d.by[set]; }
        }
        return d;
    };

    while (!g_stop && !(rounds && round >= *rounds)) {
        if (phase_mate) {  // mating drills: each player against its own mutation, the better one stays
            std::vector<int> by(mate_sets().size(), 0);
            long mates = 0, plies = 0, kept = 0;
            for (int k = 0; k < kPlayers && !g_stop; ++k) {
                const std::uint64_t parent = slot[static_cast<std::size_t>(k)];
                const std::uint64_t child = pop.mutate(parent, profile);
                const Drills dp = drills_of(pop.get(parent)), dc = drills_of(pop.get(child));
                const bool take = dc.mates > dp.mates || (dc.mates == dp.mates && dc.plies <= dp.plies);  // ties: the mutation
                pop.remove(take ? parent : child);
                slot[static_cast<std::size_t>(k)] = take ? child : parent;
                const Drills& d = take ? dc : dp;
                kept += take;
                mates += d.mates;
                plies += d.plies;
                for (std::size_t s = 0; s < by.size(); ++s) by[s] += d.by[s];
            }
            if (g_stop) break;
            const double per_set = static_cast<double>(kPlayers) * mate_drills / static_cast<double>(mate_sets().size());
            const double success = 100.0 * static_cast<double>(mates) / (static_cast<double>(kPlayers) * mate_drills);
            double neurons = 0;
            for (auto s : slot) neurons += static_cast<double>(pop.get(s).nodes.size() - kInputs);
            neurons /= kPlayers;
            mcsv << round << ',' << mate_round << ',' << success;
            for (int b : by) mcsv << ',' << 100.0 * b / per_set;
            mcsv << ',' << (mates ? static_cast<double>(plies) / static_cast<double>(mates) : 0.0) << ',' << kept << ',' << neurons << ','
                 << secs() << '\n';
            mcsv.flush();
            std::cout << std::fixed << std::setprecision(1) << "drills " << mate_round << " (round " << round << "): " << success
                      << "% mates;";
            for (std::size_t s = 0; s < by.size(); ++s) std::cout << ' ' << mate_sets()[s].name << ' ' << 100.0 * by[s] / per_set << '%';
            std::cout << "; " << (mates ? static_cast<double>(plies) / static_cast<double>(mates) : 0.0) << " plies/mate, " << kept
                      << " mutations kept, neurons " << neurons << "  " << std::setprecision(0) << secs() << "s\n" << std::flush;
            if (!puzzles.empty() && puzzle_every && round % puzzle_every == 0) puzzle_check(round);
            ++mate_round;
            if (success >= mate_until || mate_round >= mate_max) {
                phase_mate = 0;
                std::cout << "mating drills done after " << mate_round << " rounds (" << std::setprecision(1) << success
                          << "% mates): round robin from round " << round + 1 << "\n";
            }
            ++round;
            pop.next_generation();
            save();
            continue;
        }
        // this round's pairings in a shuffled order; which of the two starts as white, at random
        std::vector<std::pair<int, int>> pairs;
        for (int a = 0; a < kPlayers; ++a)
            for (int b = a + 1; b < kPlayers; ++b) pairs.push_back({a, b});
        Rng r(run_seed ^ (round * 0x9E3779B97F4A7C15ULL));
        for (std::size_t i = pairs.size(); i > 1; --i) std::swap(pairs[i - 1], pairs[r.below(i)]);
        std::vector<int> wins(kPlayers, 0);
        long games = 0, decisive = 0, mutations = 0, plies = 0;
        std::map<End, long> ends;
        auto last_print = std::chrono::steady_clock::now();
        std::size_t done = 0;
        for (auto [a, b] : pairs) {
            if (g_stop) break;
            int white = r.below(2) ? a : b, black = white == a ? b : a;
            for (int g = 0; g < kMaxDraws; ++g) {
                Net<D>& wn = pop.get(slot[static_cast<std::size_t>(white)]);
                Net<D>& bn = pop.get(slot[static_cast<std::size_t>(black)]);
                const Game res = play(net_player(wn), net_player(bn));
                ++games;
                plies += res.plies;
                ++ends[res.end];
                if (res.winner != 0) {
                    ++decisive;
                    const int winner = res.winner > 0 ? white : black, loser = res.winner > 0 ? black : white;
                    ++wins[static_cast<std::size_t>(winner)];
                    mutate_slot(loser);
                    ++mutations;
                    break;
                }
                if (g == kMaxDraws - 1) {  // three draws: both change
                    mutate_slot(a);
                    mutate_slot(b);
                    mutations += 2;
                }
                std::swap(white, black);  // colors switch after each game
            }
            ++done;
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - last_print).count() >= 60) {
                last_print = std::chrono::steady_clock::now();
                std::cout << "  round " << round << ": " << done << "/" << pairs.size() << " matches, " << games << " games ("
                          << decisive << " decisive), " << std::fixed << std::setprecision(0) << secs() << "s\n" << std::flush;
            }
        }
        if (g_stop && done < pairs.size()) break;  // an unfinished round is not logged; the slots are saved below

        // the round's best player against fixed opponents
        const int best = static_cast<int>(std::max_element(wins.begin(), wins.end()) - wins.begin());
        int res[2][3] = {};  // [random, greedy][win, draw, loss]
        for (int opp = 0; opp < 2; ++opp) {
            Rng br(run_seed ^ 0xB5E11E5ULL ^ (round << 8) ^ static_cast<std::uint64_t>(opp));
            Net<D>& bn = pop.get(slot[static_cast<std::size_t>(best)]);
            const Chooser other = opp == 0 ? random_player(br) : greedy_player(br);
            for (int g = 0; g < 2 * baseline; ++g) {
                const bool as_white = g % 2 == 0;
                const Game gm = as_white ? play(net_player(bn), other) : play(other, net_player(bn));
                const int me = as_white ? 1 : -1;
                ++res[opp][gm.winner == 0 ? 1 : gm.winner == me ? 0 : 2];
            }
        }
        double neurons = 0;
        for (auto s : slot) neurons += static_cast<double>(pop.get(s).nodes.size() - kInputs);
        neurons /= kPlayers;
        csv << round << ',' << games << ',' << decisive << ',' << ends[End::Checkmate] << ',' << ends[End::Stalemate] << ','
            << ends[End::Insufficient] << ',' << ends[End::Quiet] << ',' << ends[End::Repetition] << ','
            << static_cast<double>(plies) / std::max(1L, games) << ',' << mutations << ',' << neurons << ',' << best << ','
            << wins[static_cast<std::size_t>(best)] << ',' << res[0][0] << ',' << res[0][1] << ',' << res[0][2] << ',' << res[1][0]
            << ',' << res[1][1] << ',' << res[1][2] << ',' << secs() << '\n';
        csv.flush();
        std::cout << std::fixed << std::setprecision(1) << "round " << round << ": " << games << " games, "
                  << 100.0 * decisive / std::max(1L, games) << "% decisive (checkmate " << ends[End::Checkmate] << ", draws: stalemate "
                  << ends[End::Stalemate] << ", insufficient " << ends[End::Insufficient] << ", 21 moves " << ends[End::Quiet]
                  << ", repetition " << ends[End::Repetition] << "), " << static_cast<double>(plies) / std::max(1L, games)
                  << " plies/game, neurons " << neurons << "; best slot " << best << " (" << wins[static_cast<std::size_t>(best)]
                  << " wins): vs random " << res[0][0] << "/" << res[0][1] << "/" << res[0][2] << ", vs greedy " << res[1][0] << "/"
                  << res[1][1] << "/" << res[1][2] << " (W/D/L)  " << std::setprecision(0) << secs() << "s\n" << std::flush;
        if (!puzzles.empty() && puzzle_every && round % puzzle_every == 0) puzzle_check(round);
        ++round;
        pop.next_generation();
        save();
        save_net(out / "best.jnet", pop.get(slot[static_cast<std::size_t>(best)]));
    }
    save();
    std::cout << "stopped at round " << round << "; saved " << out.string() << "\n";
    return 0;
}
