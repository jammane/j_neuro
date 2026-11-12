// Example trainer: evolve nets that guess the next card of a shoe, using jneuro::Population.
//
// A fully shuffled 5-deck shoe (260 cards) is dealt from the first card to the last. Before
// each next card the net sees the current card and makes two guesses: the next card's suit
// (3 points if right) and whether its rank is higher, lower or equal (higher or lower: 1 point
// if right; equal: 6 points if right; a wrong guess scores 0). Equal is right with probability
// 19/259 = 7.3%, so "always equal" pays 0.44 points per guess: less than a random higher/lower
// guess (0.46). Equal only pays when counting shows the current rank is likely to repeat. The net is reset only at the start of a shoe, so s_neurons can carry the
// history of cards already seen.
//
// Each generation every net plays the same --shoes shoes (default 4). The top 10 nets are
// kept, the other 30 culled, and each survivor is mutated 3 times. Runs until interrupted
// (SIGINT / SIGTERM), then saves the population.
//
// Usage: highlow_trainer [--out DIR] [--seed N] [--resume FILE.jpnet] [--shoes N]
//                        [--path-mix N] [--in-qty-max N] [--edit PCT] [--op PCT] [--grow-gen PCT]
//                        [--grow-net PCT] [--stop-at GENERATIONS] [--console-every SECONDS]
//
// --in-qty-max (odd, >= 3; default 3) is the m_neuron slot count and the longest s_neuron
// history, so it also sets how far back a lag s_neuron can look. --stop-at N stops and saves
// once N generations have been played.

#include <jneuro/jneuro.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

using namespace jneuro;
using D = std::uint32_t;
using Pop = Population<D>;
namespace fs = std::filesystem;

namespace {

// ------------------------------------------------------------------ settings
constexpr int kPopulation = 40;
constexpr int kKeep = 10;
constexpr int kChildrenPerSurvivor = 3;
constexpr int kDecks = 5;
constexpr int kRanks = 13;  // 2..10, J, Q, K, A (ace high)
constexpr int kSuits = 4;
constexpr int kShoeSize = kDecks * kRanks * kSuits;  // 260
constexpr int kGuesses = kShoeSize - 1;              // one per card except the last
// Points are kept exactly as integer units (kUnit units = 1 point).
constexpr int kUnit = 1;
constexpr int kSuitPoints = 3 * kUnit, kHighLowPoints = 1 * kUnit, kEqualPoints = 6 * kUnit;

enum class Rel { Lower = -1, Equal = 0, Higher = 1 };
constexpr int kMaxSuddenDeath = 10;  // then remaining tied nets are picked at random

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// ------------------------------------------------------------------ game
struct Card {
    int rank;  // 0..12 -> 2..A
    int suit;  // 0..3
};
using Shoe = std::vector<Card>;

Shoe make_shoe(std::uint64_t seed) {
    Shoe s;
    for (int d = 0; d < kDecks; ++d)
        for (int r = 0; r < kRanks; ++r)
            for (int u = 0; u < kSuits; ++u) s.push_back({r, u});
    Rng rng(seed);
    for (std::size_t i = s.size() - 1; i > 0; --i)  // full Fisher-Yates
        std::swap(s[i], s[static_cast<std::size_t>(rng.below(i + 1))]);
    return s;
}

struct Result {
    int points = 0, suit_ok = 0, rel_ok = 0, equal_guesses = 0, equal_ok = 0;
    Result& operator+=(const Result& o) {
        points += o.points; suit_ok += o.suit_ok; rel_ok += o.rel_ok;
        equal_guesses += o.equal_guesses; equal_ok += o.equal_ok;
        return *this;
    }
};

/// Score one shoe given a guesser (current card -> suit guess, higher/lower/equal guess).
template <class Guess>
Result score_shoe(const Shoe& shoe, Guess guess) {
    Result r;
    for (int i = 0; i < kGuesses; ++i) {
        const Card cur = shoe[i], next = shoe[i + 1];
        const auto [suit, rel] = guess(cur);
        if (suit == next.suit) { r.points += kSuitPoints; ++r.suit_ok; }
        const Rel actual = next.rank > cur.rank ? Rel::Higher : next.rank < cur.rank ? Rel::Lower : Rel::Equal;
        if (rel == Rel::Equal) ++r.equal_guesses;
        if (rel == actual) {
            ++r.rel_ok;
            if (rel == Rel::Equal) { r.points += kEqualPoints; ++r.equal_ok; }
            else r.points += kHighLowPoints;
        }
    }
    return r;
}

Result play_shoe(Net<D>& net, const Shoe& shoe) {
    net.reset();  // history spans the whole shoe
    std::vector<double> in(2);
    return score_shoe(shoe, [&](const Card& c) {
        in[0] = static_cast<double>(c.rank + 2);  // 2..14
        in[1] = static_cast<double>(c.suit);
        const auto y = net.step(in);
        const int suit = static_cast<int>(std::ceil(std::clamp(y[0], 0.0, 3.0)));  // bound 0..3, round up
        // nearest of -1 (lower), 0 (equal), 1 (higher): <= -0.5 lower, >= 0.5 higher, else equal
        const Rel rel = y[1] <= -0.5 ? Rel::Lower : y[1] >= 0.5 ? Rel::Higher : Rel::Equal;
        return std::pair<int, Rel>{suit, rel};
    });
}

/// Best no-memory baseline: always the same suit; for the relation, the guess with the highest
/// expected points given only the current card (a full shoe minus that card): 20 cards of each
/// other rank, 19 of its own.
Result baseline_shoe(const Shoe& shoe) {
    return score_shoe(shoe, [](const Card& c) {
        const int higher = 20 * (kRanks - 1 - c.rank), lower = 20 * c.rank, equal = 19;  // counts
        const long ev_h = 1L * higher * kHighLowPoints, ev_l = 1L * lower * kHighLowPoints,
                   ev_e = 1L * equal * kEqualPoints;
        const Rel rel = (ev_e >= ev_h && ev_e >= ev_l) ? Rel::Equal : ev_h >= ev_l ? Rel::Higher : Rel::Lower;
        return std::pair<int, Rel>{0, rel};
    });
}

// ------------------------------------------------------------------ helpers
struct Counts {
    double neurons = 0, v = 0, s = 0, m = 0;
};

Counts count(const Net<D>& net) {
    Counts c;
    c.neurons = static_cast<double>(net.nodes.size() - net.n_inputs);
    for (const auto& n : net.nodes) {
        if (!n.neuron) continue;
        switch (type_of(*n.neuron)) {
            case NeuronType::V: ++c.v; break;
            case NeuronType::S: ++c.s; break;
            case NeuronType::M: ++c.m; break;
        }
    }
    return c;
}

std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
    return buf;
}

/// Edits and op changes "up to"; optional exactly-one growth attempt; constant folding.
Profile make_profile(double edit_pct, double op_pct, bool grow) {
    Profile p = Profile::defaults();
    p.set_overall("edit_max_pct", edit_pct);
    p.set_overall("op_change_max_pct", op_pct);
    p.set_overall("growth_max_pct", grow ? 1e-9 : 0.0);  // exact: max(1, ~0) = 1 attempt
    p.set_overall("removal_max_pct", 0.0);
    p.apply_patch(nlohmann::json{{"overall", {{"growth_budget", "exact"}, {"constant_folding", "on"}}}});
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    fs::path out = fs::path("runs") / ("highlow-" + timestamp());
    std::optional<std::uint64_t> seed;
    std::optional<fs::path> resume;
    int shoes = 4;
    std::uint32_t path_mix = 2, in_qty_max = 3;
    std::optional<std::uint64_t> stop_at;
    double edit_pct = 20.0, op_pct = 1.0, grow_gen = 10.0, grow_net = 10.0, console_every = 2.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--resume") resume = v;
        else if (a == "--shoes") shoes = std::stoi(v);
        else if (a == "--path-mix") path_mix = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--in-qty-max") in_qty_max = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--stop-at") stop_at = std::stoull(v);
        else if (a == "--edit") edit_pct = std::stod(v);
        else if (a == "--op") op_pct = std::stod(v);
        else if (a == "--grow-gen") grow_gen = std::stod(v);
        else if (a == "--grow-net") grow_net = std::stod(v);
        else if (a == "--console-every") console_every = std::stod(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (shoes < 1) { std::cerr << "--shoes must be >= 1\n"; return 2; }
    fs::create_directories(out);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Pop pop = resume ? Pop::load(*resume) : Pop(in_qty_max, 2, 2, Priority::Speed, std::nullopt, path_mix);
    if (!resume) pop.create(kPopulation);
    const Profile steady = make_profile(edit_pct, op_pct, false), grow_one = make_profile(edit_pct, op_pct, true);
    const std::uint64_t deal_seed = seed ? *seed : fresh_seed();
    Rng deals(deal_seed);
    Rng growth_rng(deal_seed ^ 0x9E3779B97F4A7C15ull);

    const bool new_csv = !fs::exists(out / "log.csv");
    std::ofstream csv(out / "log.csv", std::ios::app);
    if (new_csv)
        csv << "generation,top10_avg_points,best_points,all_avg_points,baseline_points,best_suit_pct,"
               "best_rel_pct,best_equal_guess_pct,best_equal_hit_pct,top10_suit_pct,top10_rel_pct,"
               "top10_equal_guess_pct,top10_avg_neurons,top10_avg_v,top10_avg_s,top10_avg_m,"
               "sudden_death_shoes,growth_children,elapsed_s\n";
    {
        std::ofstream info(out / "run_info.txt", std::ios::app);
        info << "start " << timestamp() << " deal_seed " << deal_seed << " population_seed " << pop.seed()
             << " shoes " << shoes << " path_mix " << pop.path_mix() << " edit_pct " << edit_pct << " op_pct "
             << op_pct << " grow_gen_pct " << grow_gen << " grow_net_pct " << grow_net << " constant_folding on"
             << " in_qty_max " << pop.get(pop.ids().front()).in_qty_max
             << (stop_at ? " stop_at " + std::to_string(*stop_at) : std::string()) << (resume ? " resumed_from " + resume->string() : std::string())
             << "\n";
    }
    std::cout << "high/low trainer: output in " << out.string() << " (deal seed " << deal_seed
              << "). Points are per shoe.\n"
              << std::setw(8) << "gen" << std::setw(9) << "top10" << std::setw(7) << "best" << std::setw(9)
              << "allavg" << std::setw(9) << "baseline" << std::setw(8) << "suit%" << std::setw(8) << "hle%" << std::setw(8) << "eq-g%"
              << std::setw(9) << "neurons" << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    auto last_print = t0 - std::chrono::hours(1);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());

    while (!g_stop && !(stop_at && pop.generation() >= *stop_at)) {
        const bool growth_gen = growth_rng.u() * 100.0 < grow_gen;
        int growth_children = 0;

        // the same shoes for every net this generation
        std::vector<Shoe> gen_shoes;
        Result base;
        for (int i = 0; i < shoes; ++i) {
            gen_shoes.push_back(make_shoe(deals.next_u64()));
            base += baseline_shoe(gen_shoes.back());
        }

        const std::vector<std::uint64_t> ids = pop.ids();
        std::vector<Result> res(ids.size());
        std::atomic<std::size_t> next{0};
        auto worker = [&] {
            for (std::size_t k; (k = next++) < ids.size();)
                for (const Shoe& s : gen_shoes) res[k] += play_shoe(pop.get(ids[k]), s);
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < std::min<unsigned>(threads, static_cast<unsigned>(ids.size())); ++t)
            pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        std::vector<std::size_t> order(ids.size());
        for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(),
                         [&](std::size_t a, std::size_t b) { return res[a].points > res[b].points; });

        // A tie across the keep/cull line is broken by sudden-death shoes, single elimination
        // (as in the blackjack trainer); they do not count toward scores.
        int sudden_death = 0;
        if (res[order[kKeep - 1]].points == res[order[kKeep]].points) {
            const int tied = res[order[kKeep]].points;
            std::size_t lo = 0, hi;
            while (res[order[lo]].points != tied) ++lo;
            for (hi = lo; hi < order.size() && res[order[hi]].points == tied; ++hi) {}
            std::vector<std::size_t> alive(order.begin() + lo, order.begin() + hi), advanced, eliminated;
            std::size_t slots = kKeep - lo;
            while (alive.size() > slots) {
                if (sudden_death == kMaxSuddenDeath) {
                    for (std::size_t i = alive.size() - 1; i > 0; --i)
                        std::swap(alive[i], alive[static_cast<std::size_t>(deals.below(i + 1))]);
                    break;
                }
                const Shoe s = make_shoe(deals.next_u64());
                ++sudden_death;
                std::vector<std::pair<int, std::size_t>> r;
                for (std::size_t k : alive) r.emplace_back(play_shoe(pop.get(ids[k]), s).points, k);
                std::stable_sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.first > b.first; });
                std::vector<std::size_t> still, gone;
                for (std::size_t i = 0; i < r.size();) {
                    std::size_t j = i;
                    while (j < r.size() && r[j].first == r[i].first) ++j;
                    if (slots == 0 || !still.empty()) {
                        for (std::size_t x = i; x < j; ++x) gone.push_back(r[x].second);
                    } else if (j - i <= slots) {
                        for (std::size_t x = i; x < j; ++x) advanced.push_back(r[x].second);
                        slots -= j - i;
                    } else {
                        for (std::size_t x = i; x < j; ++x) still.push_back(r[x].second);
                    }
                    i = j;
                }
                eliminated.insert(eliminated.begin(), gone.begin(), gone.end());
                alive = std::move(still);
            }
            std::size_t pos = lo;
            for (std::size_t k : advanced) order[pos++] = k;
            for (std::size_t k : alive) order[pos++] = k;
            for (std::size_t k : eliminated) order[pos++] = k;
        }

        // stats (points per shoe; accuracy over all guesses this generation)
        Result top;
        Counts sum;
        double all_points = 0;
        for (const auto& r : res) all_points += r.points;
        for (int k = 0; k < kKeep; ++k) {
            top += res[order[k]];
            const Counts c = count(pop.get(ids[order[k]]));
            sum.neurons += c.neurons; sum.v += c.v; sum.s += c.s; sum.m += c.m;
        }
        const Result& best = res[order[0]];
        const double n = kKeep, per = static_cast<double>(shoes) * kUnit;  // per shoe, in points
        auto pct = [](int ok, int total) { return total ? 100.0 * ok / total : 0.0; };
        const std::uint64_t gen = pop.generation();
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::ostringstream row;
        row << std::fixed << std::setprecision(2) << gen << ',' << top.points / n / per << ',' << best.points / per
            << ',' << all_points / ids.size() / per << ',' << base.points / per << ','
            << pct(best.suit_ok, kGuesses * shoes) << ',' << pct(best.rel_ok, kGuesses * shoes) << ','
            << pct(best.equal_guesses, kGuesses * shoes) << ',' << pct(best.equal_ok, best.equal_guesses) << ','
            << pct(top.suit_ok, kGuesses * shoes * kKeep) << ',' << pct(top.rel_ok, kGuesses * shoes * kKeep) << ','
            << pct(top.equal_guesses, kGuesses * shoes * kKeep) << ','
            << sum.neurons / n << ',' << sum.v / n << ',' << sum.s / n << ',' << sum.m / n << ',' << sudden_death
            << ',';

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_print).count() >= console_every) {
            last_print = now;
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << gen << std::setw(9)
                      << top.points / n / per << std::setw(7) << best.points / per << std::setw(9)
                      << all_points / ids.size() / per << std::setw(9) << base.points / per << std::setw(8)
                      << pct(best.suit_ok, kGuesses * shoes) << std::setw(8) << pct(best.rel_ok, kGuesses * shoes)
                      << std::setw(8) << pct(best.equal_guesses, kGuesses * shoes)
                      << std::setw(9) << sum.neurons / n << std::endl;
            save_net(out / "best.jnet", pop.get(ids[order[0]]));
        }

        // cull the bottom 30, refill with 3 mutated offspring of each of the top 10
        for (std::size_t r = kKeep; r < order.size(); ++r) pop.remove(ids[order[r]]);
        for (int r = 0; r < kKeep; ++r)
            for (int c = 0; c < kChildrenPerSurvivor; ++c) {
                const bool grows = growth_gen && growth_rng.u() * 100.0 < grow_net;
                growth_children += grows;
                pop.mutate(ids[order[r]], grows ? grow_one : steady);
            }
        pop.next_generation();
        row << growth_children << ',' << elapsed << '\n';
        csv << row.str();
        csv.flush();

        if (gen % 1000 == 999) {
            // Reloading continues exactly but empties the population's in-memory run log.
            pop.save(out / "population.jpnet");
            pop = Pop::load(out / "population.jpnet");
        }
    }

    pop.save(out / "population.jpnet");
    std::cout << "stopped at generation " << pop.generation() << "; saved " << (out / "population.jpnet").string()
              << " (resume with --resume " << (out / "population.jpnet").string() << ")\n";
    return 0;
}
