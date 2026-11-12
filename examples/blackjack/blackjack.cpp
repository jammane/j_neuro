// Example trainer: evolve nets that play blackjack, using jneuro::Population.
//
// Each generation every net plays the same --deals deals (default 200). The top 10 nets are kept, the other
// 30 are culled, and each survivor is mutated 3 times to refill the population to 40.
// Runs until interrupted (Ctrl+C / SIGINT / SIGTERM), then saves the population.
//
// Usage: blackjack_trainer [--out DIR] [--seed N] [--resume FILE.jpnet] [--rules new|old]
//                          [--deals N] [--stop-at GENERATIONS] [--console-every SECONDS]
// --deals: deals per generation (default 200). Each generation the log also scores the simple
// hit rule (hit below 12, or below 17 vs a dealer 7+; never doubles or splits) and the double rule
// (the hit rule, plus double on a two-card 10 or 11 when the dealer's card is lower) on the same deals.
//   new rules (default):   [--profile FILE] [--edit PCT] [--op PCT] [--grow-gen PCT] [--grow-net PCT]
//                          [--tap-weight W]
//   old rules:             [--alter PCT] [--grow PCT] [--big-alter PCT] [--big-grow PCT]
//
// New rules: mutation uses the profile file (default profiles/expansion.json: edits on up to
// 20% of neurons, op changes up to 1%, exactly one growth attempt in 50% of generations for 80%
// of nets, constant folding); --edit, --op, --grow-gen, --grow-net (in %) and --tap-weight
// override its values when given.
//
// Old rules: up to --alter % edited and op-changed, --grow % added and removed; nets over 50
// neurons use the --big-* caps instead.

#include <jneuro/jneuro.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

#include "game.hpp"

using namespace jneuro;
using namespace bj;
using Pop = Population<D>;
namespace fs = std::filesystem;

namespace {

// ------------------------------------------------------------------ settings
constexpr int kPopulation = 40;
constexpr int kKeep = 10;
constexpr int kChildrenPerSurvivor = 3;
int kHandsPerNet = 200;  // deals per generation (--deals)
constexpr std::uint32_t kInQtyMax = 3;
constexpr int kMaxSuddenDeath = 10;  // then remaining tied nets are picked at random
constexpr std::size_t kBigNet = 50;   // above this many neurons, mutate with the small caps

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// ------------------------------------------------------------------ helpers
struct Counts {
    double nodes = 0, neurons = 0, v = 0, s = 0, m = 0;
};

Counts count(const Net<D>& net) {
    Counts c;
    c.nodes = static_cast<double>(net.nodes.size());
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

/// New rules: a saved profile file (spec/09 JSON).
Profile load_profile(const fs::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot read profile " + file.string());
    std::stringstream text;
    text << in.rdbuf();
    return profile_from_json(text.str());
}

/// Neurons added by growth so far (sum of the growth methods' mutation counts).
std::uint64_t growth_added(const Pop& pop) {
    static const char* const kGrowth[] = {"slot_fill", "v_insert", "s_insert", "m_after_m", "v_tap", "s_tap",
                                          "m_tap",     "m_clone",  "m_insert", "v_link",    "s_link"};
    const auto counts = pop.mutation_counts();
    std::uint64_t n = 0;
    for (const char* m : kGrowth) n += counts.at(m);
    return n;
}

Profile make_profile(double alter_pct, double grow_pct) {
    Profile p = Profile::defaults();
    p.set_overall("edit_max_pct", alter_pct);       // "altered": parameter edits ...
    p.set_overall("op_change_max_pct", alter_pct);  // ... and operation changes
    p.set_overall("growth_max_pct", grow_pct);      // "add"
    p.set_overall("removal_max_pct", grow_pct);     // "remove"
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    fs::path out = fs::path("runs") / ("blackjack-" + timestamp());
    std::optional<std::uint64_t> seed;
    std::optional<fs::path> resume;
    double alter = 22.0, grow = 12.0, big_alter = 5.0, big_grow = 1.5, console_every = 2.0;
    fs::path profile_file = "profiles/expansion.json";
    std::optional<std::uint64_t> stop_at;
    std::optional<double> edit_pct, op_pct, grow_gen, grow_net, tap_weight;  // overrides of the profile
    std::string rules = "new";
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--resume") resume = v;
        else if (a == "--alter") alter = std::stod(v);
        else if (a == "--grow") grow = std::stod(v);
        else if (a == "--big-alter") big_alter = std::stod(v);
        else if (a == "--big-grow") big_grow = std::stod(v);
        else if (a == "--console-every") console_every = std::stod(v);
        else if (a == "--rules") rules = v;
        else if (a == "--edit") edit_pct = std::stod(v);
        else if (a == "--op") op_pct = std::stod(v);
        else if (a == "--grow-gen") grow_gen = std::stod(v);
        else if (a == "--grow-net") grow_net = std::stod(v);
        else if (a == "--tap-weight") tap_weight = std::stod(v);
        else if (a == "--profile") profile_file = v;
        else if (a == "--stop-at") stop_at = std::stoull(v);
        else if (a == "--deals") kHandsPerNet = std::stoi(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    if (rules != "new" && rules != "old") { std::cerr << "--rules must be new or old\n"; return 2; }
    const bool new_rules = rules == "new";
    fs::create_directories(out);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Pop pop = resume ? Pop::load(*resume) : Pop(kInQtyMax, 9, 1, Priority::Speed);
    if (!resume) pop.create(kPopulation);
    const Profile profile = make_profile(alter, grow);
    const Profile big_profile = make_profile(big_alter, big_grow);  // nets over kBigNet neurons
    Profile new_profile = new_rules ? load_profile(profile_file) : Profile::defaults();
    if (edit_pct) new_profile.set_overall("edit_max_pct", *edit_pct);
    if (op_pct) new_profile.set_overall("op_change_max_pct", *op_pct);
    if (grow_gen) new_profile.set_overall("growth_gen_prob", *grow_gen / 100.0);
    if (grow_net) new_profile.set_overall("growth_net_prob", *grow_net / 100.0);
    if (tap_weight)
        new_profile.apply_patch(nlohmann::json{{"overall", {{"add_method_weights",
            {{"v_tap", *tap_weight}, {"s_tap", *tap_weight}, {"m_tap", *tap_weight}}}}}});
    const std::uint64_t deal_seed = seed ? *seed : fresh_seed();
    Rng deals(deal_seed);

    const bool new_csv = !fs::exists(out / "log.csv");
    std::ofstream csv(out / "log.csv", std::ios::app);
    if (new_csv)
        csv << "generation,top10_avg_score,best_score,all_avg_score,top10_avg_nodes,top10_avg_neurons,"
               "top10_avg_v,top10_avg_s,top10_avg_m,unwinnable,best_win_pct_winnable,top10_win_pct_winnable,"
               "sudden_death_deals,hitrule_score,hitrule_win_pct_winnable,doublerule_score,"
               "doublerule_win_pct_winnable,growth_added,elapsed_s\n";
    {
        std::ofstream info(out / "run_info.txt", std::ios::app);
        info << "start " << timestamp() << " deals " << kHandsPerNet << " deal_seed " << deal_seed << " population_seed " << pop.seed()
             << " rules " << rules;
        if (new_rules)
            info << " profile " << profile_file.string() << " effective_profile " << profile_to_json(new_profile);
        else
            info << " alter_pct " << alter << " grow_pct " << grow << " big_alter_pct " << big_alter
                 << " big_grow_pct " << big_grow << " big_net_neurons " << kBigNet;
        info << " in_qty_max " << kInQtyMax
             << (resume ? " resumed_from " + resume->string() : std::string()) << "\n";
    }
    std::cout << "blackjack trainer: output in " << out.string() << " (deal seed " << deal_seed
              << "). Ctrl+C to stop and save.\n"
              << std::setw(8) << "gen" << std::setw(10) << "top10avg" << std::setw(7) << "best"
              << std::setw(9) << "allavg" << std::setw(9) << "nodes" << std::setw(9) << "neurons"
              << std::setw(7) << "v" << std::setw(7) << "s" << std::setw(7) << "m" << std::setw(7) << "unwin"
              << std::setw(8) << "best%" << std::setw(8) << "top10%" << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    auto last_print = t0 - std::chrono::hours(1);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());

    while (!g_stop && !(stop_at && pop.generation() >= *stop_at)) {
        // new rules: is this a growth generation? (decided up front so the log row can report it)

        // the same deals for every net this generation, and the hit rule's score on them
        std::vector<Deal> gen_deals;
        int n_winnable = 0, rule_won = 0, dbl_won = 0;
        double rule_score = 0, dbl_score = 0;
        for (int i = 0; i < kHandsPerNet; ++i) {
            gen_deals.emplace_back(deals.next_u64());
            n_winnable += winnable(gen_deals.back());
            const double s = play_hit_rule(gen_deals.back());
            rule_score += s;
            rule_won += s > 0;
            const double sd = play_double_rule(gen_deals.back());
            dbl_score += sd;
            dbl_won += sd > 0;
        }

        const std::vector<std::uint64_t> ids = pop.ids();
        std::vector<double> scores(ids.size(), 0.0);  // scores are sums of halves: exact in binary64
        std::vector<int> won(ids.size(), 0);           // won: deals with a positive score
        std::atomic<std::size_t> next{0};
        auto worker = [&] {
            for (std::size_t k; (k = next++) < ids.size();) {
                Net<D>& net = pop.get(ids[k]);
                for (const Deal& d : gen_deals) {
                    const double s = play_deal(net, d);
                    scores[k] += s;
                    won[k] += s > 0;
                }
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < std::min<unsigned>(threads, static_cast<unsigned>(ids.size())); ++t)
            pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        std::vector<std::size_t> order(ids.size());
        for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });

        // A tie across the keep/cull line is broken by sudden-death deals, single elimination:
        // the tied nets all play one new deal; those scoring below the rest are out, any group
        // that fits in the open spots advances, and the group straddling the line plays again.
        // Sudden-death deals only rank the tied nets; they do not count toward scores.
        int sudden_death = 0;
        if (order.size() > static_cast<std::size_t>(kKeep) && scores[order[kKeep - 1]] == scores[order[kKeep]]) {
            const double tied = scores[order[kKeep]];
            std::size_t lo = 0, hi;
            while (scores[order[lo]] != tied) ++lo;
            for (hi = lo; hi < order.size() && scores[order[hi]] == tied; ++hi) {}
            std::vector<std::size_t> alive(order.begin() + lo, order.begin() + hi), advanced, eliminated;
            std::size_t slots = kKeep - lo;
            while (alive.size() > slots) {
                if (sudden_death == kMaxSuddenDeath) {  // e.g. identical copies: pick at random
                    for (std::size_t i = alive.size() - 1; i > 0; --i)
                        std::swap(alive[i], alive[static_cast<std::size_t>(deals.below(i + 1))]);
                    break;
                }
                const Deal d(deals.next_u64());
                ++sudden_death;
                std::vector<std::pair<double, std::size_t>> r;
                for (std::size_t k : alive) r.emplace_back(play_deal(pop.get(ids[k]), d), k);
                std::stable_sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.first > b.first; });
                std::vector<std::size_t> still, out;
                for (std::size_t i = 0; i < r.size();) {
                    std::size_t j = i;
                    while (j < r.size() && r[j].first == r[i].first) ++j;
                    if (slots == 0 || !still.empty()) {
                        for (std::size_t x = i; x < j; ++x) out.push_back(r[x].second);
                    } else if (j - i <= slots) {
                        for (std::size_t x = i; x < j; ++x) advanced.push_back(r[x].second);
                        slots -= j - i;
                    } else {
                        for (std::size_t x = i; x < j; ++x) still.push_back(r[x].second);
                    }
                    i = j;
                }
                eliminated.insert(eliminated.begin(), out.begin(), out.end());  // later = ranked higher
                alive = std::move(still);
            }
            std::size_t pos = lo;
            for (std::size_t k : advanced) order[pos++] = k;
            for (std::size_t k : alive) order[pos++] = k;
            for (std::size_t k : eliminated) order[pos++] = k;
        }

        Counts sum;
        double top_score = 0, all_score = 0, top_won = 0;
        for (std::size_t k = 0; k < ids.size(); ++k) all_score += scores[k];
        for (int r = 0; r < kKeep; ++r) {
            const std::size_t k = order[r];
            top_score += scores[k];
            top_won += won[k];
            const Counts c = count(pop.get(ids[k]));
            sum.nodes += c.nodes; sum.neurons += c.neurons; sum.v += c.v; sum.s += c.s; sum.m += c.m;
        }
        const double n = kKeep;
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::uint64_t gen = pop.generation();
        auto pct_of_winnable = [&](double w) { return n_winnable ? 100.0 * w / n_winnable : 0.0; };
        const double best_pct = pct_of_winnable(won[order[0]]), top_pct = pct_of_winnable(top_won / n);
        std::ostringstream row;
        row << std::fixed << std::setprecision(2) << gen << ',' << top_score / n << ',' << scores[order[0]] << ','
            << all_score / ids.size() << ',' << sum.nodes / n << ',' << sum.neurons / n << ',' << sum.v / n << ','
            << sum.s / n << ',' << sum.m / n << ',' << kHandsPerNet - n_winnable << ',' << best_pct << ','
            << top_pct << ',' << sudden_death << ',' << rule_score << ',' << pct_of_winnable(rule_won) << ','
            << dbl_score << ',' << pct_of_winnable(dbl_won) << ',';  // growth_added and elapsed added after mutation

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_print).count() >= console_every) {
            last_print = now;
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << gen << std::setw(10) << top_score / n
                      << std::setw(7) << scores[order[0]] << std::setw(9) << all_score / ids.size()
                      << std::setw(9) << sum.nodes / n << std::setw(9) << sum.neurons / n << std::setw(7)
                      << sum.v / n << std::setw(7) << sum.s / n << std::setw(7) << sum.m / n << std::setw(7)
                      << kHandsPerNet - n_winnable << std::setw(8) << best_pct << std::setw(8) << top_pct << std::endl;
            save_net(out / "best.jnet", pop.get(ids[order[0]]));
        }

        // cull the bottom 30, refill with 3 mutated offspring of each of the top 10
        for (std::size_t r = kKeep; r < order.size(); ++r) pop.remove(ids[order[r]]);
        const std::uint64_t added_before = growth_added(pop);
        for (int r = 0; r < kKeep; ++r) {
            const std::uint64_t parent = ids[order[r]];
            const bool big = count(pop.get(parent)).neurons > kBigNet;
            for (int c = 0; c < kChildrenPerSurvivor; ++c) {
                if (!new_rules) { pop.mutate(parent, big ? big_profile : profile); continue; }
                pop.mutate(parent, new_profile);  // growth gating: spec/06
            }
        }
        pop.next_generation();
        row << growth_added(pop) - added_before << ',' << elapsed << '\n';  // same fixed formatting as the rest of the row
        csv << row.str();
        csv.flush();

        if (gen % 1000 == 999) {
            // Reloading continues exactly (same ids, nets, RNG state) but empties the population's
            // in-memory run log, which otherwise grows by ~60 operations per generation.
            pop.save(out / "population.jpnet");
            pop = Pop::load(out / "population.jpnet");
        }
    }

    pop.save(out / "population.jpnet");
    std::cout << "stopped at generation " << pop.generation() << "; saved " << (out / "population.jpnet").string()
              << " (resume with --resume " << (out / "population.jpnet").string() << ")\n";
    return 0;
}
