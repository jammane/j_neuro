// Example trainer: evolve nets that play blackjack, using jneuro::Population.
//
// Each generation every net plays the same 50 deals. The top 10 nets are kept, the other
// 30 are culled, and each survivor is mutated 3 times to refill the population to 40.
// Runs until interrupted (Ctrl+C / SIGINT / SIGTERM), then saves the population.
//
// Usage: blackjack_trainer [--out DIR] [--seed N] [--resume FILE.jpnet]
//                          [--alter PCT] [--grow PCT] [--console-every SECONDS]

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
constexpr int kHandsPerNet = 50;
constexpr int kDecks = 5;          // a 5-deck shoe
constexpr int kRanks = 14;         // 1..10, J, Q, K, A
constexpr int kCardsPerHand = 7;
constexpr int kMaxHands = 4;       // limit on split hands per deal
constexpr std::uint32_t kInQtyMax = 3;

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// ------------------------------------------------------------------ cards
// rank index: 0..9 -> "1".."10", 10 J, 11 Q, 12 K, 13 A
bool is_ace(int r) { return r == 13; }
int points(int r) { return r <= 9 ? r + 1 : (r <= 12 ? 10 : 11); }
double input_value(int r) { return static_cast<double>(points(r)); }  // A = 11, J/Q/K = 10

int best_total(const std::vector<int>& cards, std::size_t n) {
    int total = 0, soft_aces = 0;
    for (std::size_t i = 0; i < n; ++i) {
        total += points(cards[i]);
        if (is_ace(cards[i])) ++soft_aces;
    }
    while (total > 21 && soft_aces > 0) { total -= 10; --soft_aces; }  // ace counted as 1
    return total;
}

/// One deal: a freshly shuffled 5-deck shoe; cards are drawn from its top.
class Shoe {
public:
    explicit Shoe(std::uint64_t seed) : rng_(seed) {
        for (int r = 0; r < kRanks; ++r)
            for (int c = 0; c < 4 * kDecks; ++c) cards_.push_back(r);
    }
    int draw() {  // partial Fisher-Yates: only shuffles as far as cards are drawn
        const std::size_t j = pos_ + static_cast<std::size_t>(rng_.below(cards_.size() - pos_));
        std::swap(cards_[pos_], cards_[j]);
        return cards_[pos_++];
    }

private:
    Rng rng_;
    std::vector<int> cards_;
    std::size_t pos_ = 0;
};

struct Hand {
    std::vector<int> cards;  // 7 cards drawn up front
    std::size_t revealed = 2;
    bool doubled = false;
    bool done = false;
};

enum Action { Stand = 0, Hit = 1, Double = 2, Split = 3 };

Action decode(double y) {  // bound to 0..3, round up
    const double c = std::ceil(std::clamp(y, 0.0, 3.0));
    return static_cast<Action>(static_cast<int>(c));
}

/// Play one deal with `net`; returns the score (win 1, doubled win 2, per hand).
int play_deal(Net<D>& net, std::uint64_t deal_seed) {
    Shoe shoe(deal_seed);
    std::vector<Hand> hands(1);
    std::vector<int> dealer;
    for (int i = 0; i < kCardsPerHand; ++i) hands[0].cards.push_back(shoe.draw());
    for (int i = 0; i < kCardsPerHand; ++i) dealer.push_back(shoe.draw());

    net.reset();  // s_neuron history covers one deal
    std::vector<double> in(9);
    for (std::size_t h = 0; h < hands.size(); ++h) {
        while (!hands[h].done) {
            Hand& hand = hands[h];
            in[0] = input_value(dealer[0]);
            in[1] = input_value(dealer[1]);
            for (int i = 0; i < kCardsPerHand; ++i)
                in[2 + i] = static_cast<std::size_t>(i) < hand.revealed ? input_value(hand.cards[i]) : 0.0;
            switch (decode(net.step(in)[0])) {
                case Stand:
                    hand.done = true;
                    break;
                case Hit:
                    if (hand.revealed < kCardsPerHand) ++hand.revealed;
                    if (hand.revealed == kCardsPerHand || best_total(hand.cards, hand.revealed) > 21) hand.done = true;
                    break;
                case Double:
                    if (hand.revealed < kCardsPerHand) ++hand.revealed;
                    hand.doubled = true;
                    hand.done = true;
                    break;
                case Split: {
                    const bool valid = hand.revealed == 2 && hand.cards[0] == hand.cards[1] &&
                                       hands.size() < static_cast<std::size_t>(kMaxHands);
                    if (!valid) { hand.done = true; break; }  // invalid split: stand
                    Hand fresh;
                    fresh.cards.push_back(hand.cards[0]);      // first card starts the new hand
                    hand.cards.erase(hand.cards.begin());
                    hand.cards.push_back(shoe.draw());         // existing hand back to 7 cards
                    for (int i = 1; i < kCardsPerHand; ++i) fresh.cards.push_back(shoe.draw());
                    hands.push_back(std::move(fresh));         // played after the current hand
                    break;                                     // decide again on 2 revealed cards
                }
            }
        }
    }

    std::size_t dn = 2;  // dealer stands on 17 or more, hits on 16 or less
    while (best_total(dealer, dn) < 17 && dn < static_cast<std::size_t>(kCardsPerHand)) ++dn;
    const int dealer_total = best_total(dealer, dn);

    int score = 0;
    for (const Hand& hand : hands) {
        const int t = best_total(hand.cards, hand.revealed);
        if (t <= 21 && (dealer_total > 21 || t > dealer_total)) score += hand.doubled ? 2 : 1;
    }
    return score;
}

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
    double alter = 22.0, grow = 12.0, console_every = 2.0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--resume") resume = v;
        else if (a == "--alter") alter = std::stod(v);
        else if (a == "--grow") grow = std::stod(v);
        else if (a == "--console-every") console_every = std::stod(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    fs::create_directories(out);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    Pop pop = resume ? Pop::load(*resume) : Pop(kInQtyMax, 9, 1, Priority::Speed);
    if (!resume) pop.create(kPopulation);
    const Profile profile = make_profile(alter, grow);
    const std::uint64_t deal_seed = seed ? *seed : fresh_seed();
    Rng deals(deal_seed);

    const bool new_csv = !fs::exists(out / "log.csv");
    std::ofstream csv(out / "log.csv", std::ios::app);
    if (new_csv)
        csv << "generation,top10_avg_score,best_score,all_avg_score,top10_avg_nodes,top10_avg_neurons,"
               "top10_avg_v,top10_avg_s,top10_avg_m,elapsed_s\n";
    {
        std::ofstream info(out / "run_info.txt", std::ios::app);
        info << "start " << timestamp() << " deal_seed " << deal_seed << " population_seed " << pop.seed()
             << " alter_pct " << alter << " grow_pct " << grow << " in_qty_max " << kInQtyMax
             << (resume ? " resumed_from " + resume->string() : std::string()) << "\n";
    }
    std::cout << "blackjack trainer: output in " << out.string() << " (deal seed " << deal_seed
              << "). Ctrl+C to stop and save.\n"
              << std::setw(8) << "gen" << std::setw(10) << "top10avg" << std::setw(7) << "best"
              << std::setw(9) << "allavg" << std::setw(9) << "nodes" << std::setw(9) << "neurons"
              << std::setw(7) << "v" << std::setw(7) << "s" << std::setw(7) << "m" << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    auto last_print = t0 - std::chrono::hours(1);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());

    while (!g_stop) {
        // the same 50 deals for every net this generation
        std::vector<std::uint64_t> deal_seeds(kHandsPerNet);
        for (auto& s : deal_seeds) s = deals.next_u64();

        const std::vector<std::uint64_t> ids = pop.ids();
        std::vector<int> scores(ids.size(), 0);
        std::atomic<std::size_t> next{0};
        auto worker = [&] {
            for (std::size_t k; (k = next++) < ids.size();) {
                Net<D>& net = pop.get(ids[k]);
                for (std::uint64_t s : deal_seeds) scores[k] += play_deal(net, s);
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < std::min<unsigned>(threads, static_cast<unsigned>(ids.size())); ++t)
            pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        std::vector<std::size_t> order(ids.size());
        for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });

        Counts sum;
        double top_score = 0, all_score = 0;
        for (std::size_t k = 0; k < ids.size(); ++k) all_score += scores[k];
        for (int r = 0; r < kKeep; ++r) {
            const std::size_t k = order[r];
            top_score += scores[k];
            const Counts c = count(pop.get(ids[k]));
            sum.nodes += c.nodes; sum.neurons += c.neurons; sum.v += c.v; sum.s += c.s; sum.m += c.m;
        }
        const double n = kKeep;
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const std::uint64_t gen = pop.generation();
        std::ostringstream row;
        row << std::fixed << std::setprecision(2) << gen << ',' << top_score / n << ',' << scores[order[0]] << ','
            << all_score / ids.size() << ',' << sum.nodes / n << ',' << sum.neurons / n << ',' << sum.v / n << ','
            << sum.s / n << ',' << sum.m / n << ',' << elapsed << '\n';
        csv << row.str();
        csv.flush();

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_print).count() >= console_every) {
            last_print = now;
            std::cout << std::fixed << std::setprecision(2) << std::setw(8) << gen << std::setw(10) << top_score / n
                      << std::setw(7) << scores[order[0]] << std::setw(9) << all_score / ids.size()
                      << std::setw(9) << sum.nodes / n << std::setw(9) << sum.neurons / n << std::setw(7)
                      << sum.v / n << std::setw(7) << sum.s / n << std::setw(7) << sum.m / n << std::endl;
            save_net(out / "best.jnet", pop.get(ids[order[0]]));
        }

        // cull the bottom 30, refill with 3 mutated offspring of each of the top 10
        for (std::size_t r = kKeep; r < order.size(); ++r) pop.remove(ids[order[r]]);
        for (int r = 0; r < kKeep; ++r)
            for (int c = 0; c < kChildrenPerSurvivor; ++c) pop.mutate(ids[order[r]], profile);
        pop.next_generation();

        if (gen % 1000 == 999) pop.save(out / "population.jpnet");
    }

    pop.save(out / "population.jpnet");
    std::cout << "stopped at generation " << pop.generation() << "; saved " << (out / "population.jpnet").string()
              << " (resume with --resume " << (out / "population.jpnet").string() << ")\n";
    return 0;
}
