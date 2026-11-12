// Example trainer: evolve nets that play blackjack, using jneuro::Population.
//
// Each generation every net plays the same 200 deals. The top 10 nets are kept, the other
// 30 are culled, and each survivor is mutated 3 times to refill the population to 40.
// Runs until interrupted (Ctrl+C / SIGINT / SIGTERM), then saves the population.
//
// Usage: blackjack_trainer [--out DIR] [--seed N] [--resume FILE.jpnet] [--rules new|old]
//                          [--console-every SECONDS]
//   new rules (default):   [--edit PCT] [--op PCT] [--grow-gen PCT] [--grow-net PCT] [--tap-weight W]
//   old rules:             [--alter PCT] [--grow PCT] [--big-alter PCT] [--big-grow PCT]
//
// New rules: every child gets parameter edits on up to --edit % of its neurons (default 20)
// and operation changes on up to --op % (default 1, i.e. 5% of the 20%). Growth is rare: each
// generation has a --grow-gen % chance (default 10) of being a growth generation, and then
// each child has a --grow-net % chance (default 10) of exactly one addition attempt. No
// quota-based removal; constant neurons are folded away (spec/07).
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
constexpr int kHandsPerNet = 200;
constexpr int kDecks = 5;          // a 5-deck shoe
constexpr int kRanks = 13;         // 2..10, J, Q, K, A
constexpr int kCardsPerHand = 7;
constexpr int kMaxHands = 4;       // limit on split hands per deal
constexpr std::uint32_t kInQtyMax = 3;
constexpr int kMaxSuddenDeath = 10;  // then remaining tied nets are picked at random
constexpr std::size_t kBigNet = 50;   // above this many neurons, mutate with the small caps

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// ------------------------------------------------------------------ cards
// rank index: 0..8 -> "2".."10", 9 J, 10 Q, 11 K, 12 A
bool is_ace(int r) { return r == 12; }
int points(int r) { return r <= 8 ? r + 2 : (r <= 11 ? 10 : 11); }
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

/// One deal. Every card is drawn up front, in a fixed order: player, dealer, then one block
/// of 7 per possible split. Split k always uses block k, so every net that takes the same
/// splits sees the same cards. The dealer's play does not depend on the player's.
struct Deal {
    std::vector<int> player, dealer, split_cards;
    int dealer_total = 0;

    explicit Deal(std::uint64_t seed) {
        Shoe shoe(seed);
        for (int i = 0; i < kCardsPerHand; ++i) player.push_back(shoe.draw());
        for (int i = 0; i < kCardsPerHand; ++i) dealer.push_back(shoe.draw());
        for (int i = 0; i < (kMaxHands - 1) * kCardsPerHand; ++i) split_cards.push_back(shoe.draw());
        std::size_t dn = 2;  // dealer stands on 17 or more, hits on 16 or less
        while (best_total(dealer, dn) < 17 && dn < static_cast<std::size_t>(kCardsPerHand)) ++dn;
        dealer_total = best_total(dealer, dn);
    }
};

/// The player's side of a deal in progress; `h` is the hand being decided.
struct Table {
    std::vector<Hand> hands;
    std::size_t h = 0;
    std::size_t next_split_card = 0;

    explicit Table(const Deal& d) : hands(1) { hands[0].cards = d.player; }
    bool finished() const { return h >= hands.size(); }

    void act(const Deal& d, Action a) {
        Hand& hand = hands[h];
        switch (a) {
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
                hand.cards.push_back(d.split_cards[next_split_card++]);  // existing hand back to 7
                for (int i = 1; i < kCardsPerHand; ++i) fresh.cards.push_back(d.split_cards[next_split_card++]);
                hands.push_back(std::move(fresh));         // played after the current hand
                break;                                     // decide again on 2 revealed cards
            }
        }
        while (!finished() && hands[h].done) ++h;
    }

    int score(const Deal& d) const {  // win 1, doubled win 2, per hand
        int s = 0;
        for (const Hand& hand : hands) {
            const int t = best_total(hand.cards, hand.revealed);
            if (t <= 21 && (d.dealer_total > 21 || t > d.dealer_total)) s += hand.doubled ? 2 : 1;
        }
        return s;
    }
};

/// Play one deal with `net`; returns the score.
int play_deal(Net<D>& net, const Deal& deal) {
    Table table(deal);
    net.reset();  // s_neuron history covers one deal
    std::vector<double> in(9);
    while (!table.finished()) {
        const Hand& hand = table.hands[table.h];
        in[0] = input_value(deal.dealer[0]);
        in[1] = input_value(deal.dealer[1]);
        for (int i = 0; i < kCardsPerHand; ++i)
            in[2 + i] = static_cast<std::size_t>(i) < hand.revealed ? input_value(hand.cards[i]) : 0.0;
        table.act(deal, decode(net.step(in)[0]));
    }
    return table.score(deal);
}

/// True if some sequence of decisions wins at least one hand (exhaustive search).
bool winnable(const Deal& deal, const Table& table) {
    if (table.finished()) return table.score(deal) > 0;
    for (Action a : {Stand, Hit, Double, Split}) {
        Table next = table;
        next.act(deal, a);
        if (winnable(deal, next)) return true;
    }
    return false;
}
bool winnable(const Deal& deal) { return winnable(deal, Table(deal)); }

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

/// New rules: edits and op changes "up to", optional exactly-one growth attempt, folding.
Profile make_new_profile(double edit_pct, double op_pct, bool grow, double tap_weight) {
    Profile p = Profile::defaults();
    p.set_overall("edit_max_pct", edit_pct);
    p.set_overall("op_change_max_pct", op_pct);
    p.set_overall("growth_max_pct", grow ? 1e-9 : 0.0);  // exact: max(1, ~0) = 1 attempt
    p.set_overall("removal_max_pct", 0.0);
    p.apply_patch(nlohmann::json{{"overall", {{"growth_budget", "exact"}, {"constant_folding", "on"},
        {"add_method_weights", {{"v_tap", tap_weight}, {"s_tap", tap_weight}, {"m_tap", tap_weight}}}}}});
    return p;
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
    double edit_pct = 20.0, op_pct = 1.0, grow_gen = 10.0, grow_net = 10.0, tap_weight = 25.0;
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
    const Profile steady = make_new_profile(edit_pct, op_pct, false, tap_weight),
                  grow_one = make_new_profile(edit_pct, op_pct, true, tap_weight);
    const std::uint64_t deal_seed = seed ? *seed : fresh_seed();
    Rng deals(deal_seed);
    Rng growth_rng(deal_seed ^ 0x9E3779B97F4A7C15ull);  // growth-generation and growth-child draws

    const bool new_csv = !fs::exists(out / "log.csv");
    std::ofstream csv(out / "log.csv", std::ios::app);
    if (new_csv)
        csv << "generation,top10_avg_score,best_score,all_avg_score,top10_avg_nodes,top10_avg_neurons,"
               "top10_avg_v,top10_avg_s,top10_avg_m,unwinnable,best_win_pct_winnable,top10_win_pct_winnable,"
               "sudden_death_deals,growth_children,elapsed_s\n";
    {
        std::ofstream info(out / "run_info.txt", std::ios::app);
        info << "start " << timestamp() << " deal_seed " << deal_seed << " population_seed " << pop.seed()
             << " rules " << rules;
        if (new_rules)
            info << " edit_pct " << edit_pct << " op_pct " << op_pct << " grow_gen_pct " << grow_gen
                 << " grow_net_pct " << grow_net << " tap_weight " << tap_weight << " constant_folding on";
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

    while (!g_stop) {
        // new rules: is this a growth generation? (decided up front so the log row can report it)
        const bool growth_gen = new_rules && growth_rng.u() * 100.0 < grow_gen;
        int growth_children = 0;

        // the same 200 deals for every net this generation
        std::vector<Deal> gen_deals;
        int n_winnable = 0;
        for (int i = 0; i < kHandsPerNet; ++i) {
            gen_deals.emplace_back(deals.next_u64());
            n_winnable += winnable(gen_deals.back());
        }

        const std::vector<std::uint64_t> ids = pop.ids();
        std::vector<int> scores(ids.size(), 0), won(ids.size(), 0);  // won: deals with a winning hand
        std::atomic<std::size_t> next{0};
        auto worker = [&] {
            for (std::size_t k; (k = next++) < ids.size();) {
                Net<D>& net = pop.get(ids[k]);
                for (const Deal& d : gen_deals) {
                    const int s = play_deal(net, d);
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
            const int tied = scores[order[kKeep]];
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
                std::vector<std::pair<int, std::size_t>> r;
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
            << top_pct << ',' << sudden_death << ',';  // growth_children and elapsed added after mutation

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
        for (int r = 0; r < kKeep; ++r) {
            const std::uint64_t parent = ids[order[r]];
            const bool big = count(pop.get(parent)).neurons > kBigNet;
            for (int c = 0; c < kChildrenPerSurvivor; ++c) {
                if (!new_rules) { pop.mutate(parent, big ? big_profile : profile); continue; }
                const bool grows = growth_gen && growth_rng.u() * 100.0 < grow_net;
                growth_children += grows;
                pop.mutate(parent, grows ? grow_one : steady);
            }
        }
        pop.next_generation();
        row << growth_children << ',' << elapsed << '\n';  // same fixed formatting as the rest of the row
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
