// Blackjack analysis: how good are the nets of a saved population, and what do they do?
//
// Plays every net, the simple hit rule and the simple double rule on the same fresh deals, then
// prints each net's score and win % with its mix of actions, and for the best net its structure
// and its decision table (most common action per player total and dealer card).
//
// Usage: blackjack_analyze POPULATION.jpnet [--deals N] [--seed N]

#include "game.hpp"

#include <array>
#include <atomic>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

using namespace bj;
using Pop = Population<D>;

namespace {

struct Stats {
    long score = 0, won = 0;
    std::array<long, 4> actions{};  // stand, hit, double, split (as decided by the net)
    // decision table: [player total 4..21][dealer card 2..11][action]
    std::array<std::array<std::array<long, 4>, 12>, 22> table{};
};

template <class Play>
Stats run(const std::vector<Deal>& deals, Play play) {
    Stats st;
    auto observe = [&](const Hand& hand, const Deal& deal, Action a) {
        ++st.actions[a];
        const int total = best_total(hand.cards, hand.revealed), up = points(deal.dealer[0]);
        if (total <= 21) ++st.table[total][up][a];
    };
    for (const Deal& d : deals) {
        const int s = play(d, observe);
        st.score += s;
        st.won += s > 0;
    }
    return st;
}

const char kAct[] = {'S', 'H', 'D', 'P'};

void print_table(const Stats& st, const std::string& title) {
    std::cout << "\n" << title << " (most common action; S stand, H hit, D double, P split; . = never seen)\n"
              << "  total |";
    for (int up = 2; up <= 11; ++up) std::cout << std::setw(3) << (up == 11 ? std::string("A") : std::to_string(up));
    std::cout << "\n  ------+" << std::string(30, '-') << "\n";
    for (int total = 5; total <= 21; ++total) {
        std::cout << "  " << std::setw(5) << total << " |";
        for (int up = 2; up <= 11; ++up) {
            const auto& c = st.table[total][up];
            const long n = c[0] + c[1] + c[2] + c[3];
            if (n == 0) { std::cout << "  ."; continue; }
            const int best = static_cast<int>(std::max_element(c.begin(), c.end()) - c.begin());
            std::cout << "  " << kAct[best];
        }
        std::cout << "\n";
    }
}

std::string mix(const Stats& st) {
    const double n = static_cast<double>(st.actions[0] + st.actions[1] + st.actions[2] + st.actions[3]);
    std::ostringstream o;
    o << std::fixed << std::setprecision(0);
    for (int a = 0; a < 4; ++a) o << kAct[a] << ' ' << std::setw(3) << 100.0 * st.actions[a] / n << "%  ";
    return o.str();
}

void structure(const Net<D>& net) {
    std::map<std::string, int> ops;
    int v = 0, s = 0, m = 0, open = 0;
    for (const auto& node : net.nodes) {
        if (!node.neuron) continue;
        if (const auto* x = std::get_if<VNeuron>(&*node.neuron)) {
            ++v;
            ops["v " + (x->inversion == Inversion::None ? std::string() : to_string(x->inversion) + " ") +
                to_string(x->op) + (x->input ? "" : " (constant)")]++;
        } else if (const auto* x = std::get_if<SNeuron>(&*node.neuron)) {
            ++s;
            ops["s " + to_string(x->op) + " k=" + std::to_string(x->k)]++;
        } else {
            const auto& mn = std::get<MNeuron>(*node.neuron);
            ++m;
            int conn = 0;
            for (const auto& sl : mn.slots) { if (sl) ++conn; else ++open; }
            ops["m " + to_string(mn.op) + " (" + std::to_string(conn) + " connected)"]++;
        }
    }
    // paths from inputs, and path lengths in neurons (min/max), by depth order
    std::vector<NodeId> order(net.nodes.size());
    for (NodeId i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    std::vector<double> paths(net.nodes.size(), 0);
    std::vector<int> shortest(net.nodes.size(), 1 << 30), longest(net.nodes.size(), -1);
    for (NodeId i : order) {
        if (!net.nodes[i].neuron) { paths[i] = 1; shortest[i] = longest[i] = 0; continue; }
        for (NodeId c : connections(net.nodes[i].neuron))
            if (longest[c] >= 0) {
                paths[i] += paths[c];
                shortest[i] = std::min(shortest[i], shortest[c] + 1);
                longest[i] = std::max(longest[i], longest[c] + 1);
            }
    }
    const NodeId out = net.n_inputs;
    std::set<std::uint64_t> depths;
    for (const auto& node : net.nodes) depths.insert(node.depth);
    std::cout << "  neurons " << v + s + m << " (v " << v << ", s " << s << ", m " << m << "), open m slots " << open
              << ", distinct depths " << depths.size() << "\n  input-to-output paths " << paths[out]
              << ", path length " << shortest[out] << "-" << longest[out] << " neurons\n  neurons by kind and op:\n";
    for (auto& [name, n] : ops) std::cout << "    " << std::setw(3) << n << "  " << name << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: blackjack_analyze POPULATION.jpnet [--deals N] [--seed N]\n"; return 2; }
    int n_deals = 100000;
    std::uint64_t seed = 12345;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--deals") n_deals = std::stoi(v);
        else if (a == "--seed") seed = std::stoull(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    Pop pop = Pop::load(argv[1]);
    Rng r(seed);
    std::vector<Deal> deals;
    long winnable_n = 0;
    for (int i = 0; i < n_deals; ++i) {
        deals.emplace_back(r.next_u64());
        winnable_n += winnable(deals.back());
    }
    auto rule = [&](auto fn) {
        return run(deals, [&](const Deal& d, auto& observe) {
            Table t(d);
            while (!t.finished()) {
                const Action a = fn(t.hands[t.h], d);
                observe(t.hands[t.h], d, a);
                t.act(d, a);
            }
            return t.score(d);
        });
    };
    const Stats hit = rule(hit_rule), dbl = rule(double_rule);

    const auto ids = pop.ids();
    std::vector<Stats> nets(ids.size());
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        for (std::size_t k; (k = next++) < ids.size();) {
            Net<D> net = pop.get(ids[k]).copy_genome();
            nets[k] = run(deals, [&](const Deal& d, auto& observe) { return play_deal(net, d, observe); });
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    std::vector<std::size_t> order(ids.size());
    for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return nets[a].score > nets[b].score; });

    auto line = [&](const std::string& who, const Stats& st) {
        std::cout << "  " << std::left << std::setw(16) << who << std::right << std::fixed << std::setprecision(3)
                  << std::setw(8) << static_cast<double>(st.score) / n_deals << std::setprecision(1) << std::setw(9)
                  << 100.0 * st.won / winnable_n << "%   " << mix(st) << "\n";
    };
    std::cout << pop.size() << " nets from " << argv[1] << " (generation " << pop.generation() << "), " << n_deals
              << " fresh deals (seed " << seed << "), " << std::fixed << std::setprecision(1)
              << 100.0 * (n_deals - winnable_n) / n_deals << "% unwinnable\n\n"
              << "  " << std::left << std::setw(16) << "player" << std::right << std::setw(8) << "pts/deal" << std::setw(10)
              << "win%wnbl" << "   actions\n";
    line("double rule", dbl);
    line("hit rule", hit);
    for (std::size_t r2 = 0; r2 < order.size(); ++r2) {
        const auto k = order[r2];
        if (r2 < 10 || r2 == order.size() - 1)
            line("net " + std::to_string(ids[k]) + (r2 == 0 ? " (best)" : ""), nets[k]);
        else if (r2 == 10)
            std::cout << "  ...\n";
    }
    const auto best = order[0];
    std::cout << "\nBest net " << ids[best] << ":\n";
    structure(pop.get(ids[best]));
    print_table(nets[best], "Best net's decisions");
    print_table(dbl, "Double rule");
    return 0;
}
