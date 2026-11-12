// Expansion test: how does mutation alone grow a net? No game, no scoring, no selection.
//
// Starts --lineages independent copies of a minimal net (default: the blackjack shape, 9 inputs,
// 1 output, in_qty_max 3, path_mix 2) and mutates each one generation after generation. Each
// generation it logs, averaged over the lineages:
//   - how often each mutation operation was used (jneuro::mutation_operations());
//   - neurons, input-to-output paths (log10), and width: connections crossing 10 depth cuts.
// At the end it prints, per operation, the average change in log10(paths), neurons and mid-depth
// width in lineage-generations where the operation was used versus where it was not.
//
// Usage: expansion_test [--out DIR] [--seed N] [--lineages N] [--generations N]
//                       [--inputs N] [--outputs N] [--in-qty-max N] [--path-mix N]
//                       [--edit PCT] [--op PCT] [--growth PCT] [--removal PCT] [--depth-shift PCT]
//                       [--tap-weight W] [--clone-weight W] [--patch JSON]
//                       [--grow-gen PCT] [--grow-net PCT] [--profile FILE]
// --profile loads a saved profile (e.g. profiles/expansion.json) in place of the settings built
// from the flags above; --patch, --grow-gen and --grow-net still apply on top of it.
// --grow-gen / --grow-net set the profile's growth_gen_prob / growth_net_prob (in %).
// --patch applies a profile merge patch last, e.g.
//   --patch '{"overall":{"width_peak":0.4,"add_method_weights":{"v_insert":10}}}' 
// Growth defaults to one exact attempt per mutation. --growth PCT makes it an "up to" percentage;
// --growth-exact PCT an exact percentage of the net's neurons (at least 1 attempt).

#include <jneuro/jneuro.hpp>

#include <atomic>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <thread>
#include <vector>

using namespace jneuro;
using D = std::uint32_t;
namespace fs = std::filesystem;

namespace {

constexpr int kCuts = 10;

struct Shape {
    double neurons = 0, log10_paths = 0, open_slots = 0;
    std::array<double, kCuts> width{};
    std::array<double, kCuts> open_by_depth{};  // empty m slots, by the m_neuron's depth (10 equal bands)
};

Shape measure(const Net<D>& net) {
    Shape s;
    s.neurons = static_cast<double>(net.nodes.size() - net.n_inputs);
    // paths from any input to each node (counts can be huge: keep them as doubles)
    std::vector<NodeId> order(net.nodes.size());
    for (NodeId i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    std::vector<double> paths(net.nodes.size(), 0.0);
    for (NodeId i : order) {
        if (!net.nodes[i].neuron) { paths[i] = 1.0; continue; }
        for (NodeId c : connections(net.nodes[i].neuron)) paths[i] += paths[c];
    }
    double total = 0;
    for (NodeId o = net.n_inputs; o < net.n_inputs + net.n_outputs; ++o) total += paths[o];
    s.log10_paths = total > 0 ? std::log10(total) : 0.0;
    const double dmax = static_cast<double>(Net<D>::d_max);
    for (const auto& node : net.nodes)
        if (node.neuron)
            if (const auto* m = std::get_if<MNeuron>(&*node.neuron)) {
                const double e = static_cast<double>(std::count(m->slots.begin(), m->slots.end(), std::nullopt));
                s.open_slots += e;
                const int band = std::min(kCuts - 1, static_cast<int>(kCuts * static_cast<double>(node.depth) / (dmax + 1.0)));
                s.open_by_depth[band] += e;
            }
    for (NodeId i = 0; i < net.nodes.size(); ++i)
        for (NodeId c : connections(net.nodes[i].neuron)) {
            const double a = static_cast<double>(net.nodes[c].depth), b = static_cast<double>(net.nodes[i].depth);
            for (int k = 0; k < kCuts; ++k) {
                const double cut = dmax * (k + 1) / (kCuts + 1);
                if (a < cut && cut <= b) s.width[k] += 1;
            }
        }
    return s;
}

double mid_width(const Shape& s) {  // the middle four cuts
    return (s.width[3] + s.width[4] + s.width[5] + s.width[6]) / 4.0;
}

std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    fs::path out = fs::path("runs") / ("expansion-" + timestamp());
    std::uint64_t seed = 1;
    int lineages = 20, generations = 5000;
    std::uint32_t n_in = 9, n_out = 1, q = 3, path_mix = 2;
    double edit = 20, op = 1, growth = 0, removal = 0, depth_shift = 5, tap_weight = -1, clone_weight = -1;
    std::string user_patch;
    double grow_gen = -1, grow_net = -1;  // % chance a generation grows; % chance a lineage then grows
    std::string profile_file;
    bool growth_exact = true;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--lineages") lineages = std::stoi(v);
        else if (a == "--generations") generations = std::stoi(v);
        else if (a == "--inputs") n_in = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--outputs") n_out = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--in-qty-max") q = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--path-mix") path_mix = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--edit") edit = std::stod(v);
        else if (a == "--op") op = std::stod(v);
        else if (a == "--growth") { growth = std::stod(v); growth_exact = false; }
        else if (a == "--growth-exact") { growth = std::stod(v); growth_exact = true; }
        else if (a == "--removal") removal = std::stod(v);
        else if (a == "--depth-shift") depth_shift = std::stod(v);
        else if (a == "--tap-weight") tap_weight = std::stod(v);
        else if (a == "--clone-weight") clone_weight = std::stod(v);
        else if (a == "--patch") user_patch = v;
        else if (a == "--grow-gen") grow_gen = std::stod(v);
        else if (a == "--grow-net") grow_net = std::stod(v);
        else if (a == "--profile") profile_file = v;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    fs::create_directories(out);

    // The training profile: up-to edits and op changes, constant folding, library defaults for
    // insert_depth (midpoint), width_bias (center), method weights and slot_retarget_prob.
    Profile p = Profile::defaults();
    p.set_overall("edit_max_pct", edit);
    p.set_overall("op_change_max_pct", op);
    p.set_overall("removal_max_pct", removal);
    p.set_overall("depth_shift_max_pct", depth_shift);
    // exact with growth 0: one attempt per generation; exact with a %: that share of the neurons
    p.set_overall("growth_max_pct", growth_exact ? (growth > 0 ? growth : 1e-9) : growth);
    nlohmann::json patch = {{"overall", {{"constant_folding", "on"}, {"growth_budget", growth_exact ? "exact" : "up_to"}}}};
    if (tap_weight >= 0)
        patch["overall"]["add_method_weights"].update({{"v_tap", tap_weight}, {"s_tap", tap_weight}, {"m_tap", tap_weight}});
    if (clone_weight >= 0) patch["overall"]["add_method_weights"]["m_clone"] = clone_weight;
    p.apply_patch(patch);
    if (!profile_file.empty()) {  // a saved profile replaces the settings above
        std::ifstream in(profile_file);
        std::stringstream text;
        text << in.rdbuf();
        p = profile_from_json(text.str());
    }
    if (!user_patch.empty()) p.apply_patch(nlohmann::json::parse(user_patch));
    if (grow_gen >= 0) p.set_overall("growth_gen_prob", grow_gen / 100.0);
    if (grow_net >= 0) p.set_overall("growth_net_prob", grow_net / 100.0);

    const auto& ops = mutation_operations();
    std::vector<Net<D>> nets;
    std::vector<Rng> rngs;
    Rng seeder(seed);
    for (int l = 0; l < lineages; ++l) {
        Rng r(seeder.next_u64());
        nets.push_back(Net<D>::create(n_in, n_out, q, Priority::Speed, r, path_mix));
        rngs.push_back(r);
    }
    {
        std::ofstream info(out / "run_info.txt");
        info << "seed " << seed << " lineages " << lineages << " generations " << generations << " inputs " << n_in
             << " outputs " << n_out << " in_qty_max " << q << " path_mix " << path_mix
             << (profile_file.empty() ? std::string() : " profile_file " + profile_file) << "\nprofile "
             << profile_to_json(p) << "\n";
    }
    std::ofstream csv(out / "log.csv");
    csv << "generation,neurons,log10_paths,mid_width,open_slots";
    for (int k = 0; k < kCuts; ++k) csv << ",width_" << k;
    for (int k = 0; k < kCuts; ++k) csv << ",open_" << k;
    for (const auto& o : ops) csv << "," << o;
    csv << "\n";

    // per-operation effect tally: sums of deltas when the op was used / not used
    struct Effect { double n = 0, paths = 0, neurons = 0, width = 0; };
    std::map<std::string, std::array<Effect, 2>> effect;  // [0] not used, [1] used

    std::vector<Shape> before(lineages);
    for (int l = 0; l < lineages; ++l) before[l] = measure(nets[l]);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    for (int g = 1; g <= generations; ++g) {
        const bool growth_gen = is_growth_generation(seed, static_cast<std::uint64_t>(g), p);  // spec/06
        std::vector<MutationCounts> used(lineages);
        std::vector<Shape> after(lineages);
        std::atomic<int> next{0};
        auto worker = [&] {
            for (int l; (l = next++) < lineages;) {
                nets[l] = mutate(nets[l], p, static_cast<std::uint64_t>(g), rngs[l], &used[l], growth_gen);
                after[l] = measure(nets[l]);
            }
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < std::min<unsigned>(threads, static_cast<unsigned>(lineages)); ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        Shape avg;
        std::map<std::string, double> op_avg;
        for (int l = 0; l < lineages; ++l) {
            avg.neurons += after[l].neurons;
            avg.log10_paths += after[l].log10_paths;
            avg.open_slots += after[l].open_slots;
            for (int k = 0; k < kCuts; ++k) avg.width[k] += after[l].width[k];
            for (int k = 0; k < kCuts; ++k) avg.open_by_depth[k] += after[l].open_by_depth[k];
            const double dp = after[l].log10_paths - before[l].log10_paths, dn = after[l].neurons - before[l].neurons,
                         dw = mid_width(after[l]) - mid_width(before[l]);
            for (const auto& o : ops) {
                const auto it = used[l].find(o);
                const std::uint64_t c = it == used[l].end() ? 0 : it->second;
                op_avg[o] += static_cast<double>(c);
                Effect& e = effect[o][c > 0];
                e.n += 1; e.paths += dp; e.neurons += dn; e.width += dw;
            }
            before[l] = after[l];
        }
        const double n = lineages;
        csv << std::fixed << std::setprecision(4) << g << ',' << avg.neurons / n << ',' << avg.log10_paths / n;
        Shape a = avg;
        for (auto& w : a.width) w /= n;
        for (auto& w : a.open_by_depth) w /= n;
        csv << ',' << mid_width(a) << ',' << avg.open_slots / n;
        for (int k = 0; k < kCuts; ++k) csv << ',' << a.width[k];
        for (int k = 0; k < kCuts; ++k) csv << ',' << a.open_by_depth[k];
        for (const auto& o : ops) csv << ',' << op_avg[o] / n;
        csv << '\n';
        if (g % 500 == 0 || g == generations)
            std::cout << std::fixed << "gen " << g << ": neurons " << std::setprecision(1) << avg.neurons / n
                      << ", log10 paths " << std::setprecision(2) << avg.log10_paths / n << ", open slots "
                      << std::setprecision(1) << avg.open_slots / n << "\n  width by depth cut:" << [&] {
                             std::ostringstream w;
                             w << std::fixed << std::setprecision(1);
                             for (double x : a.width) w << ' ' << x;
                             w << "\n  open slots by depth:";
                             for (double x : a.open_by_depth) w << ' ' << x;
                             return w.str();
                         }() << std::endl;
    }

    std::ostringstream rep;
    rep << "Per operation, average change per lineage-generation when used vs not used:\n"
        << std::left << std::setw(16) << "operation" << std::right << std::setw(10) << "used%" << std::setw(14)
        << "dlog10paths" << std::setw(12) << "(unused)" << std::setw(11) << "dneurons" << std::setw(11) << "(unused)"
        << std::setw(11) << "dmidwidth" << std::setw(11) << "(unused)" << "\n";
    for (const auto& o : ops) {
        const Effect& u = effect[o][1];
        const Effect& x = effect[o][0];
        auto avg = [](double s, double k) { return k > 0 ? s / k : 0.0; };
        rep << std::left << std::setw(16) << o << std::right << std::fixed << std::setprecision(1) << std::setw(10)
            << 100.0 * u.n / (u.n + x.n) << std::setprecision(3) << std::setw(14) << avg(u.paths, u.n) << std::setw(12)
            << avg(x.paths, x.n) << std::setw(11) << avg(u.neurons, u.n) << std::setw(11) << avg(x.neurons, x.n)
            << std::setw(11) << avg(u.width, u.n) << std::setw(11) << avg(x.width, x.n) << "\n";
    }
    std::cout << rep.str();
    std::ofstream(out / "effects.txt") << rep.str();
    std::cout << "output in " << out.string() << "\n";
    return 0;
}
