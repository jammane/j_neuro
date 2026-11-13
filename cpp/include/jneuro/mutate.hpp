// Mutation (spec/06; removal phase in spec/07). Mirrors python/jneuro/mutate.py step for step,
// including the exact order of every random draw.
#pragma once

#include <jneuro/profile.hpp>
#include <jneuro/removal.hpp>

#include <cmath>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace jneuro {

/// Successful uses per mutation operation (informative; not part of the spec's results).
using MutationCounts = std::map<std::string, std::uint64_t>;

/// Every operation name that mutation counts, in a fixed order.
inline const std::vector<std::string>& mutation_operations() {
    static const std::vector<std::string> ops = {
        "op_change", "v_weight", "s_k", "m_slots", "rewire", "m_retarget", "slot_fill", "v_insert",
        "s_insert", "m_after_m", "v_tap", "s_tap", "m_tap", "m_clone", "m_insert", "v_link", "s_link", "type_swap", "bypass", "remove_silenced",
        "depth_shift", "fold_constant"};
    return ops;
}

/// Mutated offspring (spec/06). If `counts` is given, successful uses of each operation are added.
/// `growth_generation` false skips phase 3 (see is_growth_generation).
template <class D>
Net<D> mutate(const Net<D>& parent, const Profile& profile, std::uint64_t generation, Rng& rng,
              MutationCounts* counts = nullptr, bool growth_generation = true);

/// Whether `generation` is a growth generation for a population seeded `seed` (spec/06): a
/// stateless draw, the same on replay and after a save and load.
inline bool is_growth_generation(std::uint64_t seed, std::uint64_t generation, const Profile& profile) {
    const double p = profile.overall("growth_gen_prob");
    if (p >= 1.0) return true;
    Rng r(seed ^ (generation * 0x9E3779B97F4A7C15ull));
    return r.u() < p;
}

/// Give pct % of the v_neurons a new random weight; only weights change (spec/06).
template <class D>
Net<D> inject_diversity(const Net<D>& parent, double pct, Rng& rng, double sigma_reset = 1.0);

// ------------------------------------------------------------------ implementation

namespace detail {

template <class D>
class Mutation {
public:
    Mutation(const Net<D>& parent, const Profile& p, std::uint64_t gen, Rng& rng, bool growth_generation = true)
        : net_(parent.copy_genome()), p_(p), gen_(gen), rng_(rng), q_(net_.in_qty_max),
          removed_(net_.nodes.size(), false), growth_generation_(growth_generation) {}

    MutationCounts counts;  // operation -> successful uses

    Net<D> run() {
        run_phase("op_change_max_pct", &Mutation::op_change);
        run_phase("edit_max_pct", &Mutation::edit);
        const int g = grows();
        if (g == 1) run_phase("growth_max_pct", &Mutation::growth);
        else if (g == -1 && p_.no_growth_swap() == "on" && type_swap()) ++counts["type_swap"];
        run_phase("removal_max_pct", &Mutation::removal);
        run_phase("depth_shift_max_pct", &Mutation::depth_shift);
        if (p_.constant_folding() == "on") {
            const std::uint64_t folded = fold_constants(
                net_, removed_, gen_, [this](NeuronType t) { return p_.effective("min_age_for_removal", t); });
            if (folded) counts["fold_constant"] += folded;
        }
        garbage_collect(net_, removed_);
        compact(net_, removed_);
        return std::move(net_);
    }

private:
    using Attempt = std::optional<NeuronType> (Mutation::*)();
    static constexpr std::array<NeuronType, 3> kTypes{NeuronType::V, NeuronType::S, NeuronType::M};

    /// Phase 3 runs only in a growth generation, and then with growth_net_prob (spec/06).
    /// 1: grow; 0: not a growth generation; -1: a growth generation, but this offspring's draw failed.
    int grows() {
        if (!growth_generation_) return 0;
        const double p = p_.overall("growth_net_prob");
        if (p >= 1.0 || rng_.u() < p) return 1;  // no draw when 1.0
        return -1;
    }

    /// Turn one 1:1 neuron into the other kind, same input: v -> s or s -> v (spec/06).
    bool type_swap() {
        std::vector<NodeId> cands;
        for (NodeId i : live_neurons())
            if (!std::holds_alternative<MNeuron>(*net_.nodes[i].neuron)) cands.push_back(i);
        if (cands.empty()) return false;
        const NodeId i = pick(cands);
        const Neuron old = *net_.nodes[i].neuron;
        if (const auto* v = std::get_if<VNeuron>(&old)) {
            net_.nodes[i].neuron = SNeuron{v->input, static_cast<AggOp>(rng_.choice(kSOps)), 1};
        } else {
            const VOp op = static_cast<VOp>(rng_.choice(4));
            net_.nodes[i].neuron = VNeuron{std::get<SNeuron>(old).input, Inversion::None, op, identity_weight(op)};
        }
        if (!rules_ok()) {  // a path would be short of v or s
            net_.nodes[i].neuron = old;
            return false;
        }
        return true;
    }

    Net<D> net_;
    const Profile& p_;
    std::uint64_t gen_;
    Rng& rng_;
    std::uint32_t q_;
    Removed removed_;
    PathRuleTracker<D> paths_;
    std::vector<NodeId> order_;  // depth_order() cache
    bool growth_generation_ = true;
    std::array<std::optional<std::int64_t>, 3> caps_{};
    std::array<std::int64_t, 3> used_{};

    static std::size_t ti(NeuronType t) { return static_cast<std::size_t>(t); }
    bool live(NodeId i) const { return i >= removed_.size() || !removed_[i]; }
    bool cap_reached(NeuronType t) const { return caps_[ti(t)] && used_[ti(t)] >= *caps_[ti(t)]; }
    NeuronType type_at(NodeId i) const { return type_of(*net_.nodes[i].neuron); }

    void append(Node<D> n) {
        net_.nodes.push_back(std::move(n));
        removed_.push_back(false);
    }

    std::vector<NodeId> live_neurons() const {
        std::vector<NodeId> out;
        for (NodeId i = net_.n_inputs; i < net_.nodes.size(); ++i) if (live(i)) out.push_back(i);
        return out;
    }
    std::vector<NodeId> live_nodes() const {
        std::vector<NodeId> out;
        for (NodeId i = 0; i < net_.nodes.size(); ++i) if (live(i)) out.push_back(i);
        return out;
    }
    template <class T>
    T pick(const std::vector<T>& v) { return v[static_cast<std::size_t>(rng_.below(v.size()))]; }

    void run_phase(const std::string& field, Attempt attempt) {
        const double n = static_cast<double>(live_neurons().size());
        auto limit = [n](double pct) { return pct == 0 ? 0.0 : std::max(1.0, pct * n / 100.0); };  // spec/06
        const bool growth = field == "growth_max_pct";
        const bool fill = growth && p_.growth_budget() == "fill";
        const bool exact = growth && (p_.growth_budget() == "exact" || fill);
        // "up to" draws a fraction of the cap; exact uses the whole cap
        auto draw = [&](double pct) { return rng_.prob_round(exact ? limit(pct) : rng_.u() * limit(pct)); };
        const std::int64_t b = draw(p_.overall(field));
        for (NeuronType t : kTypes) {
            if (p_.has_override(field, t))
                caps_[ti(t)] = draw(p_.override_value(field, t));
            else
                caps_[ti(t)] = std::nullopt;
            used_[ti(t)] = 0;
        }
        // fill: b is a target in neurons added; skipped attempts are retried, up to 10 * b tries.
        // With open_slot_fill_pct, links into empty m slots count toward the same target.
        const double open_pct = fill ? p_.overall("open_slot_fill_pct") : 0.0;
        std::map<NodeId, std::int64_t> quota, fails;
        std::int64_t added = 0;
        for (std::int64_t i = 0; fill ? (added < b && i < 10 * b) : i < b;) {
            if (open_pct > 0) {
                std::optional<NodeId> r;
                for (NodeId j : live_neurons()) {
                    const auto* m = std::get_if<MNeuron>(&*net_.nodes[j].neuron);
                    if (!m) continue;
                    const auto e = std::count(m->slots.begin(), m->slots.end(), std::nullopt);
                    if (!quota.count(j)) {
                        if (static_cast<std::size_t>(e) == m->slots.size()) continue;  // no connected slot
                        quota[j] = round_half_away(open_pct * static_cast<double>(e) / 100.0);
                        fails[j] = 0;
                    }
                    if (!r && quota[j] > 0 && fails[j] < 3 && e > 0) r = j;
                }
                if (r) {  // a link (spec/06 open-slot fill); not counted in tries
                    const NeuronType t = rng_.below(2) == 0 ? NeuronType::V : NeuronType::S;
                    if (live_nodes().size() + 1 > static_cast<std::uint64_t>(Net<D>::d_max)) break;  // node cap
                    if (link_into(t, *r, pick(empty_slots(*r)))) {
                        --quota[*r];
                        ++added;
                        ++counts[t == NeuronType::V ? "v_link" : "s_link"];
                    } else {
                        ++fails[*r];
                    }
                    continue;
                }
            }
            ++i;
            op_ = nullptr;
            if (auto t = (this->*attempt)()) {
                ++used_[ti(*t)];
                ++counts[op_];
                added += std::string_view(op_) == "slot_fill" ? 2 : 1;
            }
        }
    }

    std::optional<NodeId> pick_neuron() {
        auto lst = live_neurons();
        if (lst.empty()) return std::nullopt;
        NodeId id = pick(lst);
        if (cap_reached(type_at(id))) return std::nullopt;
        return id;
    }

    // phase 1
    const char* op_ = nullptr;  // name of the operation being attempted (for counts)

    std::optional<NeuronType> op_change() {
        op_ = "op_change";
        auto id = pick_neuron();
        if (!id) return std::nullopt;
        Neuron& n = *net_.nodes[*id].neuron;
        if (auto* v = std::get_if<VNeuron>(&n)) {
            const std::uint64_t cur = static_cast<std::uint64_t>(v->inversion) * 4 + static_cast<std::uint64_t>(v->op);
            std::uint64_t j = rng_.below(11);
            if (j >= cur) ++j;
            v->inversion = static_cast<Inversion>(j / 4);
            v->op = static_cast<VOp>(j % 4);
            v->weight = identity_weight(v->op);
            return NeuronType::V;
        }
        auto change = [&](AggOp& op, std::uint64_t n_ops) {  // a different op, uniformly
            std::uint64_t j = rng_.below(n_ops - 1);
            if (j >= static_cast<std::uint64_t>(op)) ++j;
            op = static_cast<AggOp>(j);
        };
        if (auto* s = std::get_if<SNeuron>(&n)) {
            change(s->op, kSOps);  // s: also lag (spec/04)
            if (s->op == AggOp::Median && s->k % 2 == 0) --s->k;
            return NeuronType::S;
        }
        change(std::get<MNeuron>(n).op, kMOps);
        return NeuronType::M;
    }

    // phase 2
    std::optional<NeuronType> edit() {
        auto id = pick_neuron();
        if (!id) return std::nullopt;
        Neuron& n = *net_.nodes[*id].neuron;
        const NeuronType t = type_of(n);
        if (t != NeuronType::M && rng_.u() < p_.effective("rewire_prob", t)) {
            op_ = "rewire";
            return rewire(*id, t);
        }
        op_ = t == NeuronType::V ? "v_weight" : t == NeuronType::S ? "s_k" : "m_slots";
        if (auto* v = std::get_if<VNeuron>(&n)) {
            const double z = rng_.normal();
            const double sw = p_.effective("sigma_weight", NeuronType::V);
            double w2;
            if (v->op == VOp::Add || v->op == VOp::Sub) {
                w2 = v->weight + sw * z;
            } else {
                const double a = sw * z;
                const double f = a >= 0 ? 1.0 + a : 1.0 / (1.0 - a);
                w2 = v->weight * f;
            }
            if (rng_.u() < p_.effective("sign_flip_prob", NeuronType::V)) w2 = -w2;
            if (!std::isfinite(w2)) return std::nullopt;
            v->weight = w2;
            return NeuronType::V;
        }
        if (auto* s = std::get_if<SNeuron>(&n)) {
            const double z = rng_.normal();
            const std::int64_t d = round_half_away(p_.effective("sigma_k", NeuronType::S) * z);
            std::int64_t k2 = std::clamp<std::int64_t>(static_cast<std::int64_t>(s->k) + d, 1, q_);
            if (s->op == AggOp::Median && k2 % 2 == 0) k2 = k2 > s->k ? k2 - 1 : k2 + 1;
            if (k2 == s->k) return std::nullopt;
            s->k = static_cast<std::uint32_t>(k2);
            return NeuronType::S;
        }
        const double pr = p_.effective("slot_retarget_prob", NeuronType::M);
        if (pr > 0 && rng_.u() < pr) {  // no draw when off (spec/06)
            op_ = "m_retarget";
            return retarget(*id);
        }
        const double z = rng_.normal();
        const std::uint64_t c = std::min<std::uint64_t>(
            q_, 1 + static_cast<std::uint64_t>(std::floor(std::fabs(p_.effective("sigma_slots", NeuronType::M) * z))));
        const D my_depth = net_.nodes[*id].depth;
        bool changed = false;
        for (std::uint64_t r = 0; r < c; ++r) {
            auto& slots = std::get<MNeuron>(*net_.nodes[*id].neuron).slots;
            std::size_t i;
            if (p_.slot_edit() == "biased") {  // add or cut first, then which slot (spec/06)
                const bool add = rng_.u() < p_.overall("slot_add_prob");
                std::vector<std::size_t> pool;
                for (std::size_t s = 0; s < slots.size(); ++s)
                    if (!slots[s].has_value() == add) pool.push_back(s);
                if (pool.empty()) continue;
                i = pool[static_cast<std::size_t>(rng_.below(pool.size()))];
            } else {
                i = static_cast<std::size_t>(rng_.below(q_));
            }
            if (!slots[i]) {
                std::vector<NodeId> cands;
                for (NodeId j : live_nodes())
                    if (net_.nodes[j].depth < my_depth && std::find(slots.begin(), slots.end(), j) == slots.end())
                        cands.push_back(j);
                if (cands.empty()) continue;
                slots[i] = pick(cands);
                if (!rules_ok()) {
                    slots[i] = std::nullopt;
                    continue;
                }
                changed = true;
            } else {
                const auto old = slots[i];
                if (p_.prune() == "static" && !constant_values(net_, removed_)[*old])
                    continue;  // only static (constant) sources may be cut
                slots[i] = std::nullopt;
                if (!path_ok()) {
                    slots[i] = old;
                    continue;
                }
                changed = true;
            }
        }
        return changed ? std::optional<NeuronType>(NeuronType::M) : std::nullopt;
    }

    std::optional<NodeId>& input_ref(NodeId id) {
        Neuron& n = *net_.nodes[id].neuron;
        if (auto* v = std::get_if<VNeuron>(&n)) return v->input;
        return std::get<SNeuron>(n).input;
    }

    std::optional<NeuronType> rewire(NodeId id, NeuronType t) {
        const std::optional<NodeId> old = input_ref(id);
        if (old) {
            const auto rd = readers(net_, &removed_);
            if (rd[*old].size() == 1 && rd[*old][0] == id) return std::nullopt;  // would orphan old
        }
        const D my_depth = net_.nodes[id].depth;
        const std::uint64_t lo = span_floor(my_depth);
        std::vector<NodeId> cands;
        for (NodeId j : live_nodes())
            if (net_.nodes[j].depth >= lo && net_.nodes[j].depth < my_depth && (!old || j != *old)) cands.push_back(j);
        if (cands.empty()) return std::nullopt;
        input_ref(id) = pick(cands);
        if (!path_ok() || !rules_ok()) {
            input_ref(id) = old;
            return std::nullopt;
        }
        return t;
    }

    /// Shallowest source depth a link, rewire or retarget may reach from a node at `depth`
    /// (spec/06 link_max_span).
    std::uint64_t span_floor(std::uint64_t depth) const {
        const auto span = static_cast<std::uint64_t>(p_.overall("link_max_span") * static_cast<double>(Net<D>::d_max));
        return depth > span ? depth - span : 0;
    }

    /// The structural rules a mutation may not break besides the path rule (spec/03).
    bool rules_ok() {
        const auto& order = depth_order();
        return mix_ok(net_, &removed_, &order) && run_ok(net_, &removed_, &order);
    }

    /// Every node id (dead ones too) by depth, ties by id. Kept between calls and re-sorted only when
    /// a scan finds it stale: n ids < n in strictly increasing (depth, id) order are exactly 0..n-1.
    const std::vector<NodeId>& depth_order() {
        const std::size_t n = net_.nodes.size();
        auto before = [&](NodeId a, NodeId b) {
            return net_.nodes[a].depth != net_.nodes[b].depth ? net_.nodes[a].depth < net_.nodes[b].depth : a < b;
        };
        bool fresh = order_.size() == n;
        for (std::size_t k = 0; fresh && k < n; ++k)
            fresh = order_[k] < n && (k == 0 || before(order_[k - 1], order_[k]));
        if (!fresh) {
            order_.resize(n);
            for (NodeId i = 0; i < n; ++i) order_[i] = i;
            std::stable_sort(order_.begin(), order_.end(),
                             [&](NodeId a, NodeId b) { return net_.nodes[a].depth < net_.nodes[b].depth; });
        }
        return order_;
    }

    /// The path rule on the net as it stands (or with `cut` taken out), from the incremental cache.
    bool path_ok(std::optional<NodeId> cut = std::nullopt) { return paths_.ok(net_, &removed_, cut); }

    /// An m_neuron to receive new width: uniform, or weighted toward mid-depth (spec/06 width_bias).
    NodeId pick_receiver(const std::vector<NodeId>& recv, const char* peak_field = "width_peak",
                         bool by_empty = false) {
        const std::string& bias = p_.width_bias();
        auto empty_slots = [&](NodeId i) {
            const auto& s = std::get<MNeuron>(*net_.nodes[i].neuron).slots;
            return static_cast<double>(std::count(s.begin(), s.end(), std::nullopt));
        };
        if (by_empty && bias == "none") {  // links: weight by the number of empty slots
            std::vector<double> w;
            for (NodeId i : recv) w.push_back(empty_slots(i));
            return recv[rng_.weighted(w)];
        }
        if (bias == "center" || bias == "deficit") {
            const double peak = p_.overall(peak_field);  // taps: tap_peak; others: width_peak
            const double d_max = static_cast<double>(Net<D>::d_max);
            std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;  // (source depth, reader depth)
            if (bias == "deficit")
                for (NodeId i = 0; i < net_.nodes.size(); ++i)
                    if (!removed_[i])
                        for (NodeId c : connections(net_.nodes[i].neuron))
                            edges.emplace_back(net_.nodes[c].depth, net_.nodes[i].depth);
            std::vector<double> w;
            for (NodeId i : recv) {
                const std::uint64_t d = net_.nodes[i].depth;
                const double x = static_cast<double>(d) / d_max;
                double tri = x < peak ? x / peak : (1.0 - x) / (1.0 - peak);
                if (bias == "deficit") {  // bell target over current width at R's depth
                    std::uint64_t width = 0;
                    for (auto& [a, b] : edges) width += (a < d && d <= b);
                    tri = tri / static_cast<double>(std::max<std::uint64_t>(width, 1));
                }
                double weight = tri + 1e-6;  // > 0, so an output-only net still has a pick
                if (by_empty) weight *= empty_slots(i);
                w.push_back(weight);
            }
            return recv[rng_.weighted(w)];
        }
        return pick(recv);
    }

    /// Move one non-critical m slot to another node no deeper than the one it reads (spec/06).
    std::optional<NeuronType> retarget(NodeId id) {
        auto& slots = std::get<MNeuron>(*net_.nodes[id].neuron).slots;
        std::vector<std::size_t> conn;
        for (std::size_t i = 0; i < slots.size(); ++i) if (slots[i]) conn.push_back(i);
        if (conn.empty()) return std::nullopt;
        const std::size_t i = pick(conn);
        const std::optional<NodeId> old = slots[i];
        if (p_.prune() == "static" && std::count(slots.begin(), slots.end(), old) == 1) {
            const auto rd = readers(net_, &removed_);
            if (rd[*old].size() == 1 && rd[*old][0] == id) return std::nullopt;  // would strand old's branch
        }
        slots[i] = std::nullopt;
        if (!path_ok()) { slots[i] = old; return std::nullopt; }  // critical
        std::vector<NodeId> cands;
        const std::uint64_t lo = span_floor(net_.nodes[id].depth);
        for (NodeId j : live_nodes())
            if (net_.nodes[j].depth >= lo && net_.nodes[j].depth <= net_.nodes[*old].depth && j != *old &&
                std::find(slots.begin(), slots.end(), std::optional<NodeId>(j)) == slots.end())
                cands.push_back(j);
        if (cands.empty()) { slots[i] = old; return std::nullopt; }
        slots[i] = pick(cands);
        if (!rules_ok()) { slots[i] = old; return std::nullopt; }
        return NeuronType::M;
    }

    /// Depth for a new node in [lo, hi]: uniform draw, or the midpoint (spec/06 insert_depth).
    std::uint64_t place(std::uint64_t lo, std::uint64_t hi) {
        if (p_.insert_depth() == "midpoint") return lo + (hi - lo) / 2;
        return lo + rng_.below(hi - lo + 1);
    }

    Inversion new_inversion() {
        if (rng_.u() < p_.effective("insert_inversion_prob", NeuronType::V))
            return static_cast<Inversion>(1 + rng_.below(2));
        return Inversion::None;
    }

    // phase 3
    std::optional<NeuronType> growth() {
        const auto& w = p_.add_method_weights();
        const std::size_t method = rng_.weighted({w.begin(), w.end()});
        static const char* const kNames[] = {"slot_fill", "v_insert", "s_insert", "m_after_m", "v_tap",  "s_tap",
                                             "m_tap",     "m_clone",  "m_insert", "v_link",    "s_link"};
        op_ = kNames[method];
        const std::uint64_t needed = method == 0 ? 2 : 1;
        if (live_nodes().size() + needed > static_cast<std::uint64_t>(Net<D>::d_max)) return std::nullopt;  // node cap
        if (method == 0) return slot_fill();
        // v_insert/v_tap, s_insert/s_tap, m_after_m/m_tap; m_clone, m_insert; v_link, s_link
        const NeuronType t = method == 9 ? NeuronType::V : method == 10 ? NeuronType::S
                           : method >= 7 ? NeuronType::M : kTypes[(method - 1) % 3];
        if (cap_reached(t)) return std::nullopt;
        if (method == 7) return m_clone();
        if (method >= 9) return link(t);
        if (method == 8) return path_insert(method, t);
        if (method >= 4) return input_tap(t);
        return path_insert(method, t);
    }

    /// Copy an m_neuron U read by an m_neuron R into an empty slot of R (spec/06).
    std::optional<NeuronType> m_clone() {
        auto m_reads = [&](NodeId i) {  // distinct m-written nodes R reads, ascending
            std::vector<NodeId> out;
            for (const auto& s : std::get<MNeuron>(*net_.nodes[i].neuron).slots)
                if (s && net_.nodes[*s].neuron && std::holds_alternative<MNeuron>(*net_.nodes[*s].neuron))
                    out.push_back(*s);  // input nodes have no neuron
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        };
        std::vector<NodeId> recv;
        for (NodeId i : live_neurons()) {
            const auto* m = std::get_if<MNeuron>(&*net_.nodes[i].neuron);
            if (!m || std::find(m->slots.begin(), m->slots.end(), std::nullopt) == m->slots.end()) continue;
            if (!m_reads(i).empty()) recv.push_back(i);
        }
        if (recv.empty()) return std::nullopt;
        const NodeId r = pick_receiver(recv);
        const std::vector<NodeId> ups = m_reads(r);
        const NodeId u = pick(ups);
        std::vector<std::size_t> empties;
        {
            const auto& slots = std::get<MNeuron>(*net_.nodes[r].neuron).slots;
            for (std::size_t s = 0; s < slots.size(); ++s) if (!slots[s]) empties.push_back(s);
        }
        const std::size_t slot = pick(empties);
        const NodeId x = static_cast<NodeId>(net_.nodes.size());
        MNeuron copy = std::get<MNeuron>(*net_.nodes[u].neuron);
        append(Node<D>{net_.next_birth, gen_, net_.nodes[u].depth, std::move(copy)});
        net_.next_birth += 1;
        std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = x;
        if (!rules_ok()) {  // cannot happen (paths mirror U's); kept as a guard
            std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = std::nullopt;
            net_.nodes.resize(x);
            removed_.resize(x);
            net_.next_birth -= 1;
            return std::nullopt;
        }
        return NeuronType::M;
    }

    /// A new 1:1 neuron from an existing node into an empty m slot (spec/06 v_link, s_link).
    std::optional<NeuronType> link(NeuronType t) {
        std::vector<NodeId> recv;
        for (NodeId i : live_neurons()) {
            const auto* m = std::get_if<MNeuron>(&*net_.nodes[i].neuron);
            if (!m) continue;
            bool empty = false, connected = false;
            for (const auto& s : m->slots) (s ? connected : empty) = true;
            if (empty && connected) recv.push_back(i);
        }
        if (recv.empty()) return std::nullopt;
        const NodeId r = pick_receiver(recv, "width_peak", true);
        return link_into(t, r, pick(empty_slots(r)));
    }

    std::vector<std::size_t> empty_slots(NodeId r) const {
        std::vector<std::size_t> out;
        const auto& slots = std::get<MNeuron>(*net_.nodes[r].neuron).slots;
        for (std::size_t s = 0; s < slots.size(); ++s)
            if (!slots[s]) out.push_back(s);
        return out;
    }

    /// Links steps 3-6 (spec/06): a new 1:1 neuron from an existing node into empty slot `slot` of R.
    std::optional<NeuronType> link_into(NeuronType t, NodeId r, std::size_t slot) {
        std::uint64_t hi_src = 0;  // sources no deeper than what R already reads
        for (const auto& s : std::get<MNeuron>(*net_.nodes[r].neuron).slots)
            if (s) hi_src = std::max<std::uint64_t>(hi_src, net_.nodes[*s].depth);
        const std::uint64_t lo_src = span_floor(net_.nodes[r].depth);  // longest link allowed
        std::vector<NodeId> cands;
        {
            const auto& slots = std::get<MNeuron>(*net_.nodes[r].neuron).slots;
            for (NodeId j : live_nodes())
                if (net_.nodes[j].depth >= lo_src && net_.nodes[j].depth <= hi_src &&
                    std::find(slots.begin(), slots.end(), std::optional<NodeId>(j)) == slots.end())
                    cands.push_back(j);
        }
        if (cands.empty()) return std::nullopt;
        const NodeId src = pick(cands);
        const std::uint64_t lo = static_cast<std::uint64_t>(net_.nodes[src].depth) + 1;
        const std::uint64_t hi = static_cast<std::uint64_t>(net_.nodes[r].depth) - 1;
        if (lo > hi) return std::nullopt;
        const std::uint64_t d = place(lo, hi);
        Neuron neuron;
        if (t == NeuronType::V) {
            const VOp op = static_cast<VOp>(rng_.choice(4));
            neuron = VNeuron{src, new_inversion(), op, identity_weight(op)};
        } else {
            neuron = SNeuron{src, static_cast<AggOp>(rng_.choice(kSOps)), 1};
        }
        const NodeId x = static_cast<NodeId>(net_.nodes.size());
        append(Node<D>{net_.next_birth, gen_, static_cast<D>(d), std::move(neuron)});
        net_.next_birth += 1;
        std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = x;
        if (!rules_ok()) {
            std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = std::nullopt;
            net_.nodes.resize(x);
            removed_.resize(x);
            net_.next_birth -= 1;
            return std::nullopt;
        }
        return t;
    }

    /// A new neuron reading network inputs, feeding an empty slot of an m_neuron (spec/06).
    std::optional<NeuronType> input_tap(NeuronType t) {
        std::vector<NodeId> recv;
        for (NodeId i : live_neurons()) {
            const auto* m = std::get_if<MNeuron>(&*net_.nodes[i].neuron);
            if (!m) continue;
            bool empty = false, connected = false;
            for (const auto& s : m->slots) (s ? connected : empty) = true;
            if (empty && connected) recv.push_back(i);
        }
        if (recv.empty()) return std::nullopt;
        const NodeId r = pick_receiver(recv, "tap_peak");
        std::vector<std::size_t> empties;
        std::uint64_t hi = 0;  // deepest node R reads
        {
            const auto& slots = std::get<MNeuron>(*net_.nodes[r].neuron).slots;
            for (std::size_t s = 0; s < slots.size(); ++s) {
                if (!slots[s]) empties.push_back(s);
                else hi = std::max<std::uint64_t>(hi, net_.nodes[*slots[s]].depth);
            }
        }
        const std::size_t slot = pick(empties);
        if (hi < 1) return std::nullopt;  // R reads only inputs: no room
        const std::uint64_t d = place(1, hi);
        const std::uint64_t n_in = net_.n_inputs;
        Neuron neuron;
        if (t == NeuronType::V) {
            const VOp op = static_cast<VOp>(rng_.choice(4));
            const NodeId in = static_cast<NodeId>(rng_.below(n_in));
            neuron = VNeuron{in, new_inversion(), op, identity_weight(op)};
        } else if (t == NeuronType::S) {
            const NodeId in = static_cast<NodeId>(rng_.below(n_in));
            neuron = SNeuron{in, static_cast<AggOp>(rng_.choice(kSOps)), 1};
        } else {
            const std::uint64_t filled = (q_ + 1) / 2;                         // half the slots, rounded up
            const std::uint64_t from_inputs = std::min((filled + 1) / 2, n_in);  // half of those from inputs
            std::vector<NodeId> ins;
            for (NodeId i = 0; i < n_in; ++i) ins.push_back(i);
            std::vector<std::optional<NodeId>> slots;
            for (std::uint64_t k = 0; k < from_inputs; ++k) {
                const std::size_t idx = static_cast<std::size_t>(rng_.below(ins.size()));
                slots.push_back(ins[idx]);
                ins.erase(ins.begin() + static_cast<std::ptrdiff_t>(idx));
            }
            std::vector<NodeId> cands;
            for (NodeId j : live_nodes())
                if (net_.nodes[j].depth < d && std::find(slots.begin(), slots.end(), std::optional<NodeId>(j)) == slots.end())
                    cands.push_back(j);
            const std::uint64_t n_rand = std::min<std::uint64_t>(filled - from_inputs, cands.size());
            for (std::uint64_t k = 0; k < n_rand; ++k) {
                const std::size_t idx = static_cast<std::size_t>(rng_.below(cands.size()));
                slots.push_back(cands[idx]);
                cands.erase(cands.begin() + static_cast<std::ptrdiff_t>(idx));
            }
            slots.resize(q_, std::nullopt);
            neuron = MNeuron{static_cast<AggOp>(rng_.choice(kMOps)), slots};
        }
        const NodeId x = static_cast<NodeId>(net_.nodes.size());
        append(Node<D>{net_.next_birth, gen_, static_cast<D>(d), std::move(neuron)});
        net_.next_birth += 1;
        std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = x;
        if (!rules_ok()) {  // undo: the new paths break the mix rule
            std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = std::nullopt;
            net_.nodes.resize(x);
            removed_.resize(x);
            net_.next_birth -= 1;
            return std::nullopt;
        }
        return t;
    }

    std::optional<NeuronType> path_insert(std::size_t method, NeuronType t) {
        std::vector<NodeId> cands;
        for (NodeId i : live_neurons())
            if (method != 3 || std::holds_alternative<MNeuron>(*net_.nodes[i].neuron)) cands.push_back(i);
        if (cands.empty()) return std::nullopt;
        const NodeId a = pick(cands);
        const D old_depth = net_.nodes[a].depth;
        std::uint64_t lo = 1;
        bool any = false;
        for (NodeId c : connections(*net_.nodes[a].neuron)) {
            if (!live(c)) continue;
            const std::uint64_t d = static_cast<std::uint64_t>(net_.nodes[c].depth) + 1;
            if (!any || d > lo) lo = d;
            any = true;
        }
        std::uint64_t hi = static_cast<std::uint64_t>(net_.nodes[a].depth) - 1;
        if (lo > hi) {
            if (!net_.is_hidden(a)) return std::nullopt;
            const auto rd = readers(net_, &removed_)[a];
            std::uint64_t ub = static_cast<std::uint64_t>(Net<D>::d_max) - 1;
            for (std::size_t k = 0; k < rd.size(); ++k) {
                const std::uint64_t d = static_cast<std::uint64_t>(net_.nodes[rd[k]].depth) - 1;
                if (k == 0 || d < ub) ub = d;
            }
            if (lo > ub - 1) return std::nullopt;
            net_.nodes[a].depth = static_cast<D>(ub);
            hi = ub - 1;
        }
        const std::uint64_t d_b = place(lo, hi);
        const NodeId b = static_cast<NodeId>(net_.nodes.size());
        append(Node<D>{net_.nodes[a].birth, net_.nodes[a].born_gen, static_cast<D>(d_b), net_.nodes[a].neuron});
        Neuron x;
        if (method == 1) {
            const VOp op = static_cast<VOp>(rng_.choice(4));
            const Inversion inv = new_inversion();
            x = VNeuron{b, inv, op, identity_weight(op)};
        } else if (method == 2) {
            x = SNeuron{b, static_cast<AggOp>(rng_.choice(kSOps)), 1};
        } else {
            std::vector<std::optional<NodeId>> slots(q_, std::nullopt);
            slots[0] = b;
            x = MNeuron{method == 8 ? AggOp::Min : AggOp::Max, slots};  // m_insert: min of one slot, neutral
        }
        auto& node_a = net_.nodes[a];
        const auto saved_neuron = node_a.neuron;
        const std::uint64_t saved_birth = node_a.birth, saved_gen = node_a.born_gen;
        node_a.neuron = x;
        node_a.birth = net_.next_birth++;
        node_a.born_gen = gen_;
        if (!rules_ok()) {  // undo: a run would become too long
            auto& na = net_.nodes[a];
            na.neuron = saved_neuron;
            na.birth = saved_birth;
            na.born_gen = saved_gen;
            na.depth = old_depth;
            net_.nodes.resize(b);
            removed_.resize(b);
            net_.next_birth -= 1;
            return std::nullopt;
        }
        return t;
    }

    std::optional<NeuronType> slot_fill() {
        std::vector<NodeId> recv;
        for (NodeId i : live_neurons()) {
            const auto* m = std::get_if<MNeuron>(&*net_.nodes[i].neuron);
            if (m && m->op != AggOp::Min && std::find(m->slots.begin(), m->slots.end(), std::nullopt) != m->slots.end())
                recv.push_back(i);
        }
        if (recv.empty()) return std::nullopt;
        const NodeId r = pick_receiver(recv);
        std::vector<std::size_t> empties;
        {
            const auto& slots = std::get<MNeuron>(*net_.nodes[r].neuron).slots;
            for (std::size_t s = 0; s < slots.size(); ++s) if (!slots[s]) empties.push_back(s);
        }
        const std::size_t slot = pick(empties);
        const auto& tw = p_.slot_fill_type_weights();
        const NeuronType t = kTypes[rng_.weighted({tw[0], tw[1], tw[2]})];
        if (cap_reached(t)) return std::nullopt;
        const std::uint64_t d_r = net_.nodes[r].depth;
        if (d_r < 3) return std::nullopt;
        const std::uint64_t d_w = place(2, d_r - 1);
        const std::uint64_t d_q = place(1, d_w - 1);
        std::vector<NodeId> cands;
        for (NodeId j : live_nodes()) if (net_.nodes[j].depth < d_q) cands.push_back(j);
        Neuron qn;
        if (t == NeuronType::V) {
            const NodeId in = pick(cands);
            const VOp op = static_cast<VOp>(rng_.choice(4));
            const Inversion inv = new_inversion();
            qn = VNeuron{in, inv, op, identity_weight(op)};
        } else if (t == NeuronType::S) {
            const NodeId in = pick(cands);
            qn = SNeuron{in, static_cast<AggOp>(rng_.choice(kSOps)), 1};
        } else {
            const std::uint64_t count = 1 + rng_.below(std::min<std::uint64_t>(q_, cands.size()));
            std::vector<NodeId> rem = cands;
            std::vector<std::optional<NodeId>> slots;
            for (std::uint64_t k = 0; k < count; ++k) {
                const std::size_t idx = static_cast<std::size_t>(rng_.below(rem.size()));
                slots.push_back(rem[idx]);
                rem.erase(rem.begin() + static_cast<std::ptrdiff_t>(idx));
            }
            slots.resize(q_, std::nullopt);
            qn = MNeuron{static_cast<AggOp>(rng_.choice(4)), slots};
        }
        const NodeId q_id = static_cast<NodeId>(net_.nodes.size());
        append(Node<D>{net_.next_birth, gen_, static_cast<D>(d_q), qn});
        const NodeId w_id = static_cast<NodeId>(net_.nodes.size());
        append(Node<D>{net_.next_birth + 1, gen_, static_cast<D>(d_w), VNeuron{q_id, Inversion::None, VOp::Mul, 0.0}});
        net_.next_birth += 2;
        std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = w_id;
        if (!rules_ok()) {  // undo: the new paths break the mix rule
            std::get<MNeuron>(*net_.nodes[r].neuron).slots[slot] = std::nullopt;
            net_.nodes.resize(q_id);
            removed_.resize(q_id);
            net_.next_birth -= 2;
            return std::nullopt;
        }
        return t;
    }

    // phase 4
    std::optional<NeuronType> removal() {
        auto id = pick_neuron();
        if (!id) return std::nullopt;
        const NeuronType t = type_at(*id);
        if (!net_.is_hidden(*id)) return std::nullopt;
        const double age = static_cast<double>(gen_) - static_cast<double>(net_.nodes[*id].born_gen);
        if (age < p_.effective("min_age_for_removal", t)) return std::nullopt;
        if (is_identity(net_, *id)) {
            std::vector<std::pair<NodeId, Neuron>> saved;
            const auto rd = readers(net_, &removed_);
            for (NodeId r : rd[*id]) saved.emplace_back(r, *net_.nodes[r].neuron);
            op_ = "bypass";
            bypass(net_, *id, removed_);
            if (!rules_ok()) {  // undo: paths through it would lose a v
                for (auto& [r, n] : saved) net_.nodes[r].neuron = std::move(n);
                removed_[*id] = false;
                return std::nullopt;
            }
            return t;
        }
        if (!is_silenced(net_, *id)) return std::nullopt;
        if (has_min_reader(net_, *id, removed_) || !path_ok(*id)) return std::nullopt;
        op_ = "remove_silenced";
        remove_silenced(net_, *id, removed_);
        return t;
    }

    // phase 5
    std::optional<NeuronType> depth_shift() {
        op_ = "depth_shift";
        std::vector<NodeId> hidden;
        for (NodeId i = net_.n_inputs + net_.n_outputs; i < net_.nodes.size(); ++i) if (live(i)) hidden.push_back(i);
        if (hidden.empty()) return std::nullopt;
        const NodeId id = pick(hidden);
        const NeuronType t = type_at(id);
        if (cap_reached(t)) return std::nullopt;
        const auto [lo, hi] = legal_depth_range(net_, id, &removed_);
        if (lo == hi) return std::nullopt;
        std::uint64_t d;
        if (p_.depth_shift_mode() == "center")
            d = static_cast<std::uint64_t>(lo) + (static_cast<std::uint64_t>(hi) - lo) / 2;
        else
            d = static_cast<std::uint64_t>(lo) + rng_.below(static_cast<std::uint64_t>(hi) - lo + 1);
        if (d == net_.nodes[id].depth) return std::nullopt;
        net_.nodes[id].depth = static_cast<D>(d);
        return t;
    }
};

}  // namespace detail

template <class D>
Net<D> mutate(const Net<D>& parent, const Profile& profile, std::uint64_t generation, Rng& rng,
              MutationCounts* counts, bool growth_generation) {
    detail::Mutation<D> m(parent, profile, generation, rng, growth_generation);
    Net<D> child = m.run();
    if (counts)
        for (auto& [k, v] : m.counts) (*counts)[k] += v;
    return child;
}

template <class D>
Net<D> inject_diversity(const Net<D>& parent, double pct, Rng& rng, double sigma_reset) {
    Net<D> net = parent.copy_genome();
    std::vector<NodeId> vs;
    for (NodeId i = 0; i < net.nodes.size(); ++i)
        if (net.nodes[i].neuron && std::holds_alternative<VNeuron>(*net.nodes[i].neuron)) vs.push_back(i);
    const std::int64_t want = rng.prob_round(pct * static_cast<double>(vs.size()) / 100.0);
    const std::size_t count = std::min<std::size_t>(vs.size(), static_cast<std::size_t>(std::max<std::int64_t>(want, 0)));
    for (std::size_t c = 0; c < count; ++c) {
        const std::size_t k = static_cast<std::size_t>(rng.below(vs.size()));
        const NodeId i = vs[k];
        vs.erase(vs.begin() + static_cast<std::ptrdiff_t>(k));
        std::get<VNeuron>(*net.nodes[i].neuron).weight = rng.normal() * sigma_reset;
    }
    return net;
}

}  // namespace jneuro
