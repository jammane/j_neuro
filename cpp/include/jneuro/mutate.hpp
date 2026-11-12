// Mutation (spec/06; removal phase in spec/07). Mirrors python/jneuro/mutate.py step for step,
// including the exact order of every random draw.
#pragma once

#include <jneuro/profile.hpp>
#include <jneuro/removal.hpp>

#include <cmath>

namespace jneuro {

template <class D>
Net<D> mutate(const Net<D>& parent, const Profile& profile, std::uint64_t generation, Rng& rng);

/// Give pct % of the v_neurons a new random weight; only weights change (spec/06).
template <class D>
Net<D> inject_diversity(const Net<D>& parent, double pct, Rng& rng, double sigma_reset = 1.0);

// ------------------------------------------------------------------ implementation

namespace detail {

template <class D>
class Mutation {
public:
    Mutation(const Net<D>& parent, const Profile& p, std::uint64_t gen, Rng& rng)
        : net_(parent.copy_genome()), p_(p), gen_(gen), rng_(rng), q_(net_.in_qty_max),
          removed_(net_.nodes.size(), false) {}

    Net<D> run() {
        run_phase("op_change_max_pct", &Mutation::op_change);
        run_phase("edit_max_pct", &Mutation::edit);
        run_phase("growth_max_pct", &Mutation::growth);
        run_phase("removal_max_pct", &Mutation::removal);
        run_phase("depth_shift_max_pct", &Mutation::depth_shift);
        if (p_.constant_folding() == "on")
            fold_constants(net_, removed_, gen_,
                           [this](NeuronType t) { return p_.effective("min_age_for_removal", t); });
        garbage_collect(net_, removed_);
        compact(net_, removed_);
        return std::move(net_);
    }

private:
    using Attempt = std::optional<NeuronType> (Mutation::*)();
    static constexpr std::array<NeuronType, 3> kTypes{NeuronType::V, NeuronType::S, NeuronType::M};

    Net<D> net_;
    const Profile& p_;
    std::uint64_t gen_;
    Rng& rng_;
    std::uint32_t q_;
    Removed removed_;
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
        const bool exact = field == "growth_max_pct" && p_.growth_budget() == "exact";
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
        for (std::int64_t i = 0; i < b; ++i)
            if (auto t = (this->*attempt)()) ++used_[ti(*t)];
    }

    std::optional<NodeId> pick_neuron() {
        auto lst = live_neurons();
        if (lst.empty()) return std::nullopt;
        NodeId id = pick(lst);
        if (cap_reached(type_at(id))) return std::nullopt;
        return id;
    }

    // phase 1
    std::optional<NeuronType> op_change() {
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
        auto change = [&](AggOp& op) {
            std::uint64_t j = rng_.below(3);
            if (j >= static_cast<std::uint64_t>(op)) ++j;
            op = static_cast<AggOp>(j);
        };
        if (auto* s = std::get_if<SNeuron>(&n)) {
            change(s->op);
            if (s->op == AggOp::Median && s->k % 2 == 0) --s->k;
            return NeuronType::S;
        }
        change(std::get<MNeuron>(n).op);
        return NeuronType::M;
    }

    // phase 2
    std::optional<NeuronType> edit() {
        auto id = pick_neuron();
        if (!id) return std::nullopt;
        Neuron& n = *net_.nodes[*id].neuron;
        const NeuronType t = type_of(n);
        if (t != NeuronType::M && rng_.u() < p_.effective("rewire_prob", t)) return rewire(*id, t);
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
        const double z = rng_.normal();
        const std::uint64_t c = std::min<std::uint64_t>(
            q_, 1 + static_cast<std::uint64_t>(std::floor(std::fabs(p_.effective("sigma_slots", NeuronType::M) * z))));
        const D my_depth = net_.nodes[*id].depth;
        bool changed = false;
        for (std::uint64_t r = 0; r < c; ++r) {
            const std::size_t i = static_cast<std::size_t>(rng_.below(q_));
            auto& slots = std::get<MNeuron>(*net_.nodes[*id].neuron).slots;
            if (!slots[i]) {
                std::vector<NodeId> cands;
                for (NodeId j : live_nodes())
                    if (net_.nodes[j].depth < my_depth && std::find(slots.begin(), slots.end(), j) == slots.end())
                        cands.push_back(j);
                if (cands.empty()) continue;
                slots[i] = pick(cands);
                if (!mix_ok(net_, &removed_)) {
                    slots[i] = std::nullopt;
                    continue;
                }
                changed = true;
            } else {
                const auto old = slots[i];
                slots[i] = std::nullopt;
                if (!path_rule_ok(net_, &removed_)) {
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
        std::vector<NodeId> cands;
        for (NodeId j : live_nodes())
            if (net_.nodes[j].depth < my_depth && (!old || j != *old)) cands.push_back(j);
        if (cands.empty()) return std::nullopt;
        input_ref(id) = pick(cands);
        if (!path_rule_ok(net_, &removed_) || !mix_ok(net_, &removed_)) {
            input_ref(id) = old;
            return std::nullopt;
        }
        return t;
    }

    Inversion new_inversion() {
        if (rng_.u() < p_.effective("insert_inversion_prob", NeuronType::V))
            return static_cast<Inversion>(1 + rng_.below(2));
        return Inversion::None;
    }

    // phase 3
    std::optional<NeuronType> growth() {
        const auto& w = p_.add_method_weights();
        const std::size_t method = rng_.weighted({w[0], w[1], w[2], w[3]});
        const std::uint64_t needed = method == 0 ? 2 : 1;
        if (live_nodes().size() + needed > static_cast<std::uint64_t>(Net<D>::d_max)) return std::nullopt;  // node cap
        if (method == 0) return slot_fill();
        const NeuronType t = method == 1 ? NeuronType::V : method == 2 ? NeuronType::S : NeuronType::M;
        if (cap_reached(t)) return std::nullopt;
        return path_insert(method, t);
    }

    std::optional<NeuronType> path_insert(std::size_t method, NeuronType t) {
        std::vector<NodeId> cands;
        for (NodeId i : live_neurons())
            if (method != 3 || std::holds_alternative<MNeuron>(*net_.nodes[i].neuron)) cands.push_back(i);
        if (cands.empty()) return std::nullopt;
        const NodeId a = pick(cands);
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
        const std::uint64_t d_b = lo + rng_.below(hi - lo + 1);
        const NodeId b = static_cast<NodeId>(net_.nodes.size());
        append(Node<D>{net_.nodes[a].birth, net_.nodes[a].born_gen, static_cast<D>(d_b), net_.nodes[a].neuron});
        Neuron x;
        if (method == 1) {
            const VOp op = static_cast<VOp>(rng_.choice(4));
            const Inversion inv = new_inversion();
            x = VNeuron{b, inv, op, identity_weight(op)};
        } else if (method == 2) {
            x = SNeuron{b, static_cast<AggOp>(rng_.choice(4)), 1};
        } else {
            std::vector<std::optional<NodeId>> slots(q_, std::nullopt);
            slots[0] = b;
            x = MNeuron{AggOp::Max, slots};
        }
        auto& node_a = net_.nodes[a];
        node_a.neuron = x;
        node_a.birth = net_.next_birth++;
        node_a.born_gen = gen_;
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
        const NodeId r = pick(recv);
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
        const std::uint64_t d_w = 2 + rng_.below(d_r - 2);
        const std::uint64_t d_q = 1 + rng_.below(d_w - 1);
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
            qn = SNeuron{in, static_cast<AggOp>(rng_.choice(4)), 1};
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
        if (!mix_ok(net_, &removed_)) {  // undo: the new paths break the mix rule
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
            bypass(net_, *id, removed_);
            if (!mix_ok(net_, &removed_)) {  // undo: paths through it would lose a v
                for (auto& [r, n] : saved) net_.nodes[r].neuron = std::move(n);
                removed_[*id] = false;
                return std::nullopt;
            }
            return t;
        }
        if (!is_silenced(net_, *id)) return std::nullopt;
        if (has_min_reader(net_, *id, removed_) || is_immune(net_, *id, &removed_)) return std::nullopt;
        remove_silenced(net_, *id, removed_);
        return t;
    }

    // phase 5
    std::optional<NeuronType> depth_shift() {
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
Net<D> mutate(const Net<D>& parent, const Profile& profile, std::uint64_t generation, Rng& rng) {
    return detail::Mutation<D>(parent, profile, generation, rng).run();
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
