// Net::create (spec/03) and Net::validate (spec/03, spec/09).
#pragma once

#include <jneuro/analysis.hpp>

#include <cmath>
#include <set>
#include <string>

namespace jneuro {

namespace detail {
inline void check_params(std::uint32_t n_inputs, std::uint32_t n_outputs, std::uint32_t q) {
    if (q < 3 || q % 2 == 0) throw ValidationError("in_qty_max must be an odd integer >= 3");
    if (n_inputs < 1 || n_outputs < 1) throw ValidationError("n_inputs and n_outputs must be >= 1");
}

inline void check_path_mix(std::uint32_t p) {
    if (p > 255) throw ValidationError("path_mix must be in [0, 255]");
}

inline void check_max_run(std::uint32_t k) {
    if (k > 255) throw ValidationError("max_run must be in [0, 255]");
}

inline void check_neuron(const Neuron& n, std::uint32_t q, NodeId i) {
    const std::string at = "node " + std::to_string(i) + ": ";
    if (auto* v = std::get_if<VNeuron>(&n)) {
        if (!std::isfinite(v->weight)) throw ValidationError(at + "weight must be finite");
    } else if (auto* s = std::get_if<SNeuron>(&n)) {
        if (s->k < 1 || s->k > q) throw ValidationError(at + "k range");
        if (s->op == AggOp::Median && s->k % 2 == 0) throw ValidationError(at + "median k must be odd");
    } else {
        if (std::get<MNeuron>(n).op == AggOp::Lag) throw ValidationError(at + "lag is an s_neuron op");
        if (std::get<MNeuron>(n).slots.size() != q)
            throw ValidationError(at + "slot count must equal in_qty_max");
    }
}
}  // namespace detail

template <class D>
Net<D> Net<D>::create(std::uint32_t n_inputs, std::uint32_t n_outputs, std::uint32_t q,
                      Priority priority, Rng& rng, std::uint32_t path_mix, std::uint32_t max_run) {
    detail::check_params(n_inputs, n_outputs, q);
    detail::check_path_mix(path_mix);
    detail::check_max_run(max_run);
    const std::uint32_t k = max_run;
    Net net;
    net.priority = priority;
    net.in_qty_max = q;
    net.n_inputs = n_inputs;
    net.n_outputs = n_outputs;
    net.path_mix = path_mix;
    net.max_run = max_run;
    for (std::uint32_t i = 0; i < n_inputs; ++i) net.nodes.push_back(Node<D>{0, 0, 0, std::nullopt});
    for (std::uint32_t o = 0; o < n_outputs; ++o) net.nodes.push_back(Node<D>{0, 0, d_max, std::nullopt});

    auto slots = [&](const std::vector<NodeId>& ids, std::size_t from, std::size_t to) {
        std::vector<std::optional<NodeId>> s;
        for (std::size_t i = from; i < to; ++i) s.push_back(ids[i]);
        s.resize(q, std::nullopt);
        return s;
    };
    auto identity_v = [&](NodeId src) {
        const VOp op = static_cast<VOp>(rng.choice(4));
        return Neuron{VNeuron{src, Inversion::None, op, identity_weight(op)}};
    };

    // merge tree (n > Q); with a run limit, a level of identity v_neurons after every k m levels
    std::vector<NodeId> level;
    for (NodeId i = 0; i < n_inputs; ++i) level.push_back(i);
    std::vector<std::vector<NodeId>> hidden_levels;
    std::size_t m_run = 0, m_tree = 0;
    auto v_level = [&] {
        std::vector<NodeId> next;
        for (NodeId src : level) {
            net.nodes.push_back(Node<D>{0, 0, 0, identity_v(src)});
            next.push_back(static_cast<NodeId>(net.nodes.size() - 1));
        }
        hidden_levels.push_back(next);
        level = next;
        m_run = 0;
    };
    while (level.size() > q) {
        if (k && m_run == k) v_level();
        std::vector<NodeId> next;
        for (std::size_t c = 0; c < level.size(); c += q) {
            AggOp op = static_cast<AggOp>(rng.choice(4));
            net.nodes.push_back(Node<D>{0, 0, 0,
                MNeuron{op, slots(level, c, std::min<std::size_t>(c + q, level.size()))}});
            next.push_back(static_cast<NodeId>(net.nodes.size() - 1));
        }
        hidden_levels.push_back(next);
        level = next;
        ++m_run;
        ++m_tree;
    }
    if (k && m_run == k && level.size() > 1) v_level();  // the merge m would make the run k + 1
    const std::size_t n_tree = hidden_levels.size();

    // each output's chain: [merge m], extra m (min), s ..., v ...; the last one writes the output
    enum class Part { Merge, Min, S, V, SepV };
    std::vector<Part> chain;
    if (level.size() > 1) chain.push_back(Part::Merge);
    const std::size_t m_have = chain.empty() ? 0 : m_tree + 1;
    for (std::size_t i = m_have; i < path_mix; ++i) chain.push_back(Part::Min);
    for (std::uint32_t i = 0; i < path_mix; ++i) chain.push_back(Part::S);
    const std::uint32_t n_v = (path_mix > 0 || !chain.empty()) ? path_mix : 1;
    for (std::uint32_t i = 0; i < n_v; ++i) chain.push_back(Part::V);
    std::vector<std::vector<Part>> chains;
    std::size_t longest = 0;
    for (std::uint32_t o = 0; o < n_outputs; ++o) {
        // a random order (the merge m, which reads F, stays first) ...
        std::vector<Part> c = chain;
        const std::size_t first = !c.empty() && c[0] == Part::Merge ? 1 : 0;
        for (std::size_t i = c.size(); i-- > first + 1;) {
            const std::size_t j = first + static_cast<std::size_t>(rng.below(i - first + 1));
            std::swap(c[i], c[j]);
        }
        // ... then separators wherever a run would exceed k: a min m between 1:1 neurons, an
        // identity v between m neurons (both pass their input through unchanged)
        std::vector<Part> out;
        std::size_t run_m = m_run, run_1 = 0;
        for (Part part : c) {
            if (part == Part::Merge || part == Part::Min) {
                if (k && run_m == k) { out.push_back(Part::SepV); run_1 = 1; run_m = 0; }
                ++run_m; run_1 = 0;
            } else {
                if (k && run_1 == k) { out.push_back(Part::Min); run_m = 1; run_1 = 0; }
                ++run_1; run_m = 0;
            }
            out.push_back(part);
        }
        longest = std::max(longest, out.size());
        chains.push_back(std::move(out));
    }
    const D step = static_cast<D>(d_max / static_cast<D>(n_tree + longest));  // n_tree + longest - 1 hidden levels
    for (std::size_t li = 0; li < n_tree; ++li)
        for (NodeId id : hidden_levels[li]) net.nodes[id].depth = static_cast<D>(step * (li + 1));
    for (std::uint32_t o = 0; o < n_outputs; ++o) {
        const std::vector<Part>& chain_o = chains[o];
        std::vector<NodeId> prev = level;
        for (std::size_t j = 0; j < chain_o.size(); ++j) {
            Neuron neuron;
            switch (chain_o[j]) {
                case Part::Merge: {
                    AggOp op = static_cast<AggOp>(rng.choice(4));
                    neuron = MNeuron{op, slots(prev, 0, prev.size())};
                    break;
                }
                case Part::Min: neuron = MNeuron{AggOp::Min, slots(prev, 0, prev.size())}; break;
                case Part::S: neuron = SNeuron{prev[0], AggOp::Mean, 1}; break;
                case Part::V:
                case Part::SepV: neuron = identity_v(prev[0]); break;
            }
            if (j + 1 == chain_o.size()) {
                net.nodes[n_inputs + o].neuron = std::move(neuron);
            } else {
                const NodeId id = static_cast<NodeId>(net.nodes.size());
                net.nodes.push_back(Node<D>{0, 0, static_cast<D>(step * (n_tree + 1 + j)), std::move(neuron)});
                prev = {id};
            }
        }
    }
    if (net.nodes.size() > d_max) throw ValidationError("node cap: too many nodes for the depth type");
    if (const char* e = depth_rule_error(net)) throw ValidationError(std::string("create: ") + e);
    for (std::size_t i = 0; i < net.nodes.size(); ++i) net.nodes[i].birth = i;
    net.next_birth = net.nodes.size();
    return net;
}

/// Spec 0.5 nets: re-space depths so the s gap holds (spec/09 reading 0.5 nets). Every connection
/// stays strictly shallower than its reader, so outputs are unchanged.
template <class D>
void respace_05(Net<D>& net) {
    const std::size_t n = net.nodes.size();
    std::vector<NodeId> order(n);
    for (NodeId i = 0; i < n; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    std::vector<std::uint64_t> nd(n, 0);
    const std::uint64_t dmax = Net<D>::d_max;
    for (NodeId id : order) {
        const auto& node = net.nodes[id];
        if (net.is_input(id) || !node.neuron) { nd[id] = 0; continue; }
        std::uint64_t floor = 0;
        for (NodeId c : connections(*node.neuron))
            if (c < n) floor = std::max(floor, nd[c] + 1);
        if (floor == 0) floor = 1;
        if (std::holds_alternative<SNeuron>(*node.neuron)) floor += std::uint64_t{net.in_qty_max} + 1;
        if (net.is_output(id)) {
            if (floor > dmax) throw ValidationError("0.5 conversion: no depth room for the s gap");
            nd[id] = dmax;
        } else {
            nd[id] = std::max<std::uint64_t>(node.depth, floor);
            if (nd[id] > dmax - 1) throw ValidationError("0.5 conversion: no depth room for the s gap");
        }
    }
    for (NodeId i = 0; i < n; ++i) net.nodes[i].depth = static_cast<D>(nd[i]);
}

template <class D>
void Net<D>::validate() const {
    detail::check_params(n_inputs, n_outputs, in_qty_max);
    detail::check_path_mix(path_mix);
    detail::check_max_run(max_run);
    const std::size_t n = nodes.size();
    if (n < std::size_t{n_inputs} + n_outputs) throw ValidationError("fewer nodes than inputs + outputs");
    if (n > d_max) throw ValidationError("node cap: more than D_MAX nodes");
    std::set<std::uint64_t> births;
    for (NodeId i = 0; i < n; ++i) {
        const auto& node = nodes[i];
        const std::string at = "node " + std::to_string(i) + ": ";
        if (is_input(i)) {
            if (node.depth != 0 || node.neuron) throw ValidationError(at + "input must have depth 0 and no neuron");
        } else if (is_output(i)) {
            if (node.depth != d_max) throw ValidationError(at + "output depth must be D_MAX");
        } else if (node.depth < 1 || node.depth > d_max - 1) {
            throw ValidationError(at + "hidden depth must be in [1, D_MAX-1]");
        }
        if (!is_input(i)) {
            if (!node.neuron) throw ValidationError(at + "missing neuron");
            detail::check_neuron(*node.neuron, in_qty_max, i);
            if (const auto* m = std::get_if<MNeuron>(&*node.neuron); m && !m->pos.empty()) {
                if (m->pos.size() != m->slots.size()) throw ValidationError(at + "slot positions");
                for (std::size_t s = 0; s < m->slots.size(); ++s)
                    if (!m->slots[s] && m->pos[s] != 0) throw ValidationError(at + "empty slot with a position");
            }
            if (const auto* v = std::get_if<VNeuron>(&*node.neuron); v && !v->input && v->input_pos != 0)
                throw ValidationError(at + "missing input with a position");
            if (const auto* s = std::get_if<SNeuron>(&*node.neuron); s && !s->input && s->input_pos != 0)
                throw ValidationError(at + "missing input with a position");
        }
        if (node.birth >= next_birth || !births.insert(node.birth).second)
            throw ValidationError(at + "birth must be unique and < next_birth");
    }
    if (const char* e = depth_rule_error(*this)) throw ValidationError(std::string("depth rule: ") + e);
    if (!path_rule_ok(*this)) throw ValidationError("path rule: some input has no path to some output");
    if (!mix_ok(*this)) throw ValidationError("mix rule: some path has too few v, s or m neurons");
    if (!run_ok(*this)) throw ValidationError("run rule: some path has too many 1:1 or many:1 neurons in a row");
    const auto reach = reaching_output(*this);
    for (NodeId i = n_inputs + n_outputs; i < n; ++i)
        if (!reach[i]) throw ValidationError("dead node " + std::to_string(i));
}

}  // namespace jneuro
