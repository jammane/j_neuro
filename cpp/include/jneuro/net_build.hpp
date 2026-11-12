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

inline void check_neuron(const Neuron& n, std::uint32_t q, NodeId i) {
    const std::string at = "node " + std::to_string(i) + ": ";
    if (auto* v = std::get_if<VNeuron>(&n)) {
        if (!std::isfinite(v->weight)) throw ValidationError(at + "weight must be finite");
    } else if (auto* s = std::get_if<SNeuron>(&n)) {
        if (s->k < 1 || s->k > q) throw ValidationError(at + "k range");
        if (s->op == AggOp::Median && s->k % 2 == 0) throw ValidationError(at + "median k must be odd");
    } else {
        if (std::get<MNeuron>(n).slots.size() != q)
            throw ValidationError(at + "slot count must equal in_qty_max");
    }
}
}  // namespace detail

template <class D>
Net<D> Net<D>::create(std::uint32_t n_inputs, std::uint32_t n_outputs, std::uint32_t q,
                      Priority priority, Rng& rng, std::uint32_t path_mix) {
    detail::check_params(n_inputs, n_outputs, q);
    detail::check_path_mix(path_mix);
    Net net;
    net.priority = priority;
    net.in_qty_max = q;
    net.n_inputs = n_inputs;
    net.n_outputs = n_outputs;
    net.path_mix = path_mix;
    for (std::uint32_t i = 0; i < n_inputs; ++i) net.nodes.push_back(Node<D>{0, 0, 0, std::nullopt});
    for (std::uint32_t o = 0; o < n_outputs; ++o) net.nodes.push_back(Node<D>{0, 0, d_max, std::nullopt});

    auto slots = [&](const std::vector<NodeId>& ids, std::size_t from, std::size_t to) {
        std::vector<std::optional<NodeId>> s;
        for (std::size_t i = from; i < to; ++i) s.push_back(ids[i]);
        s.resize(q, std::nullopt);
        return s;
    };

    // merge tree (n > Q)
    std::vector<NodeId> level;
    for (NodeId i = 0; i < n_inputs; ++i) level.push_back(i);
    std::vector<std::vector<NodeId>> hidden_levels;
    while (level.size() > q) {
        std::vector<NodeId> next;
        for (std::size_t c = 0; c < level.size(); c += q) {
            NodeId id = static_cast<NodeId>(net.nodes.size());
            AggOp op = static_cast<AggOp>(rng.choice(4));
            net.nodes.push_back(Node<D>{0, 0, 0,
                MNeuron{op, slots(level, c, std::min<std::size_t>(c + q, level.size()))}});
            next.push_back(id);
        }
        hidden_levels.push_back(next);
        level = next;
    }
    const std::size_t n_tree = hidden_levels.size();

    // each output's chain: [merge m], extra m (min), s ..., v ...; the last one writes the output
    enum class Part { Merge, Min, S, V };
    std::vector<Part> chain;
    if (level.size() > 1) chain.push_back(Part::Merge);
    const std::size_t m_have = chain.empty() ? 0 : n_tree + 1;
    for (std::size_t i = m_have; i < path_mix; ++i) chain.push_back(Part::Min);
    for (std::uint32_t i = 0; i < path_mix; ++i) chain.push_back(Part::S);
    const std::uint32_t n_v = (path_mix > 0 || !chain.empty()) ? path_mix : 1;
    for (std::uint32_t i = 0; i < n_v; ++i) chain.push_back(Part::V);
    const std::size_t n_levels = n_tree + chain.size() - 1;
    const D step = static_cast<D>(d_max / static_cast<D>(n_levels + 1));
    for (std::size_t li = 0; li < n_tree; ++li)
        for (NodeId id : hidden_levels[li]) net.nodes[id].depth = static_cast<D>(step * (li + 1));
    for (std::uint32_t o = 0; o < n_outputs; ++o) {
        std::vector<NodeId> prev = level;
        for (std::size_t j = 0; j < chain.size(); ++j) {
            Neuron neuron;
            switch (chain[j]) {
                case Part::Merge: {
                    AggOp op = static_cast<AggOp>(rng.choice(4));
                    neuron = MNeuron{op, slots(prev, 0, prev.size())};
                    break;
                }
                case Part::Min: neuron = MNeuron{AggOp::Min, slots(prev, 0, prev.size())}; break;
                case Part::S: neuron = SNeuron{prev[0], AggOp::Mean, 1}; break;
                case Part::V: {
                    VOp op = static_cast<VOp>(rng.choice(4));
                    neuron = VNeuron{prev[0], Inversion::None, op, identity_weight(op)};
                    break;
                }
            }
            if (j + 1 == chain.size()) {
                net.nodes[n_inputs + o].neuron = std::move(neuron);
            } else {
                const NodeId id = static_cast<NodeId>(net.nodes.size());
                net.nodes.push_back(Node<D>{0, 0, static_cast<D>(step * (n_tree + 1 + j)), std::move(neuron)});
                prev = {id};
            }
        }
    }
    if (net.nodes.size() > d_max) throw ValidationError("node cap: too many nodes for the depth type");
    for (std::size_t i = 0; i < net.nodes.size(); ++i) net.nodes[i].birth = i;
    net.next_birth = net.nodes.size();
    return net;
}

template <class D>
void Net<D>::validate() const {
    detail::check_params(n_inputs, n_outputs, in_qty_max);
    detail::check_path_mix(path_mix);
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
            for (NodeId c : connections(*node.neuron)) {
                if (c >= n) throw ValidationError(at + "bad reference");
                if (nodes[c].depth >= node.depth) throw ValidationError(at + "connection depth must be smaller");
            }
        }
        if (node.birth >= next_birth || !births.insert(node.birth).second)
            throw ValidationError(at + "birth must be unique and < next_birth");
    }
    if (!path_rule_ok(*this)) throw ValidationError("path rule: some input has no path to some output");
    if (!mix_ok(*this)) throw ValidationError("mix rule: some path has too few v, s or m neurons");
    const auto reach = reaching_output(*this);
    for (NodeId i = n_inputs + n_outputs; i < n; ++i)
        if (!reach[i]) throw ValidationError("dead node " + std::to_string(i));
}

}  // namespace jneuro
