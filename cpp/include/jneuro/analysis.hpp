// Structural analysis (spec/03 path rule, spec/07 immunity, spec/06 legal depth ranges).
// `removed` (optional) marks node ids treated as absent; ids beyond its size are live.
#pragma once

#include <jneuro/net.hpp>

#include <utility>
#include <vector>

namespace jneuro {

using Removed = std::vector<bool>;

/// Connected node ids of a neuron, in slot order (duplicates kept).
inline std::vector<NodeId> connections(const Neuron& n) {
    std::vector<NodeId> out;
    if (auto* m = std::get_if<MNeuron>(&n)) {
        for (auto& s : m->slots) if (s) out.push_back(*s);
    } else if (auto* v = std::get_if<VNeuron>(&n)) {
        if (v->input) out.push_back(*v->input);
    } else if (auto& s = std::get<SNeuron>(n); s.input) {
        out.push_back(*s.input);
    }
    return out;
}

inline std::vector<NodeId> connections(const std::optional<Neuron>& n) {
    return n ? connections(*n) : std::vector<NodeId>{};
}

namespace detail {
inline bool alive(const Removed* removed, NodeId i) {
    return !removed || i >= removed->size() || !(*removed)[i];
}
}  // namespace detail

/// For each node: ascending ids of live neurons connected to it (each once).
template <class D>
std::vector<std::vector<NodeId>> readers(const Net<D>& net, const Removed* removed = nullptr) {
    std::vector<std::vector<NodeId>> out(net.nodes.size());
    for (NodeId i = 0; i < net.nodes.size(); ++i) {
        if (!detail::alive(removed, i)) continue;
        for (NodeId c : connections(net.nodes[i].neuron))
            if (out[c].empty() || out[c].back() != i) out[c].push_back(i);
    }
    return out;
}

/// Live nodes from which some output is reachable (outputs included).
template <class D>
std::vector<bool> reaching_output(const Net<D>& net, const Removed* removed = nullptr) {
    std::vector<bool> marked(net.nodes.size(), false);
    std::vector<NodeId> stack;
    for (NodeId o = net.n_inputs; o < net.n_inputs + net.n_outputs; ++o)
        if (detail::alive(removed, o)) { marked[o] = true; stack.push_back(o); }
    while (!stack.empty()) {
        NodeId i = stack.back();
        stack.pop_back();
        for (NodeId c : connections(net.nodes[i].neuron))
            if (detail::alive(removed, c) && !marked[c]) { marked[c] = true; stack.push_back(c); }
    }
    return marked;
}

/// Every input reaches every output. `cut` is a node no path may pass through.
template <class D>
bool path_rule_ok(const Net<D>& net, const Removed* removed = nullptr,
                  std::optional<NodeId> cut = std::nullopt) {
    auto ok = [&](NodeId i) { return detail::alive(removed, i) && (!cut || *cut != i); };
    const auto rd = readers(net, removed);
    for (NodeId i = 0; i < net.n_inputs; ++i) {
        if (!ok(i)) return false;
        std::vector<bool> seen(net.nodes.size(), false);
        std::vector<NodeId> stack{i};
        seen[i] = true;
        while (!stack.empty()) {
            NodeId x = stack.back();
            stack.pop_back();
            for (NodeId r : rd[x])
                if (ok(r) && !seen[r]) { seen[r] = true; stack.push_back(r); }
        }
        for (NodeId o = net.n_inputs; o < net.n_inputs + net.n_outputs; ++o)
            if (!seen[o]) return false;
    }
    return true;
}

/// True if removing `id` would break the path rule (spec/07).
template <class D>
bool is_immune(const Net<D>& net, NodeId id, const Removed* removed = nullptr) {
    return !path_rule_ok(net, removed, id);
}

/// [lo, hi] depths `id` may take without breaking any neuron (spec/06).
template <class D>
std::pair<D, D> legal_depth_range(const Net<D>& net, NodeId id, const Removed* removed = nullptr) {
    D lo = 1;
    bool any = false;
    for (NodeId c : connections(net.nodes[id].neuron)) {
        if (!detail::alive(removed, c)) continue;
        D d = static_cast<D>(net.nodes[c].depth + 1);
        if (!any || d > lo) lo = d;
        any = true;
    }
    D hi = static_cast<D>(Net<D>::d_max - 1);
    const auto rd = readers(net, removed);
    for (NodeId r : rd[id]) {
        D d = static_cast<D>(net.nodes[r].depth - 1);
        if (d < hi) hi = d;
    }
    return {lo, hi};
}

}  // namespace jneuro
