// Structural analysis (spec/03 path rule, spec/07 immunity, spec/06 legal depth ranges).
// `removed` (optional) marks node ids treated as absent; ids beyond its size are live.
#pragma once

#include <jneuro/net.hpp>

#include <algorithm>
#include <array>
#include <optional>
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

/// Mix rule (spec/03): every input-to-output path has >= path_mix v, s and m neurons.
template <class D>
bool mix_ok(const Net<D>& net, const Removed* removed = nullptr) {
    if (net.path_mix == 0) return true;
    const std::size_t n = net.nodes.size();
    std::vector<NodeId> order;
    for (NodeId i = 0; i < n; ++i)
        if (detail::alive(removed, i)) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    using Triple = std::array<std::uint64_t, 3>;  // fewest (v, s, m) from any input
    std::vector<std::optional<Triple>> fewest(n);
    for (NodeId i : order) {
        const auto& neuron = net.nodes[i].neuron;
        if (!neuron) { fewest[i] = Triple{0, 0, 0}; continue; }
        std::optional<Triple> best;
        for (NodeId c : connections(*neuron)) {
            if (!detail::alive(removed, c) || !fewest[c]) continue;
            if (!best) best = fewest[c];
            else for (int k = 0; k < 3; ++k) (*best)[k] = std::min((*best)[k], (*fewest[c])[k]);
        }
        if (!best) continue;
        ++(*best)[neuron->index()];  // variant order: VNeuron, SNeuron, MNeuron
        fewest[i] = best;
    }
    for (NodeId o = net.n_inputs; o < net.n_inputs + net.n_outputs; ++o)
        if (detail::alive(removed, o) && fewest[o])
            for (int k = 0; k < 3; ++k)
                if ((*fewest[o])[k] < net.path_mix) return false;
    return true;
}

/// Run rule (spec/03): no input-to-output path has more than max_run 1:1 (v, s) or many:1 (m)
/// neurons in a row.
template <class D>
bool run_ok(const Net<D>& net, const Removed* removed = nullptr) {
    const std::uint64_t k = net.max_run;
    if (k == 0) return true;
    const std::size_t n = net.nodes.size();
    const auto reach = reaching_output(net, removed);
    std::vector<NodeId> order;
    for (NodeId i = 0; i < n; ++i)
        if (detail::alive(removed, i)) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    std::vector<std::optional<std::uint64_t>> run_1(n), run_m(n);  // longest run ending here, from an input
    for (NodeId i : order) {
        const auto& neuron = net.nodes[i].neuron;
        if (!neuron) { run_1[i] = run_m[i] = 0; continue; }
        std::optional<std::uint64_t> best_1, best_m;
        for (NodeId c : connections(*neuron)) {
            if (!detail::alive(removed, c) || !run_1[c]) continue;
            best_1 = std::max(best_1.value_or(0), *run_1[c]);
            best_m = std::max(best_m.value_or(0), *run_m[c]);
        }
        if (!best_1) continue;
        if (std::holds_alternative<MNeuron>(*neuron)) { run_m[i] = *best_m + 1; run_1[i] = 0; }
        else { run_1[i] = *best_1 + 1; run_m[i] = 0; }
        if (reach[i] && (*run_1[i] > k || *run_m[i] > k)) return false;
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
