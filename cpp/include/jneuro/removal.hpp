// Removal candidates, bypass, cascades, garbage collection and compaction (spec/07).
// Functions mark node ids in `removed` (tombstones); `compact` deletes them.
#pragma once

#include <jneuro/analysis.hpp>

#include <deque>

namespace jneuro {

template <class D>
bool is_identity(const Net<D>& net, NodeId id) {
    const auto* v = std::get_if<VNeuron>(&*net.nodes[id].neuron);
    return v && v->input && v->inversion == Inversion::None && v->weight == identity_weight(v->op);
}

/// Output is always +0.0.
template <class D>
bool is_silenced(const Net<D>& net, NodeId id) {
    const Neuron& n = *net.nodes[id].neuron;
    const auto* v = std::get_if<VNeuron>(&n);
    if (v && v->op == VOp::Mul && v->weight == 0.0) return true;
    if (!connections(n).empty()) return false;
    if (v) return eval_v(*v, 0.0) == 0.0;
    return true;  // connection-less s or m neuron
}

template <class D>
bool has_min_reader(const Net<D>& net, NodeId id, const Removed& removed) {
    const auto rd = readers(net, &removed);
    for (NodeId r : rd[id]) {
        const auto* m = std::get_if<MNeuron>(&*net.nodes[r].neuron);
        if (m && m->op == AggOp::Min) return true;
    }
    return false;
}

namespace detail {
inline void redirect(Neuron& n, NodeId old, std::optional<NodeId> to) {
    if (auto* m = std::get_if<MNeuron>(&n)) {
        for (auto& s : m->slots) if (s == old) s = to;
    } else if (auto* v = std::get_if<VNeuron>(&n)) {
        if (v->input == old) v->input = to;
    } else {
        auto& s = std::get<SNeuron>(n);
        if (s.input == old) s.input = to;
    }
}
template <class D>
void fit(const Net<D>& net, Removed& removed) {
    if (removed.size() < net.nodes.size()) removed.resize(net.nodes.size(), false);
}
}  // namespace detail

/// Remove an identity neuron; its readers read its input directly.
template <class D>
void bypass(Net<D>& net, NodeId id, Removed& removed) {
    detail::fit(net, removed);
    const auto src = std::get<VNeuron>(*net.nodes[id].neuron).input;
    const auto rd = readers(net, &removed);
    for (NodeId r : rd[id]) detail::redirect(*net.nodes[r].neuron, id, src);
    removed[id] = true;
}

/// Remove a silenced neuron and cascade downstream (spec/07).
template <class D>
void remove_silenced(Net<D>& net, NodeId id, Removed& removed) {
    detail::fit(net, removed);
    std::deque<NodeId> queue{id};
    while (!queue.empty()) {
        const NodeId y = queue.front();
        queue.pop_front();
        const auto rd = readers(net, &removed)[y];
        for (NodeId r : rd) detail::redirect(*net.nodes[r].neuron, y, std::nullopt);
        removed[y] = true;
        for (NodeId r : rd)
            if (net.is_hidden(r) && connections(*net.nodes[r].neuron).empty() && is_silenced(net, r) &&
                !has_min_reader(net, r, removed))
                queue.push_back(r);
    }
}

/// Mark every live hidden node that reaches no output as removed.
template <class D>
void garbage_collect(Net<D>& net, Removed& removed) {
    detail::fit(net, removed);
    const auto reach = reaching_output(net, &removed);
    for (NodeId i = net.n_inputs + net.n_outputs; i < net.nodes.size(); ++i)
        if (!removed[i] && !reach[i]) removed[i] = true;
}

/// Delete removed nodes, keep order, renumber ids and connections.
template <class D>
void compact(Net<D>& net, Removed& removed) {
    detail::fit(net, removed);
    std::vector<NodeId> map(net.nodes.size(), 0);
    std::vector<Node<D>> keep;
    for (NodeId i = 0; i < net.nodes.size(); ++i) {
        if (removed[i]) continue;
        map[i] = static_cast<NodeId>(keep.size());
        keep.push_back(std::move(net.nodes[i]));
    }
    auto remap = [&](std::optional<NodeId>& r) { if (r) r = map[*r]; };
    for (auto& node : keep) {
        if (!node.neuron) continue;
        if (auto* m = std::get_if<MNeuron>(&*node.neuron)) for (auto& s : m->slots) remap(s);
        else if (auto* v = std::get_if<VNeuron>(&*node.neuron)) remap(v->input);
        else remap(std::get<SNeuron>(*node.neuron).input);
    }
    net.nodes = std::move(keep);
    removed.assign(net.nodes.size(), false);
    net.reset();
}

}  // namespace jneuro
