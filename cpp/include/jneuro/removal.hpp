// Removal candidates, bypass, cascades, garbage collection and compaction (spec/07).
// Functions mark node ids in `removed` (tombstones); `compact` deletes them.
#pragma once

#include <jneuro/analysis.hpp>

#include <jneuro/aggregate.hpp>
#include <jneuro/numerics.hpp>

#include <algorithm>
#include <deque>
#include <optional>

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

/// For each live node: its output if provably the same at every step from reset (spec/07).
template <class D>
std::vector<std::optional<double>> constant_values(const Net<D>& net, const Removed& removed) {
    std::vector<NodeId> order;
    for (NodeId i = 0; i < net.nodes.size(); ++i)
        if (detail::alive(&removed, i)) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    std::vector<std::optional<double>> c(net.nodes.size());
    for (NodeId i : order) {
        if (!net.nodes[i].neuron) continue;
        const Neuron& n = *net.nodes[i].neuron;
        if (auto* v = std::get_if<VNeuron>(&n)) {
            if (!v->input) c[i] = eval_v(*v, 0.0);
            else if (c[*v->input]) c[i] = eval_v(*v, *c[*v->input]);
            else if (v->op == VOp::Mul && v->weight == 0.0) c[i] = 0.0;
        } else if (auto* s = std::get_if<SNeuron>(&n)) {
            if (!s->input || (c[*s->input] && *c[*s->input] == 0.0)) c[i] = 0.0;  // canonical: +0 only
        } else {
            const auto& m = std::get<MNeuron>(n);
            std::vector<double> values;
            std::vector<bool> present;
            bool all = true;
            for (const auto& s : m.slots) {
                if (s && !c[*s]) { all = false; break; }
                values.push_back(s ? *c[*s] : 0.0);
                present.push_back(s.has_value());
            }
            if (all) c[i] = canon(aggregate(m.op, values, present));
        }
    }
    return c;
}

/// Replace constant hidden neurons by constant v_neurons (spec/07); no RNG, no quota.
/// Returns the number of neurons folded.
template <class D, class MinAge>
std::uint64_t fold_constants(Net<D>& net, Removed& removed, std::uint64_t generation, MinAge min_age) {
    detail::fit(net, removed);
    const auto c = constant_values(net, removed);
    std::vector<NodeId> cands;
    for (NodeId i = 0; i < net.nodes.size(); ++i) {
        if (removed[i] || !net.is_hidden(i) || !c[i] || connections(net.nodes[i].neuron).empty()) continue;
        const double age = static_cast<double>(generation) - static_cast<double>(net.nodes[i].born_gen);
        if (age >= min_age(type_of(*net.nodes[i].neuron))) cands.push_back(i);
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth > net.nodes[b].depth; });
    std::uint64_t folded = 0;
    for (NodeId i : cands) {
        Neuron old = *net.nodes[i].neuron;
        net.nodes[i].neuron = VNeuron{std::nullopt, Inversion::None, VOp::Add, *c[i]};
        if (!path_rule_ok(net, &removed)) net.nodes[i].neuron = std::move(old);
        else ++folded;
    }
    return folded;
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
