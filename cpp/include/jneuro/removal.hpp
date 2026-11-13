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

/// Output is always +0.0 (and so is everything a live reader reads from it: an s_neuron whose
/// length node is read is not silenced, spec/07).
template <class D>
bool is_silenced(const Net<D>& net, NodeId id, const Removed* removed = nullptr) {
    const Neuron& n = *net.nodes[id].neuron;
    const auto* v = std::get_if<VNeuron>(&n);
    if (v && v->op == VOp::Mul && v->weight == 0.0) return true;
    if (!connections(n).empty()) return false;
    if (v) return eval_v(*v, 0.0) == 0.0;
    if (std::holds_alternative<SNeuron>(n))
        for (NodeId r = 0; r < net.nodes.size(); ++r) {
            if (!detail::alive(removed, r) || !net.nodes[r].neuron) continue;
            bool length_read = false;
            for_each_conn(*net.nodes[r].neuron, [&](NodeId c, std::uint32_t p) { length_read = length_read || (c == id && p == 1); });
            if (length_read) return false;
        }
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
/// Replace every connection to node `old` (any position) by the connection (to, to_pos), or by
/// an empty slot / missing input when `to` is nullopt.
inline void redirect(Neuron& n, NodeId old, std::optional<NodeId> to, std::uint32_t to_pos = 0) {
    if (!to) to_pos = 0;
    if (auto* m = std::get_if<MNeuron>(&n)) {
        for (std::size_t i = 0; i < m->slots.size(); ++i)
            if (m->slots[i] == old) m->set(i, to, to_pos);
    } else if (auto* v = std::get_if<VNeuron>(&n)) {
        if (v->input == old) { v->input = to; v->input_pos = to_pos; }
    } else {
        auto& s = std::get<SNeuron>(n);
        if (s.input == old) { s.input = to; s.input_pos = to_pos; }
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
    const auto& x = std::get<VNeuron>(*net.nodes[id].neuron);
    const auto src = x.input;
    const std::uint32_t src_pos = x.input_pos;  // the input may itself be a history node
    const auto rd = readers(net, &removed);
    for (NodeId r : rd[id]) detail::redirect(*net.nodes[r].neuron, id, src, src_pos);
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
            if (net.is_hidden(r) && connections(*net.nodes[r].neuron).empty() && is_silenced(net, r, &removed) &&
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
    // a connection's constant: the node's at position 0; for history node j of S, +0.0 when S's
    // input is missing or its constant is +0.0 (the history is then always 0), else unknown
    auto cc = [&](auto&& self, NodeId r, std::uint32_t pos) -> std::optional<double> {
        if (pos == 0) return c[r];
        if (pos == 1) return std::nullopt;  // the length node counts up after a reset
        const auto& s = std::get<SNeuron>(*net.nodes[r].neuron);
        if (!s.input) return 0.0;
        const auto in = self(self, *s.input, s.input_pos);
        if (in && *in == 0.0) return 0.0;
        return std::nullopt;
    };
    auto cconn = [&](NodeId r, std::uint32_t pos) { return cc(cc, r, pos); };
    for (NodeId i : order) {
        if (!net.nodes[i].neuron) continue;
        const Neuron& n = *net.nodes[i].neuron;
        if (auto* v = std::get_if<VNeuron>(&n)) {
            const auto in = v->input ? cconn(*v->input, v->input_pos) : std::optional<double>(0.0);
            if (!v->input) c[i] = eval_v(*v, 0.0);
            else if (in) c[i] = eval_v(*v, *in);
            else if (v->op == VOp::Mul && v->weight == 0.0) c[i] = 0.0;
        } else if (auto* s = std::get_if<SNeuron>(&n)) {
            if (!s->input) c[i] = 0.0;
            else if (const auto in = cconn(*s->input, s->input_pos); in && *in == 0.0) c[i] = 0.0;  // +0 only
        } else {
            const auto& m = std::get<MNeuron>(n);
            std::vector<double> values;
            std::vector<bool> present;
            bool all = true;
            for (std::size_t k = 0; k < m.slots.size(); ++k) {
                const auto& s = m.slots[k];
                const auto cv = s ? cconn(*s, m.slot_pos(k)) : std::optional<double>(0.0);
                if (s && !cv) { all = false; break; }
                values.push_back(s ? *cv : 0.0);
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
    PathRuleTracker<D> paths;
    for (NodeId i : cands) {
        Neuron old = *net.nodes[i].neuron;
        net.nodes[i].neuron = VNeuron{std::nullopt, Inversion::None, VOp::Add, *c[i]};
        // restore if the path rule breaks, or a depth rule (an s_neuron whose history is read)
        if (!paths.ok(net, &removed) || history_floors(net, &removed)[i] > 0) net.nodes[i].neuron = std::move(old);
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
