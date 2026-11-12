// Adding and removing inputs and outputs of an existing net (spec/12).
// Mirrors python/jneuro/modify.py. Every function validates the given net, leaves it unchanged
// and returns a new, valid net in the reset state.
#pragma once

#include <jneuro/net_build.hpp>
#include <jneuro/removal.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace jneuro {

/// Delete output `index` and every hidden node that only fed it.
template <class D>
Net<D> remove_output(const Net<D>& net, std::uint32_t index);

/// Delete input `index`, the neurons only it reaches, and what only fed them.
template <class D>
Net<D> remove_input(const Net<D>& net, std::uint32_t index);

/// New input at input index `position` (default: last), connected to every output through one
/// chain ending in a v_neuron W. With `active = false`, W is `mul 0` and every existing output
/// stays bit-for-bit unchanged; with `active = true`, W is an identity v_neuron.
template <class D>
Net<D> add_input(const Net<D>& net, std::optional<std::uint32_t> position = std::nullopt,
                 bool active = false, std::uint64_t generation = 0);

/// New output whose neuron copies output `source`'s; it starts with the same values.
template <class D>
Net<D> add_output_clone(const Net<D>& net, std::uint32_t source,
                        std::optional<std::uint32_t> position = std::nullopt, std::uint64_t generation = 0);

/// New output fed by its own minimal chain from the inputs, built as `create` would.
template <class D>
Net<D> add_output_fresh(const Net<D>& net, Rng& rng, std::optional<std::uint32_t> position = std::nullopt,
                        std::uint64_t generation = 0);

// ------------------------------------------------------------------ implementation

namespace detail::modify {

template <class F>
void remap(std::optional<Neuron>& n, F&& f) {
    if (!n) return;
    auto one = [&](std::optional<NodeId>& r) { if (r) r = f(*r); };
    if (auto* m = std::get_if<MNeuron>(&*n)) for (auto& s : m->slots) one(s);
    else if (auto* v = std::get_if<VNeuron>(&*n)) one(v->input);
    else one(std::get<SNeuron>(*n).input);
}

/// Insert a node at id `at`; every id >= at moves up by 1.
template <class D>
void insert_node(Net<D>& net, NodeId at, Node<D> node) {
    auto shift = [at](NodeId c) -> std::optional<NodeId> { return c >= at ? c + 1 : c; };
    for (auto& n : net.nodes) remap(n.neuron, shift);
    remap(node.neuron, shift);
    net.nodes.insert(net.nodes.begin() + at, std::move(node));
}

template <class D>
Node<D> new_node(Net<D>& net, std::uint64_t generation, D depth, std::optional<Neuron> neuron) {
    return Node<D>{net.next_birth++, generation, depth, std::move(neuron)};
}

template <class D>
void check_cap(const Net<D>& net) {
    if (net.nodes.size() > Net<D>::d_max) throw ValidationError("node cap: too many nodes for the depth type");
}

inline std::uint32_t position(std::optional<std::uint32_t> p, std::uint32_t count) {
    const std::uint32_t v = p ? *p : count;
    if (v > count) throw ValidationError("position " + std::to_string(v) + " out of range");
    return v;
}

inline Neuron identity_v(NodeId src) { return VNeuron{src, Inversion::None, VOp::Mul, 1.0}; }

inline std::vector<bool> forward(const std::vector<std::vector<NodeId>>& rd, const std::vector<NodeId>& starts) {
    std::vector<bool> seen(rd.size(), false);
    std::vector<NodeId> stack;
    for (NodeId s : starts) if (!seen[s]) { seen[s] = true; stack.push_back(s); }
    while (!stack.empty()) {
        const NodeId x = stack.back();
        stack.pop_back();
        for (NodeId r : rd[x])
            if (!seen[r]) { seen[r] = true; stack.push_back(r); }
    }
    return seen;
}

/// Nodes from which `start` is reachable (start included).
template <class D>
std::vector<bool> reverse(const Net<D>& net, NodeId start) {
    std::vector<bool> seen(net.nodes.size(), false);
    std::vector<NodeId> stack{start};
    seen[start] = true;
    while (!stack.empty()) {
        const NodeId x = stack.back();
        stack.pop_back();
        for (NodeId c : connections(net.nodes[x].neuron))
            if (!seen[c]) { seen[c] = true; stack.push_back(c); }
    }
    return seen;
}

using Triple = std::array<std::uint64_t, 3>;  // (v, s, m)

/// For each node: fewest (v, s, m) neurons on any path from it to an output, its own included.
template <class D>
std::vector<std::optional<Triple>> fewest_to_output(const Net<D>& net) {
    const auto rd = readers(net);
    std::vector<NodeId> order;
    for (NodeId i = net.n_inputs; i < net.nodes.size(); ++i) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth > net.nodes[b].depth; });
    std::vector<std::optional<Triple>> fewest(net.nodes.size());
    for (NodeId i : order) {
        Triple own{0, 0, 0};
        own[net.nodes[i].neuron->index()] = 1;
        if (net.is_output(i)) { fewest[i] = own; continue; }
        std::optional<Triple> best;
        for (NodeId r : rd[i]) {
            if (!fewest[r]) continue;
            if (!best) best = fewest[r];
            else for (int t = 0; t < 3; ++t) (*best)[t] = std::min((*best)[t], (*fewest[r])[t]);
        }
        if (best) {
            for (int t = 0; t < 3; ++t) (*best)[t] += own[t];
            fewest[i] = best;
        }
    }
    return fewest;
}

}  // namespace detail::modify

template <class D>
Net<D> remove_output(const Net<D>& net, std::uint32_t index) {
    net.validate();
    if (index >= net.n_outputs) throw ValidationError("output index out of range");
    if (net.n_outputs < 2) throw ValidationError("cannot remove the only output");
    Net<D> out = net.copy_genome();
    Removed removed(out.nodes.size(), false);
    removed[out.n_inputs + index] = true;
    garbage_collect(out, removed);
    compact(out, removed);
    out.n_outputs -= 1;
    out.validate();
    return out;
}

template <class D>
Net<D> remove_input(const Net<D>& net, std::uint32_t index) {
    net.validate();
    if (index >= net.n_inputs) throw ValidationError("input index out of range");
    if (net.n_inputs < 2) throw ValidationError("cannot remove the only input");
    Net<D> out = net.copy_genome();
    const auto rd = readers(out);
    std::vector<NodeId> others;
    for (NodeId i = 0; i < out.n_inputs; ++i) if (i != index) others.push_back(i);
    const auto a = detail::modify::forward(rd, {index});
    const auto b = detail::modify::forward(rd, others);
    Removed doomed(out.nodes.size(), false);
    for (NodeId i = 0; i < out.nodes.size(); ++i) doomed[i] = a[i] && !b[i];
    for (NodeId i = 0; i < out.nodes.size(); ++i) {
        if (doomed[i]) continue;
        detail::modify::remap(out.nodes[i].neuron,
                              [&](NodeId c) -> std::optional<NodeId> { if (doomed[c]) return std::nullopt; return c; });
    }
    garbage_collect(out, doomed);
    compact(out, doomed);
    out.n_inputs -= 1;
    out.validate();
    return out;
}

template <class D>
Net<D> add_output_clone(const Net<D>& net, std::uint32_t source, std::optional<std::uint32_t> position,
                        std::uint64_t generation) {
    net.validate();
    if (source >= net.n_outputs) throw ValidationError("source output out of range");
    const std::uint32_t p = detail::modify::position(position, net.n_outputs);
    Net<D> out = net.copy_genome();
    std::optional<Neuron> neuron = out.nodes[out.n_inputs + source].neuron;
    auto node = detail::modify::new_node(out, generation, Net<D>::d_max, std::move(neuron));
    detail::modify::insert_node(out, out.n_inputs + p, std::move(node));
    out.n_outputs += 1;
    detail::modify::check_cap(out);
    out.reset();
    out.validate();
    return out;
}

template <class D>
Net<D> add_output_fresh(const Net<D>& net, Rng& rng, std::optional<std::uint32_t> position,
                        std::uint64_t generation) {
    net.validate();
    const std::uint32_t p = detail::modify::position(position, net.n_outputs);
    const Net<D> g = Net<D>::create(net.n_inputs, 1, net.in_qty_max, net.priority, rng, net.path_mix, net.max_run);
    Net<D> out = net.copy_genome();
    const NodeId n_in = out.n_inputs;
    const NodeId o = n_in + p;
    detail::modify::insert_node(out, o, detail::modify::new_node(out, generation, Net<D>::d_max, std::nullopt));
    out.n_outputs += 1;
    const NodeId base = static_cast<NodeId>(out.nodes.size());
    auto map = [&](NodeId c) -> std::optional<NodeId> {
        if (c < n_in) return c;
        if (c == n_in) return o;
        return base + (c - n_in - 1);
    };
    std::optional<Neuron> top = g.nodes[n_in].neuron;
    detail::modify::remap(top, map);
    out.nodes[o].neuron = std::move(top);
    for (std::size_t i = n_in + 1; i < g.nodes.size(); ++i) {
        std::optional<Neuron> n = g.nodes[i].neuron;
        detail::modify::remap(n, map);
        out.nodes.push_back(detail::modify::new_node(out, generation, g.nodes[i].depth, std::move(n)));
    }
    detail::modify::check_cap(out);
    out.reset();
    out.validate();
    return out;
}

template <class D>
Net<D> add_input(const Net<D>& net, std::optional<std::uint32_t> position, bool active, std::uint64_t generation) {
    namespace mod = detail::modify;
    net.validate();
    const std::uint32_t p = mod::position(position, net.n_inputs);
    Net<D> out = net.copy_genome();
    const std::uint32_t q = out.in_qty_max;
    const std::uint32_t k_run = out.max_run;
    constexpr D d_max = Net<D>::d_max;
    mod::insert_node(out, p, mod::new_node(out, generation, D{0}, std::nullopt));
    out.n_inputs += 1;
    const NodeId n_in = out.n_inputs;
    const std::uint32_t n_out = out.n_outputs;
    const NodeId new_input = p;

    // 1. receivers: greedy cover of the outputs
    std::vector<std::vector<bool>> outs(out.nodes.size(), std::vector<bool>(n_out, false));
    for (std::uint32_t o = 0; o < n_out; ++o) {
        const auto up = mod::reverse(out, n_in + o);
        for (NodeId x = 0; x < out.nodes.size(); ++x) if (up[x]) outs[x][o] = true;
    }
    std::vector<NodeId> cands;
    for (NodeId i = n_in; i < out.nodes.size(); ++i) {
        const auto* m = std::get_if<MNeuron>(&*out.nodes[i].neuron);
        if (!m || (!active && m->op == AggOp::Min)) continue;
        if (std::any_of(m->slots.begin(), m->slots.end(), [](const auto& s) { return !s; })) cands.push_back(i);
    }
    std::vector<bool> uncovered(n_out, true);
    std::vector<NodeId> receivers;
    for (;;) {
        std::optional<NodeId> best;
        std::tuple<std::size_t, D, NodeId> best_key{};
        for (NodeId c : cands) {
            std::size_t gain = 0;
            for (std::uint32_t o = 0; o < n_out; ++o) gain += outs[c][o] && uncovered[o];
            if (!gain) continue;
            // more gain first, then smaller depth, then smaller id
            const std::tuple<std::size_t, D, NodeId> key{n_out - gain, out.nodes[c].depth, c};
            if (!best || key < best_key) { best = c; best_key = key; }
        }
        if (!best) break;
        receivers.push_back(*best);
        for (std::uint32_t o = 0; o < n_out; ++o) if (outs[*best][o]) uncovered[o] = false;
    }

    // 2. a neutral median in front of every output no receiver reaches
    for (std::uint32_t o = 0; o < n_out; ++o) {
        if (!uncovered[o]) continue;
        const NodeId oid = n_in + o;
        Neuron n = *out.nodes[oid].neuron;
        D lo = 1;
        bool any = false;
        for (NodeId c : connections(n)) {
            const D d = static_cast<D>(out.nodes[c].depth + 1);
            if (!any || d > lo) lo = d;
            any = true;
        }
        const D hi = static_cast<D>(d_max - 1);
        if (lo > hi) throw ValidationError("output " + std::to_string(o) + ": no depth room for a receiver");
        const NodeId b = static_cast<NodeId>(out.nodes.size());
        const bool need_sep = std::holds_alternative<MNeuron>(n) && k_run > 0;
        out.nodes.push_back(Node<D>{out.nodes[oid].birth, out.nodes[oid].born_gen,
                                    static_cast<D>(lo + (hi - lo) / 2), n});
        out.nodes[oid] = mod::new_node(out, generation, d_max, std::nullopt);
        NodeId src = b;
        if (need_sep) {
            const D db = out.nodes[b].depth;
            if (db >= hi) throw ValidationError("output " + std::to_string(o) + ": no depth room for a separator");
            out.nodes.push_back(mod::new_node(out, generation, static_cast<D>(db + 1 + (hi - db - 1) / 2),
                                              std::optional<Neuron>{mod::identity_v(b)}));
            src = static_cast<NodeId>(out.nodes.size() - 1);
        }
        std::vector<std::optional<NodeId>> slots(q, std::nullopt);
        for (std::uint32_t s = 0; s < (q + 1) / 2; ++s) slots[s] = src;
        out.nodes[oid].neuron = MNeuron{AggOp::Median, std::move(slots)};
        receivers.push_back(oid);
    }

    // 3. chain contents
    const auto fewest = mod::fewest_to_output(out);
    const std::uint64_t pm = out.path_mix;
    mod::Triple need{0, 0, 0};
    for (NodeId r : receivers)
        for (int t = 0; t < 3; ++t)
            need[t] = std::max(need[t], (*fewest[r])[t] < pm ? pm - (*fewest[r])[t] : 0);
    enum class Part { M, S, V, W };
    std::uint64_t left_m = need[2], left_s = need[1], left_v = need[0] > 0 ? need[0] - 1 : 0;
    std::vector<Part> parts;
    while (left_m || left_s || left_v) {
        if (left_m) { parts.push_back(Part::M); --left_m; }
        if (left_s) { parts.push_back(Part::S); --left_s; }
        if (left_v) { parts.push_back(Part::V); --left_v; }
    }
    parts.push_back(Part::W);
    if (k_run) {
        std::vector<Part> walked;
        std::uint64_t run_1 = 0, run_m = 0;
        for (Part part : parts) {
            if (part == Part::M) {
                if (run_m == k_run) { walked.push_back(Part::V); run_1 = 1; run_m = 0; }
                ++run_m; run_1 = 0;
            } else {
                if (run_1 == k_run) { walked.push_back(Part::M); run_m = 1; run_1 = 0; }
                ++run_1; run_m = 0;
            }
            walked.push_back(part);
        }
        parts = std::move(walked);
    }

    // 4. build
    D d = d_max;
    for (NodeId r : receivers) d = std::min(d, out.nodes[r].depth);
    const D step = static_cast<D>(d / static_cast<D>(parts.size() + 1));
    if (step == 0) throw ValidationError("no depth room for the new input's chain");
    NodeId prev = new_input;
    for (std::size_t j = 0; j < parts.size(); ++j) {
        Neuron neuron;
        switch (parts[j]) {
            case Part::M: {
                std::vector<std::optional<NodeId>> slots(q, std::nullopt);
                slots[0] = prev;
                neuron = MNeuron{AggOp::Min, std::move(slots)};
                break;
            }
            case Part::S: neuron = SNeuron{prev, AggOp::Mean, 1}; break;
            case Part::V: neuron = mod::identity_v(prev); break;
            case Part::W: neuron = VNeuron{prev, Inversion::None, VOp::Mul, active ? 1.0 : 0.0}; break;
        }
        out.nodes.push_back(mod::new_node(out, generation, static_cast<D>(step * static_cast<D>(j + 1)),
                                          std::optional<Neuron>{std::move(neuron)}));
        prev = static_cast<NodeId>(out.nodes.size() - 1);
    }

    // 5. connect
    for (NodeId r : receivers) {
        auto& slots = std::get<MNeuron>(*out.nodes[r].neuron).slots;
        *std::find(slots.begin(), slots.end(), std::nullopt) = prev;
    }
    mod::check_cap(out);
    out.reset();
    out.validate();
    return out;
}

}  // namespace jneuro
