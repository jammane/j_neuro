// Net data model and evaluation (spec/03, spec/04, spec/05).
#pragma once

#include <jneuro/aggregate.hpp>
#include <jneuro/numerics.hpp>
#include <jneuro/rng.hpp>
#include <jneuro/types.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <unordered_map>
#include <variant>
#include <vector>

namespace jneuro {

struct VNeuron {
    std::optional<NodeId> input;
    Inversion inversion = Inversion::None;
    VOp op = VOp::Mul;
    double weight = 1.0;
};

struct SNeuron {
    std::optional<NodeId> input;
    AggOp op = AggOp::Mean;
    std::uint32_t k = 1;
};

struct MNeuron {
    AggOp op = AggOp::Mean;
    std::vector<std::optional<NodeId>> slots;
};

using Neuron = std::variant<VNeuron, SNeuron, MNeuron>;

inline NeuronType type_of(const Neuron& n) { return static_cast<NeuronType>(n.index()); }

template <class D>
struct Node {
    std::uint64_t birth = 0;
    std::uint64_t born_gen = 0;
    D depth = 0;
    std::optional<Neuron> neuron;
};

/// History ring of an s_neuron (spec/04). Oldest-to-newest order starts at `head`.
struct Ring {
    std::vector<double> buf;
    std::size_t head = 0;
    std::uint32_t filled = 0;
};

/// Evaluate one v_neuron given its (canonical) input value.
double eval_v(const VNeuron& n, double x);

template <class D>
class Net {
public:
    using depth_t = D;
    static constexpr DepthType depth_type = depth_type_of<D>();
    static constexpr D d_max = std::numeric_limits<D>::max();

    Priority priority = Priority::Balanced;
    std::uint32_t in_qty_max = 5;
    std::uint32_t n_inputs = 0;
    std::uint32_t n_outputs = 0;
    std::uint32_t path_mix = 0;  ///< min v, s and m neurons on every path (spec/03 mix rule), 0..255
    std::uint64_t next_birth = 0;
    std::vector<Node<D>> nodes;

    bool is_input(NodeId i) const { return i < n_inputs; }
    bool is_output(NodeId i) const { return i >= n_inputs && i < n_inputs + n_outputs; }
    bool is_hidden(NodeId i) const { return i >= n_inputs + n_outputs; }

    /// Minimal net satisfying the path and mix rules (spec/03).
    static Net create(std::uint32_t n_inputs, std::uint32_t n_outputs, std::uint32_t in_qty_max,
                      Priority priority, Rng& rng, std::uint32_t path_mix = 2);

    /// Throws ValidationError unless every invariant of spec/03 and spec/09 holds.
    void validate() const;

    /// Steps after a reset before every s_neuron history on any path is filled (spec/05).
    std::uint64_t get_warmup_steps() const;

    /// Evaluate one step (spec/05). Call reset() after editing nodes directly.
    std::vector<double> step(const std::vector<double>& inputs);
    /// Clear all s_neuron state and cached evaluation plans.
    void reset() { state_ = State{}; }

    /// Copy of the genome in the reset state.
    Net copy_genome() const {
        Net n = *this;
        n.reset();
        return n;
    }

private:
    struct State {
        bool ready = false;
        std::vector<NodeId> order;                 // SPEED / BALANCED: cached evaluation order
        std::vector<double> values;                // SPEED / BALANCED: all node values
        std::unordered_map<NodeId, Ring> rings;    // s_neuron histories
    };
    State state_;

    std::vector<NodeId> evaluation_order() const;
    double eval_node(NodeId id, const Neuron& n, auto&& value_of);
};

// ------------------------------------------------------------------ implementation

inline double eval_v(const VNeuron& n, double x) {
    double y = x;
    switch (n.inversion) {
        case Inversion::None: break;
        case Inversion::Bitnot: y = bitnot(x); break;
        case Inversion::Recip: y = ieee_div(1.0, x); break;
    }
    double z = 0.0;
    switch (n.op) {
        case VOp::Mul: z = y * n.weight; break;
        case VOp::Div: z = ieee_div(y, n.weight); break;
        case VOp::Add: z = y + n.weight; break;
        case VOp::Sub: z = y - n.weight; break;
    }
    return canon(z);
}

template <class D>
std::vector<NodeId> Net<D>::evaluation_order() const {
    std::vector<NodeId> order;
    order.reserve(nodes.size() - n_inputs);
    for (NodeId i = n_inputs; i < nodes.size(); ++i) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return nodes[a].depth < nodes[b].depth; });
    return order;
}

template <class D>
std::uint64_t Net<D>::get_warmup_steps() const {
    std::vector<std::uint64_t> w(nodes.size(), 0);
    for (NodeId id : evaluation_order()) {
        const Neuron& n = *nodes[id].neuron;
        std::uint64_t best = 0;
        auto see = [&](const std::optional<NodeId>& r) { if (r && w[*r] > best) best = w[*r]; };
        if (auto* m = std::get_if<MNeuron>(&n)) for (auto& s : m->slots) see(s);
        else if (auto* v = std::get_if<VNeuron>(&n)) see(v->input);
        else see(std::get<SNeuron>(n).input);
        const auto* s = std::get_if<SNeuron>(&n);
        w[id] = (s ? s->k : 0) + best;
    }
    std::uint64_t out = 0;
    for (NodeId o = n_inputs; o < n_inputs + n_outputs; ++o) out = std::max(out, w[o]);
    return out;
}

template <class D>
double Net<D>::eval_node(NodeId id, const Neuron& neuron, auto&& value_of) {
    auto in = [&](const std::optional<NodeId>& r) { return r ? value_of(*r) : 0.0; };
    if (auto* v = std::get_if<VNeuron>(&neuron)) return eval_v(*v, in(v->input));
    if (auto* s = std::get_if<SNeuron>(&neuron)) {
        Ring& ring = state_.rings[id];
        if (ring.buf.size() != s->k) ring = Ring{std::vector<double>(s->k, 0.0), 0, 0};
        const double x = in(s->input);
        std::vector<double> h(s->k);
        std::vector<bool> present(s->k);
        for (std::uint32_t i = 0; i < s->k; ++i) {
            h[i] = ring.buf[(ring.head + i) % s->k];
            present[i] = i >= s->k - ring.filled;
        }
        // lag: the input from k steps ago (+0.0 if unfilled); otherwise x minus the aggregate
        const double out = s->op == AggOp::Lag ? canon(h[0]) : canon(x - aggregate(s->op, h, present));
        ring.buf[ring.head] = x;
        ring.head = (ring.head + 1) % s->k;
        ring.filled = std::min(ring.filled + 1, s->k);
        return out;
    }
    const auto& m = std::get<MNeuron>(neuron);
    std::vector<double> vals(m.slots.size());
    std::vector<bool> present(m.slots.size());
    for (std::size_t i = 0; i < m.slots.size(); ++i) {
        present[i] = m.slots[i].has_value();
        vals[i] = in(m.slots[i]);
    }
    return canon(aggregate(m.op, vals, present));
}

template <class D>
std::vector<double> Net<D>::step(const std::vector<double>& inputs) {
    if (inputs.size() != n_inputs) throw std::invalid_argument("step: wrong number of inputs");
    std::vector<double> outputs(n_outputs);

    if (priority == Priority::Memory) {
        // No cached plan; each value is freed once its last reader has been evaluated.
        const auto order = evaluation_order();
        std::vector<std::size_t> last_use(nodes.size(), 0);
        for (std::size_t p = 0; p < order.size(); ++p)
            std::visit([&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                if constexpr (std::is_same_v<T, MNeuron>) {
                    for (auto& s : n.slots) if (s) last_use[*s] = p;
                } else {
                    if (n.input) last_use[*n.input] = p;
                }
            }, *nodes[order[p]].neuron);
        std::unordered_map<NodeId, double> live;
        for (NodeId i = 0; i < n_inputs; ++i) live[i] = canon(inputs[i]);
        auto value_of = [&](NodeId r) { return live.at(r); };
        for (std::size_t p = 0; p < order.size(); ++p) {
            const NodeId id = order[p];
            const double v = eval_node(id, *nodes[id].neuron, value_of);
            if (is_output(id)) outputs[id - n_inputs] = v;
            live[id] = v;
            std::visit([&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                auto drop = [&](NodeId r) { if (last_use[r] == p && !is_output(r)) live.erase(r); };
                if constexpr (std::is_same_v<T, MNeuron>) {
                    for (auto& s : n.slots) if (s && live.count(*s)) drop(*s);
                } else {
                    if (n.input && live.count(*n.input)) drop(*n.input);
                }
            }, *nodes[id].neuron);
        }
        return outputs;
    }

    // SPEED / BALANCED: cached plan, one contiguous value array.
    if (!state_.ready) {
        state_.order = evaluation_order();
        state_.values.assign(nodes.size(), 0.0);
        state_.ready = true;
    }
    auto& values = state_.values;
    for (NodeId i = 0; i < n_inputs; ++i) values[i] = canon(inputs[i]);
    auto value_of = [&](NodeId r) { return values[r]; };
    for (NodeId id : state_.order) values[id] = eval_node(id, *nodes[id].neuron, value_of);
    for (std::uint32_t o = 0; o < n_outputs; ++o) outputs[o] = values[n_inputs + o];
    return outputs;
}

}  // namespace jneuro
