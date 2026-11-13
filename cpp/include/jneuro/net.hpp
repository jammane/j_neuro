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

// A connection is (node, position): position 0 reads the node, position 1 the length node of the
// node's s_neuron (how many history entries are filled), position j + 1 its history node j (spec/03). The node ids live in `input` / `slots` as before, so every
// structural analysis works on them unchanged; the positions sit beside them.

struct VNeuron {
    std::optional<NodeId> input;
    Inversion inversion = Inversion::None;
    VOp op = VOp::Mul;
    double weight = 1.0;
    std::uint32_t input_pos = 0;  ///< position of the input connection (0 if missing)
};

struct SNeuron {
    std::optional<NodeId> input;
    AggOp op = AggOp::Mean;
    std::uint32_t k = 1;
    std::uint32_t input_pos = 0;
};

struct MNeuron {
    AggOp op = AggOp::Mean;
    std::vector<std::optional<NodeId>> slots;
    std::vector<std::uint32_t> pos;  ///< per slot position; empty means all 0

    std::uint32_t slot_pos(std::size_t i) const { return pos.empty() ? 0 : pos[i]; }
    /// Set slot i to a connection (or empty: node nullopt, position 0).
    void set(std::size_t i, std::optional<NodeId> node, std::uint32_t p = 0) {
        slots[i] = node;
        if (!node) p = 0;
        if (p != 0 && pos.empty()) pos.assign(slots.size(), 0);
        if (!pos.empty()) pos[i] = p;
        if (!pos.empty() && std::all_of(pos.begin(), pos.end(), [](std::uint32_t x) { return x == 0; })) pos.clear();
    }
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
    std::uint32_t max_run = 0;   ///< max 1:1 or many:1 neurons in a row on a path (spec/03 run rule); 0 off
    std::uint64_t next_birth = 0;
    std::vector<Node<D>> nodes;

    bool is_input(NodeId i) const { return i < n_inputs; }
    bool is_output(NodeId i) const { return i >= n_inputs && i < n_inputs + n_outputs; }
    bool is_hidden(NodeId i) const { return i >= n_inputs + n_outputs; }

    /// Minimal net satisfying the path and mix rules (spec/03).
    static Net create(std::uint32_t n_inputs, std::uint32_t n_outputs, std::uint32_t in_qty_max,
                      Priority priority, Rng& rng, std::uint32_t path_mix = 2, std::uint32_t max_run = 3);

    /// Throws ValidationError unless every invariant of spec/03 and spec/09 holds.
    void validate() const;

    /// Steps after a reset before every s_neuron history on any path is filled (spec/05).
    std::uint64_t get_warmup_steps() const;

    /// Evaluate one step (spec/05). Call reset() after editing nodes directly.
    std::vector<double> step(const std::vector<double>& inputs);
    /// Clear all s_neuron state and cached evaluation plans.
    void reset() { state_ = State{}; }
    /// Clear all s_neuron state but keep the cached evaluation plan and warmup. Same results as
    /// reset(), faster; only valid if nodes have not been edited since the last step (else reset()).
    void reset_state() {
        for (Ring& r : state_.rings) {
            std::fill(r.buf.begin(), r.buf.end(), 0.0);
            r.head = 0;
            r.filled = 0;
        }
    }

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
        std::uint64_t warmup = 0;                  // get_warmup_steps(), valid when ready
        std::vector<Ring> rings;                   // s_neuron histories, by node id
        std::vector<double> vals, scratch;         // reused per neuron: no allocation per step
        std::vector<bool> present;
        std::vector<std::pair<NodeId, double>> pushes;  // s_neuron inputs, pushed at the end of the step
    };
    State state_;

    std::vector<NodeId> evaluation_order() const;
    std::uint64_t warmup_of(const std::vector<NodeId>& order) const;
    double eval_node(NodeId id, const Neuron& n, auto&& value_of);
    /// Length node (pos 1) or history node pos - 1 of the s_neuron at `s`, as it stood at the start
    /// of the step (spec/04).
    double history_value(NodeId s, std::uint32_t pos) const;
    void push_histories();
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
    return state_.ready ? state_.warmup : warmup_of(evaluation_order());
}

template <class D>
std::uint64_t Net<D>::warmup_of(const std::vector<NodeId>& order) const {
    std::vector<std::uint64_t> w(nodes.size(), 0);
    // W of a connection: W(node) at position 0; j + W(input of S) for history node j of S (spec/05)
    auto w_rec = [&](auto&& self, NodeId r, std::uint32_t pos) -> std::uint64_t {
        if (pos == 0) return w[r];
        if (pos == 1) return 0;  // the length node is exact from the first step
        const auto& s = std::get<SNeuron>(*nodes[r].neuron);
        return (pos - 1) + (s.input ? self(self, *s.input, s.input_pos) : 0);
    };
    auto w_conn = [&](NodeId r, std::uint32_t pos) { return w_rec(w_rec, r, pos); };
    for (NodeId id : order) {
        const Neuron& n = *nodes[id].neuron;
        std::uint64_t best = 0;
        auto see = [&](const std::optional<NodeId>& r, std::uint32_t pos) {
            if (r && w_conn(*r, pos) > best) best = w_conn(*r, pos);
        };
        if (auto* m = std::get_if<MNeuron>(&n)) for (std::size_t i = 0; i < m->slots.size(); ++i) see(m->slots[i], m->slot_pos(i));
        else if (auto* v = std::get_if<VNeuron>(&n)) see(v->input, v->input_pos);
        else see(std::get<SNeuron>(n).input, std::get<SNeuron>(n).input_pos);
        const auto* s = std::get_if<SNeuron>(&n);
        w[id] = (s ? s->k : 0) + best;
    }
    std::uint64_t out = 0;
    for (NodeId o = n_inputs; o < n_inputs + n_outputs; ++o) out = std::max(out, w[o]);
    return out;
}

template <class D>
double Net<D>::eval_node(NodeId id, const Neuron& neuron, auto&& value_of) {
    auto in = [&](const std::optional<NodeId>& r, std::uint32_t pos) {
        return r ? (pos == 0 ? value_of(*r) : history_value(*r, pos)) : 0.0;
    };
    if (auto* v = std::get_if<VNeuron>(&neuron)) return eval_v(*v, in(v->input, v->input_pos));
    auto& vals = state_.vals;
    auto& present = state_.present;
    if (auto* s = std::get_if<SNeuron>(&neuron)) {
        if (state_.rings.size() <= id) state_.rings.resize(nodes.size());
        Ring& ring = state_.rings[id];
        if (ring.buf.size() != s->k) ring = Ring{std::vector<double>(s->k, 0.0), 0, 0};
        const double x = in(s->input, s->input_pos);
        auto& h = vals;
        h.resize(s->k);
        present.resize(s->k);
        for (std::uint32_t i = 0; i < s->k; ++i) {
            h[i] = ring.buf[(ring.head + i) % s->k];
            present[i] = i >= s->k - ring.filled;
        }
        // lag: the input from k steps ago (+0.0 if unfilled); otherwise x minus the aggregate
        const double out =
            s->op == AggOp::Lag ? canon(h[0]) : canon(x - aggregate(s->op, h, present, state_.scratch));
        state_.pushes.push_back({id, x});  // pushed after the step: history nodes read the old ring
        return out;
    }
    const auto& m = std::get<MNeuron>(neuron);
    vals.resize(m.slots.size());
    present.resize(m.slots.size());
    for (std::size_t i = 0; i < m.slots.size(); ++i) {
        present[i] = m.slots[i].has_value();
        vals[i] = in(m.slots[i], m.slot_pos(i));
    }
    return canon(aggregate(m.op, vals, present, state_.scratch));
}

template <class D>
double Net<D>::history_value(NodeId s, std::uint32_t pos) const {
    if (s >= state_.rings.size()) return 0.0;  // not yet stepped: nothing filled
    const Ring& ring = state_.rings[s];
    if (pos == 1) return static_cast<double>(ring.filled);  // the length node
    const std::uint32_t j = pos - 1;
    const std::size_t k = ring.buf.size();
    if (k == 0 || j > ring.filled) return 0.0;  // unfilled
    return ring.buf[(ring.head + k - j) % k];   // h[k - j], oldest-first from head
}

template <class D>
void Net<D>::push_histories() {
    for (const auto& [id, x] : state_.pushes) {
        Ring& ring = state_.rings[id];
        const auto k = static_cast<std::uint32_t>(ring.buf.size());
        ring.buf[ring.head] = x;
        ring.head = (ring.head + 1) % k;
        ring.filled = std::min(ring.filled + 1, k);
    }
    state_.pushes.clear();
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
        push_histories();
        return outputs;
    }

    // SPEED / BALANCED: cached plan, one contiguous value array.
    if (!state_.ready) {
        state_.order = evaluation_order();
        state_.values.assign(nodes.size(), 0.0);
        state_.warmup = warmup_of(state_.order);
        state_.ready = true;
    }
    auto& values = state_.values;
    for (NodeId i = 0; i < n_inputs; ++i) values[i] = canon(inputs[i]);
    auto value_of = [&](NodeId r) { return values[r]; };
    for (NodeId id : state_.order) values[id] = eval_node(id, *nodes[id].neuron, value_of);
    push_histories();
    for (std::uint32_t o = 0; o < n_outputs; ++o) outputs[o] = values[n_inputs + o];
    return outputs;
}

}  // namespace jneuro
