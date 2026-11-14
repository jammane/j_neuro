// Recording evaluation and gradients for the blackjack tuner (examples/blackjack/tune.cpp).
//
// Tape evaluates a net step by step exactly as Net::step does (same order, same arithmetic), but
// keeps every node's value at every step of a deal, so the gradient of an output can be taken back
// through all steps (s_neuron histories reach back into earlier steps). Connections carry positions
// (spec/03): a length node reads how many history entries are filled, a history node the s_neuron's
// input from earlier steps.
//
// Derivatives, where the forward pass is not smooth:
//   v  bitnot: flipping every bit maps each binade linearly onto another, so its slope there is
//      2^(exponent of ~x - exponent of x) (0 where x or ~x is not a normal number); recip: -1/x^2.
//   m/s min, max, median: the whole gradient goes to the value the aggregate picked (first index
//   on ties); mean: 1/n to each slot. Absent slots and unfilled history are constants.
//   Any non-finite value or derivative gives 0.
#pragma once

#include "game.hpp"

#include <cmath>

namespace bj {

inline std::vector<NodeId> eval_order(const Net<D>& net) {
    std::vector<NodeId> order;
    for (NodeId i = net.n_inputs; i < net.nodes.size(); ++i) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    return order;
}

/// Index of the value min, max or median picked (first on ties), or -1 if it is an absent slot.
inline int picked(AggOp op, const std::vector<double>& v, const std::vector<bool>& present, double r) {
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (op == AggOp::Min && !present[i]) continue;
        if (v[i] == r) return present[i] ? static_cast<int>(i) : -1;
    }
    return -1;
}

inline double finite_or_0(double x) { return std::isfinite(x) ? x : 0.0; }

class Tape {
public:
    explicit Tape(const Net<D>& net) : net_(&net), order_(eval_order(net)) {}

    std::size_t steps() const { return vals_.size(); }
    const std::vector<double>& at(std::size_t t) const { return vals_[t]; }
    double output(std::size_t t) const { return vals_[t][net_->n_inputs]; }
    void clear() { vals_.clear(); xin_.clear(); }
    void pop() { vals_.pop_back(); xin_.pop_back(); }

    /// The value connection (r, pos) read at step t (spec/03, spec/04): the node at position 0, the
    /// length node (filled entries at the start of the step) at 1, history node pos - 1 above that.
    double conn_value(std::size_t t, NodeId r, std::uint32_t pos) const {
        if (pos == 0) return vals_[t][r];
        const auto& s = std::get<SNeuron>(*net_->nodes[r].neuron);
        if (pos == 1) return static_cast<double>(std::min<std::size_t>(t, s.k));
        const std::size_t j = pos - 1;
        return t >= j ? xin_[t - j][r] : 0.0;  // the s_neuron's input j steps ago, 0 if unfilled
    }

    /// One step; returns output 0.
    double push(const std::vector<double>& in) {
        const Net<D>& net = *net_;
        const std::size_t t = vals_.size();
        vals_.emplace_back(net.nodes.size(), 0.0);
        xin_.emplace_back(net.nodes.size(), 0.0);
        auto& v = vals_.back();
        for (NodeId i = 0; i < net.n_inputs; ++i) v[i] = canon(in[i]);
        auto val = [&](const std::optional<NodeId>& r, std::uint32_t pos) { return r ? conn_value(t, *r, pos) : 0.0; };
        for (NodeId id : order_) {
            const Neuron& n = *net.nodes[id].neuron;
            if (const auto* x = std::get_if<VNeuron>(&n)) {
                v[id] = eval_v(*x, val(x->input, x->input_pos));
            } else if (const auto* s = std::get_if<SNeuron>(&n)) {
                const double x = val(s->input, s->input_pos);
                xin_.back()[id] = x;
                history(t, id, *s, buf_, pres_);
                v[id] = s->op == AggOp::Lag ? canon(buf_[0]) : canon(x - aggregate(s->op, buf_, pres_));
            } else {
                const auto& m = std::get<MNeuron>(n);
                slots(t, m, buf_, pres_);
                v[id] = canon(aggregate(m.op, buf_, pres_));
            }
        }
        return v[net.n_inputs];
    }

    /// Gradient of a loss whose derivative at output 0 of step T is `e`. Adds dL/dweight to
    /// `wgrad[id]` for each v_neuron and dL/dvalue (summed over steps) to `node[id]` for every node.
    /// A history read passes its gradient to the s_neuron's input at the step the value was pushed;
    /// a length read passes none (it counts steps, not values).
    void backward(std::size_t T, double e, std::vector<double>& wgrad, std::vector<double>& node) const {
        const Net<D>& net = *net_;
        const std::size_t n = net.nodes.size();
        std::vector<std::vector<double>> g(T + 1, std::vector<double>(n, 0.0));
        g[T][net.n_inputs] = e;
        std::vector<double> buf;
        std::vector<bool> pres;
        // dL/d(value of connection (r, pos) at step t) += gv
        auto add = [&](auto&& self, std::size_t t, NodeId r, std::uint32_t pos, double gv) -> void {
            gv = finite_or_0(gv);
            if (gv == 0.0) return;
            if (pos == 0) { g[t][r] += gv; return; }
            if (pos == 1) return;
            const std::size_t j = pos - 1;
            if (t < j) return;  // unfilled: a constant 0
            const auto& s = std::get<SNeuron>(*net.nodes[r].neuron);
            if (s.input) self(self, t - j, *s.input, s.input_pos, gv);
        };
        auto add_conn = [&](std::size_t t, NodeId r, std::uint32_t pos, double gv) { add(add, t, r, pos, gv); };
        for (std::size_t t = T + 1; t-- > 0;) {
            auto& gt = g[t];
            for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
                const NodeId id = *it;
                const double gv = gt[id];
                if (gv == 0.0 || !std::isfinite(gv)) continue;  // an overflowed gradient carries nothing
                node[id] += gv;
                const Neuron& nr = *net.nodes[id].neuron;
                if (const auto* x = std::get_if<VNeuron>(&nr)) {
                    const double in = x->input ? conn_value(t, *x->input, x->input_pos) : 0.0;
                    double dx = 0, dw = 0;
                    v_derivs(*x, in, dx, dw);
                    if (x->input) add_conn(t, *x->input, x->input_pos, gv * dx);
                    wgrad[id] += finite_or_0(gv * dw);
                } else if (const auto* s = std::get_if<SNeuron>(&nr)) {
                    if (!s->input) continue;
                    const NodeId u = *s->input;
                    const std::uint32_t up = s->input_pos;
                    const std::int64_t first = static_cast<std::int64_t>(t) - static_cast<std::int64_t>(s->k);
                    if (s->op == AggOp::Lag) {
                        if (first >= 0) add_conn(static_cast<std::size_t>(first), u, up, gv);
                        continue;
                    }
                    add_conn(t, u, up, gv);  // the current input
                    history(t, id, *s, buf, pres);
                    if (s->op == AggOp::Mean) {
                        for (std::uint32_t i = 0; i < s->k; ++i)
                            if (pres[i]) add_conn(static_cast<std::size_t>(first + i), u, up, -gv / s->k);
                    } else {
                        const int k = picked(s->op, buf, pres, aggregate(s->op, buf, pres));
                        if (k >= 0) add_conn(static_cast<std::size_t>(first + k), u, up, -gv);
                    }
                } else {
                    const auto& m = std::get<MNeuron>(nr);
                    slots(t, m, buf, pres);
                    if (m.op == AggOp::Mean) {
                        for (std::size_t i = 0; i < m.slots.size(); ++i)
                            if (pres[i]) add_conn(t, *m.slots[i], m.slot_pos(i), gv / static_cast<double>(m.slots.size()));
                    } else {
                        const int k = picked(m.op, buf, pres, aggregate(m.op, buf, pres));
                        if (k >= 0) add_conn(t, *m.slots[static_cast<std::size_t>(k)], m.slot_pos(static_cast<std::size_t>(k)), gv);
                    }
                }
            }
        }
    }

private:
    const Net<D>* net_;
    std::vector<NodeId> order_;
    std::vector<std::vector<double>> vals_;  // node values per step
    std::vector<std::vector<double>> xin_;   // each s_neuron's input value per step (what it pushed)
    std::vector<double> buf_;
    std::vector<bool> pres_;

    /// The history the s_neuron `id` sees at step t, oldest first (as its ring holds it).
    void history(std::size_t t, NodeId id, const SNeuron& s, std::vector<double>& buf, std::vector<bool>& pres) const {
        buf.assign(s.k, 0.0);
        pres.assign(s.k, false);
        for (std::uint32_t i = 0; i < s.k; ++i) {
            const std::int64_t st = static_cast<std::int64_t>(t) - s.k + i;
            if (st < 0) continue;
            pres[i] = true;
            buf[i] = xin_[static_cast<std::size_t>(st)][id];
        }
    }

    void slots(std::size_t t, const MNeuron& m, std::vector<double>& buf, std::vector<bool>& pres) const {
        buf.resize(m.slots.size());
        pres.resize(m.slots.size());
        for (std::size_t i = 0; i < m.slots.size(); ++i) {
            pres[i] = m.slots[i].has_value();
            buf[i] = pres[i] ? conn_value(t, *m.slots[i], m.slot_pos(i)) : 0.0;
        }
    }

    static void v_derivs(const VNeuron& n, double x, double& dx, double& dw) {
        double y = x, dy = 1.0;
        switch (n.inversion) {
            case Inversion::None: break;
            case Inversion::Bitnot:  // linear within a binade: slope 2^(exponent of ~x - exponent of x)
                y = bitnot(x);
                dy = (std::isnormal(x) && std::isnormal(y)) ? std::ldexp(1.0, std::ilogb(y) - std::ilogb(x)) : 0.0;
                break;
            case Inversion::Recip: y = 1.0 / x; dy = -1.0 / (x * x); break;
        }
        const double w = n.weight;
        double z = 0, dz_dy = 0, dz_dw = 0;
        switch (n.op) {
            case VOp::Mul: z = y * w; dz_dy = w; dz_dw = y; break;
            case VOp::Div: z = y / w; dz_dy = 1.0 / w; dz_dw = -y / (w * w); break;
            case VOp::Add: z = y + w; dz_dy = 1.0; dz_dw = 1.0; break;
            case VOp::Sub: z = y - w; dz_dy = 1.0; dz_dw = -1.0; break;
        }
        if (!std::isfinite(z) || !std::isfinite(y)) { dx = dw = 0.0; return; }
        dx = finite_or_0(dz_dy * dy);
        dw = finite_or_0(dz_dw);
    }
};

}  // namespace bj
