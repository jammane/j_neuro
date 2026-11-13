// Recording evaluation and gradients for the blackjack tuner (examples/blackjack/tune.cpp).
//
// Tape evaluates a net step by step exactly as Net::step does (same order, same arithmetic), but
// keeps every node's value at every step of a deal, so the gradient of an output can be taken back
// through all steps (s_neuron histories reach back into earlier steps).
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
    void clear() { vals_.clear(); }
    void pop() { vals_.pop_back(); }

    /// One step; returns output 0.
    double push(const std::vector<double>& in) {
        const Net<D>& net = *net_;
        const std::size_t t = vals_.size();
        std::vector<double> v(net.nodes.size(), 0.0);
        for (NodeId i = 0; i < net.n_inputs; ++i) v[i] = canon(in[i]);
        auto val = [&](const std::optional<NodeId>& r) { return r ? v[*r] : 0.0; };
        for (NodeId id : order_) {
            const Neuron& n = *net.nodes[id].neuron;
            if (const auto* x = std::get_if<VNeuron>(&n)) {
                v[id] = eval_v(*x, val(x->input));
            } else if (const auto* s = std::get_if<SNeuron>(&n)) {
                history(t, *s, buf_, pres_);
                v[id] = s->op == AggOp::Lag ? canon(buf_[0]) : canon(val(s->input) - aggregate(s->op, buf_, pres_));
            } else {
                const auto& m = std::get<MNeuron>(n);
                slots(m, v, buf_, pres_);
                v[id] = canon(aggregate(m.op, buf_, pres_));
            }
        }
        vals_.push_back(std::move(v));
        return vals_.back()[net.n_inputs];
    }

    /// Gradient of a loss whose derivative at output 0 of step T is `e`. Adds dL/dweight to
    /// `wgrad[id]` for each v_neuron and dL/dvalue (summed over steps) to `node[id]` for every node.
    void backward(std::size_t T, double e, std::vector<double>& wgrad, std::vector<double>& node) const {
        const Net<D>& net = *net_;
        const std::size_t n = net.nodes.size();
        std::vector<std::vector<double>> g(T + 1, std::vector<double>(n, 0.0));
        g[T][net.n_inputs] = e;
        std::vector<double> buf;
        std::vector<bool> pres;
        for (std::size_t t = T + 1; t-- > 0;) {
            auto& gt = g[t];
            const auto& vt = vals_[t];
            for (auto it = order_.rbegin(); it != order_.rend(); ++it) {
                const NodeId id = *it;
                const double gv = gt[id];
                if (gv == 0.0 || !std::isfinite(gv)) continue;  // an overflowed gradient carries nothing
                node[id] += gv;
                const Neuron& nr = *net.nodes[id].neuron;
                if (const auto* x = std::get_if<VNeuron>(&nr)) {
                    const double in = x->input ? vt[*x->input] : 0.0;
                    double dx = 0, dw = 0;
                    v_derivs(*x, in, dx, dw);
                    if (x->input) gt[*x->input] += finite_or_0(gv * dx);
                    wgrad[id] += finite_or_0(gv * dw);
                } else if (const auto* s = std::get_if<SNeuron>(&nr)) {
                    if (!s->input) continue;
                    const NodeId u = *s->input;
                    const std::int64_t first = static_cast<std::int64_t>(t) - static_cast<std::int64_t>(s->k);
                    if (s->op == AggOp::Lag) {
                        if (first >= 0) g[static_cast<std::size_t>(first)][u] += gv;
                        continue;
                    }
                    gt[u] += gv;  // the current input
                    history(t, *s, buf, pres);
                    if (s->op == AggOp::Mean) {
                        for (std::uint32_t i = 0; i < s->k; ++i)
                            if (pres[i]) g[static_cast<std::size_t>(first + i)][u] -= gv / s->k;
                    } else {
                        const int k = picked(s->op, buf, pres, aggregate(s->op, buf, pres));
                        if (k >= 0) g[static_cast<std::size_t>(first + k)][u] -= gv;
                    }
                } else {
                    const auto& m = std::get<MNeuron>(nr);
                    slots(m, vt, buf, pres);
                    if (m.op == AggOp::Mean) {
                        for (std::size_t i = 0; i < m.slots.size(); ++i)
                            if (pres[i]) gt[*m.slots[i]] += gv / static_cast<double>(m.slots.size());
                    } else {
                        const int k = picked(m.op, buf, pres, aggregate(m.op, buf, pres));
                        if (k >= 0) gt[*m.slots[static_cast<std::size_t>(k)]] += gv;
                    }
                }
            }
        }
    }

private:
    const Net<D>* net_;
    std::vector<NodeId> order_;
    std::vector<std::vector<double>> vals_;
    std::vector<double> buf_;
    std::vector<bool> pres_;

    /// The history an s_neuron sees at step t, oldest first (as its ring holds it).
    void history(std::size_t t, const SNeuron& s, std::vector<double>& buf, std::vector<bool>& pres) const {
        buf.assign(s.k, 0.0);
        pres.assign(s.k, false);
        for (std::uint32_t i = 0; i < s.k; ++i) {
            const std::int64_t st = static_cast<std::int64_t>(t) - s.k + i;
            if (st < 0) continue;
            pres[i] = true;
            buf[i] = s.input ? vals_[static_cast<std::size_t>(st)][*s.input] : 0.0;
        }
    }

    static void slots(const MNeuron& m, const std::vector<double>& v, std::vector<double>& buf, std::vector<bool>& pres) {
        buf.resize(m.slots.size());
        pres.resize(m.slots.size());
        for (std::size_t i = 0; i < m.slots.size(); ++i) {
            pres[i] = m.slots[i].has_value();
            buf[i] = pres[i] ? v[*m.slots[i]] : 0.0;
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
