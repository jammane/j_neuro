// Structural analysis (spec/03 path rule, spec/07 immunity, spec/06 legal depth ranges).
// `removed` (optional) marks node ids treated as absent; ids beyond its size are live.
#pragma once

#include <jneuro/net.hpp>

#include <algorithm>
#include <array>
#include <functional>
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

/// Calls f(id) for each connected node id of a neuron, in slot order (duplicates kept), without
/// allocating.
template <class F>
void for_each_connection(const Neuron& n, F&& f) {
    if (auto* m = std::get_if<MNeuron>(&n)) {
        for (auto& s : m->slots) if (s) f(*s);
    } else if (auto* v = std::get_if<VNeuron>(&n)) {
        if (v->input) f(*v->input);
    } else if (auto& s = std::get<SNeuron>(n); s.input) {
        f(*s.input);
    }
}

/// Calls f(id, position) for each connection of a neuron, in slot order (spec/03 connections).
template <class F>
void for_each_conn(const Neuron& n, F&& f) {
    if (auto* m = std::get_if<MNeuron>(&n)) {
        for (std::size_t i = 0; i < m->slots.size(); ++i) if (m->slots[i]) f(*m->slots[i], m->slot_pos(i));
    } else if (auto* v = std::get_if<VNeuron>(&n)) {
        if (v->input) f(*v->input, v->input_pos);
    } else if (auto& s = std::get<SNeuron>(n); s.input) {
        f(*s.input, s.input_pos);
    }
}

/// Structural depth (spec/03 structural order): the depth, or for an s_neuron the bottom of its
/// gap, depth - (in_qty_max + 1). Its input is shallower and every reader of it or of its length and
/// history nodes is deeper, so ascending structural depth visits each node after what it reads.
template <class D>
std::uint64_t struct_depth(const Net<D>& net, NodeId i) {
    const std::uint64_t d = net.nodes[i].depth;
    const auto& n = net.nodes[i].neuron;
    if (!n || !std::holds_alternative<SNeuron>(*n)) return d;
    const std::uint64_t g = std::uint64_t{net.in_qty_max} + 1;
    return d > g ? d - g : 0;
}

namespace detail {
inline bool alive(const Removed* removed, NodeId i) {
    return !removed || i >= removed->size() || !(*removed)[i];
}
}  // namespace detail

/// Depth of a connection: depth(node) - position (spec/03).
template <class D>
std::uint64_t conn_depth(const Net<D>& net, NodeId node, std::uint32_t pos) {
    return static_cast<std::uint64_t>(net.nodes[node].depth) - pos;
}

/// The first broken depth rule of spec/03 (connections shallower than their reader, the s gap,
/// history positions within k of a hidden s_neuron), or nullptr. Live nodes only.
template <class D>
const char* depth_rule_error(const Net<D>& net, const Removed* removed = nullptr) {
    const std::uint64_t q = net.in_qty_max;
    for (NodeId i = net.n_inputs; i < net.nodes.size(); ++i) {
        if (!detail::alive(removed, i) || !net.nodes[i].neuron) continue;
        const std::uint64_t d = net.nodes[i].depth;
        const char* err = nullptr;
        for_each_conn(*net.nodes[i].neuron, [&](NodeId c, std::uint32_t p) {
            if (err) return;
            if (c >= net.nodes.size()) { err = "bad reference"; return; }
            if (p > 0) {
                const auto& src = net.nodes[c];
                const auto* s = src.neuron ? std::get_if<SNeuron>(&*src.neuron) : nullptr;
                if (!net.is_hidden(c) || !s || p > s->k + 1 || p > src.depth) {
                    err = "history connection: no such history node";
                    return;
                }
            }
            if (conn_depth(net, c, p) >= d) err = "connection depth must be smaller";
        });
        if (err) return err;
        if (const auto* s = std::get_if<SNeuron>(&*net.nodes[i].neuron)) {
            const std::uint64_t in = s->input ? conn_depth(net, *s->input, s->input_pos) : 0;
            if (in + q + 1 >= d) return "s gap: an s_neuron's input must be shallower than its depth - (in_qty_max + 1)";
        }
    }
    return nullptr;
}

/// Largest history position read from each node (spec/03 history floor), by node id.
template <class D>
std::vector<std::uint32_t> history_floors(const Net<D>& net, const Removed* removed = nullptr) {
    std::vector<std::uint32_t> hf(net.nodes.size(), 0);
    for (NodeId i = 0; i < net.nodes.size(); ++i) {
        if (!detail::alive(removed, i) || !net.nodes[i].neuron) continue;
        for_each_conn(*net.nodes[i].neuron, [&](NodeId c, std::uint32_t p) {
            if (c < hf.size()) hf[c] = std::max(hf[c], p);
        });
    }
    return hf;
}

/// For each node: ascending ids of live neurons connected to it (each once).
template <class D>
std::vector<std::vector<NodeId>> readers(const Net<D>& net, const Removed* removed = nullptr) {
    std::vector<std::vector<NodeId>> out(net.nodes.size());
    for (NodeId i = 0; i < net.nodes.size(); ++i) {
        if (!detail::alive(removed, i)) continue;
        if (!net.nodes[i].neuron) continue;
        for_each_connection(*net.nodes[i].neuron, [&](NodeId c) {
            if (out[c].empty() || out[c].back() != i) out[c].push_back(i);
        });
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
        if (!net.nodes[i].neuron) continue;
        for_each_connection(*net.nodes[i].neuron, [&](NodeId c) {
            if (detail::alive(removed, c) && !marked[c]) { marked[c] = true; stack.push_back(c); }
        });
    }
    return marked;
}

namespace detail {
/// Input-reach bitsets: word w of node x's set is bits[x * words + w]; bit i means input i reaches x.
inline std::size_t reach_words(std::uint32_t n_inputs) { return (n_inputs + 63) / 64; }

/// True if every output's set holds every input.
inline bool outputs_full(const std::vector<std::uint64_t>& bits, std::size_t words, std::uint32_t n_inputs,
                         std::uint32_t n_outputs) {
    for (NodeId o = n_inputs; o < n_inputs + n_outputs; ++o)
        for (std::size_t w = 0; w < words; ++w) {
            const std::uint32_t lo = static_cast<std::uint32_t>(w * 64);
            const std::uint64_t full = n_inputs - lo >= 64 ? ~0ull : (1ull << (n_inputs - lo)) - 1;
            if (bits[o * words + w] != full) return false;
        }
    return true;
}
}  // namespace detail

/// Every input reaches every output. `cut` is a node no path may pass through.
///
/// One pass in depth order carrying, per node, the set of inputs that reach it (a neuron reads only
/// strictly shallower nodes, spec/03). Same answer as one search per input, in O(edges * inputs/64).
template <class D>
bool path_rule_ok(const Net<D>& net, const Removed* removed = nullptr,
                  std::optional<NodeId> cut = std::nullopt) {
    auto ok = [&](NodeId i) { return detail::alive(removed, i) && (!cut || *cut != i); };
    for (NodeId i = 0; i < net.n_inputs; ++i)
        if (!ok(i)) return false;
    if (net.n_inputs == 0) return true;
    const std::size_t n = net.nodes.size(), words = detail::reach_words(net.n_inputs);
    std::vector<NodeId> order;
    order.reserve(n - net.n_inputs);
    for (NodeId i = net.n_inputs; i < n; ++i)
        if (ok(i)) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return struct_depth(net, a) < struct_depth(net, b); });
    std::vector<std::uint64_t> bits(n * words, 0);  // dead and cut nodes stay empty
    for (NodeId i = 0; i < net.n_inputs; ++i) bits[i * words + i / 64] |= 1ull << (i % 64);
    for (NodeId x : order) {
        std::uint64_t* dst = &bits[x * words];
        for_each_connection(*net.nodes[x].neuron, [&](NodeId c) {
            const std::uint64_t* src = &bits[c * words];
            for (std::size_t w = 0; w < words; ++w) dst[w] |= src[w];
        });
    }
    return detail::outputs_full(bits, words, net.n_inputs, net.n_outputs);
}

/// Answers path_rule_ok for one net as it is edited, recomputing only what changed (an
/// implementation cache: it never changes a result, only how fast it is found).
///
/// It keeps each node's input-reach set and the connections it last saw. Each call compares the
/// net with those, recomputes the nodes whose connections or liveness changed, and carries a change
/// downstream (in depth order) only while a set actually changes, so a cut that another slot
/// covers costs one node. Edits between calls need no notification, undo included. `cut` is
/// applied as a trial and rolled back.
template <class D>
class PathRuleTracker {
public:
    bool ok(const Net<D>& net, const Removed* removed, std::optional<NodeId> cut = std::nullopt) {
        for (NodeId i = 0; i < net.n_inputs; ++i)
            if (!detail::alive(removed, i) || (cut && *cut == i)) return false;
        if (net.n_inputs == 0) return true;
        sync(net, removed);
        if (!cut) return detail::outputs_full(bits_, words_, net.n_inputs, net.n_outputs);
        log_.clear();
        cut_ = cut;
        push(net, *cut);
        propagate(net, true);
        const bool result = detail::outputs_full(bits_, words_, net.n_inputs, net.n_outputs);
        for (auto it = log_.rbegin(); it != log_.rend(); ++it)
            std::copy(it->second.begin(), it->second.end(), bits_.begin() + it->first * words_);
        cut_.reset();
        return result;
    }

private:
    std::size_t words_ = 0;
    bool built_ = false;
    std::vector<std::uint64_t> bits_;
    std::vector<std::vector<NodeId>> conn_;     // connections as last seen; empty if dead
    std::vector<char> alive_;                   // liveness as last seen
    std::vector<std::vector<NodeId>> readers_;  // superset of each node's readers (stale ids harmless)
    std::vector<char> queued_;
    std::vector<std::pair<std::uint64_t, NodeId>> heap_;  // min-heap on (structural depth, id)
    std::vector<NodeId> scratch_;
    std::vector<std::uint64_t> tmp_;
    std::optional<NodeId> cut_;
    std::vector<std::pair<NodeId, std::vector<std::uint64_t>>> log_;  // trial: old sets to restore

    void add_reader(NodeId c, NodeId x) {
        auto& r = readers_[c];
        if (std::find(r.begin(), r.end(), x) == r.end()) r.push_back(x);
    }

    void push(const Net<D>& net, NodeId x) {
        if (queued_[x]) return;
        queued_[x] = 1;
        heap_.emplace_back(struct_depth(net, x), x);
        std::push_heap(heap_.begin(), heap_.end(), std::greater<>{});
    }

    void sync(const Net<D>& net, const Removed* removed) {
        const std::size_t n = net.nodes.size();
        if (!built_) {
            words_ = detail::reach_words(net.n_inputs);
            built_ = true;
        }
        const std::size_t old_n = conn_.size();
        if (n != old_n) {
            bits_.resize(n * words_, 0);
            conn_.resize(n);
            alive_.resize(n, 0);
            readers_.resize(n);
            queued_.resize(n, 0);
            if (n > old_n)
                for (std::size_t x = old_n; x < n; ++x) {
                    std::fill(bits_.begin() + x * words_, bits_.begin() + (x + 1) * words_, 0);
                    conn_[x].clear();
                    readers_[x].clear();
                }
        }
        for (NodeId x = 0; x < n; ++x) {
            const bool a = detail::alive(removed, x);
            scratch_.clear();
            if (a && net.nodes[x].neuron)
                for_each_connection(*net.nodes[x].neuron, [&](NodeId c) { scratch_.push_back(c); });
            const bool fresh = x >= old_n;
            if (!fresh && a == static_cast<bool>(alive_[x]) && scratch_ == conn_[x]) continue;
            alive_[x] = a;
            conn_[x] = scratch_;
            for (NodeId c : scratch_) add_reader(c, x);
            push(net, x);
        }
        propagate(net, false);
    }

    void propagate(const Net<D>& net, bool trial) {
        const std::size_t n = net.nodes.size();
        tmp_.assign(words_, 0);
        while (!heap_.empty()) {
            std::pop_heap(heap_.begin(), heap_.end(), std::greater<>{});
            const NodeId x = heap_.back().second;
            heap_.pop_back();
            queued_[x] = 0;
            std::fill(tmp_.begin(), tmp_.end(), 0);
            if (!alive_[x] || (cut_ && *cut_ == x)) {
                // dead or cut: no input reaches it
            } else if (x < net.n_inputs) {
                tmp_[x / 64] = 1ull << (x % 64);
            } else {
                for (NodeId c : conn_[x])
                    if (c < n)
                        for (std::size_t w = 0; w < words_; ++w) tmp_[w] |= bits_[c * words_ + w];
            }
            auto cur = bits_.begin() + x * words_;
            if (std::equal(tmp_.begin(), tmp_.end(), cur)) continue;
            if (trial) log_.emplace_back(x, std::vector<std::uint64_t>(cur, cur + words_));
            std::copy(tmp_.begin(), tmp_.end(), cur);
            for (NodeId r : readers_[x])
                if (r < n) push(net, r);
        }
    }
};

namespace detail {
/// Live node ids in structural order: by structural depth, ties by id (spec/03).
template <class D>
std::vector<NodeId> live_depth_order(const Net<D>& net, const Removed* removed) {
    std::vector<NodeId> order;
    for (NodeId i = 0; i < net.nodes.size(); ++i)
        if (alive(removed, i)) order.push_back(i);
    std::stable_sort(order.begin(), order.end(),
                     [&](NodeId a, NodeId b) { return struct_depth(net, a) < struct_depth(net, b); });
    return order;
}
}  // namespace detail

/// Mix rule (spec/03): every input-to-output path has >= path_mix v, s and m neurons.
template <class D>
bool mix_ok(const Net<D>& net, const Removed* removed = nullptr, const std::vector<NodeId>* order = nullptr) {
    if (net.path_mix == 0) return true;
    const std::size_t n = net.nodes.size();
    std::vector<NodeId> own;
    if (!order) {
        own = detail::live_depth_order(net, removed);
        order = &own;
    }
    using Triple = std::array<std::uint64_t, 3>;  // fewest (v, s, m) from any input
    std::vector<std::optional<Triple>> fewest(n);
    for (NodeId i : *order) {
        if (!detail::alive(removed, i)) continue;
        const auto& neuron = net.nodes[i].neuron;
        if (!neuron) { fewest[i] = Triple{0, 0, 0}; continue; }
        std::optional<Triple> best;
        for_each_connection(*neuron, [&](NodeId c) {
            if (!detail::alive(removed, c) || !fewest[c]) return;
            if (!best) best = fewest[c];
            else for (int k = 0; k < 3; ++k) (*best)[k] = std::min((*best)[k], (*fewest[c])[k]);
        });
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
bool run_ok(const Net<D>& net, const Removed* removed = nullptr, const std::vector<NodeId>* order = nullptr) {
    const std::uint64_t k = net.max_run;
    if (k == 0) return true;
    const std::size_t n = net.nodes.size();
    const auto reach = reaching_output(net, removed);
    std::vector<NodeId> own;
    if (!order) {
        own = detail::live_depth_order(net, removed);
        order = &own;
    }
    std::vector<std::optional<std::uint64_t>> run_1(n), run_m(n);  // longest run ending here, from an input
    for (NodeId i : *order) {
        if (!detail::alive(removed, i)) continue;
        const auto& neuron = net.nodes[i].neuron;
        if (!neuron) { run_1[i] = run_m[i] = 0; continue; }
        std::optional<std::uint64_t> best_1, best_m;
        for_each_connection(*neuron, [&](NodeId c) {
            if (!detail::alive(removed, c) || !run_1[c]) return;
            best_1 = std::max(best_1.value_or(0), *run_1[c]);
            best_m = std::max(best_m.value_or(0), *run_m[c]);
        });
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

/// gap(X) of spec/06: in_qty_max + 1 for an s_neuron (length node and longest history), else 0.
template <class D>
std::uint64_t s_gap(const Net<D>& net, const Neuron& n) {
    return std::holds_alternative<SNeuron>(n) ? std::uint64_t{net.in_qty_max} + 1 : 0;
}

/// floor(X) of spec/06: the shallowest depth node `id` may have, given what its neuron reads
/// (live connections; a missing s input counts as depth 0) and its own s gap.
template <class D>
std::uint64_t depth_floor(const Net<D>& net, NodeId id, const Removed* removed = nullptr) {
    const Neuron& n = *net.nodes[id].neuron;
    std::uint64_t f = 1;
    for_each_conn(n, [&](NodeId c, std::uint32_t p) {
        if (detail::alive(removed, c)) f = std::max(f, conn_depth(net, c, p) + 1);
    });
    return f + s_gap(net, n);
}

/// ceil(X) of spec/06: the deepest depth node `id` may have, given every live connection (id, j)
/// and its reader's s gap: depth(r) - gap(r) + j - 1; D_MAX - 1 if nothing reads it.
template <class D>
std::uint64_t depth_ceil(const Net<D>& net, NodeId id, const Removed* removed = nullptr) {
    std::uint64_t hi = static_cast<std::uint64_t>(Net<D>::d_max) - 1;
    for (NodeId r = 0; r < net.nodes.size(); ++r) {
        if (!detail::alive(removed, r) || !net.nodes[r].neuron) continue;
        const Neuron& n = *net.nodes[r].neuron;
        for_each_conn(n, [&](NodeId c, std::uint32_t j) {
            if (c != id) return;
            const std::uint64_t lim = static_cast<std::uint64_t>(net.nodes[r].depth) - s_gap(net, n) + j - 1;
            hi = std::min(hi, lim);
        });
    }
    return hi;
}

/// [lo, hi] depths `id` may take without breaking any neuron (spec/06).
template <class D>
std::pair<D, D> legal_depth_range(const Net<D>& net, NodeId id, const Removed* removed = nullptr) {
    return {static_cast<D>(depth_floor(net, id, removed)), static_cast<D>(depth_ceil(net, id, removed))};
}

}  // namespace jneuro
