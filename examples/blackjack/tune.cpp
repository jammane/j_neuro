// Example tool: tune one trained blackjack net on blocks of dealt hands, guided by gradients.
//
//   1 deal      All hands are dealt up front, in blocks of --block-size, and written to deals.csv.
//               A hand is used only if some sequence of decisions scores 0 or more (a push counts).
//   2 series    For each hand, the move series that score 0 or more.
//   3 targets   The series that changes the fewest of the net's own decisions (iterative deepening on
//               the number of changes). A hand the net already plays to 0 or more keeps its own moves.
//   4 stress    Back-propagate, for every decision whose output is outside its action's range (narrowed
//               by --margin), how far the output must move. Per neuron: stress = sum of |pull| over
//               decisions, conflict = 1 - |sum of pulls| / stress (decisions pulling opposite ways).
//   5 decide    If the most stressed neuron's conflict is >= --conflict, try adding neurons there first;
//               otherwise try adjusting first. Adding at a stressed node N: a new m_neuron goes after
//               N's neuron (N's old neuron now writes a new node B, the m's first slot, and N's readers
//               read the m) or before it (on the connection N's neuron reads; for an m_neuron its most
//               stressed slot); the other slots start as copies of that first connection
//               (v(.., mul 1)), so the m starts neutral; then slot by slot a new s or v neuron
//               reading an existing node (through a chain with enough v, s and m neurons for the mix
//               and run rules) replaces the filler, with the m's op and the weight chosen to correct
//               N on the decisions pulling it up against those pulling it down.
//   6 adjust    Keep the best candidate if it is better: more of the block's hands work, then fewer
//               broken earlier hands, then lower loss; never more broken earlier hands than the
//               acceptance rule allows.
//   7 loop      Back to 4 while it improves. When every hand of the block works and the rule holds,
//               the block is accepted (back to 1 for the next block). If nothing improves, the block
//               is stuck: the net goes back to how it was before the block and the block goes to the
//               end of the queue. Finish when the queue is empty, or when every queued block has
//               failed since the last accepted one.
//
// Acceptance rule (broken = hands in accepted blocks the net no longer plays to 0 or more):
//   block 1: all its hands work.  block 2: all its hands work and broken < 50% of all hands.
//   later blocks: all their hands work and the broken share is strictly lower than after the
//   previous accepted block (0% stays 0%).
//
// Usage: blackjack_tune [--pop FILE.jpnet] [--net-id N] [--blocks 100] [--block-size 20] [--seed N]
//                       [--holdout N] [--max-rounds N] [--margin M] [--conflict C] [--out DIR]

#include "tape.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <thread>
#include <functional>
#include <cstdlib>

using namespace bj;
using Pop = Population<D>;
namespace fs = std::filesystem;

namespace {

double g_margin = 0.1;
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

constexpr int kMaxChanges = 8;          // phase 3: most decisions changed in a target series
constexpr long kSearchBudget = 200000;  // phase 3: decisions visited per hand
constexpr std::size_t kTopOps = 5;      // phase 6: neurons whose operations are tried
constexpr std::size_t kTopGrow = 3;     // phase 5: neurons where growth is tried
constexpr std::size_t kSources = 4;     // phase 5: sources tried per growth site

struct Step {
    std::vector<double> in;
    Action target;
};

struct HandRec {
    std::uint64_t seed;
    Deal deal;
    std::vector<Step> path;  // the target series (phase 3)
    bool has_path = false;
    explicit HandRec(std::uint64_t s) : seed(s), deal(s) {}
};

/// How far y is outside the output range that decodes to `a`, narrowed by the margin.
double error(double y, Action a) {
    const double m = g_margin, inf = std::numeric_limits<double>::infinity();
    double lo = -inf, hi = inf;
    switch (a) {
        case Stand: hi = -m; break;
        case Hit: lo = m; hi = 1 - m; break;
        case Double: lo = 1 + m; hi = 2 - m; break;
        case Split: lo = 2 + m; break;
    }
    return y < lo ? y - lo : (y > hi ? y - hi : 0.0);
}

/// Per-decision loss: squared within the output range's width (3), linear beyond it, so a node
/// clamped to the largest double cannot make the loss infinite. dloss is its derivative.
constexpr double kHuber = 3.0;
double loss_of(double e) {
    const double a = std::fabs(e);
    return a <= kHuber ? 0.5 * e * e : kHuber * (a - 0.5 * kHuber);
}
double dloss(double e) { return std::clamp(e, -kHuber, kHuber); }

/// Play a deal on the tape; records each decision (inputs and the net's action) if `path` is given.
double play(Tape& tape, const Deal& deal, std::vector<Step>* path) {
    Table table(deal);
    tape.clear();
    std::vector<double> in;
    while (!table.finished()) {
        decision_inputs(table, deal, in);
        const Action a = decode(tape.push(in));
        if (path) path->push_back({in, a});
        table.act(deal, a);
    }
    return table.score(deal);
}

/// Phase 2-3: depth-first over move series, at most `limit` decisions changed from the net's.
struct Searcher {
    const Deal& deal;
    Tape tape;
    int limit;
    long nodes = 0;
    std::vector<Step> path;

    bool dfs(const Table& t, int cost) {
        if (++nodes > kSearchBudget) return false;
        if (t.finished()) return t.score(deal) >= 0;
        std::vector<double> in;
        decision_inputs(t, deal, in);
        const Action mine = decode(tape.push(in));
        const bool split_ok = split_valid(t);
        const Action eff = (mine == Split && !split_ok) ? Stand : mine;  // an invalid split stands
        std::array<Action, 4> order{eff, Stand, Hit, Double};
        std::size_t n = 1;
        for (Action a : {Stand, Hit, Double, Split})
            if (a != eff && (a != Split || split_ok)) order[n++] = a;
        for (std::size_t i = 0; i < n; ++i) {
            const Action a = order[i];
            const int c = cost + (a != eff);
            if (c > limit) continue;
            Table next = t;
            next.act(deal, a);
            path.push_back({in, a == eff ? mine : a});
            if (dfs(next, c)) { tape.pop(); return true; }
            path.pop_back();
        }
        tape.pop();
        return false;
    }
};

bool target_path(const Net<D>& net, const Deal& deal, std::vector<Step>& out) {
    for (int limit = 1; limit <= kMaxChanges; ++limit) {
        Searcher s{deal, Tape(net), limit};
        if (s.dfs(Table(deal), 0)) { out = std::move(s.path); return true; }
        if (s.nodes > kSearchBudget) return false;
    }
    return false;
}

void prepare_targets(const Net<D>& net, const std::vector<HandRec*>& hands) {
    Tape tape(net);
    for (HandRec* h : hands) {
        h->path.clear();
        if (play(tape, h->deal, &h->path) >= 0) { h->has_path = true; continue; }
        h->path.clear();
        h->has_path = target_path(net, h->deal, h->path);
    }
}

struct Eval {
    int block_works = 0;
    int ledger_broken = 0;
    double loss = 0;
};

Eval evaluate(const Net<D>& net, const std::vector<HandRec*>& block, const std::vector<HandRec*>& ledger) {
    Tape tape(net);
    Eval ev;
    auto one = [&](const HandRec& h) {
        const bool works = play(tape, h.deal, nullptr) >= 0;
        if (h.has_path) {
            tape.clear();
            for (const Step& st : h.path) {
                ev.loss += loss_of(error(tape.push(st.in), st.target));
            }
        }
        return works;
    };
    for (const HandRec* h : block) ev.block_works += one(*h);
    for (const HandRec* h : ledger) ev.ledger_broken += !one(*h);
    return ev;
}

std::vector<Eval> evaluate_all(const std::vector<Net<D>>& cands, const std::vector<HandRec*>& block,
                               const std::vector<HandRec*>& ledger, unsigned threads) {
    std::vector<Eval> out(cands.size());
    std::atomic<std::size_t> next{0};
    auto worker = [&] {
        for (std::size_t k; (k = next++) < cands.size();) out[k] = evaluate(cands[k], block, ledger);
    };
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    return out;
}

/// Phase 4: gradients of the loss over every target decision.
struct Grads {
    std::vector<double> w;                         // dL/dweight by node id (v_neurons)
    std::vector<std::vector<double>> dec_node;     // per decision with an error: dL/dvalue by node id
    std::vector<std::vector<double>> dec_vals;     // per decision: node values at its step
    std::vector<std::pair<const HandRec*, std::size_t>> dec_at;  // per decision: its hand and step
    std::vector<std::pair<NodeId, std::uint32_t>> hist;  // length and history nodes (S, p) of hidden s_neurons
    std::vector<std::vector<double>> dec_hval;           // per decision: the value of each of `hist`
};

Grads gradients(const Net<D>& net, const std::vector<HandRec*>& hands) {
    Grads g;
    g.w.assign(net.nodes.size(), 0.0);
    for (NodeId s = net.n_inputs + net.n_outputs; s < net.nodes.size(); ++s)
        if (const auto* sn = std::get_if<SNeuron>(&*net.nodes[s].neuron))
            for (std::uint32_t p = 1; p <= sn->k + 1; ++p) g.hist.push_back({s, p});
    Tape tape(net);
    for (const HandRec* h : hands) {
        if (!h->has_path) continue;
        tape.clear();
        std::vector<double> e(h->path.size());
        for (std::size_t t = 0; t < h->path.size(); ++t) e[t] = dloss(error(tape.push(h->path[t].in), h->path[t].target));
        for (std::size_t t = 0; t < e.size(); ++t) {
            if (e[t] == 0.0) continue;
            std::vector<double> node(net.nodes.size(), 0.0);
            tape.backward(t, e[t], g.w, node);
            g.dec_node.push_back(std::move(node));
            g.dec_vals.push_back(tape.at(t));
            g.dec_at.push_back({h, t});
            std::vector<double> hv;
            for (const auto& [s, p] : g.hist) hv.push_back(tape.conn_value(t, s, p));
            g.dec_hval.push_back(std::move(hv));
        }
    }
    return g;
}

struct Stress {
    NodeId id;
    double stress, conflict;
};

std::vector<Stress> stress_of(const Net<D>& net, const Grads& g) {
    std::vector<Stress> out;
    for (NodeId i = net.n_inputs; i < net.nodes.size(); ++i) {
        double sum = 0, abs = 0;
        for (const auto& d : g.dec_node) { sum += d[i]; abs += std::fabs(d[i]); }
        if (abs > 0 && std::isfinite(abs)) out.push_back({i, abs, 1.0 - std::fabs(sum) / abs});
    }
    std::stable_sort(out.begin(), out.end(), [](const Stress& a, const Stress& b) { return a.stress > b.stress; });
    return out;
}

bool valid(const Net<D>& net) {
    try {
        net.validate();
        return true;
    } catch (const ValidationError&) {
        return false;
    }
}

// ------------------------------------------------------------------ phase 6 candidates

std::vector<Net<D>> weight_candidates(const Net<D>& net, const std::vector<double>& g) {
    double gmax = 0;
    for (NodeId i = 0; i < net.nodes.size(); ++i)
        if (net.nodes[i].neuron && std::holds_alternative<VNeuron>(*net.nodes[i].neuron)) gmax = std::max(gmax, std::fabs(g[i]));
    std::vector<Net<D>> out;
    if (gmax == 0 || !std::isfinite(gmax)) return out;
    for (double s : {1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001}) {  // the largest weight change
        Net<D> c = net.copy_genome();
        for (NodeId i = 0; i < c.nodes.size(); ++i)
            if (c.nodes[i].neuron)
                if (auto* v = std::get_if<VNeuron>(&*c.nodes[i].neuron)) v->weight -= s * g[i] / gmax;
        out.push_back(std::move(c));
    }
    return out;
}

std::vector<Net<D>> op_candidates(const Net<D>& net, const std::vector<Stress>& st) {
    std::vector<Net<D>> out;
    for (std::size_t k = 0; k < std::min(kTopOps, st.size()); ++k) {
        const NodeId id = st[k].id;
        const Neuron& n = *net.nodes[id].neuron;
        auto with = [&](Neuron x) {
            Net<D> c = net.copy_genome();
            c.nodes[id].neuron = std::move(x);
            out.push_back(std::move(c));
        };
        if (const auto* v = std::get_if<VNeuron>(&n)) {
            for (int inv = 0; inv < 3; ++inv)
                for (int op = 0; op < 4; ++op) {
                    VNeuron x = *v;
                    x.inversion = static_cast<Inversion>(inv);
                    x.op = static_cast<VOp>(op);
                    if (x.inversion != v->inversion || x.op != v->op) with(x);
                }
        } else if (const auto* s = std::get_if<SNeuron>(&n)) {
            for (std::uint64_t op = 0; op < kSOps; ++op)
                if (static_cast<AggOp>(op) != s->op) { SNeuron x = *s; x.op = static_cast<AggOp>(op); with(x); }
        } else {
            const auto& m = std::get<MNeuron>(n);
            for (std::uint64_t op = 0; op < kMOps; ++op)
                if (static_cast<AggOp>(op) != m.op) { MNeuron x = m; x.op = static_cast<AggOp>(op); with(x); }
        }
    }
    return out;
}

std::map<std::string, long> g_rejects;  // why growth was not possible (TUNE_DEBUG summary)

bool valid_or_count(const Net<D>& net) {
    try {
        net.validate();
        return true;
    } catch (const ValidationError& e) {
        ++g_rejects[std::string(e.what()).substr(0, 60)];
        return false;
    }
}

/// Neurons reading `id` at position 0 (its own value).
std::vector<NodeId> live_readers(const Net<D>& net, NodeId id) {
    std::vector<NodeId> out;
    for (NodeId r = 0; r < net.nodes.size(); ++r) {
        if (!net.nodes[r].neuron) continue;
        bool reads = false;
        for_each_conn(*net.nodes[r].neuron, [&](NodeId c, std::uint32_t p) { reads = reads || (c == id && p == 0); });
        if (reads) out.push_back(r);
    }
    return out;
}

/// Up to n values spread over the sorted, distinct, finite values of v.
std::vector<double> quantiles(std::vector<double> v, std::size_t n) {
    v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    std::vector<double> out;
    const std::size_t k = std::min(n, v.size());
    for (std::size_t q = 0; q < k; ++q) out.push_back(v[(q * (v.size() - 1)) / std::max<std::size_t>(1, k - 1)]);
    return out;
}

/// Per node and kind (0 v, 1 s, 2 m): the fewest neurons of that kind on any path from an input
/// to the node (counting the node), and on any path from the node to an output (not counting it).
struct MixCounts {
    std::array<std::vector<int>, 3> up, down;
    std::vector<int> run1, runm;  // longest run of one-in / many-in neurons ending at the node
};

MixCounts mix_counts(const Net<D>& net) {
    const std::size_t n = net.nodes.size();
    constexpr int kFar = 1 << 20;
    MixCounts mc;
    std::vector<NodeId> order(n);
    for (NodeId i = 0; i < n; ++i) order[i] = i;
    // structural order (spec/03): a history reader can be shallower than its s_neuron
    std::stable_sort(order.begin(), order.end(), [&](NodeId a, NodeId b) { return struct_depth(net, a) < struct_depth(net, b); });
    auto kind = [&](NodeId i) { return net.nodes[i].neuron ? static_cast<int>(type_of(*net.nodes[i].neuron)) : -1; };
    const auto rd = readers(net);
    mc.run1.assign(n, 0);
    mc.runm.assign(n, 0);
    for (NodeId id : order) {
        if (!net.nodes[id].neuron) continue;
        int b1 = 0, bm = 0;
        for (NodeId c : connections(*net.nodes[id].neuron)) { b1 = std::max(b1, mc.run1[c]); bm = std::max(bm, mc.runm[c]); }
        if (kind(id) == 2) mc.runm[id] = bm + 1;
        else mc.run1[id] = b1 + 1;
    }
    for (int t = 0; t < 3; ++t) {
        mc.up[t].assign(n, kFar);
        mc.down[t].assign(n, kFar);
        for (NodeId id : order) {
            if (!net.nodes[id].neuron) { mc.up[t][id] = 0; continue; }
            int best = kFar;
            for (NodeId c : connections(*net.nodes[id].neuron)) best = std::min(best, mc.up[t][c]);
            mc.up[t][id] = (best == kFar ? 0 : best) + (kind(id) == t);
        }
        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            const NodeId id = *it;
            if (net.is_output(id)) { mc.down[t][id] = 0; continue; }
            int best = kFar;
            for (NodeId r : rd[id]) best = std::min(best, mc.down[t][r] + (kind(r) == t));
            mc.down[t][id] = best;
        }
    }
    return mc;
}

using Pick = std::function<std::size_t(const std::vector<Net<D>>&)>;  // best candidate, or size() for none

/// Phase 5: add an m_neuron in front of the stressed node N. N's old neuron moves to a new node B,
/// which feeds the m's first slot; the other slots start as v(B, mul 1), so the m starts neutral
/// (op(B, B, ...) = B). Then, slot by slot, a new s or v neuron reading an existing node replaces
/// the filler, chosen (with the m's op, at the first slot) to correct the m on the decisions
/// pulling N up against those pulling it down.
/// Where phase 5 puts the new m_neuron, relative to the stressed node N's neuron.
enum class Where { After, Before };

std::optional<Net<D>> grow_at(const Net<D>& net, NodeId N, const Grads& g, const Pick& pick, Where where) {
    auto depth = [&](NodeId i) { return static_cast<std::uint64_t>(net.nodes[i].depth); };
    const std::uint32_t q = net.in_qty_max;
    if (q < 2) return std::nullopt;
    std::vector<std::size_t> dec;
    for (std::size_t d = 0; d < g.dec_node.size(); ++d)
        if (g.dec_node[d][N] != 0) dec.push_back(d);
    if (dec.size() < 2) { ++g_rejects["fewer than 2 decisions pull on the node"]; return std::nullopt; }

    Net<D> base = net.copy_genome();
    NodeId M = N;                 // the new m_neuron's node
    NodeId ref = N;               // the connection the m wraps: its slot 0, copied into the fillers
    std::uint32_t ref_pos = 0;
    std::uint64_t dm = 0;         // the m's depth
    std::array<int, 3> down_extra{0, 0, 0};  // neurons between the m and N's readers (Before: N itself)
    std::vector<std::optional<NodeId>> slots;
    if (where == Where::After) {
        // N's neuron moves to B; the m takes N's node, so N's readers read the m
        const std::uint64_t lo = depth_floor(net, N);  // B's shallowest depth (N's neuron, its s gap included)
        dm = depth(N);
        if (dm < lo + 2) {  // room for B and the fillers below the m; N may move deeper
            if (!net.is_hidden(N)) { ++g_rejects["no depth room (after, output)"]; return std::nullopt; }
            std::uint64_t ub = std::numeric_limits<std::uint64_t>::max();
            for (NodeId r : live_readers(net, N)) ub = std::min(ub, depth(r) - s_gap(net, *net.nodes[r].neuron) - 1);
            if (ub < lo + 2) { ++g_rejects["no depth room (after)"]; return std::nullopt; }
            dm = ub;
        }
        const std::uint64_t d_b = lo + (dm - lo) / 3, d_f = lo + 2 * (dm - lo) / 3;
        base.nodes[N].depth = static_cast<D>(dm);
        const NodeId B = static_cast<NodeId>(base.nodes.size());
        base.nodes.push_back(Node<D>{base.nodes[N].birth, base.nodes[N].born_gen, static_cast<D>(d_b), base.nodes[N].neuron});
        slots.push_back(B);
        for (std::uint32_t i = 1; i < q; ++i) {
            slots.push_back(static_cast<NodeId>(base.nodes.size()));
            base.nodes.push_back(Node<D>{base.next_birth++, 0, static_cast<D>(d_f), VNeuron{B, Inversion::None, VOp::Mul, 1.0}});
        }
        base.nodes[N].neuron = MNeuron{AggOp::Min, slots};
        base.nodes[N].birth = base.next_birth++;
        base.nodes[N].born_gen = 0;
        for (auto& node : base.nodes) {  // N's length and history move with its neuron to B (spec/06)
            if (!node.neuron) continue;
            if (auto* m = std::get_if<MNeuron>(&*node.neuron)) {
                for (std::size_t i = 0; i < m->slots.size(); ++i)
                    if (m->slots[i] == N && m->slot_pos(i) > 0) m->set(i, B, m->slot_pos(i));
            } else if (auto* v = std::get_if<VNeuron>(&*node.neuron)) {
                if (v->input == N && v->input_pos > 0) v->input = B;
            } else if (auto& s = std::get<SNeuron>(*node.neuron); s.input == N && s.input_pos > 0) {
                s.input = B;
            }
        }
        ref = B;
    } else {
        // the m goes on the connection N's neuron reads (an m_neuron: its most stressed connected slot)
        const Neuron& nn = *net.nodes[N].neuron;
        std::optional<std::size_t> slot;
        if (const auto* m = std::get_if<MNeuron>(&nn)) {
            double best_stress = -1;
            for (std::size_t i = 0; i < m->slots.size(); ++i) {
                if (!m->slots[i]) continue;
                double st = 0;
                for (const auto& dn : g.dec_node) st += std::fabs(dn[*m->slots[i]]);
                if (st > best_stress) { best_stress = st; slot = i; }
            }
            if (!slot) { ++g_rejects["no connection to wrap (before)"]; return std::nullopt; }
            ref = *m->slots[*slot];
            ref_pos = m->slot_pos(*slot);
        } else {
            const auto& in = std::holds_alternative<VNeuron>(nn) ? std::get<VNeuron>(nn).input : std::get<SNeuron>(nn).input;
            if (!in) { ++g_rejects["no connection to wrap (before)"]; return std::nullopt; }
            ref = *in;
            ref_pos = std::holds_alternative<VNeuron>(nn) ? std::get<VNeuron>(nn).input_pos : std::get<SNeuron>(nn).input_pos;
        }
        const std::uint64_t dc = depth(ref) - ref_pos;     // the wrapped connection's depth
        const std::uint64_t gn = s_gap(net, nn);            // an s_neuron N keeps its gap above the m
        if (depth(N) < dc + gn + 3) { ++g_rejects["no depth room (before)"]; return std::nullopt; }
        dm = depth(N) - 1 - gn;
        const std::uint64_t d_f = dc + (dm - dc) / 2;
        slots.push_back(ref);
        for (std::uint32_t i = 1; i < q; ++i) {
            slots.push_back(static_cast<NodeId>(base.nodes.size()));
            base.nodes.push_back(Node<D>{base.next_birth++, 0, static_cast<D>(d_f),
                                         VNeuron{ref, Inversion::None, VOp::Mul, 1.0, ref_pos}});
        }
        M = static_cast<NodeId>(base.nodes.size());
        MNeuron mm{AggOp::Min, slots};
        mm.set(0, ref, ref_pos);
        base.nodes.push_back(Node<D>{base.next_birth++, 0, static_cast<D>(dm), mm});
        Neuron& bn = *base.nodes[N].neuron;
        if (auto* m = std::get_if<MNeuron>(&bn)) m->set(*slot, M, 0);
        else if (auto* v = std::get_if<VNeuron>(&bn)) { v->input = M; v->input_pos = 0; }
        else { auto& sn = std::get<SNeuron>(bn); sn.input = M; sn.input_pos = 0; }
        down_extra[static_cast<std::size_t>(type_of(nn))] = 1;
    }
    if (!valid_or_count(base)) return std::nullopt;

    // sources: existing nodes, and length and history nodes of s_neurons, whose values best line
    // up with the pulls on N; the best kSources of each kind
    struct Src {
        double score;
        NodeId node;
        std::uint32_t pos;
    };
    auto score_of = [&](auto&& val) -> std::optional<double> {
        double wsum = 0, mean = 0;
        for (std::size_t d : dec) { const double a = std::fabs(g.dec_node[d][N]); wsum += a; mean += a * val(d); }
        mean /= wsum;
        double var = 0, cov = 0;
        for (std::size_t d : dec) {
            const double a = std::fabs(g.dec_node[d][N]), x = val(d) - mean;
            var += a * x * x;
            cov += g.dec_node[d][N] * x;
        }
        if (!(var > 0) || !std::isfinite(var) || !std::isfinite(cov)) return std::nullopt;
        return std::fabs(cov) / (std::sqrt(var / wsum) * wsum);
    };
    std::vector<Src> nodes_ranked, hist_ranked;
    for (NodeId s = 0; s < net.nodes.size(); ++s) {
        if (s == N || depth(s) + 2 > dm) continue;
        if (auto sc = score_of([&](std::size_t d) { return g.dec_vals[d][s]; })) nodes_ranked.push_back({*sc, s, 0});
    }
    for (std::size_t i = 0; i < g.hist.size(); ++i) {
        const auto [s, p] = g.hist[i];
        if (s == N || depth(s) - p + 2 > dm) continue;  // N's own history moves to B; skip it
        if (auto sc = score_of([&](std::size_t d) { return g.dec_hval[d][i]; })) hist_ranked.push_back({*sc, s, p});
    }
    std::vector<Src> ranked;
    for (auto* list : {&nodes_ranked, &hist_ranked}) {
        std::stable_sort(list->begin(), list->end(), [](const Src& a, const Src& b) { return a.score > b.score; });
        if (list->size() > kSources) list->resize(kSources);
        ranked.insert(ranked.end(), list->begin(), list->end());
    }
    if (ranked.empty()) { ++g_rejects["no source varies with the pulls"]; return base; }

    // Each new slot is fed by a chain from its source with as many s, v and m neurons as the
    // path needs for the mix rule (fewest on any path into the source, plus fewest from N to an
    // output, plus the m itself); at most 2 one-in neurons in a row, single-slot min m between.
    const auto mc = mix_counts(net);
    const int pm = static_cast<int>(net.path_mix);
    enum XKind { kAdd, kMul, kDiff, kPrev };
    Net<D> best = base;
    for (std::uint32_t k = 1; k < q; ++k) {
        const NodeId F = *slots[k];
        std::vector<Net<D>> cands;
        std::vector<AggOp> ops{std::get<MNeuron>(*best.nodes[M].neuron).op};
        if (k == 1) ops = {AggOp::Min, AggOp::Max, AggOp::Mean, AggOp::Median};
        for (AggOp op : ops) {
            Net<D> b0 = best.copy_genome();
            std::get<MNeuron>(*b0.nodes[M].neuron).op = op;
            cands.push_back(b0);  // the filler stays
            for (const auto& rs : ranked) {
                const NodeId S = rs.node;              // the node, or the s_neuron of a length / history node
                const std::uint32_t sp = rs.pos;
                const std::uint64_t sd = depth(S) - sp;  // the source connection's depth
                for (XKind xk : {kAdd, kMul, kDiff, kPrev}) {
                    const int x_type = (xk == kAdd || xk == kMul) ? 0 : 1;
                    int need[3];
                    for (int t = 0; t < 3; ++t)
                        need[t] = std::max(0, pm - mc.up[t][S] - mc.down[t][N] - down_extra[static_cast<std::size_t>(t)] - (t == 2) - (t == x_type));
                    {
                        // from S toward the m: alternate one-in (s, v) and many-in (single-slot min)
                        // neurons, never letting a run reach max_run (counting the run ending at
                        // S); extra identity neurons separate runs where the counts run out
                        const int lim = net.max_run == 0 ? 1 << 20 : static_cast<int>(net.max_run);
                        std::vector<int> seq;  // 0 v, 1 s, 2 m
                        int ns = need[1], nv = need[0], nm = need[2];
                        int r1 = mc.run1[S], rm = mc.runm[S];
                        auto put = [&](int t) {
                            seq.push_back(t);
                            if (t == 2) { ++rm; r1 = 0; } else { ++r1; rm = 0; }
                        };
                        while (ns + nv + nm > 0) {
                            const bool one_ok = ns + nv > 0 && r1 + 1 <= lim - 1;  // leave room for X
                            const bool m_ok = nm > 0 && rm + 1 <= lim - 1;        // and for the m
                            if (one_ok && (r1 == 0 || !m_ok)) { if (ns > 0) { --ns; put(1); } else { --nv; put(0); } }
                            else if (m_ok) { --nm; put(2); }
                            else if (ns + nv > 0) put(2);  // a separator m
                            else put(0);                    // a separator v
                        }
                        if (r1 + 1 > lim) put(2);           // X must not end a run that is too long
                        const std::uint64_t len = seq.size() + 1;  // the chain plus the slot's neuron
                        if (dm < sd + len + 1) { ++g_rejects["no depth room for the chain"]; continue; }
                        Net<D> c = b0.copy_genome();
                        auto d_at = [&](std::uint64_t i) { return static_cast<D>(sd + (i + 1) * (dm - sd) / (len + 1)); };
                        NodeId prev = S;
                        std::uint32_t prev_pos = sp;  // the first neuron reads the source connection
                        for (std::size_t i = 0; i < seq.size(); ++i) {
                            Neuron nn;
                            if (seq[i] == 0) nn = VNeuron{prev, Inversion::None, VOp::Mul, 1.0, prev_pos};
                            else if (seq[i] == 1) nn = SNeuron{prev, AggOp::Mean, 1, prev_pos};
                            else {
                                MNeuron mm{AggOp::Min, std::vector<std::optional<NodeId>>(q, std::nullopt)};
                                mm.set(0, prev, prev_pos);
                                nn = mm;
                            }
                            const NodeId id = static_cast<NodeId>(c.nodes.size());
                            c.nodes.push_back(Node<D>{c.next_birth++, 0, d_at(i), std::move(nn)});
                            prev = id;
                            prev_pos = 0;
                        }
                        if (xk == kAdd) c.nodes[F].neuron = VNeuron{prev, Inversion::None, VOp::Add, 0.0, prev_pos};
                        else if (xk == kMul) c.nodes[F].neuron = VNeuron{prev, Inversion::None, VOp::Mul, 1.0, prev_pos};
                        else c.nodes[F].neuron = SNeuron{prev, xk == kDiff ? AggOp::Mean : AggOp::Lag, 1, prev_pos};
                        c.nodes[F].depth = d_at(seq.size());
                        if (!valid_or_count(c)) continue;
                        if (xk == kDiff || xk == kPrev) { cands.push_back(std::move(c)); continue; }
                        // weights at the breakpoints, from the chain's actual values on the decisions
                        std::vector<double> ws;
                        {
                            Tape tape(c);
                            for (std::size_t d : dec) {
                                const auto& [h, t] = g.dec_at[d];
                                tape.clear();
                                for (std::size_t i = 0; i <= t; ++i) tape.push(h->path[i].in);
                                const double xin = tape.conn_value(t, prev, prev_pos), bv = tape.conn_value(t, ref, ref_pos);
                                ws.push_back(xk == kAdd ? bv - xin : bv / xin);
                            }
                        }
                        for (double w : quantiles(ws, 8)) {
                            Net<D> cw = c;
                            std::get<VNeuron>(*cw.nodes[F].neuron).weight = w;
                            cands.push_back(std::move(cw));
                        }
                    }
                }
            }
        }
        const std::size_t i = pick(cands);
        if (i >= cands.size()) break;
        best = cands[i].copy_genome();
    }
    return best;
}

// ------------------------------------------------------------------ bookkeeping

/// Most broken earlier hands the acceptance rule allows for the next block (spec in the header).
long allowed_broken(int accepted, long ledger, long x, long broken_prev) {
    if (accepted == 0) return 0;
    const long n = ledger + x;
    if (accepted == 1) return (n - 1) / 2;   // broken < 50% of all hands
    if (broken_prev == 0) return 0;          // 0% stays 0%
    return (broken_prev * n - 1) / ledger;   // broken / n < broken_prev / ledger
}

std::size_t neurons(const Net<D>& net) { return net.nodes.size() - net.n_inputs; }

/// Even depths: hidden nodes keep their depth order (ties by id) and are spread evenly over
/// [1, D_MAX - 1]. Every connection stays strictly shallower than its reader and the gaps are far
/// wider than an s gap, so outputs are unchanged (nets without length or history reads only).
void respace_even(Net<D>& net) {
    std::vector<NodeId> hidden;
    for (NodeId i = net.n_inputs + net.n_outputs; i < net.nodes.size(); ++i) hidden.push_back(i);
    std::stable_sort(hidden.begin(), hidden.end(), [&](NodeId a, NodeId b) { return net.nodes[a].depth < net.nodes[b].depth; });
    const std::uint64_t step = static_cast<std::uint64_t>(Net<D>::d_max) / (hidden.size() + 1);
    for (std::size_t k = 0; k < hidden.size(); ++k) net.nodes[hidden[k]].depth = static_cast<D>(step * (k + 1));
    net.reset();
    net.validate();
}

/// Connections reading a length or history node (position >= 1), and those reading a length node.
std::size_t history_reads(const Net<D>& net) {
    std::size_t n = 0;
    for (const auto& node : net.nodes)
        if (node.neuron) for_each_conn(*node.neuron, [&](NodeId, std::uint32_t p) { n += p > 0; });
    return n;
}
/// The tape must reproduce Net::step exactly; checked on the given hands.
bool tape_matches(const Net<D>& net, const std::vector<HandRec*>& hands, std::size_t count) {
    Tape tape(net);
    Net<D> n = net.copy_genome();
    for (std::size_t i = 0; i < std::min(count, hands.size()); ++i) {
        std::vector<Step> path;
        play(tape, hands[i]->deal, &path);
        n.reset_state();
        for (std::size_t t = 0; t < path.size(); ++t)
            if (to_bits(n.step(path[t].in)[0]) != to_bits(tape.output(t))) return false;
    }
    return true;
}

std::size_t length_reads(const Net<D>& net) {
    std::size_t n = 0;
    for (const auto& node : net.nodes)
        if (node.neuron) for_each_conn(*node.neuron, [&](NodeId, std::uint32_t p) { n += p == 1; });
    return n;
}

struct Holdout {
    double pts = 0, even_pct = 0;
};

Holdout holdout(const Net<D>& net0, const std::vector<Deal>& deals) {
    Net<D> net = net0.copy_genome();
    Holdout h;
    long even = 0;
    for (const Deal& d : deals) {
        const double s = play_deal(net, d);
        h.pts += s;
        even += s >= 0;
    }
    if (!deals.empty()) { h.pts /= deals.size(); h.even_pct = 100.0 * even / deals.size(); }
    return h;
}

std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
    return buf;
}

std::string type_name(const Net<D>& net, NodeId i) {
    const Neuron& n = *net.nodes[i].neuron;
    if (const auto* v = std::get_if<VNeuron>(&n))
        return "v " + (v->inversion == Inversion::None ? std::string() : to_string(v->inversion) + " ") + to_string(v->op);
    if (const auto* s = std::get_if<SNeuron>(&n)) return "s " + to_string(s->op);
    return "m " + to_string(std::get<MNeuron>(n).op);
}

}  // namespace

int main(int argc, char** argv) {
    fs::path pop_file = "runs/bulk-bj-control/population.jpnet";
    fs::path out = fs::path("runs") / ("tune-" + timestamp());
    std::optional<std::uint64_t> net_id;
    int n_blocks = 100, x = 20, max_rounds = 100, n_holdout = 10000;
    std::uint64_t seed = 777;
    double conflict_at = 0.5;
    bool open = false;               // open-card blackjack: every card face up (game.hpp)
    std::string start_kind = "pop";  // pop: the population's best net (widened with add_input if open); fresh
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--pop") pop_file = v;
        else if (a == "--net-id") net_id = std::stoull(v);
        else if (a == "--blocks") n_blocks = std::stoi(v);
        else if (a == "--block-size") x = std::stoi(v);
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--holdout") n_holdout = std::stoi(v);
        else if (a == "--max-rounds") max_rounds = std::stoi(v);
        else if (a == "--margin") g_margin = std::stod(v);
        else if (a == "--conflict") conflict_at = std::stod(v);
        else if (a == "--open") open = v == "1" || v == "on" || v == "yes";
        else if (a == "--start") start_kind = v;
        else if (a == "--out") out = v;
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    fs::create_directories(out);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const unsigned threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    const auto t0 = std::chrono::steady_clock::now();
    auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

    // phase 1: deal every block up front
    std::vector<HandRec> hands;
    hands.reserve(static_cast<std::size_t>(n_blocks) * x);
    {
        std::ofstream f(out / "deals.csv");
        f << "index,seed,usable,block\n";
        Rng r(seed);
        for (long i = 0; hands.size() < static_cast<std::size_t>(n_blocks) * x; ++i) {
            const std::uint64_t s = r.next_u64();
            const bool ok = can_break_even(Deal(s));
            f << i << ',' << s << ',' << ok << ',' << (ok ? static_cast<long>(hands.size() / x) : -1L) << '\n';
            if (ok) hands.emplace_back(s);
        }
    }
    std::vector<std::vector<HandRec*>> blocks(n_blocks);
    for (std::size_t i = 0; i < hands.size(); ++i) blocks[i / x].push_back(&hands[i]);

    // the starting net: the population's net that plays the most dealt hands to 0 or more
    if (start_kind != "pop" && start_kind != "fresh") { std::cerr << "--start must be pop or fresh\n"; return 2; }
    Net<D> net;
    if (start_kind == "fresh") {
        // a minimal net (spec/03) with the trainer's parameters
        Rng r(seed);
        net = Net<D>::create(static_cast<std::uint32_t>(open ? kOpenInputs : kInputs), 1, 3, Priority::Speed, r, 2, 3);
    } else {
        Pop pop = Pop::load(pop_file);
        long best = -1;
        for (auto id : pop.ids()) {  // played as the normal 9-input game
            if (net_id && id != *net_id) continue;
            Net<D> n = pop.get(id).copy_genome();
            long even = 0;
            for (const HandRec& h : hands) even += play_deal(n, h.deal) >= 0;
            if (even > best) { best = even; net = pop.get(id).copy_genome(); }
        }
        if (best < 0) { std::cerr << "no such net\n"; return 2; }
        // open cards: append the 21 new inputs, silenced (spec/12 add_input, active = false), so the
        // net plays exactly as before until tuning gives them weight
        while (open && net.n_inputs < kOpenInputs) {
            respace_even(net);  // room for add_input's medians in front of outputs (spec/12)
            net = add_input(net, std::nullopt, false, 0);
        }
    }
    g_open_cards = open;
    // TUNE_GRADCHECK=history: first make the net read length and history nodes (library mutation
    // with history_read_prob), so the checks below cover them
    if (const char* gc = std::getenv("TUNE_GRADCHECK"); gc && std::string(gc) == "history") {
        Profile hp = Profile::defaults();
        hp.set_overall("history_read_prob", 0.6);
        hp.set_overall("rewire_prob", 0.6);
        hp.set_overall("edit_max_pct", 40.0);
        hp.set_overall("growth_max_pct", 20.0);
        hp.set_overall("removal_max_pct", 0.0);
        Rng r(seed);
        for (int gen = 1; gen <= 60 && history_reads(net) < 6; ++gen) net = mutate(net, hp, gen, r);
        std::cerr << "  grown to " << neurons(net) << " neurons, " << history_reads(net) << " length/history reads ("
                  << length_reads(net) << " length)\n";
    }
    // the tape must reproduce Net::step exactly
    {
        Tape tape(net);
        Net<D> n = net.copy_genome();
        for (std::size_t i = 0; i < std::min<std::size_t>(500, hands.size()); ++i) {
            std::vector<Step> path;
            play(tape, hands[i].deal, &path);
            n.reset_state();
            for (std::size_t t = 0; t < path.size(); ++t)
                if (to_bits(n.step(path[t].in)[0]) != to_bits(tape.output(t))) { std::cerr << "tape differs from Net::step\n"; return 1; }
        }
    }
    if (std::getenv("TUNE_GRADCHECK")) {  // analytic dL/dw against central differences, on block 0
        prepare_targets(net, blocks[0]);
        const Grads g = gradients(net, blocks[0]);
        auto loss = [&](const Net<D>& n) { return evaluate(n, blocks[0], {}).loss; };
        for (NodeId i = 0; i < net.nodes.size(); ++i) {
            if (!net.nodes[i].neuron || !std::holds_alternative<VNeuron>(*net.nodes[i].neuron)) continue;
            const double w = std::get<VNeuron>(*net.nodes[i].neuron).weight, h = 1e-6 * std::max(1.0, std::fabs(w));
            Net<D> a = net.copy_genome(), b = net.copy_genome();
            std::get<VNeuron>(*a.nodes[i].neuron).weight = w + h;
            std::get<VNeuron>(*b.nodes[i].neuron).weight = w - h;
            std::cerr << "  node " << std::setw(3) << i << "  " << std::setw(14) << type_name(net, i) << "  analytic "
                      << std::setw(13) << g.w[i] << "  numeric " << std::setw(13) << (loss(a) - loss(b)) / (2 * h) << "\n";
        }
        return 0;
    }
    std::vector<Deal> hold;
    {
        Rng r(seed ^ 0x9e3779b97f4a7c15ULL);
        for (int i = 0; i < n_holdout; ++i) hold.emplace_back(r.next_u64());
    }
    save_net(out / "start.jnet", net);
    {
        long can = 0;
        for (const Deal& d : hold) can += can_break_even(d);
        std::cout << "ceiling: " << std::fixed << std::setprecision(1) << 100.0 * can / std::max<std::size_t>(1, hold.size())
                  << "% of holdout hands can reach 0 or more\n";
    }
    const Holdout h0 = holdout(net, hold);
    std::cout << "start: " << neurons(net) << " neurons, " << hands.size() << " usable hands in " << n_blocks << " blocks of "
              << x << ", holdout " << std::fixed << std::setprecision(4) << h0.pts << " pts/deal, "
              << std::setprecision(1) << h0.even_pct << "% >= 0\n";

    std::ofstream log(out / "attempts.csv"), slog(out / "stress.csv");
    log << "attempt,block,result,rounds,block_works,ledger_broken,ledger,broken_pct,neurons,weight_steps,op_changes,"
           "growths,history_reads,holdout_pts,holdout_even_pct,seconds\n";
    slog << "attempt,block,round,rank,node,kind,stress,conflict,decided\n";
    log << "0,-1,start,0,0,0,0,0," << neurons(net) << ",0,0,0," << history_reads(net) << "," << h0.pts << ',' << h0.even_pct << ",0\n";

    std::deque<int> queue;
    for (int b = 0; b < n_blocks; ++b) queue.push_back(b);
    std::vector<HandRec*> ledger;
    std::set<int> failed;  // blocks stuck since the last accepted block
    int accepted = 0, attempt = 0;
    long broken_prev = 0;
    while (!queue.empty() && !g_stop) {
        const int blk = queue.front();
        queue.pop_front();
        ++attempt;
        const Net<D> before = net.copy_genome();
        const long allowed = allowed_broken(accepted, static_cast<long>(ledger.size()), x, broken_prev);
        std::vector<HandRec*> all = blocks[blk];
        all.insert(all.end(), ledger.begin(), ledger.end());
        int rounds = 0, n_w = 0, n_o = 0, n_g = 0, n_before = 0;
        bool ok = false;
        Eval cur;
        for (; rounds < max_rounds && !g_stop; ++rounds) {
            prepare_targets(net, all);                       // phases 2-3
            cur = evaluate(net, blocks[blk], ledger);
            if (cur.block_works == x && cur.ledger_broken <= allowed) { ok = true; break; }
            const Grads g = gradients(net, all);             // phase 4
            const auto st = stress_of(net, g);
            if (st.empty()) break;
            // phase 5: grow first where the top neuron is pulled both ways, else adjust first
            const bool grow_first = st[0].conflict >= conflict_at;
            for (std::size_t k = 0; k < std::min<std::size_t>(5, st.size()); ++k)
                slog << attempt << ',' << blk << ',' << rounds << ',' << k << ',' << st[k].id << ",\"" << type_name(net, st[k].id)
                     << "\"," << st[k].stress << ',' << st[k].conflict << ',' << (grow_first ? "grow" : "adjust") << '\n';
            // phase 6 rule: more block hands working, then fewer broken earlier hands, then lower loss;
            // never more broken earlier hands than the acceptance rule allows
            auto better = [&](const Eval& c, const Eval& ref) {
                if (c.ledger_broken > allowed) return false;
                if (c.block_works != ref.block_works) return c.block_works > ref.block_works;
                if (c.ledger_broken != ref.ledger_broken) return c.ledger_broken < ref.ledger_broken;
                return c.loss < ref.loss;
            };
            // the best candidate (better than `ref` if given), or size() for none
            auto best_of = [&](const std::vector<Net<D>>& cands, const Eval* ref) {
                const auto ev = evaluate_all(cands, blocks[blk], ledger, threads);
                std::size_t best = cands.size();
                for (std::size_t k = 0; k < cands.size(); ++k) {
                    if (best == cands.size() ? (ref ? better(ev[k], *ref) : ev[k].ledger_broken <= allowed)
                                             : better(ev[k], ev[best]))
                        best = k;
                }
                return best;
            };
            const Pick pick = [&](const std::vector<Net<D>>& c) { return best_of(c, nullptr); };
            auto try_kind = [&](int kind) -> bool {
                std::vector<Net<D>> cands;
                std::vector<char> before;  // growth: whether each candidate's m went before N's neuron
                if (kind == 0) cands = weight_candidates(net, g.w);
                else if (kind == 1) cands = op_candidates(net, st);
                else {
                    std::size_t sites = 0;
                    for (const Stress& s : st) {
                        if (sites == kTopGrow) break;
                        if (grow_first && s.conflict < conflict_at) continue;
                        ++sites;
                        for (Where w : {Where::After, Where::Before})  // the m after N's neuron, or before it
                            if (auto c = grow_at(net, s.id, g, pick, w)) {
                                cands.push_back(std::move(*c));
                                before.push_back(w == Where::Before);
                            }
                    }
                }
                if (cands.empty()) return false;
                const std::size_t best = best_of(cands, &cur);
                if (best == cands.size()) return false;
                net = cands[best].copy_genome();
                (kind == 0 ? n_w : kind == 1 ? n_o : n_g) += 1;
                if (kind == 2) n_before += before[best];
                if (kind == 2 && !tape_matches(net, blocks[blk], 30)) {
                    std::cerr << "tape differs from Net::step after a growth\n";
                    std::exit(1);
                }
                return true;
            };
            const std::array<int, 3> order = grow_first ? std::array<int, 3>{2, 0, 1} : std::array<int, 3>{0, 1, 2};
            bool improved = false;
            for (int kind : order)
                if ((improved = try_kind(kind))) break;
            if (!improved) {                                 // stuck
                if (std::getenv("TUNE_DEBUG")) {
                    double gn = 0;
                    for (double v : g.w) gn = std::max(gn, std::fabs(v));
                    std::cerr << "  stuck: loss " << cur.loss << ", max |dL/dw| " << gn << ", decisions with error "
                              << g.dec_node.size() << "\n";
                    Tape tape(net);
                    for (HandRec* hp : blocks[blk]) {
                        if (play(tape, hp->deal, nullptr) >= 0) continue;
                        std::cerr << "  broken hand " << hp->seed << (hp->has_path ? "" : " (no target series)") << ":";
                        tape.clear();
                        for (const Step& stp : hp->path) {
                            const double y = tape.push(stp.in);
                            std::cerr << "  [";
                            for (double v : stp.in) std::cerr << v << ' ';
                            std::cerr << "] y=" << y << " want " << "SHDP"[stp.target] << " e=" << error(y, stp.target);
                        }
                        std::cerr << "\n";
                    }
                }
                break;
            }
        }
        Holdout h{};
        if (ok) {
            ledger.insert(ledger.end(), blocks[blk].begin(), blocks[blk].end());
            broken_prev = cur.ledger_broken;
            ++accepted;
            failed.clear();
            h = holdout(net, hold);
        } else {
            net = before.copy_genome();
            if (!g_stop) {
                queue.push_back(blk);
                failed.insert(blk);
            }
        }
        const double pct = ledger.empty() ? 0.0 : 100.0 * (ok ? cur.ledger_broken : broken_prev) / ledger.size();
        log << attempt << ',' << blk << ',' << (ok ? "accepted" : "stuck") << ',' << rounds << ',' << cur.block_works << ','
            << cur.ledger_broken << ',' << ledger.size() << ',' << pct << ',' << neurons(net) << ',' << n_w << ',' << n_o
            << ',' << n_g << ',' << history_reads(net) << ',' << (ok ? h.pts : 0.0) << ',' << (ok ? h.even_pct : 0.0) << ',' << secs() << '\n';
        log.flush();
        slog.flush();
        std::cout << "attempt " << std::setw(4) << attempt << "  block " << std::setw(3) << blk << "  "
                  << (ok ? "accepted" : "stuck   ") << "  rounds " << std::setw(3) << rounds << "  works " << std::setw(2)
                  << cur.block_works << "/" << x << "  broken " << std::setw(4) << (ok ? cur.ledger_broken : broken_prev) << "/"
                  << std::setw(4) << ledger.size() << " (" << std::fixed << std::setprecision(1) << pct << "%)  neurons "
                  << neurons(net) << "  w/o/g " << n_w << "/" << n_o << "/" << n_g << " (before " << n_before << ")  hist " << history_reads(net);
        if (ok) std::cout << "  holdout " << std::setprecision(4) << h.pts << " pts, " << std::setprecision(1) << h.even_pct << "% >=0";
        std::cout << "  " << std::setprecision(0) << secs() << "s\n" << std::flush;
        if (!queue.empty() && failed.size() == queue.size()) {
            std::cout << "every queued block has failed since the last accepted one: finished\n";
            break;
        }
    }
    if (std::getenv("TUNE_DEBUG"))
        for (const auto& [why, n] : g_rejects) std::cerr << "  growth rejected " << n << "x: " << why << "\n";
    save_net(out / "tuned.jnet", net);
    const Holdout h1 = holdout(net, hold);
    std::cout << "end: " << accepted << "/" << n_blocks << " blocks accepted, " << neurons(net) << " neurons, holdout "
              << std::fixed << std::setprecision(4) << h1.pts << " pts/deal (start " << h0.pts << "), "
              << std::setprecision(1) << h1.even_pct << "% >= 0 (start " << h0.even_pct << "%)\n  " << out.string() << "\n";
    return 0;
}
