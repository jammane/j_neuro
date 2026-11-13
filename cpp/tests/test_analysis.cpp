#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace th;
using N16 = Net<std::uint16_t>;

static N16 diamond() {
    return make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {3, 4, 5})},
                                             {10, V(0)}, {10, V(0)}, {20, V(1)}});
}

static std::vector<bool> removed_set(const N16& net, std::initializer_list<NodeId> ids) {
    std::vector<bool> r(net.nodes.size(), false);
    for (auto i : ids) r[i] = true;
    return r;
}

TEST_CASE("connections and readers") {
    auto net = diamond();
    CHECK(connections(*net.nodes[2].neuron) == std::vector<NodeId>{3, 4, 5});
    CHECK(connections(V(none)).empty());
    auto r = readers(net);
    CHECK(r[0] == std::vector<NodeId>{3, 4});
    CHECK(r[1] == std::vector<NodeId>{5});
    CHECK(r[3] == std::vector<NodeId>{2});
    CHECK(r[2].empty());
}

TEST_CASE("path rule") {
    auto net = diamond();
    CHECK(path_rule_ok(net));
    net.nodes[5].neuron = V(0);
    CHECK_FALSE(path_rule_ok(net));
}

TEST_CASE("path rule with removed and cut") {
    auto net = diamond();
    auto r3 = removed_set(net, {3});
    auto r34 = removed_set(net, {3, 4});
    CHECK(path_rule_ok(net, &r3));
    CHECK_FALSE(path_rule_ok(net, &r34));
    CHECK(path_rule_ok(net, nullptr, NodeId{3}));
    CHECK_FALSE(path_rule_ok(net, nullptr, NodeId{5}));
}

TEST_CASE("immunity") {
    auto net = diamond();
    CHECK_FALSE(is_immune(net, 3));
    CHECK_FALSE(is_immune(net, 4));
    CHECK(is_immune(net, 5));
    CHECK(is_immune(net, 2));
    net.nodes[4].neuron = V(1);
    CHECK_FALSE(is_immune(net, 5));
    CHECK(is_immune(net, 3));
}

TEST_CASE("legal depth range") {
    auto net = diamond();
    CHECK(legal_depth_range(net, 3) == std::pair<std::uint16_t, std::uint16_t>{1, D16 - 1});
    auto n2 = make_net<std::uint16_t>(1, 1, 3, {{D16, V(2)}, {50, V(3)}, {10, V(0)}});
    CHECK(legal_depth_range(n2, 3) == std::pair<std::uint16_t, std::uint16_t>{1, 49});
    CHECK(legal_depth_range(n2, 2) == std::pair<std::uint16_t, std::uint16_t>{11, D16 - 1});
}

TEST_CASE("reaching output") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0)}, {5, V(0)}, {6, S(2)}});
    CHECK(reaching_output(net) == std::vector<bool>{true, true, false, false});
}

// The original path rule: one search per input. The bitset pass and PathRuleTracker must agree
// with it on every net, every removed set and every cut.
template <class D>
static bool path_rule_reference(const Net<D>& net, const Removed* removed, std::optional<NodeId> cut) {
    auto ok = [&](NodeId i) { return detail::alive(removed, i) && (!cut || *cut != i); };
    const auto rd = readers(net, removed);
    for (NodeId i = 0; i < net.n_inputs; ++i) {
        if (!ok(i)) return false;
        std::vector<bool> seen(net.nodes.size(), false);
        std::vector<NodeId> stack{i};
        seen[i] = true;
        while (!stack.empty()) {
            NodeId x = stack.back();
            stack.pop_back();
            for (NodeId r : rd[x])
                if (ok(r) && !seen[r]) { seen[r] = true; stack.push_back(r); }
        }
        for (NodeId o = net.n_inputs; o < net.n_inputs + net.n_outputs; ++o)
            if (!seen[o]) return false;
    }
    return true;
}

TEST_CASE("path rule: bitset pass and tracker match the per-input search") {
    using N32 = Net<std::uint32_t>;
    Profile p = Profile::defaults();
    p.set_overall("growth_max_pct", 20.0);
    p.set_overall("growth_gen_prob", 1.0);
    p.set_overall("growth_net_prob", 1.0);
    std::uint64_t checks = 0, failing = 0;
    for (std::uint32_t n_in : {3u, 70u, 130u}) {  // one, two and three bitset words
        Rng rng(1000 + n_in);
        N32 net = std::get<N32>(create(n_in, 4, DepthType::U32, 5, Priority::Speed, rng));
        for (std::uint64_t g = 1; g <= 40; ++g) net = mutate(net, p, g, rng);
        // Random edits on a working copy; the tracker is never told what changed.
        N32 w = net;
        Removed removed(w.nodes.size(), false);
        PathRuleTracker<std::uint32_t> tracker;
        for (int step = 0; step < 3000; ++step) {
            const NodeId x = static_cast<NodeId>(w.n_inputs + rng.below(w.nodes.size() - w.n_inputs));
            const auto kind = rng.below(10);
            if (kind < 5) {  // set a slot or input: empty it, or point it at a shallower node
                std::optional<NodeId> to;
                if (rng.below(2)) {
                    const NodeId c = static_cast<NodeId>(rng.below(w.nodes.size()));
                    if (w.nodes[c].depth < w.nodes[x].depth) to = c;
                }
                Neuron& n = *w.nodes[x].neuron;
                if (auto* m = std::get_if<MNeuron>(&n)) m->slots[rng.below(m->slots.size())] = to;
                else if (auto* v = std::get_if<VNeuron>(&n)) v->input = to;
                else std::get<SNeuron>(n).input = to;
            } else if (kind < 7) {  // tombstone or revive a hidden node
                if (w.is_hidden(x)) removed[x] = !removed[x];
            } else if (kind < 8) {  // restore the original net (an undo of everything)
                w = net;
                removed.assign(w.nodes.size(), false);
            } else {  // a cut query
                const NodeId cut = static_cast<NodeId>(rng.below(w.nodes.size()));
                const bool ref = path_rule_reference(w, &removed, cut);
                CHECK(path_rule_ok(w, &removed, cut) == ref);
                CHECK(tracker.ok(w, &removed, cut) == ref);
            }
            const bool ref = path_rule_reference(w, &removed, std::nullopt);
            REQUIRE(path_rule_ok(w, &removed) == ref);
            REQUIRE(tracker.ok(w, &removed) == ref);
            ++checks;
            failing += !ref;
        }
    }
    CHECK(failing > 0);           // the edits broke the rule often enough to test both answers
    CHECK(failing < checks);
}
