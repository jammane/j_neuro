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
