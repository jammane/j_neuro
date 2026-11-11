#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace th;
using N16 = Net<std::uint16_t>;

static void check_same_outputs(N16 before, N16 after, std::uint32_t n_in) {
    auto rows = sample_inputs(n_in);
    CHECK(outputs_for(before, rows) == outputs_for(after, rows));
}

static std::vector<bool> fresh(const N16& net) { return std::vector<bool>(net.nodes.size(), false); }
static std::vector<NodeId> ids(const std::vector<bool>& r) {
    std::vector<NodeId> out;
    for (NodeId i = 0; i < r.size(); ++i) if (r[i]) out.push_back(i);
    return out;
}

TEST_CASE("identity and silenced predicates") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, M(AggOp::Mean, {2, 3, 4})},
                                                 {10, V(0, VOp::Mul, 1.0)}, {10, V(0, VOp::Add, 0.0)},
                                                 {10, V(0, VOp::Mul, 0.0, Inversion::Recip)}});
    CHECK(is_identity(net, 2));
    CHECK(is_identity(net, 3));
    CHECK_FALSE(is_identity(net, 4));
    CHECK(is_silenced(net, 4));
    CHECK_FALSE(is_silenced(net, 2));
    net.nodes[3].neuron = V(0, VOp::Sub, -0.0);
    CHECK(is_identity(net, 3));
    net.nodes[3].neuron = V(0, VOp::Add, 0.0, Inversion::Bitnot);
    CHECK_FALSE(is_identity(net, 3));
}

TEST_CASE("connectionless constants") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, M(AggOp::Mean, {0, 2, 3})},
                                                 {5, V(none, VOp::Add, 2.0)}, {5, V(none, VOp::Div, 3.0)}});
    CHECK_FALSE(is_silenced(net, 2));
    CHECK(is_silenced(net, 3));
    net.nodes[2].neuron = S(none, AggOp::Max, 3);
    CHECK(is_silenced(net, 2));
    net.nodes[2].neuron = M(AggOp::Max, {none, none, none});
    CHECK(is_silenced(net, 2));
    net.nodes[2].neuron = V(none, VOp::Add, 5.0, Inversion::Bitnot);
    CHECK(is_silenced(net, 2));
}

TEST_CASE("bypass exact") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {3, 1, none})},
                                                 {10, V(4, VOp::Div, 1.0)}, {5, V(0, VOp::Mul, 2.0)}});
    auto before = net;
    auto removed = fresh(net);
    bypass(net, 3, removed);
    CHECK(ids(removed) == std::vector<NodeId>{3});
    CHECK(std::get<MNeuron>(*net.nodes[2].neuron).slots == Slots{4, 1, none});
    compact(net, removed);
    CHECK(net.nodes.size() == 4);
    net.validate();
    check_same_outputs(before, net, 2);
}

TEST_CASE("silenced cascade downstream") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {4, 1, 0})},
                                                 {10, V(0, VOp::Mul, 0.0)}, {20, V(3, VOp::Mul, 5.0)}});
    auto before = net;
    auto removed = fresh(net);
    remove_silenced(net, 3, removed);
    CHECK(ids(removed) == std::vector<NodeId>{3, 4});
    CHECK(std::get<MNeuron>(*net.nodes[2].neuron).slots == Slots{none, 1, 0});
    compact(net, removed);
    net.validate();
    check_same_outputs(before, net, 2);
}

TEST_CASE("cascade stops at min reader") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Min, {4, 1, 0})},
                                                 {10, V(0, VOp::Mul, 0.0)}, {20, V(3, VOp::Mul, 5.0)}});
    auto before = net;
    auto removed = fresh(net);
    remove_silenced(net, 3, removed);
    CHECK(ids(removed) == std::vector<NodeId>{3});
    CHECK_FALSE(std::get<VNeuron>(*net.nodes[4].neuron).input.has_value());
    compact(net, removed);
    check_same_outputs(before, net, 2);
}

TEST_CASE("cascade stops at nonzero constant") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {4, 1, 0})},
                                                 {10, V(0, VOp::Mul, 0.0)}, {20, V(3, VOp::Add, 2.0)}});
    auto before = net;
    auto removed = fresh(net);
    remove_silenced(net, 3, removed);
    CHECK(ids(removed) == std::vector<NodeId>{3});
    compact(net, removed);
    check_same_outputs(before, net, 2);
}

TEST_CASE("garbage collect") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0)}, {5, V(0)}, {6, S(2)}});
    auto removed = fresh(net);
    garbage_collect(net, removed);
    CHECK(ids(removed) == std::vector<NodeId>{2, 3});
    compact(net, removed);
    CHECK(net.nodes.size() == 2);
    net.validate();
}

TEST_CASE("compact keeps order and births") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, M(AggOp::Max, {3, 2, none})}, {5, V(0)}, {6, V(0)}});
    auto removed = fresh(net);
    removed[2] = true;
    std::get<MNeuron>(*net.nodes[1].neuron).slots = Slots{3, none, none};
    compact(net, removed);
    CHECK(net.nodes.size() == 3);
    CHECK(net.nodes[2].birth == 3);
    CHECK(std::get<MNeuron>(*net.nodes[1].neuron).slots == Slots{2, none, none});
    CHECK(path_rule_ok(net));
}
