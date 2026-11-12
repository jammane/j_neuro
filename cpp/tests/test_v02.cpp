// Spec 0.2.0 features: finite clamping, warm-up, node cap, rewiring, inversion on insert,
// diversity injection, run-log diversity entries.
#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace th;
using N16 = Net<std::uint16_t>;
static const double INF = std::numeric_limits<double>::infinity();
static const double MAXD = std::numeric_limits<double>::max();
static const char* PHASES[] = {"op_change_max_pct", "edit_max_pct", "growth_max_pct",
                               "removal_max_pct", "depth_shift_max_pct"};

static Profile zero_profile() {
    Profile p = Profile::defaults();
    for (auto f : PHASES) p.set_overall(f, 0.0);
    return p;
}

TEST_CASE("v02: outputs always finite") {
    auto net = make_net<std::uint16_t>(1, 2, 3, {{D16, V(0, VOp::Div, 0.0)}, {D16, V(0, VOp::Mul, 1e308)}});
    CHECK(net.step({5.0}) == std::vector<double>{MAXD, MAXD});
    CHECK(net.step({INF}) == std::vector<double>{MAXD, MAXD});
    CHECK(net.step({-INF}) == std::vector<double>{-MAXD, -MAXD});
}

TEST_CASE("v02: warmup steps") {
    auto net = make_net<std::uint16_t>(1, 1, 5, {{D16, M(AggOp::Mean, {2, 3, none, none, none})},
                                                 {30, S(4, AggOp::Mean, 1)}, {10, S(0, AggOp::Mean, 2)},
                                                 {20, S(3, AggOp::Max, 3)}});
    CHECK(net.get_warmup_steps() == 6);
    CHECK(make_net<std::uint16_t>(1, 1, 3, {{D16, V(0)}}).get_warmup_steps() == 0);
    CHECK(make_net<std::uint16_t>(1, 1, 3, {{D16, S(0, AggOp::Mean, 3)}}).get_warmup_steps() == 3);
}

TEST_CASE("v02: create respects node cap") {
    Rng r(0);
    CHECK_THROWS_AS(N16::create(70000, 1, 3, Priority::Speed, r), ValidationError);
}

TEST_CASE("v02: growth stops at node cap") {
    N16 net;
    net.priority = Priority::Speed;
    net.in_qty_max = 3;
    net.n_inputs = 1;
    net.n_outputs = 1;
    net.nodes.push_back(Node<std::uint16_t>{0, 0, 0, std::nullopt});
    net.nodes.push_back(Node<std::uint16_t>{1, 0, D16, VNeuron{NodeId(D16 - 1), Inversion::None, VOp::Mul, 1.0}});
    for (std::uint32_t i = 2; i < D16; ++i)
        net.nodes.push_back(Node<std::uint16_t>{i, 0, static_cast<std::uint16_t>(i - 1),
                                                VNeuron{NodeId(i == 2 ? 0 : i - 1), Inversion::None, VOp::Mul, 1.0}});
    net.next_birth = net.nodes.size();
    net.validate();
    auto p = zero_profile();
    p.set_overall("growth_max_pct", 0.05);
    Rng r(3);
    CHECK(mutate(net, p, 1, r).nodes.size() == D16);
    net.nodes.push_back(Node<std::uint16_t>{D16, 0, 5, VNeuron{0, Inversion::None, VOp::Mul, 1.0}});
    net.next_birth = net.nodes.size();
    CHECK_THROWS_AS(net.validate(), ValidationError);
}

static N16 rewire_net() {
    return make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {3, 4, 5})}, {10, V(0)}, {10, V(0)}, {20, V(1)}});
}

static Profile rewire_profile() {
    auto p = zero_profile();
    p.set_overall("edit_max_pct", 100.0);
    p.set_overall("rewire_prob", 1.0);
    p.apply_patch(json{{"m", {{"edit_max_pct", 0.0}}}});
    return p;
}

static std::optional<NodeId> input_of(const N16& n, NodeId i) { return std::get<VNeuron>(*n.nodes[i].neuron).input; }

TEST_CASE("v02: rewire keeps every node and the path rule") {
    auto net = rewire_net();
    auto p = rewire_profile();
    bool rewired = false;
    for (std::uint64_t seed = 0; seed < 60; ++seed) {
        Rng r(seed);
        auto child = mutate(net, p, 1, r);
        child.validate();
        CHECK(path_rule_ok(child));
        CHECK(child.nodes.size() == net.nodes.size());
        rewired = rewired || input_of(child, 3) != std::optional<NodeId>(0) ||
                  input_of(child, 4) != std::optional<NodeId>(0) || input_of(child, 5) != std::optional<NodeId>(1);
    }
    CHECK(rewired);
}

TEST_CASE("v02: rewire never orphans") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {3, 4, none})}, {10, V(0)}, {20, V(1)}});
    auto p = rewire_profile();
    for (std::uint64_t seed = 0; seed < 30; ++seed) {
        Rng r(seed);
        CHECK(to_bytes(mutate(net, p, 1, r)) == to_bytes(net));
    }
}

TEST_CASE("v02: rewire prob zero never rewires") {
    auto net = rewire_net();
    auto p = rewire_profile();
    p.set_overall("rewire_prob", 0.0);
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        Rng r(seed);
        auto child = mutate(net, p, 1, r);
        CHECK(input_of(child, 3) == std::optional<NodeId>(0));
        CHECK(input_of(child, 4) == std::optional<NodeId>(0));
        CHECK(input_of(child, 5) == std::optional<NodeId>(1));
    }
}

TEST_CASE("v02: insert inversion") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, 2.0)}});
    auto p = zero_profile();
    p.set_overall("growth_max_pct", 100.0);
    p.set_overall("insert_inversion_prob", 1.0);
    p.apply_patch(json{{"overall", {{"add_method_weights", {{"slot_fill", 0}, {"v_insert", 1}, {"s_insert", 0}, {"m_after_m", 0},
        {"v_tap", 0}, {"s_tap", 0}, {"m_tap", 0}, {"m_clone", 0},
        {"m_insert", 0}, {"v_link", 0}, {"s_link", 0}}}}}});
    for (std::uint64_t seed = 0; seed < 40; ++seed) {
        Rng r(seed);
        auto child = mutate(net, p, 1, r);
        if (child.nodes.size() == 3) {
            auto& n = std::get<VNeuron>(*child.nodes[1].neuron);
            CHECK((n.inversion == Inversion::Bitnot || n.inversion == Inversion::Recip));
            CHECK(n.weight == identity_weight(n.op));
            return;
        }
    }
    FAIL("no insertion happened");
}

TEST_CASE("v02: default profile has new fields") {
    auto p = Profile::defaults();
    CHECK(p.overall("rewire_prob") == 0.1);
    CHECK(p.overall("insert_inversion_prob") == 0.05);
}

static N16 diversity_net() {
    return make_net<std::uint16_t>(1, 1, 5, {{D16, M(AggOp::Mean, {2, 3, 4, 5, none})},
                                             {10, V(0, VOp::Mul, 1.0)}, {10, V(0, VOp::Add, 0.0)},
                                             {10, V(0, VOp::Div, 1.0, Inversion::Bitnot)},
                                             {10, S(0, AggOp::Mean, 1)}});
}

TEST_CASE("v02: diversity changes only weights") {
    auto net = diversity_net();
    auto before = to_bytes(net);
    Rng r(4);
    auto child = inject_diversity(net, 100.0, r);
    CHECK(to_bytes(net) == before);
    for (std::size_t i = 0; i < net.nodes.size(); ++i) {
        if (!net.nodes[i].neuron || !std::holds_alternative<VNeuron>(*net.nodes[i].neuron)) continue;
        auto& a = std::get<VNeuron>(*net.nodes[i].neuron);
        auto& b = std::get<VNeuron>(*child.nodes[i].neuron);
        CHECK(a.input == b.input);
        CHECK(a.inversion == b.inversion);
        CHECK(a.op == b.op);
        CHECK(a.weight != b.weight);
    }
}

TEST_CASE("v02: diversity counts") {
    auto net = diversity_net();
    Rng r0(1);
    CHECK(to_bytes(inject_diversity(net, 0.0, r0)) == to_bytes(net));
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        Rng r(seed);
        auto child = inject_diversity(net, 50.0, r);
        int changed = 0;
        for (std::size_t i = 0; i < net.nodes.size(); ++i)
            if (net.nodes[i].neuron && std::holds_alternative<VNeuron>(*net.nodes[i].neuron) &&
                std::get<VNeuron>(*net.nodes[i].neuron).weight != std::get<VNeuron>(*child.nodes[i].neuron).weight)
                ++changed;
        CHECK((changed == 1 || changed == 2));
    }
    Rng a(9), b(9);
    CHECK(to_bytes(inject_diversity(net, 70.0, a, 3.0)) == to_bytes(inject_diversity(net, 70.0, b, 3.0)));
}

TEST_CASE("v02: diversity golden") {
    auto cases = load_golden("diversity");
    REQUIRE(cases.size() >= 4);
    for (auto& [name, c] : cases) {
        INFO(name);
        AnyNet any = from_bytes(hex_to_bytes(c["jnet"].get<std::string>()));
        Bytes expected = hex_to_bytes(c["expected_jnet"].get<std::string>());
        std::visit([&](const auto& net) {
            Rng r(c["seed"].get<std::uint64_t>());
            CHECK(to_bytes(inject_diversity(net, c["pct"].get<double>(), r, c["sigma_reset"].get<double>())) == expected);
        }, any);
    }
}
