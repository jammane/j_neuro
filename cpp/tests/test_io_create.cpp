#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace th;

TEST_CASE("invalid golden") {
    auto cases = load_golden("invalid");
    REQUIRE(cases.size() >= 19);
    for (auto& [name, c] : cases) {
        INFO(name);
        CHECK_THROWS_AS(genome_from_json(c["genome"].dump()), ValidationError);
    }
}

TEST_CASE("format_double matches Python repr") {
    std::vector<std::pair<double, std::string>> v = {
        {1.0, "1.0"}, {0.1, "0.1"}, {-0.0, "-0.0"}, {0.0, "0.0"}, {1e16, "1e+16"},
        {1.5e-05, "1.5e-05"}, {0.0001, "0.0001"}, {123456789.123, "123456789.123"},
        {1e22, "1e+22"}, {5e-324, "5e-324"}, {-2.5, "-2.5"},
        {9999999999999998.0, "9999999999999998.0"},
        {1.7976931348623157e308, "1.7976931348623157e+308"}, {100.0, "100.0"}, {3e-05, "3e-05"}};
    for (auto& [x, s] : v) CHECK(format_double(x) == s);
}

TEST_CASE("canonical text") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Add, 0.5)}});
    CHECK(genome_to_json(net) ==
          "{\"format\":\"j_neuro.genome\",\"spec_version\":\"0.3.0\",\"params\":{\"priority\":\"SPEED\","
          "\"depth_type\":\"U16\",\"in_qty_max\":3,\"n_inputs\":1,\"n_outputs\":1},\"next_birth\":2,"
          "\"nodes\":[{\"birth\":0,\"born_gen\":0,\"depth\":0},{\"birth\":1,\"born_gen\":0,\"depth\":65535,"
          "\"neuron\":{\"type\":\"v\",\"input\":0,\"inversion\":\"none\",\"op\":\"add\",\"weight\":0.5}}]}");
}

TEST_CASE("roundtrip all types") {
    auto net = make_net<std::uint16_t>(2, 1, 5, {{D16, M(AggOp::Median, {3, 4, 1, none, none})},
                                                 {10, V(0, VOp::Div, -1e-07, Inversion::Bitnot)},
                                                 {20, S(0, AggOp::Max, 4)}});
    auto text = genome_to_json(net);
    CHECK(std::visit([](const auto& n) { return genome_to_json(n); }, genome_from_json(text)) == text);
}

TEST_CASE("u64 depth roundtrip") {
    auto net = make_net<std::uint64_t>(1, 1, 3, {{~std::uint64_t{0}, V(2, VOp::Add, 1.0)},
                                                 {(std::uint64_t{1} << 53) + 1, V(0, VOp::Mul, 3.0)}});
    auto any = genome_from_json(genome_to_json(net));
    auto& n2 = std::get<Net<std::uint64_t>>(any);
    CHECK(n2.nodes[2].depth == (std::uint64_t{1} << 53) + 1);
}

template <class D>
static void check_create(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t q) {
    Rng r(11);
    auto net = Net<D>::create(n_in, n_out, q, Priority::Memory, r);
    net.validate();
    CHECK(path_rule_ok(net));
    CHECK(net.next_birth == net.nodes.size());
    for (std::size_t i = 0; i < net.nodes.size(); ++i) {
        CHECK(net.nodes[i].birth == i);
        CHECK(net.nodes[i].born_gen == 0);
    }
    if (n_in == 1) {
        CHECK(net.nodes.size() == n_in + n_out);
        for (std::uint32_t o = 0; o < n_out; ++o)
            CHECK(std::holds_alternative<VNeuron>(*net.nodes[n_in + o].neuron));
    } else if (n_in <= q) {
        CHECK(net.nodes.size() == n_in + n_out);
        auto& m = std::get<MNeuron>(*net.nodes[n_in].neuron);
        for (std::uint32_t i = 0; i < n_in; ++i) CHECK(m.slots[i] == std::optional<NodeId>(i));
    } else {
        CHECK(net.nodes.size() > n_in + n_out);
    }
}

TEST_CASE("create minimal") {
    std::vector<std::array<std::uint32_t, 3>> cases = {{1, 3, 3}, {3, 2, 3}, {2, 1, 5}, {7, 2, 3}, {30, 2, 3}, {5, 1, 3}};
    for (auto& c : cases) {
        check_create<std::uint16_t>(c[0], c[1], c[2]);
        check_create<std::uint32_t>(c[0], c[1], c[2]);
        check_create<std::uint64_t>(c[0], c[1], c[2]);
    }
}

TEST_CASE("create tree shape") {
    Rng r(1);
    auto net = Net<std::uint16_t>::create(7, 2, 3, Priority::Speed, r);
    REQUIRE(net.nodes.size() == 12);
    CHECK(std::get<MNeuron>(*net.nodes[9].neuron).slots == Slots{0, 1, 2});
    CHECK(std::get<MNeuron>(*net.nodes[10].neuron).slots == Slots{3, 4, 5});
    CHECK(std::get<MNeuron>(*net.nodes[11].neuron).slots == Slots{6, none, none});
    for (int i = 9; i < 12; ++i) CHECK(net.nodes[i].depth == D16 / 2);
    CHECK(std::get<MNeuron>(*net.nodes[7].neuron).slots == Slots{9, 10, 11});
}

TEST_CASE("create from counts only") {
    AnyNet any = create(4, 2);
    auto& net = std::get<Net<std::uint32_t>>(any);
    net.validate();
    CHECK(net.in_qty_max == 5);
    CHECK(net.priority == Priority::Balanced);
    Rng r(0);
    CHECK(genome_to_json(net) == genome_to_json(Net<std::uint32_t>::create(4, 2, 5, Priority::Balanced, r)));
    CHECK(net.step({1.0, 2.0, 3.0, 4.0}).size() == 2);
}

TEST_CASE("create deterministic") {
    Rng a(5), b(5);
    CHECK(genome_to_json(Net<std::uint32_t>::create(9, 3, 3, Priority::Speed, a)) ==
          genome_to_json(Net<std::uint32_t>::create(9, 3, 3, Priority::Speed, b)));
}

TEST_CASE("create rejects bad params") {
    Rng r(1);
    CHECK_THROWS_AS(Net<std::uint16_t>::create(2, 1, 4, Priority::Speed, r), ValidationError);
    CHECK_THROWS_AS(Net<std::uint16_t>::create(0, 1, 3, Priority::Speed, r), ValidationError);
}
