// spec/12: adding and removing inputs and outputs.
#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace th;

namespace {

Net<std::uint32_t> grown(std::uint64_t seed, std::uint32_t n_in, std::uint32_t n_out, std::uint32_t q,
                         std::uint32_t mix, std::uint32_t run) {
    Profile grow = Profile::defaults();
    for (const char* f : {"op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
                          "depth_shift_max_pct"})
        grow.set_overall(f, 35.0);
    Rng r(seed);
    auto net = Net<std::uint32_t>::create(n_in, n_out, q, Priority::Speed, r, mix, run);
    for (std::uint64_t g = 1; g <= 14; ++g) net = mutate(net, grow, g, r);
    return net;
}

std::vector<Net<std::uint32_t>> nets() {
    std::vector<Net<std::uint32_t>> out;
    for (std::uint32_t s = 0; s < 10; ++s) out.push_back(grown(s, 1 + s % 5, 1 + s % 4, 3 + 2 * (s % 3), s % 3, s % 4));
    return out;
}

template <class Row>
Row drop(const Row& r, std::size_t i) {
    Row o = r;
    o.erase(o.begin() + static_cast<std::ptrdiff_t>(i));
    return o;
}

}  // namespace

TEST_CASE("add_input neutral keeps outputs") {
    for (auto& net : nets()) {
        auto rows = sample_inputs(net.n_inputs);
        const auto before = outputs_for(net, rows);
        for (std::uint32_t pos = 0; pos <= net.n_inputs; ++pos) {
            auto child = add_input(net, pos, false, 7);
            REQUIRE(child.n_inputs == net.n_inputs + 1);
            CHECK(child.nodes[pos].birth == net.next_birth);
            auto rows2 = rows;
            for (std::size_t t = 0; t < rows2.size(); ++t)
                rows2[t].insert(rows2[t].begin() + pos, 3.5 * static_cast<double>(t + 1) - 9.0);
            CHECK(outputs_for(child, rows2) == before);
        }
        auto active = add_input(net, std::nullopt, true);
        const auto& w = std::get<VNeuron>(*active.nodes.back().neuron);
        CHECK(w.weight == 1.0);
    }
}

TEST_CASE("add_input uses medians when receivers are all min") {
    Rng r(3);
    auto net = Net<std::uint16_t>::create(1, 3, 3, Priority::Speed, r);
    auto child = add_input(net);
    for (std::uint32_t o = 0; o < 3; ++o)
        CHECK(std::get<MNeuron>(*child.nodes[child.n_inputs + o].neuron).op == AggOp::Median);
    auto rows = sample_inputs(1);
    auto rows2 = rows;
    for (auto& row : rows2) row.push_back(7.0);
    CHECK(outputs_for(child, rows2) == outputs_for(net, rows));
}

TEST_CASE("add_output clone and fresh keep outputs") {
    for (auto& net : nets()) {
        auto rows = sample_inputs(net.n_inputs);
        const auto before = outputs_for(net, rows);
        auto clone = add_output_clone(net, net.n_outputs - 1, 0u, 4);
        auto out = outputs_for(clone, rows);
        for (std::size_t t = 0; t < out.size(); ++t) {
            CHECK(drop(out[t], 0) == before[t]);
            CHECK(out[t].front() == out[t].back());
        }
        Rng r(5);
        auto fresh = add_output_fresh(net, r);
        auto out2 = outputs_for(fresh, rows);
        for (std::size_t t = 0; t < out2.size(); ++t) CHECK(drop(out2[t], out2[t].size() - 1) == before[t]);
    }
}

TEST_CASE("remove_output keeps the others") {
    for (auto& net : nets()) {
        if (net.n_outputs < 2) continue;
        auto rows = sample_inputs(net.n_inputs);
        const auto before = outputs_for(net, rows);
        for (std::uint32_t i = 0; i < net.n_outputs; ++i) {
            auto child = remove_output(net, i);
            auto out = outputs_for(child, rows);
            for (std::size_t t = 0; t < out.size(); ++t) CHECK(out[t] == drop(before[t], i));
        }
    }
}

TEST_CASE("remove_input cascades both ways and keeps shared neurons") {
    auto n = make_net<std::uint16_t>(2, 1, 3, {
        {D16, M(AggOp::Max, {6, 3, none})},       // 2: output
        {200, M(AggOp::Mean, {4, 5, none})},       // 3: only input 1 reaches it
        {100, V(1)},                               // 4
        {50, V(none, VOp::Add, 2.0)},              // 5: bias, feeds only 3
        {100, V(0)},                               // 6
    });
    n.validate();
    auto child = remove_input(n, 1);
    CHECK(child.n_inputs == 1);
    CHECK(child.nodes.size() == 3);
    CHECK(std::get<MNeuron>(*child.nodes[1].neuron).slots == Slots{2, none, none});

    auto shared = make_net<std::uint16_t>(2, 1, 3, {{D16, V(3)}, {100, M(AggOp::Mean, {0, 1, none})}});
    auto c2 = remove_input(shared, 0);
    CHECK(c2.nodes.size() == 3);
    CHECK(std::get<MNeuron>(*c2.nodes[2].neuron).slots == Slots{none, 0, none});
}

TEST_CASE("modify errors") {
    Rng r(0);
    auto one = Net<std::uint32_t>::create(1, 1, 5, Priority::Balanced, r);
    CHECK_THROWS_AS(remove_input(one, 0), ValidationError);
    CHECK_THROWS_AS(remove_output(one, 0), ValidationError);
    CHECK_THROWS_AS(add_input(one, 2u), ValidationError);
    CHECK_THROWS_AS(add_output_clone(one, 1), ValidationError);
    CHECK_THROWS_AS(add_output_fresh(one, r, 5u), ValidationError);
}

TEST_CASE("modify golden") {
    const auto cases = load_golden("modify");
    REQUIRE(!cases.empty());
    for (auto& [name, c] : cases) {
        INFO(name);
        AnyNet any = from_bytes(hex_to_bytes(c["jnet"].get<std::string>()));
        const auto& a = c["args"];
        const std::string op = c["op"].get<std::string>();
        const Bytes expected = hex_to_bytes(c["expected_jnet"].get<std::string>());
        std::visit([&](auto& net) {
            Bytes got;
            if (op == "add_input")
                got = to_bytes(add_input(net, a["position"].get<std::uint32_t>(), a["active"].get<bool>(),
                                         a["generation"].get<std::uint64_t>()));
            else if (op == "add_output_clone")
                got = to_bytes(add_output_clone(net, a["source"].get<std::uint32_t>(),
                                                a["position"].get<std::uint32_t>(),
                                                a["generation"].get<std::uint64_t>()));
            else if (op == "add_output_fresh") {
                Rng r(a["seed"].get<std::uint64_t>());
                got = to_bytes(add_output_fresh(net, r, a["position"].get<std::uint32_t>(),
                                                a["generation"].get<std::uint64_t>()));
            } else if (op == "remove_input")
                got = to_bytes(remove_input(net, a["index"].get<std::uint32_t>()));
            else
                got = to_bytes(remove_output(net, a["index"].get<std::uint32_t>()));
            CHECK(got == expected);
        }, any);
    }
}
