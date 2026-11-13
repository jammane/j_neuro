#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace th;
static const double INF = std::numeric_limits<double>::infinity();

TEST_CASE("eval golden") {
    auto cases = load_golden("handmade");
    REQUIRE(cases.size() >= 7);
    for (auto& [name, c] : cases) {
        for (auto& prio : c["priorities"]) {
            INFO(name << " " << prio.get<std::string>());
            AnyNet any = from_bytes(hex_to_bytes(c["jnet"].get<std::string>()));
            std::visit([&](auto& net) {
                net.priority = parse_priority(prio.get<std::string>());
                net.reset();
                for (auto& st : c["steps"]) {
                    std::vector<double> in;
                    for (auto& h : st["inputs"]) in.push_back(from_hex(h));
                    auto out = net.step(in);
                    REQUIRE(out.size() == st["outputs"].size());
                    for (std::size_t i = 0; i < out.size(); ++i)
                        CHECK(bits(out[i]) == st["outputs"][i].get<std::string>());
                }
            }, any);
        }
    }
}

TEST_CASE("depth order") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(2, VOp::Add, 1.0)}, {20, V(3, VOp::Mul, 2.0)},
                                                 {10, V(0, VOp::Mul, 3.0)}});
    CHECK(net.step({1.0}) == std::vector<double>{7.0});
}

TEST_CASE("reset clears history") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, S(0, AggOp::Mean, 1)}});
    CHECK(net.step({5.0})[0] == 5.0);
    CHECK(net.step({5.0})[0] == 0.0);
    net.reset();
    CHECK(net.step({5.0})[0] == 5.0);
}

TEST_CASE("missing input constants") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, M(AggOp::Mean, {0, 2, none})},
                                                 {5, V(none, VOp::Add, 3.0)}});
    CHECK(net.step({0.0})[0] == 1.0);
}

TEST_CASE("priority invariance and switching") {
    auto net = make_net<std::uint16_t>(2, 2, 3, {{D16, M(AggOp::Max, {4, 1, none})},
                                                 {D16, S(4, AggOp::Median, 3)},
                                                 {100, V(0, VOp::Div, 2.0, Inversion::Recip)}});
    std::vector<std::vector<double>> rows = {{1, 2}, {-3, 0.5}, {0, 0}, {7, -1}, {INF, 2}};
    std::vector<std::vector<std::vector<std::string>>> res;
    for (Priority p : {Priority::Speed, Priority::Memory, Priority::Balanced}) {
        net.priority = p;
        res.push_back(outputs_for(net, rows));
    }
    CHECK(res[0] == res[1]);
    CHECK(res[1] == res[2]);
}

TEST_CASE("outputs canonical") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, -1.0)}});
    CHECK(same(net.step({0.0})[0], 0.0));
}

TEST_CASE("reset_state matches reset, and the cached warmup matches the computed one") {
    using N32 = Net<std::uint32_t>;
    Profile p = Profile::defaults();
    p.set_overall("growth_max_pct", 20.0);
    p.set_overall("growth_gen_prob", 1.0);
    p.set_overall("growth_net_prob", 1.0);
    for (Priority pr : {Priority::Speed, Priority::Memory, Priority::Balanced}) {
        Rng rng(77);
        N32 net = std::get<N32>(create(6, 3, DepthType::U32, 5, pr, rng));
        for (std::uint64_t g = 1; g <= 60; ++g) net = mutate(net, p, g, rng);
        N32 a = net, b = net;
        const std::uint64_t warm = net.get_warmup_steps();
        for (int episode = 0; episode < 5; ++episode) {
            a.reset();
            b.reset_state();
            for (int t = 0; t < 12; ++t) {
                std::vector<double> in(6);
                for (auto& x : in) x = rng.normal() * 3.0;
                const auto ya = a.step(in), yb = b.step(in);
                REQUIRE(ya.size() == yb.size());
                for (std::size_t o = 0; o < ya.size(); ++o) CHECK(same(ya[o], yb[o]));
            }
            CHECK(b.get_warmup_steps() == warm);  // from the plan, after steps
        }
    }
}
