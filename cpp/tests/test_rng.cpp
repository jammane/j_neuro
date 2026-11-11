#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace th;

TEST_CASE("rng golden") {
    auto cases = load_golden("rng");
    REQUIRE(cases.size() == 3);
    for (auto& [name, c] : cases) {
        INFO(name);
        const std::uint64_t seed = c["seed"].get<std::uint64_t>();
        Rng r(seed);
        for (auto& h : c["next_u64"]) {
            char buf[17];
            std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(r.next_u64()));
            CHECK(std::string(buf) == h.get<std::string>());
        }
        r = Rng(seed);
        for (auto& h : c["u"]) CHECK(same(r.u(), from_hex(h)));
        for (auto& b : c["below"]) {
            r = Rng(seed);
            for (auto& v : b["values"]) CHECK(r.below(b["n"].get<std::uint64_t>()) == v.get<std::uint64_t>());
        }
        r = Rng(seed);
        for (auto& h : c["normal"]) CHECK(same(r.normal(), from_hex(h)));
        for (auto& p : c["prob_round"]) {
            r = Rng(seed);
            double x = from_hex(p["x"]);
            for (auto& v : p["values"]) CHECK(r.prob_round(x) == v.get<std::int64_t>());
        }
        for (auto& w : c["weighted"]) {
            r = Rng(seed);
            std::vector<double> ws;
            for (auto& h : w["w"]) ws.push_back(from_hex(h));
            for (auto& v : w["values"]) CHECK(r.weighted(ws) == v.get<std::size_t>());
        }
    }
}

TEST_CASE("rng state save/restore") {
    Rng r(7);
    r.next_u64();
    auto st = r.get_state();
    std::vector<std::uint64_t> a;
    for (int i = 0; i < 5; ++i) a.push_back(r.next_u64());
    r.set_state(st);
    for (int i = 0; i < 5; ++i) CHECK(r.next_u64() == a[i]);
}

TEST_CASE("rng ranges") {
    Rng r(1);
    for (int i = 0; i < 2000; ++i) {
        double x = r.u();
        CHECK((x >= 0.0 && x < 1.0));
        CHECK(r.below(7) < 7);
        CHECK(r.choice(4) < 4);
        double z = r.normal();
        CHECK((z >= -6.0 && z <= 6.0));
    }
}

TEST_CASE("weighted skips zero weights") {
    Rng r(3);
    for (int i = 0; i < 500; ++i) {
        auto k = r.weighted({0.0, 1.0, 0.0, 2.0});
        CHECK((k == 1 || k == 3));
    }
}

TEST_CASE("prob_round always draws") {
    Rng a(5), b(5);
    a.prob_round(3.0);
    b.u();
    CHECK(a.next_u64() == b.next_u64());
}

TEST_CASE("round_half_away") {
    CHECK(round_half_away(0.5) == 1);
    CHECK(round_half_away(-0.5) == -1);
    CHECK(round_half_away(1.49) == 1);
    CHECK(round_half_away(-2.5) == -3);
    CHECK(round_half_away(0.0) == 0);
}
