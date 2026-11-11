#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

using namespace th;
static const double INF = std::numeric_limits<double>::infinity();
static const double NaN = std::numeric_limits<double>::quiet_NaN();

TEST_CASE("canon") {
    CHECK(same(canon(NaN), 0.0));
    CHECK(same(canon(-0.0), 0.0));
    CHECK(canon(INF) == std::numeric_limits<double>::max());
    CHECK(canon(-INF) == -std::numeric_limits<double>::max());
    CHECK(same(canon(-2.5), -2.5));
}

TEST_CASE("bits roundtrip") {
    CHECK(to_bits(1.0) == 0x3FF0000000000000ull);
    CHECK(same(from_bits(0xC00FFFFFFFFFFFFFull), -3.9999999999999996));
}

TEST_CASE("bitnot") {
    CHECK(same(bitnot(1.0), -3.9999999999999996));
    CHECK(same(bitnot(2.0), -1.9999999999999998));
    CHECK(std::isnan(bitnot(0.0)));
    CHECK(same(bitnot(-INF), from_bits(0x000FFFFFFFFFFFFFull)));
    for (double x : {1.0, -3.5, 1e-300, 7e300}) CHECK(same(bitnot(bitnot(x)), x));
}

TEST_CASE("ieee_div") {
    CHECK(ieee_div(1.0, 0.0) == INF);
    CHECK(ieee_div(-1.0, 0.0) == -INF);
    CHECK(ieee_div(1.0, -0.0) == -INF);
    CHECK(std::isnan(ieee_div(0.0, 0.0)));
    CHECK(std::isnan(ieee_div(NaN, 2.0)));
    CHECK(ieee_div(6.0, 3.0) == 2.0);
    CHECK(same(ieee_div(1.0, -INF), -0.0));
}

TEST_CASE("compensated_mean") {
    CHECK(compensated_mean({1e16, 1.0, -1e16}) == 1.0 / 3.0);
    CHECK(compensated_mean({1e308, 1e308, -1e308}) == INF);
    CHECK(std::isnan(compensated_mean({INF, -INF, 0.0})));
    CHECK(compensated_mean({0.0, 0.0, 1.0}) == 1.0 / 3.0);
    CHECK(compensated_mean({2.0}) == 2.0);
    CHECK(same(compensated_mean(std::vector<double>(10, 0.1)), 0.1));
}
