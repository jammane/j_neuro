#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <limits>

using namespace th;
static const double INF = std::numeric_limits<double>::infinity();
static const bool T = true, F = false;

TEST_CASE("mean counts empties") {
    CHECK(aggregate(AggOp::Mean, {3.0, 5.0, 0.0, 0.0, 0.0}, {T, T, F, F, F}) == 8.0 / 5.0);
}

TEST_CASE("median counts empties") {
    CHECK(same(aggregate(AggOp::Median, {3.0, 5.0, 0.0, 0.0, 0.0}, {T, T, F, F, F}), 0.0));
    CHECK(aggregate(AggOp::Median, {9.0, -1.0, 4.0}, {T, T, T}) == 4.0);
}

TEST_CASE("min skips empties") {
    CHECK(aggregate(AggOp::Min, {3.0, 5.0, 0.0}, {T, T, F}) == 3.0);
    CHECK(aggregate(AggOp::Min, {3.0, 0.0, 5.0}, {T, T, T}) == 0.0);
    CHECK(same(aggregate(AggOp::Min, {0.0, 0.0, 0.0}, {F, F, F}), 0.0));
}

TEST_CASE("max counts empties") {
    CHECK(same(aggregate(AggOp::Max, {-3.0, -5.0, 0.0}, {T, T, F}), 0.0));
    CHECK(aggregate(AggOp::Max, {-3.0, -5.0, -1.0}, {T, T, T}) == -1.0);
}

TEST_CASE("aggregate infinities") {
    CHECK(aggregate(AggOp::Min, {INF, -INF, 0.0}, {T, T, F}) == -INF);
    CHECK(aggregate(AggOp::Max, {INF, -INF, 0.0}, {T, T, F}) == INF);
}
