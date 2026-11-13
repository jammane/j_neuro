// Shared aggregation for s_neurons and m_neurons (spec/04).
#pragma once

#include <jneuro/types.hpp>

#include <vector>

namespace jneuro {

/// values: canonical values in order; present: connected slot / filled history entry.
/// Non-present values are +0.0 and count for mean, median and max; min skips them.
double aggregate(AggOp op, const std::vector<double>& values, const std::vector<bool>& present);
/// The same, with a caller-owned buffer for median's sorted copy (no allocation once it is large enough).
double aggregate(AggOp op, const std::vector<double>& values, const std::vector<bool>& present,
                 std::vector<double>& scratch);

}  // namespace jneuro
