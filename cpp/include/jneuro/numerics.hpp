// Exact binary64 helpers (spec/02).
#pragma once

#include <cfloat>
#include <cstdint>
#include <limits>
#include <vector>

static_assert(std::numeric_limits<double>::is_iec559, "j_neuro requires IEEE-754 binary64");
static_assert(FLT_EVAL_METHOD == 0, "j_neuro requires FLT_EVAL_METHOD == 0 (no x87 excess precision)");
#if defined(__FAST_MATH__)
#error "j_neuro must not be compiled with -ffast-math (spec/02)"
#endif

namespace jneuro {

std::uint64_t to_bits(double x);
double from_bits(std::uint64_t b);

/// NaN and -0.0 become +0.0, +-inf become +-max finite; finite values are unchanged.
double canon(double x);
/// Flip all 64 bits of the binary64 pattern.
double bitnot(double x);
/// IEEE-754 division (x/0 = +-inf or NaN).
double ieee_div(double a, double b);
/// Neumaier-compensated mean in the given order (spec/04).
double compensated_mean(const std::vector<double>& values);

}  // namespace jneuro
