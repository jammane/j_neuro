// Shared PRNG and distributions (spec/08). Bit-identical to the Python implementation.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace jneuro {

class Rng {
public:
    explicit Rng(std::uint64_t seed = 0);

    std::uint64_t next_u64();
    double u();                                   ///< uniform in [0, 1)
    std::uint64_t below(std::uint64_t n);         ///< uniform in [0, n), n >= 1
    std::uint64_t choice(std::uint64_t k) { return below(k); }
    std::size_t weighted(const std::vector<double>& weights);
    double normal();                              ///< Irwin-Hall(12) - 6
    std::int64_t prob_round(double x);

    std::array<std::uint64_t, 4> get_state() const { return s_; }
    void set_state(const std::array<std::uint64_t, 4>& s) { s_ = s; }

private:
    std::array<std::uint64_t, 4> s_{};
};

std::int64_t round_half_away(double x);

}  // namespace jneuro
