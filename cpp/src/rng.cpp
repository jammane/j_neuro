#include <jneuro/rng.hpp>

#include <cmath>
#include <stdexcept>

namespace jneuro {

namespace {
inline std::uint64_t rotl(std::uint64_t v, int r) { return (v << r) | (v >> (64 - r)); }
inline std::uint64_t splitmix64(std::uint64_t& x) {
    x += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
}  // namespace

Rng::Rng(std::uint64_t seed) {
    std::uint64_t x = seed;
    for (auto& v : s_) v = splitmix64(x);
}

std::uint64_t Rng::next_u64() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
}

double Rng::u() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

std::uint64_t Rng::below(std::uint64_t n) {
    if (n == 0) throw std::invalid_argument("below(0)");
    const std::uint64_t threshold = (0 - n) % n;
    for (;;) {
        std::uint64_t r = next_u64();
        if (r >= threshold) return r % n;
    }
}

std::size_t Rng::weighted(const std::vector<double>& w) {
    double total = 0.0;
    for (double x : w) total = total + x;
    const double x = u() * total;
    double acc = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        acc = acc + w[i];
        if (x < acc) return i;
    }
    for (std::size_t i = w.size(); i-- > 0;)
        if (w[i] > 0) return i;
    throw std::invalid_argument("weighted() needs a positive total weight");
}

double Rng::normal() {
    double s = 0.0;
    for (int i = 0; i < 12; ++i) s = s + u();
    return s - 6.0;
}

std::int64_t Rng::prob_round(double x) {
    const double f = std::floor(x);
    return static_cast<std::int64_t>(f) + (u() < (x - f) ? 1 : 0);
}

std::int64_t round_half_away(double x) {
    const auto r = static_cast<std::int64_t>(std::floor(std::fabs(x) + 0.5));
    return x >= 0 ? r : -r;
}

}  // namespace jneuro
