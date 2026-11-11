#include <jneuro/numerics.hpp>

#include <bit>
#include <cmath>

namespace jneuro {

std::uint64_t to_bits(double x) { return std::bit_cast<std::uint64_t>(x); }
double from_bits(std::uint64_t b) { return std::bit_cast<double>(b); }

double canon(double x) {
    if (x != x || x == 0.0) return 0.0;
    if (x == std::numeric_limits<double>::infinity()) return std::numeric_limits<double>::max();
    if (x == -std::numeric_limits<double>::infinity()) return -std::numeric_limits<double>::max();
    return x;
}

double bitnot(double x) { return from_bits(~to_bits(x)); }

double ieee_div(double a, double b) { return a / b; }

double compensated_mean(const std::vector<double>& values) {
    double s = 0.0;
    double c = 0.0;
    for (double v : values) {
        double t = s + v;
        if (std::fabs(s) >= std::fabs(v))
            c = c + ((s - t) + v);
        else
            c = c + ((v - t) + s);
        s = t;
    }
    double total = std::isfinite(s) ? s + c : s;
    return total / static_cast<double>(values.size());
}

}  // namespace jneuro
