#include <jneuro/types.hpp>

#include <array>

namespace jneuro {

namespace {
template <class E, std::size_t N>
E parse(const std::array<const char*, N>& names, const std::string& s, const char* what) {
    for (std::size_t i = 0; i < N; ++i)
        if (s == names[i]) return static_cast<E>(i);
    throw ValidationError(std::string("unknown ") + what + ": " + s);
}
constexpr std::array<const char*, 3> kPriority{"SPEED", "MEMORY", "BALANCED"};
constexpr std::array<const char*, 3> kDepth{"U16", "U32", "U64"};
constexpr std::array<const char*, 3> kInv{"none", "bitnot", "recip"};
constexpr std::array<const char*, 4> kVOp{"mul", "div", "add", "sub"};
constexpr std::array<const char*, 4> kAgg{"mean", "median", "min", "max"};
constexpr std::array<const char*, 3> kType{"v", "s", "m"};
}  // namespace

std::string to_string(Priority v) { return kPriority[static_cast<std::size_t>(v)]; }
std::string to_string(DepthType v) { return kDepth[static_cast<std::size_t>(v)]; }
std::string to_string(Inversion v) { return kInv[static_cast<std::size_t>(v)]; }
std::string to_string(VOp v) { return kVOp[static_cast<std::size_t>(v)]; }
std::string to_string(AggOp v) { return kAgg[static_cast<std::size_t>(v)]; }
std::string to_string(NeuronType v) { return kType[static_cast<std::size_t>(v)]; }

Priority parse_priority(const std::string& s) { return parse<Priority>(kPriority, s, "priority"); }
DepthType parse_depth_type(const std::string& s) { return parse<DepthType>(kDepth, s, "depth_type"); }
Inversion parse_inversion(const std::string& s) { return parse<Inversion>(kInv, s, "inversion"); }
VOp parse_vop(const std::string& s) { return parse<VOp>(kVOp, s, "v op"); }
AggOp parse_aggop(const std::string& s) { return parse<AggOp>(kAgg, s, "aggregation op"); }

std::uint64_t d_max(DepthType t) {
    switch (t) {
        case DepthType::U16: return 0xFFFFull;
        case DepthType::U32: return 0xFFFFFFFFull;
        case DepthType::U64: return ~std::uint64_t{0};
    }
    return 0;
}

double identity_weight(VOp op) { return (op == VOp::Mul || op == VOp::Div) ? 1.0 : 0.0; }

}  // namespace jneuro
