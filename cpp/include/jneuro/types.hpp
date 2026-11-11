// Enumerations and basic types (spec/01, spec/03, spec/04).
// Enumerator order is normative: it defines what an RNG choice index means.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace jneuro {

enum class Priority : std::uint8_t { Speed, Memory, Balanced };
enum class DepthType : std::uint8_t { U16, U32, U64 };
enum class Inversion : std::uint8_t { None, Bitnot, Recip };
enum class VOp : std::uint8_t { Mul, Div, Add, Sub };
enum class AggOp : std::uint8_t { Mean, Median, Min, Max };
enum class NeuronType : std::uint8_t { V, S, M };

using NodeId = std::uint32_t;

/// A genome or parameter set violates the spec.
struct ValidationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::string to_string(Priority);
std::string to_string(DepthType);
std::string to_string(Inversion);
std::string to_string(VOp);
std::string to_string(AggOp);
std::string to_string(NeuronType);

Priority parse_priority(const std::string&);
DepthType parse_depth_type(const std::string&);
Inversion parse_inversion(const std::string&);
VOp parse_vop(const std::string&);
AggOp parse_aggop(const std::string&);

std::uint64_t d_max(DepthType);
double identity_weight(VOp);

template <class D>
constexpr DepthType depth_type_of();
template <> constexpr DepthType depth_type_of<std::uint16_t>() { return DepthType::U16; }
template <> constexpr DepthType depth_type_of<std::uint32_t>() { return DepthType::U32; }
template <> constexpr DepthType depth_type_of<std::uint64_t>() { return DepthType::U64; }

}  // namespace jneuro
