// SHA-256 (FIPS 180-4), implemented here so no hashing dependency is needed.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace jneuro {

using Bytes = std::vector<std::uint8_t>;
using Digest = std::array<std::uint8_t, 32>;

Digest sha256(const std::uint8_t* data, std::size_t len);
inline Digest sha256(const Bytes& b) { return sha256(b.data(), b.size()); }

std::string bytes_to_hex(const std::uint8_t* data, std::size_t len);
inline std::string bytes_to_hex(const Bytes& b) { return bytes_to_hex(b.data(), b.size()); }
inline std::string bytes_to_hex(const Digest& d) { return bytes_to_hex(d.data(), d.size()); }
/// Throws std::invalid_argument on odd length or non-hex characters.
Bytes hex_to_bytes(const std::string& hex);

}  // namespace jneuro
