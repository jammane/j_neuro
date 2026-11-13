// Binary .jnet files (spec/09), version checks, and creation of any depth type.
// Profiles and run logs (JSON) live in profile.hpp and population.hpp.
#pragma once

#include <jneuro/net_build.hpp>
#include <jneuro/sha256.hpp>

#include <nlohmann/json.hpp>

#include <cstring>
#include <filesystem>
#include <string>
#include <variant>

namespace jneuro {

inline constexpr const char* kSpecVersion = "0.6.0";
inline constexpr std::uint16_t kSpecMajor = 0, kSpecMinor = 6, kSpecPatch = 0;
inline constexpr std::uint16_t kNetFormatVersion = 2;  ///< .jnet; 1 = spec 0.5 (read and converted)
inline constexpr std::uint16_t kPopFormatVersion = 1;  ///< .jpnet

/// Spec 0.5 genomes are still read: nets are converted on load (spec/09 reading 0.5 nets).
inline bool is_legacy_05(std::uint16_t major, std::uint16_t minor) { return major == 0 && minor == 5; }

/// Same MAJOR; while MAJOR is 0, also the same MINOR (spec/README).
bool version_compatible(const std::string& v);
bool version_compatible(std::uint16_t major, std::uint16_t minor);
/// Validate a JSON document's format and spec_version (throws ValidationError).
void check_format(const nlohmann::json& d, const char* fmt, bool accept_05 = false);
/// A new 64-bit seed from the operating system's entropy source.
std::uint64_t fresh_seed();

using AnyNet = std::variant<Net<std::uint16_t>, Net<std::uint32_t>, Net<std::uint64_t>>;

/// The canonical .jnet bytes of a net.
template <class D>
Bytes to_bytes(const Net<D>& net);
/// Parse and fully validate .jnet bytes. Throws ValidationError.
AnyNet from_bytes(const Bytes& data);
template <class D>
void save_net(const std::filesystem::path& path, const Net<D>& net);
AnyNet load_net(const std::filesystem::path& path);

Bytes read_file(const std::filesystem::path& path);
void write_file(const std::filesystem::path& path, const Bytes& data);

/// Minimal net from input/output counts (spec/03). Defaults match the Python implementation.
AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type, std::uint32_t in_qty_max,
              Priority priority, Rng& rng, std::uint32_t path_mix = 2, std::uint32_t max_run = 3);
AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type = DepthType::U32,
              std::uint32_t in_qty_max = 5, Priority priority = Priority::Balanced, std::uint32_t path_mix = 2,
              std::uint32_t max_run = 3);

namespace detail {

/// Little-endian field writer (spec/09 binary conventions).
class Writer {
public:
    void u(std::uint64_t v, std::size_t width) {
        for (std::size_t i = 0; i < width; ++i) buf_.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
    }
    void f64(double x) {
        std::uint64_t b;
        std::memcpy(&b, &x, 8);
        u(b, 8);
    }
    void raw(const std::uint8_t* p, std::size_t n) { buf_.insert(buf_.end(), p, p + n); }
    void raw(const std::string& s) { raw(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()); }
    void header(const char magic[4]) {
        raw(reinterpret_cast<const std::uint8_t*>(magic), 4);
        u(std::memcmp(magic, "JNET", 4) == 0 ? kNetFormatVersion : kPopFormatVersion, 2);
        u(kSpecMajor, 2);
        u(kSpecMinor, 2);
        u(kSpecPatch, 2);
    }
    /// Append the SHA-256 of everything written so far.
    Bytes finish() {
        const Digest d = sha256(buf_);
        Bytes out = buf_;
        out.insert(out.end(), d.begin(), d.end());
        return out;
    }

private:
    Bytes buf_;
};

/// Checksum-verified little-endian reader.
class Reader {
public:
    Reader(const Bytes& data, std::string what);
    std::uint64_t u(std::size_t width);
    double f64();
    const std::uint8_t* take(std::size_t n);
    std::uint8_t enum_byte(std::uint8_t count, const char* field);
    /// Checks magic, format and spec version; returns true for a spec 0.5 file (to convert).
    bool header(const char magic[4]);
    void done() const;

private:
    const Bytes& data_;
    std::size_t end_;
    std::size_t pos_ = 0;
    std::string what_;
};

}  // namespace detail

// ------------------------------------------------------------------ implementation

template <class D>
Bytes to_bytes(const Net<D>& net) {
    constexpr std::size_t W = sizeof(D);
    const std::uint64_t none = static_cast<std::uint64_t>(Net<D>::d_max);
    detail::Writer w;
    w.header("JNET");
    w.u(static_cast<std::uint8_t>(net.priority), 1);
    w.u(static_cast<std::uint8_t>(Net<D>::depth_type), 1);
    w.u(net.in_qty_max, 4);
    w.u(net.n_inputs, 4);
    w.u(net.n_outputs, 4);
    w.u(net.path_mix, 1);
    w.u(net.max_run, 1);
    w.u(net.next_birth, 8);
    w.u(net.nodes.size(), W);
    auto ref = [&](const std::optional<NodeId>& r, std::uint32_t pos) {  // a conn: id, position
        w.u(r ? *r : none, W);
        w.u(r ? pos : 0, 4);
    };
    for (const auto& node : net.nodes) {
        w.u(node.birth, 8);
        w.u(node.born_gen, 8);
        w.u(node.depth, W);
        if (!node.neuron) {
            w.u(0, 1);
        } else if (auto* v = std::get_if<VNeuron>(&*node.neuron)) {
            w.u(1, 1);
            ref(v->input, v->input_pos);
            w.u(static_cast<std::uint8_t>(v->inversion), 1);
            w.u(static_cast<std::uint8_t>(v->op), 1);
            w.f64(v->weight);
        } else if (auto* s = std::get_if<SNeuron>(&*node.neuron)) {
            w.u(2, 1);
            ref(s->input, s->input_pos);
            w.u(static_cast<std::uint8_t>(s->op), 1);
            w.u(s->k, 4);
        } else {
            const auto& m = std::get<MNeuron>(*node.neuron);
            w.u(3, 1);
            w.u(static_cast<std::uint8_t>(m.op), 1);
            for (std::size_t i = 0; i < m.slots.size(); ++i) ref(m.slots[i], m.slot_pos(i));
        }
    }
    return w.finish();
}

template <class D>
void save_net(const std::filesystem::path& path, const Net<D>& net) {
    write_file(path, to_bytes(net));
}

}  // namespace jneuro
