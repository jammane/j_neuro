// Genome JSON (spec/09) and convenience creation of any depth type.
#pragma once

#include <jneuro/net_build.hpp>

#include <nlohmann/json.hpp>

#include <string>
#include <variant>

namespace jneuro {

inline constexpr const char* kSpecVersion = "0.3.0";

/// Same MAJOR; while MAJOR is 0, also the same MINOR (spec/README).
bool version_compatible(const std::string& v);

using AnyNet = std::variant<Net<std::uint16_t>, Net<std::uint32_t>, Net<std::uint64_t>>;

/// Validate a document's format and spec_version (throws ValidationError).
void check_format(const nlohmann::json& d, const char* fmt);
/// A new 64-bit seed from the operating system's entropy source.
std::uint64_t fresh_seed();

/// Canonical text of a finite binary64 value (Python repr rules, spec/09).
std::string format_double(double x);

/// Canonical (byte-comparable) genome text.
template <class D>
std::string genome_to_json(const Net<D>& net);

/// Parse and fully validate a genome. Throws ValidationError.
AnyNet genome_from_json(const std::string& text);

/// Minimal net from input/output counts (spec/03). Defaults match the Python implementation.
AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type, std::uint32_t in_qty_max,
              Priority priority, Rng& rng);
AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type = DepthType::U32,
              std::uint32_t in_qty_max = 5, Priority priority = Priority::Balanced);

// ------------------------------------------------------------------ implementation

namespace detail {
inline void append_ref(std::string& s, const std::optional<NodeId>& r) {
    s += r ? std::to_string(*r) : "null";
}
}  // namespace detail

template <class D>
std::string genome_to_json(const Net<D>& net) {
    std::string s;
    s += "{\"format\":\"j_neuro.genome\",\"spec_version\":\"";
    s += kSpecVersion;
    s += "\",\"params\":{\"priority\":\"" + to_string(net.priority) + "\",\"depth_type\":\"" +
         to_string(Net<D>::depth_type) + "\",\"in_qty_max\":" + std::to_string(net.in_qty_max) +
         ",\"n_inputs\":" + std::to_string(net.n_inputs) + ",\"n_outputs\":" +
         std::to_string(net.n_outputs) + "},\"next_birth\":" + std::to_string(net.next_birth) +
         ",\"nodes\":[";
    for (std::size_t i = 0; i < net.nodes.size(); ++i) {
        const auto& n = net.nodes[i];
        if (i) s += ',';
        s += "{\"birth\":" + std::to_string(n.birth) + ",\"born_gen\":" + std::to_string(n.born_gen) +
             ",\"depth\":" + std::to_string(static_cast<std::uint64_t>(n.depth));
        if (n.neuron) {
            s += ",\"neuron\":{";
            if (auto* v = std::get_if<VNeuron>(&*n.neuron)) {
                s += "\"type\":\"v\",\"input\":";
                detail::append_ref(s, v->input);
                s += ",\"inversion\":\"" + to_string(v->inversion) + "\",\"op\":\"" + to_string(v->op) +
                     "\",\"weight\":" + format_double(v->weight);
            } else if (auto* sn = std::get_if<SNeuron>(&*n.neuron)) {
                s += "\"type\":\"s\",\"input\":";
                detail::append_ref(s, sn->input);
                s += ",\"op\":\"" + to_string(sn->op) + "\",\"k\":" + std::to_string(sn->k);
            } else {
                const auto& m = std::get<MNeuron>(*n.neuron);
                s += "\"type\":\"m\",\"op\":\"" + to_string(m.op) + "\",\"slots\":[";
                for (std::size_t j = 0; j < m.slots.size(); ++j) {
                    if (j) s += ',';
                    detail::append_ref(s, m.slots[j]);
                }
                s += ']';
            }
            s += '}';
        }
        s += '}';
    }
    s += "]}";
    return s;
}

}  // namespace jneuro
