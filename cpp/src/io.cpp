#include <jneuro/io.hpp>

#include <charconv>
#include <cmath>
#include <algorithm>
#include <cstdlib>

namespace jneuro {

using json = nlohmann::json;

std::string format_double(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("only finite values are serialized");
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof buf, x, std::chars_format::scientific);
    std::string sci(buf, res.ptr);                    // e.g. "-1.25e+03", "5e-324", "0e+00"
    std::string sign;
    if (sci[0] == '-') { sign = "-"; sci.erase(0, 1); }
    const auto epos = sci.find('e');
    std::string digits = sci.substr(0, epos);
    digits.erase(std::remove(digits.begin(), digits.end(), '.'), digits.end());
    const int exp = std::atoi(sci.c_str() + epos + 1);

    if (exp >= -4 && exp < 16) {                     // positional, like Python repr
        std::string out;
        if (exp >= 0) {
            std::string ip = digits.substr(0, std::min<std::size_t>(digits.size(), exp + 1));
            ip.append(static_cast<std::size_t>(exp + 1) - ip.size(), '0');
            std::string fp = digits.size() > static_cast<std::size_t>(exp + 1) ? digits.substr(exp + 1) : "0";
            out = ip + "." + fp;
        } else {
            out = "0." + std::string(static_cast<std::size_t>(-exp - 1), '0') + digits;
        }
        return sign + out;
    }
    std::string mant = digits.substr(0, 1);
    if (digits.size() > 1) mant += "." + digits.substr(1);
    const int a = std::abs(exp);
    std::string e = std::to_string(a);
    if (e.size() < 2) e = "0" + e;
    return sign + mant + "e" + (exp < 0 ? "-" : "+") + e;
}

namespace {

std::uint64_t get_uint(const json& j, const char* what) {
    if (!j.is_number_unsigned() && !(j.is_number_integer() && j.get<std::int64_t>() >= 0))
        throw ValidationError(std::string(what) + " must be a non-negative integer");
    return j.get<std::uint64_t>();
}

std::uint32_t get_u32(const json& j, const char* what) {
    std::uint64_t v = get_uint(j, what);
    if (v > 0xFFFFFFFFull) throw ValidationError(std::string(what) + " out of range");
    return static_cast<std::uint32_t>(v);
}

std::optional<NodeId> get_ref(const json& j, const char* what) {
    if (j.is_null()) return std::nullopt;
    return get_u32(j, what);
}

const json& at(const json& j, const char* key) {
    if (!j.is_object() || !j.contains(key)) throw ValidationError(std::string("missing field ") + key);
    return j.at(key);
}

std::string get_str(const json& j, const char* what) {
    if (!j.is_string()) throw ValidationError(std::string(what) + " must be a string");
    return j.get<std::string>();
}

Neuron neuron_from(const json& o) {
    const std::string t = get_str(at(o, "type"), "type");
    if (t == "v") {
        const json& w = at(o, "weight");
        if (!w.is_number()) throw ValidationError("weight must be a number");
        return VNeuron{get_ref(at(o, "input"), "input"), parse_inversion(get_str(at(o, "inversion"), "inversion")),
                       parse_vop(get_str(at(o, "op"), "op")), w.get<double>()};
    }
    if (t == "s")
        return SNeuron{get_ref(at(o, "input"), "input"), parse_aggop(get_str(at(o, "op"), "op")),
                       get_u32(at(o, "k"), "k")};
    if (t == "m") {
        const json& sl = at(o, "slots");
        if (!sl.is_array()) throw ValidationError("slots must be a list");
        MNeuron m{parse_aggop(get_str(at(o, "op"), "op")), {}};
        for (auto& s : sl) m.slots.push_back(get_ref(s, "slot"));
        return m;
    }
    throw ValidationError("unknown neuron type " + t);
}

template <class D>
Net<D> net_from(const json& d) {
    Net<D> net;
    const json& p = at(d, "params");
    net.priority = parse_priority(get_str(at(p, "priority"), "priority"));
    net.in_qty_max = get_u32(at(p, "in_qty_max"), "in_qty_max");
    net.n_inputs = get_u32(at(p, "n_inputs"), "n_inputs");
    net.n_outputs = get_u32(at(p, "n_outputs"), "n_outputs");
    net.next_birth = get_uint(at(d, "next_birth"), "next_birth");
    const json& nodes = at(d, "nodes");
    if (!nodes.is_array()) throw ValidationError("nodes must be a list");
    for (auto& nd : nodes) {
        Node<D> node;
        node.birth = get_uint(at(nd, "birth"), "birth");
        node.born_gen = get_uint(at(nd, "born_gen"), "born_gen");
        std::uint64_t depth = get_uint(at(nd, "depth"), "depth");
        if (depth > Net<D>::d_max) throw ValidationError("depth out of range for depth_type");
        node.depth = static_cast<D>(depth);
        if (nd.contains("neuron")) node.neuron = neuron_from(nd.at("neuron"));
        net.nodes.push_back(std::move(node));
    }
    net.validate();
    return net;
}

}  // namespace

void check_format(const json& d, const char* fmt) {
    if (!d.is_object() || !d.contains("format") || d.at("format") != fmt)
        throw ValidationError(std::string("format must be ") + fmt);
    if (!d.contains("spec_version") || !d.at("spec_version").is_string())
        throw ValidationError("missing spec_version");
    std::string v = d.at("spec_version").get<std::string>();
    std::string ours = kSpecVersion;
    if (v.substr(0, v.find('.')) != ours.substr(0, ours.find('.')))
        throw ValidationError("unsupported spec_version " + v);
}

AnyNet genome_from_json(const std::string& text) {
    json d;
    try {
        d = json::parse(text);
    } catch (const json::exception& e) {
        throw ValidationError(std::string("invalid JSON: ") + e.what());
    }
    check_format(d, "j_neuro.genome");
    switch (parse_depth_type(get_str(at(at(d, "params"), "depth_type"), "depth_type"))) {
        case DepthType::U16: return net_from<std::uint16_t>(d);
        case DepthType::U32: return net_from<std::uint32_t>(d);
        case DepthType::U64: return net_from<std::uint64_t>(d);
    }
    throw ValidationError("unknown depth_type");
}

AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type, std::uint32_t q,
              Priority priority, Rng& rng) {
    switch (depth_type) {
        case DepthType::U16: return Net<std::uint16_t>::create(n_inputs, n_outputs, q, priority, rng);
        case DepthType::U32: return Net<std::uint32_t>::create(n_inputs, n_outputs, q, priority, rng);
        case DepthType::U64: return Net<std::uint64_t>::create(n_inputs, n_outputs, q, priority, rng);
    }
    throw ValidationError("unknown depth_type");
}

AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type, std::uint32_t q,
              Priority priority) {
    Rng rng(0);
    return create(n_inputs, n_outputs, depth_type, q, priority, rng);
}

}  // namespace jneuro
