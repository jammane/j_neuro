#include <jneuro/io.hpp>

#include <algorithm>
#include <fstream>
#include <random>
#include <vector>

namespace jneuro {

using json = nlohmann::json;

bool version_compatible(std::uint16_t major, std::uint16_t minor) {
    return major == kSpecMajor && (major != 0 || minor == kSpecMinor);
}

bool version_compatible(const std::string& v) {
    std::vector<std::string> parts;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= v.size(); ++i)
        if (i == v.size() || v[i] == '.') { parts.push_back(v.substr(start, i - start)); start = i + 1; }
    if (parts.size() < 2) return false;
    try {
        return version_compatible(static_cast<std::uint16_t>(std::stoul(parts[0])),
                                  static_cast<std::uint16_t>(std::stoul(parts[1])));
    } catch (const std::exception&) {
        return false;
    }
}

void check_format(const json& d, const char* fmt, bool accept_05) {
    if (!d.is_object() || !d.contains("format") || d.at("format") != fmt)
        throw ValidationError(std::string("format must be ") + fmt);
    if (!d.contains("spec_version") || !d.at("spec_version").is_string())
        throw ValidationError("unsupported spec_version");
    const std::string v = d.at("spec_version").get<std::string>();
    if (!version_compatible(v) && !(accept_05 && v.rfind("0.5.", 0) == 0))
        throw ValidationError("unsupported spec_version");
}

std::uint64_t fresh_seed() {
    std::random_device rd;
    return (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd());
}

Bytes read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw ValidationError("cannot open " + path.string());
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void write_file(const std::filesystem::path& path, const Bytes& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

namespace detail {

Reader::Reader(const Bytes& data, std::string what) : data_(data), what_(std::move(what)) {
    if (data.size() < 32) throw ValidationError(what_ + ": file too short");
    end_ = data.size() - 32;
    const Digest d = sha256(data.data(), end_);
    if (!std::equal(d.begin(), d.end(), data.begin() + static_cast<std::ptrdiff_t>(end_)))
        throw ValidationError(what_ + ": checksum mismatch");
}

const std::uint8_t* Reader::take(std::size_t n) {
    if (n > end_ - pos_) throw ValidationError(what_ + ": truncated");
    const std::uint8_t* p = data_.data() + pos_;
    pos_ += n;
    return p;
}

std::uint64_t Reader::u(std::size_t width) {
    const std::uint8_t* p = take(width);
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < width; ++i) v |= std::uint64_t{p[i]} << (8 * i);
    return v;
}

double Reader::f64() {
    const std::uint64_t b = u(8);
    double x;
    std::memcpy(&x, &b, 8);
    return x;
}

std::uint8_t Reader::enum_byte(std::uint8_t count, const char* field) {
    const auto v = static_cast<std::uint8_t>(u(1));
    if (v >= count) throw ValidationError(what_ + ": " + field + " byte out of range");
    return v;
}

bool Reader::header(const char magic[4]) {
    if (std::memcmp(take(4), magic, 4) != 0) throw ValidationError(what_ + ": wrong magic");
    const auto format = u(2);
    const auto major = static_cast<std::uint16_t>(u(2));
    const auto minor = static_cast<std::uint16_t>(u(2));
    u(2);
    const bool legacy = is_legacy_05(major, minor);
    if (!legacy && !version_compatible(major, minor)) throw ValidationError(what_ + ": unsupported spec version");
    const bool net = std::memcmp(magic, "JNET", 4) == 0;
    const std::uint64_t want = net ? (legacy ? 1 : kNetFormatVersion) : kPopFormatVersion;
    if (format != want) throw ValidationError(what_ + ": unknown format version");
    return legacy;
}

void Reader::done() const {
    if (pos_ != end_) throw ValidationError(what_ + ": trailing bytes");
}

}  // namespace detail

namespace {

template <class D>
Net<D> parse_net(detail::Reader& r, Priority priority, bool legacy) {
    constexpr std::size_t W = sizeof(D);
    const std::uint64_t none = static_cast<std::uint64_t>(Net<D>::d_max);
    Net<D> net;
    net.priority = priority;
    net.in_qty_max = static_cast<std::uint32_t>(r.u(4));
    net.n_inputs = static_cast<std::uint32_t>(r.u(4));
    net.n_outputs = static_cast<std::uint32_t>(r.u(4));
    net.path_mix = static_cast<std::uint32_t>(r.u(1));
    net.max_run = static_cast<std::uint32_t>(r.u(1));
    net.next_birth = r.u(8);
    const std::uint64_t count = r.u(W);
    if (count > Net<D>::d_max) throw ValidationError("jnet: more than D_MAX nodes");
    if (net.in_qty_max < 3 || net.in_qty_max % 2 == 0)
        throw ValidationError("jnet: in_qty_max must be odd and >= 3");
    // a conn: node id, then (format 2) its position
    auto ref = [&](std::uint32_t& pos) -> std::optional<NodeId> {
        const std::uint64_t v = r.u(W);
        pos = legacy ? 0 : static_cast<std::uint32_t>(r.u(4));
        if (v == none) {
            if (pos != 0) throw ValidationError("jnet: empty connection with a position");
            return std::nullopt;
        }
        if (v > 0xFFFFFFFFull) throw ValidationError("jnet: node id out of range");
        return static_cast<NodeId>(v);
    };
    for (std::uint64_t i = 0; i < count; ++i) {
        Node<D> node;
        node.birth = r.u(8);
        node.born_gen = r.u(8);
        node.depth = static_cast<D>(r.u(W));
        const std::uint8_t kind = r.enum_byte(4, "neuron type");
        if (kind == 1) {
            VNeuron v;
            v.input = ref(v.input_pos);
            v.inversion = static_cast<Inversion>(r.enum_byte(3, "inversion"));
            v.op = static_cast<VOp>(r.enum_byte(4, "v op"));
            v.weight = r.f64();
            node.neuron = v;
        } else if (kind == 2) {
            SNeuron s;
            s.input = ref(s.input_pos);
            s.op = static_cast<AggOp>(r.enum_byte(kSOps, "op"));
            s.k = static_cast<std::uint32_t>(r.u(4));
            node.neuron = s;
        } else if (kind == 3) {
            MNeuron m;
            m.op = static_cast<AggOp>(r.enum_byte(kMOps, "op"));  // lag is s-only
            m.slots.resize(net.in_qty_max);
            for (std::uint32_t k = 0; k < net.in_qty_max; ++k) {
                std::uint32_t p = 0;
                const auto c = ref(p);
                m.set(k, c, p);
            }
            node.neuron = m;
        }
        net.nodes.push_back(std::move(node));
    }
    r.done();
    if (legacy) respace_05(net);
    net.validate();
    return net;
}

}  // namespace

AnyNet from_bytes(const Bytes& data) {
    detail::Reader r(data, "jnet");
    const bool legacy = r.header("JNET");
    const auto priority = static_cast<Priority>(r.enum_byte(3, "priority"));
    switch (static_cast<DepthType>(r.enum_byte(3, "depth_type"))) {
        case DepthType::U16: return parse_net<std::uint16_t>(r, priority, legacy);
        case DepthType::U32: return parse_net<std::uint32_t>(r, priority, legacy);
        case DepthType::U64: return parse_net<std::uint64_t>(r, priority, legacy);
    }
    throw ValidationError("jnet: unknown depth_type");
}

AnyNet load_net(const std::filesystem::path& path) { return from_bytes(read_file(path)); }

AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type, std::uint32_t q,
              Priority priority, Rng& rng, std::uint32_t path_mix, std::uint32_t max_run) {
    switch (depth_type) {
        case DepthType::U16: return Net<std::uint16_t>::create(n_inputs, n_outputs, q, priority, rng, path_mix, max_run);
        case DepthType::U32: return Net<std::uint32_t>::create(n_inputs, n_outputs, q, priority, rng, path_mix, max_run);
        case DepthType::U64: return Net<std::uint64_t>::create(n_inputs, n_outputs, q, priority, rng, path_mix, max_run);
    }
    throw ValidationError("unknown depth_type");
}

AnyNet create(std::uint32_t n_inputs, std::uint32_t n_outputs, DepthType depth_type, std::uint32_t q,
              Priority priority, std::uint32_t path_mix, std::uint32_t max_run) {
    Rng rng(0);
    return create(n_inputs, n_outputs, depth_type, q, priority, rng, path_mix, max_run);
}

}  // namespace jneuro
