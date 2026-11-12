// Shared test helpers: compact constructors for hand-built nets.
#pragma once

#include <jneuro/jneuro.hpp>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace th {

using namespace jneuro;
using json = nlohmann::json;
inline constexpr auto none = std::nullopt;
using Slots = std::vector<std::optional<NodeId>>;

inline std::string bits(double x) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(to_bits(x)));
    return buf;
}
inline double from_hex(const std::string& h) { return from_bits(std::stoull(h, nullptr, 16)); }
inline bool same(double a, double b) { return to_bits(a) == to_bits(b); }

inline Neuron V(std::optional<NodeId> in, VOp op = VOp::Mul, double w = 1.0,
                Inversion inv = Inversion::None) {
    return VNeuron{in, inv, op, w};
}
inline Neuron S(std::optional<NodeId> in, AggOp op = AggOp::Mean, std::uint32_t k = 1) {
    return SNeuron{in, op, k};
}
inline Neuron M(AggOp op, Slots slots) { return MNeuron{op, std::move(slots)}; }

template <class D>
Net<D> make_net(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t q,
                std::vector<std::pair<D, Neuron>> noninput, Priority prio = Priority::Speed,
                std::uint32_t path_mix = 0) {
    Net<D> net;
    net.priority = prio;
    net.path_mix = path_mix;
    net.in_qty_max = q;
    net.n_inputs = n_in;
    net.n_outputs = n_out;
    for (std::uint32_t i = 0; i < n_in; ++i) net.nodes.push_back(Node<D>{i, 0, 0, std::nullopt});
    std::uint64_t b = n_in;
    for (auto& [d, n] : noninput) net.nodes.push_back(Node<D>{b++, 0, d, n});
    net.next_birth = net.nodes.size();
    return net;
}

template <class D>
std::vector<std::vector<std::string>> outputs_for(Net<D>& net,
                                                  const std::vector<std::vector<double>>& rows) {
    net.reset();
    std::vector<std::vector<std::string>> out;
    for (auto& r : rows) {
        std::vector<std::string> o;
        for (double y : net.step(r)) o.push_back(bits(y));
        out.push_back(o);
    }
    return out;
}

inline std::vector<std::vector<double>> sample_inputs(std::uint32_t n_in, int rows = 6) {
    const double base[] = {0.0, 1.0, -2.5, 3.25, 1e-3, -7.0, 0.5, 100.0, -0.125, 42.0};
    std::vector<std::vector<double>> out;
    for (int r = 0; r < rows; ++r) {
        std::vector<double> row;
        for (std::uint32_t i = 0; i < n_in; ++i) row.push_back(base[(r * 3 + i * 7) % 10] * (1 + r));
        out.push_back(row);
    }
    return out;
}

inline std::vector<std::pair<std::string, json>> load_golden(const std::string& sub) {
    std::vector<std::pair<std::string, json>> out;
    std::filesystem::path dir = std::filesystem::path(JNEURO_GOLDEN_DIR) / sub;
    if (!std::filesystem::is_directory(dir)) return out;
    std::vector<std::filesystem::path> files;
    for (auto& e : std::filesystem::directory_iterator(dir))
        if (e.path().extension() == ".json") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        std::ifstream in(f);
        out.emplace_back(f.filename().string(), json::parse(in));
    }
    return out;
}

inline constexpr std::uint64_t D16 = 65535;

}  // namespace th
