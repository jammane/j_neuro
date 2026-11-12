#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <limits>

using namespace th;

TEST_CASE("invalid golden") {
    auto cases = load_golden("invalid");
    REQUIRE(cases.size() >= 25);
    for (auto& [name, c] : cases) {
        INFO(name);
        CHECK_THROWS_AS(from_bytes(hex_to_bytes(c["jnet"].get<std::string>())), ValidationError);
    }
}

TEST_CASE("sha256 known vectors") {
    CHECK(bytes_to_hex(sha256(Bytes{})) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const std::string abc = "abc";
    CHECK(bytes_to_hex(sha256(reinterpret_cast<const std::uint8_t*>(abc.data()), abc.size())) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::string two = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(bytes_to_hex(sha256(reinterpret_cast<const std::uint8_t*>(two.data()), two.size())) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const Bytes million(1000000, 'a');
    CHECK(bytes_to_hex(sha256(million)) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

static void put(Bytes& b, std::uint64_t v, int w) { for (int i = 0; i < w; ++i) b.push_back(static_cast<std::uint8_t>(v >> (8 * i))); }

TEST_CASE("exact layout") {
    // Bytes built by hand from spec/09, independent of the library's writer.
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Add, 0.5)}});
    Bytes b = {'J', 'N', 'E', 'T'};
    put(b, 1, 2); put(b, 0, 2); put(b, 5, 2); put(b, 0, 2);
    put(b, 0, 1); put(b, 0, 1); put(b, 3, 4); put(b, 1, 4); put(b, 1, 4); put(b, 0, 1); put(b, 2, 8);
    put(b, 2, 2);
    put(b, 0, 8); put(b, 0, 8); put(b, 0, 2); put(b, 0, 1);
    put(b, 1, 8); put(b, 0, 8); put(b, 0xFFFF, 2); put(b, 1, 1);
    put(b, 0, 2); put(b, 0, 1); put(b, 2, 1); put(b, to_bits(0.5), 8);
    const Digest h = sha256(b);
    b.insert(b.end(), h.begin(), h.end());
    CHECK(to_bytes(net) == b);
}

TEST_CASE("none sentinel and widths") {
    auto check = [](auto tag, std::size_t w) {
        using D = decltype(tag);
        auto net = make_net<D>(2, 1, 3, {{std::numeric_limits<D>::max(), M(AggOp::Max, {0, 1, none})}});
        Bytes b = to_bytes(net);
        for (std::size_t i = 0; i < w; ++i) CHECK(b[b.size() - 32 - w + i] == 0xFF);
        CHECK(b.size() == 35 + w + 3 * (16 + w + 1) + (1 + 3 * w) + 32);
    };
    check(std::uint16_t{}, 2);
    check(std::uint32_t{}, 4);
    check(std::uint64_t{}, 8);
}

TEST_CASE("roundtrip all types") {
    auto net = make_net<std::uint16_t>(2, 1, 5, {{D16, M(AggOp::Median, {3, 4, 1, none, none})},
                                                 {10, V(0, VOp::Div, -1e-07, Inversion::Bitnot)},
                                                 {20, S(0, AggOp::Max, 4)}});
    Bytes data = to_bytes(net);
    CHECK(std::visit([](const auto& n) { return to_bytes(n); }, from_bytes(data)) == data);
}

TEST_CASE("u64 depth roundtrip") {
    auto net = make_net<std::uint64_t>(1, 1, 3, {{~std::uint64_t{0}, V(2, VOp::Add, 1.0)},
                                                 {(std::uint64_t{1} << 53) + 1, V(0, VOp::Mul, 3.0)}});
    auto any = from_bytes(to_bytes(net));
    CHECK(std::get<Net<std::uint64_t>>(any).nodes[2].depth == (std::uint64_t{1} << 53) + 1);
}

TEST_CASE("weights exact") {
    for (double w : {0.1, -0.0, 5e-324, 1.7976931348623157e308, 1.0 / 3.0}) {
        auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, w)}});
        auto back = std::get<Net<std::uint16_t>>(from_bytes(to_bytes(net)));
        CHECK(to_bits(std::get<VNeuron>(*back.nodes[1].neuron).weight) == to_bits(w));
    }
}

TEST_CASE("save and load file") {
    auto dir = std::filesystem::temp_directory_path() / ("jneuro_io_" + std::to_string(fresh_seed()));
    std::filesystem::create_directories(dir);
    Rng r(3);
    auto net = Net<std::uint32_t>::create(4, 2, 5, Priority::Memory, r);
    save_net(dir / "a.jnet", net);
    CHECK(read_file(dir / "a.jnet") == to_bytes(net));
    CHECK(std::visit([](const auto& n) { return to_bytes(n); }, load_net(dir / "a.jnet")) == to_bytes(net));
    std::filesystem::remove_all(dir);
}

TEST_CASE("damage detected") {
    Bytes data = std::visit([](const auto& n) { return to_bytes(n); }, create(3, 1));
    Bytes cut(data.begin(), data.end() - 1);
    Bytes extra = data;
    extra.push_back(0);
    Bytes magic = data;
    magic[0] = 'X';
    Bytes flipped = data;
    flipped[40] ^= 1;
    for (auto* b : {&cut, &extra, &magic, &flipped}) CHECK_THROWS_AS(from_bytes(*b), ValidationError);
}

template <class D>
static void check_create(std::uint32_t n_in, std::uint32_t n_out, std::uint32_t q) {
    Rng r(11);
    auto net = Net<D>::create(n_in, n_out, q, Priority::Memory, r, 0);
    net.validate();
    CHECK(path_rule_ok(net));
    CHECK(net.next_birth == net.nodes.size());
    for (std::size_t i = 0; i < net.nodes.size(); ++i) {
        CHECK(net.nodes[i].birth == i);
        CHECK(net.nodes[i].born_gen == 0);
    }
    if (n_in == 1) {
        CHECK(net.nodes.size() == n_in + n_out);
        for (std::uint32_t o = 0; o < n_out; ++o)
            CHECK(std::holds_alternative<VNeuron>(*net.nodes[n_in + o].neuron));
    } else if (n_in <= q) {
        CHECK(net.nodes.size() == n_in + n_out);
        auto& m = std::get<MNeuron>(*net.nodes[n_in].neuron);
        for (std::uint32_t i = 0; i < n_in; ++i) CHECK(m.slots[i] == std::optional<NodeId>(i));
    } else {
        CHECK(net.nodes.size() > n_in + n_out);
    }
}

TEST_CASE("create minimal") {
    std::vector<std::array<std::uint32_t, 3>> cases = {{1, 3, 3}, {3, 2, 3}, {2, 1, 5}, {7, 2, 3}, {30, 2, 3}, {5, 1, 3}};
    for (auto& c : cases) {
        check_create<std::uint16_t>(c[0], c[1], c[2]);
        check_create<std::uint32_t>(c[0], c[1], c[2]);
        check_create<std::uint64_t>(c[0], c[1], c[2]);
    }
}

TEST_CASE("create tree shape") {
    Rng r(1);
    auto net = Net<std::uint16_t>::create(7, 2, 3, Priority::Speed, r, 0);
    REQUIRE(net.nodes.size() == 12);
    CHECK(std::get<MNeuron>(*net.nodes[9].neuron).slots == Slots{0, 1, 2});
    CHECK(std::get<MNeuron>(*net.nodes[10].neuron).slots == Slots{3, 4, 5});
    CHECK(std::get<MNeuron>(*net.nodes[11].neuron).slots == Slots{6, none, none});
    for (int i = 9; i < 12; ++i) CHECK(net.nodes[i].depth == D16 / 2);
    CHECK(std::get<MNeuron>(*net.nodes[7].neuron).slots == Slots{9, 10, 11});
}

TEST_CASE("create from counts only") {
    AnyNet any = create(4, 2);
    auto& net = std::get<Net<std::uint32_t>>(any);
    net.validate();
    CHECK(net.in_qty_max == 5);
    CHECK(net.priority == Priority::Balanced);
    Rng r(0);
    CHECK(to_bytes(net) == to_bytes(Net<std::uint32_t>::create(4, 2, 5, Priority::Balanced, r)));
    CHECK(net.step({1.0, 2.0, 3.0, 4.0}).size() == 2);
}

TEST_CASE("create deterministic") {
    Rng a(5), b(5);
    CHECK(to_bytes(Net<std::uint32_t>::create(9, 3, 3, Priority::Speed, a)) ==
          to_bytes(Net<std::uint32_t>::create(9, 3, 3, Priority::Speed, b)));
}

TEST_CASE("create rejects bad params") {
    Rng r(1);
    CHECK_THROWS_AS(Net<std::uint16_t>::create(2, 1, 4, Priority::Speed, r), ValidationError);
    CHECK_THROWS_AS(Net<std::uint16_t>::create(0, 1, 3, Priority::Speed, r), ValidationError);
}

// ---------------------------------------------------------------- mix rule (spec/03)

/// Every input-to-output path as a (v, s, m) count, by brute force.
template <class D>
static std::vector<std::array<int, 3>> path_counts(const Net<D>& net) {
    std::vector<std::vector<NodeId>> rd(net.nodes.size());
    for (NodeId i = 0; i < net.nodes.size(); ++i)
        for (NodeId c : connections(net.nodes[i].neuron)) rd[c].push_back(i);
    std::vector<std::array<int, 3>> out;
    std::function<void(NodeId, std::array<int, 3>)> walk = [&](NodeId i, std::array<int, 3> c) {
        if (net.nodes[i].neuron) ++c[net.nodes[i].neuron->index()];
        if (net.is_output(i)) out.push_back(c);
        for (NodeId r : rd[i]) walk(r, c);
    };
    for (NodeId i = 0; i < net.n_inputs; ++i) walk(i, {0, 0, 0});
    return out;
}

TEST_CASE("create meets path_mix") {
    std::vector<std::array<std::uint32_t, 3>> cases = {{1, 3, 3}, {3, 2, 3}, {2, 1, 5}, {7, 2, 3}, {9, 1, 3}, {30, 2, 3}};
    for (auto& c : cases) {
        for (std::uint32_t mix : {1u, 2u, 3u}) {
            Rng r(4);
            auto net = Net<std::uint16_t>::create(c[0], c[1], c[2], Priority::Speed, r, mix);
            net.validate();
            CHECK(net.path_mix == mix);
            CHECK(mix_ok(net));
            const auto counts = path_counts(net);
            REQUIRE(!counts.empty());
            int lowest = 1 << 30;
            for (auto& k : counts) {
                CHECK(std::min({k[0], k[1], k[2]}) >= static_cast<int>(mix));
                lowest = std::min(lowest, std::min({k[0], k[1], k[2]}));
            }
            CHECK(lowest == static_cast<int>(mix));
            CHECK(std::get<Net<std::uint16_t>>(from_bytes(to_bytes(net))).path_mix == mix);
        }
    }
}

TEST_CASE("create default path_mix chain") {
    Rng r(2);
    auto net = Net<std::uint32_t>::create(9, 1, 3, Priority::Speed, r);  // the blackjack shape
    CHECK(net.path_mix == 2);
    std::string kinds;
    for (std::size_t i = 9; i < net.nodes.size(); ++i) kinds += "VSM"[net.nodes[i].neuron->index()];
    CHECK(kinds == "VMMMMSSV");  // output v, 3 tree m, merge m, s, s, v
    for (auto& k : path_counts(net)) CHECK(k == std::array<int, 3>{2, 2, 2});
}

TEST_CASE("mix rule rejects") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(2, VOp::Add, 0.0)}, {5, M(AggOp::Min, {0, none, none})}},
                                       Priority::Speed, 1);
    CHECK_FALSE(mix_ok(net));  // no s_neuron on the path
    CHECK_THROWS_AS(net.validate(), ValidationError);
    net.path_mix = 0;
    net.validate();
    Rng r(0);
    CHECK_THROWS_AS(Net<std::uint16_t>::create(2, 1, 3, Priority::Speed, r, 256), ValidationError);
}
