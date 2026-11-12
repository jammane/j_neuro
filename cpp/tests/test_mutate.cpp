#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <set>

using namespace th;
using N16 = Net<std::uint16_t>;
using N32 = Net<std::uint32_t>;

static const char* PHASES[] = {"op_change_max_pct", "edit_max_pct", "growth_max_pct",
                               "removal_max_pct", "depth_shift_max_pct"};

// Only the given phases enabled; new v_neurons never inverted, so insertions stay neutral.
static Profile zero_profile() {
    Profile p = Profile::defaults();
    for (auto f : PHASES) p.set_overall(f, 0.0);
    p.set_overall("insert_inversion_prob", 0.0);
    return p;
}

static Profile growth_only(const std::string& method) {
    Profile p = zero_profile();
    p.set_overall("growth_max_pct", 100.0);
    p.apply_patch(json{{"overall", {{"add_method_weights",
        {{"slot_fill", method == "slot_fill" ? 1.0 : 0.0}, {"v_insert", method == "v_insert" ? 1.0 : 0.0},
         {"s_insert", method == "s_insert" ? 1.0 : 0.0}, {"m_after_m", method == "m_after_m" ? 1.0 : 0.0}}}}}});
    return p;
}

template <class D>
static std::size_t n_neurons(const Net<D>& n) { return n.nodes.size() - n.n_inputs; }

template <class D>
static Net<D> first_changed(const Net<D>& net, const Profile& p, std::uint64_t gen,
                            std::function<bool(const Net<D>&)> pred = {}) {
    for (std::uint64_t seed = 0; seed < 300; ++seed) {
        Rng r(seed);
        auto child = mutate(net, p, gen, r);
        bool ok = pred ? pred(child) : to_bytes(child) != to_bytes(net);
        if (ok) return child;
    }
    FAIL("no seed produced the expected change");
    return net;
}

static N32 base_net(std::uint64_t seed = 3, std::uint32_t n_in = 4, std::uint32_t n_out = 2, std::uint32_t q = 5) {
    Rng r(seed);
    return N32::create(n_in, n_out, q, Priority::Speed, r);
}

TEST_CASE("profile defaults") {
    auto p = Profile::defaults();
    CHECK(p.effective("edit_max_pct", NeuronType::V) == 5.0);
    CHECK(p.effective("growth_max_pct", NeuronType::M) == 1.5);
    CHECK(p.add_method_weights() == std::array<double, 4>{40, 30, 20, 10});
    CHECK(p.effective("sigma_weight", NeuronType::V) == 0.1);
}

TEST_CASE("profile overrides and patch") {
    auto p = Profile::defaults();
    p.apply_patch(json{{"m", {{"edit_max_pct", 2.0}}}, {"overall", {{"sigma_weight", 0.05}}}});
    CHECK(p.effective("edit_max_pct", NeuronType::M) == 2.0);
    CHECK(p.has_override("edit_max_pct", NeuronType::M));
    CHECK_FALSE(p.has_override("edit_max_pct", NeuronType::V));
    CHECK(p.effective("edit_max_pct", NeuronType::V) == 5.0);
    CHECK(p.effective("sigma_weight", NeuronType::V) == 0.05);
    p.apply_patch(json{{"m", {{"edit_max_pct", nullptr}}}});
    CHECK_FALSE(p.has_override("edit_max_pct", NeuronType::M));
}

TEST_CASE("profile json roundtrip") {
    auto p = Profile::defaults();
    p.apply_patch(json{{"s", {{"sigma_k", 2.0}}}});
    auto q = profile_from_json(profile_to_json(p));
    CHECK(profile_to_json(q) == profile_to_json(p));
}

TEST_CASE("parent untouched and deterministic") {
    auto net = base_net();
    auto text = to_bytes(net);
    auto p = Profile::defaults();
    for (auto f : PHASES) p.set_overall(f, 50.0);
    Rng a(9), b(9);
    auto x = mutate(net, p, 1, a);
    auto y = mutate(net, p, 1, b);
    CHECK(to_bytes(net) == text);
    CHECK(to_bytes(x) == to_bytes(y));
}

TEST_CASE("zero profile is identity") {
    auto net = base_net();
    Rng r(1);
    CHECK(to_bytes(mutate(net, zero_profile(), 4, r)) == to_bytes(net));
}

TEST_CASE("offspring starts reset") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, S(0, AggOp::Mean, 1)}});
    net.step({5.0});
    Rng r(1);
    auto child = mutate(net, zero_profile(), 1, r);
    CHECK(child.step({5.0})[0] == 5.0);
}

TEST_CASE("neutral growth") {
    for (std::string method : {"v_insert", "slot_fill"}) {
        INFO(method);
        auto net = base_net();
        auto rows = sample_inputs(net.n_inputs);
        auto p = growth_only(method);
        std::size_t grown = 0;
        for (std::uint64_t seed = 0; seed < 20; ++seed) {
            Rng r(seed);
            auto child = mutate(net, p, 1, r);
            child.validate();
            CHECK(outputs_for(child, rows) == outputs_for(net, rows));
            grown += n_neurons(child) - n_neurons(net);
        }
        CHECK(grown > 0);
    }
}

TEST_CASE("v insert identity and birth") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, 2.0)}});
    auto child = first_changed(net, growth_only("v_insert"), 7);
    REQUIRE(child.nodes.size() == 3);
    auto& out = child.nodes[1];
    auto& b = child.nodes[2];
    CHECK(std::get<VNeuron>(*out.neuron).input == std::optional<NodeId>(2));
    CHECK(out.born_gen == 7);
    CHECK(out.birth == 2);
    CHECK(b.birth == 1);
    CHECK(b.born_gen == 0);
    CHECK(std::get<VNeuron>(*b.neuron).weight == 2.0);
    CHECK((b.depth >= 1 && b.depth <= D16 - 1));
}

TEST_CASE("s insert creates k1") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, 2.0)}});
    auto child = first_changed(net, growth_only("s_insert"), 1);
    CHECK(std::get<SNeuron>(*child.nodes[1].neuron).k == 1);
}

TEST_CASE("m after m starts max and is neutral after max with empty slot") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Max, {0, 1, none})}});
    auto rows = sample_inputs(2);
    auto child = first_changed(net, growth_only("m_after_m"), 1);
    auto& m = std::get<MNeuron>(*child.nodes[2].neuron);
    CHECK(m.op == AggOp::Max);
    CHECK(m.slots == Slots{3, none, none});
    CHECK(outputs_for(child, rows) == outputs_for(net, rows));
}

TEST_CASE("slot fill skips min receivers") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Min, {0, 1, none})}});
    for (std::uint64_t seed = 0; seed < 10; ++seed) {
        Rng r(seed);
        CHECK(to_bytes(mutate(net, growth_only("slot_fill"), 1, r)) == to_bytes(net));
    }
}

TEST_CASE("growth cap per type") {
    auto net = base_net();
    auto p = growth_only("v_insert");
    p.apply_patch(json{{"v", {{"growth_max_pct", 0.0}}}});
    Rng r(1);
    CHECK(to_bytes(mutate(net, p, 1, r)) == to_bytes(net));
}

TEST_CASE("no gap skips or shifts") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(2)}, {1, V(0)}});
    for (std::uint64_t seed = 0; seed < 10; ++seed) {
        Rng r(seed);
        mutate(net, growth_only("v_insert"), 1, r).validate();
    }
}

TEST_CASE("op changes valid") {
    auto net = base_net();
    auto p = zero_profile();
    p.set_overall("op_change_max_pct", 100.0);
    first_changed(net, p, 1).validate();
}

TEST_CASE("op change resets weight") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, 3.0)}});
    auto p = zero_profile();
    p.set_overall("op_change_max_pct", 100.0);
    auto child = first_changed(net, p, 1);
    auto& n = std::get<VNeuron>(*child.nodes[1].neuron);
    CHECK_FALSE((n.inversion == Inversion::None && n.op == VOp::Mul));
    CHECK(n.weight == ((n.op == VOp::Mul || n.op == VOp::Div) ? 1.0 : 0.0));
}

TEST_CASE("edits valid") {
    auto net = make_net<std::uint16_t>(2, 1, 5, {{D16, M(AggOp::Mean, {3, 4, 5, none, none})},
                                                 {10, V(0, VOp::Mul, 2.0)}, {20, V(1, VOp::Add, 1.0)},
                                                 {30, S(4, AggOp::Median, 3)}});
    auto p = zero_profile();
    p.set_overall("edit_max_pct", 100.0);
    for (std::uint64_t seed = 0; seed < 30; ++seed) {
        Rng r(seed);
        auto child = mutate(net, p, 1, r);
        child.validate();
        for (auto& n : child.nodes)
            if (n.neuron && std::holds_alternative<SNeuron>(*n.neuron))
                CHECK(std::get<SNeuron>(*n.neuron).k % 2 == 1);
    }
}

TEST_CASE("depth shift neutral") {
    auto net = base_net(3, 9, 2, 3);
    auto rows = sample_inputs(net.n_inputs);
    auto p = zero_profile();
    p.set_overall("depth_shift_max_pct", 100.0);
    auto child = first_changed(net, p, 1);
    CHECK(outputs_for(child, rows) == outputs_for(net, rows));
    for (std::size_t i = 0; i < net.nodes.size(); ++i) CHECK(child.nodes[i].birth == net.nodes[i].birth);
}

TEST_CASE("removal of inserted identities after grace") {
    auto net = base_net(3, 9, 2, 3);
    auto rows = sample_inputs(net.n_inputs);
    auto big = net;
    for (std::uint64_t i = 0; i < 4; ++i) {
        Rng r(i);
        big = mutate(big, growth_only("v_insert"), 0, r);
    }
    REQUIRE(n_neurons(big) > n_neurons(net));
    auto prune = zero_profile();
    prune.set_overall("removal_max_pct", 100.0);
    for (std::uint64_t seed = 0; seed < 20; ++seed) {
        Rng r(seed);
        CHECK(to_bytes(mutate(big, prune, 2, r)) == to_bytes(big));
    }
    auto later = first_changed<std::uint32_t>(big, prune, 10,
                                              [&](const N32& c) { return n_neurons(c) < n_neurons(big); });
    later.validate();
    CHECK(outputs_for(later, rows) == outputs_for(net, rows));
}

TEST_CASE("immune silenced not removed") {
    auto net = make_net<std::uint16_t>(2, 1, 3, {{D16, M(AggOp::Mean, {0, 3, none})},
                                                 {10, V(1, VOp::Mul, 0.0)}});
    auto p = zero_profile();
    p.set_overall("removal_max_pct", 100.0);
    Rng r(1);
    CHECK(to_bytes(mutate(net, p, 50, r)) == to_bytes(net));
}

TEST_CASE("mutate golden") {
    for (auto& [name, c] : load_golden("mutate")) {
        INFO(name);
        AnyNet any = from_bytes(hex_to_bytes(c["jnet"].get<std::string>()));
        Profile p = profile_from_json(c["profile"].dump());
        Bytes expected = hex_to_bytes(c["expected_jnet"].get<std::string>());
        std::visit([&](auto& net) {
            Rng r(c["seed"].get<std::uint64_t>());
            CHECK(to_bytes(mutate(net, p, c["generation"].get<std::uint64_t>(), r)) == expected);
        }, any);
    }
}

TEST_CASE("create golden") {
    for (auto& [name, c] : load_golden("create")) {
        INFO(name);
        Rng r(c["seed"].get<std::uint64_t>());
        AnyNet net = create(c["n_inputs"].get<std::uint32_t>(), c["n_outputs"].get<std::uint32_t>(),
                            parse_depth_type(c["depth_type"].get<std::string>()),
                            c["in_qty_max"].get<std::uint32_t>(),
                            parse_priority(c["priority"].get<std::string>()), r,
                            c["path_mix"].get<std::uint32_t>());
        Bytes expected = hex_to_bytes(c["expected_jnet"].get<std::string>());
        CHECK(std::visit([](const auto& n) { return to_bytes(n); }, net) == expected);
    }
}

TEST_CASE("property: random chains stay valid") {
    for (std::uint64_t seed = 0; seed < 12; ++seed) {
        Rng r(1000 + seed);
        std::uint32_t n_in = 1 + static_cast<std::uint32_t>(r.below(8));
        std::uint32_t n_out = 1 + static_cast<std::uint32_t>(r.below(3));
        DepthType dt = static_cast<DepthType>(r.below(3));
        std::uint32_t q = 3 + 2 * static_cast<std::uint32_t>(r.below(3));
        AnyNet any = create(n_in, n_out, dt, q, Priority::Balanced, r);
        auto p = Profile::defaults();
        for (auto f : PHASES) p.set_overall(f, 30.0);
        p.set_overall("min_age_for_removal", 1.0);
        std::visit([&](auto& net) {
            for (std::uint64_t gen = 1; gen < 25; ++gen) {
                net = mutate(net, p, gen, r);
                net.validate();
                CHECK(path_rule_ok(net));
                net.step(std::vector<double>(net.n_inputs, 0.5));
            }
        }, any);
    }
}

TEST_CASE("property: neutral chains preserve outputs") {
    for (std::uint64_t seed = 0; seed < 12; ++seed) {
        Rng r(2000 + seed);
        std::uint32_t n_in = 2 + static_cast<std::uint32_t>(r.below(6));
        std::uint32_t n_out = 1 + static_cast<std::uint32_t>(r.below(2));
        std::uint32_t q = 3 + 2 * static_cast<std::uint32_t>(r.below(2));
        auto net = N32::create(n_in, n_out, q, Priority::Speed, r);
        auto rows = sample_inputs(net.n_inputs);
        auto ref = outputs_for(net, rows);
        auto p = zero_profile();
        p.set_overall("growth_max_pct", 40.0);
        p.set_overall("removal_max_pct", 40.0);
        p.set_overall("depth_shift_max_pct", 40.0);
        p.set_overall("min_age_for_removal", 0.0);
        p.apply_patch(json{{"overall", {{"add_method_weights",
            {{"slot_fill", 1.0}, {"v_insert", 1.0}, {"s_insert", 0.0}, {"m_after_m", 0.0}}}}}});
        for (std::uint64_t gen = 1; gen < 25; ++gen) {
            net = mutate(net, p, gen, r);
            net.validate();
            CHECK(outputs_for(net, rows) == ref);
        }
    }
}

TEST_CASE("nonzero cap is at least one") {
    auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, V(0, VOp::Mul, 2.0)}});
    auto p = zero_profile();
    p.set_overall("edit_max_pct", 1.0);
    int changed = 0;
    for (std::uint64_t s = 0; s < 60; ++s) {
        Rng r(s);
        if (to_bytes(mutate(net, p, 1, r)) != to_bytes(net)) ++changed;
    }
    CHECK(changed >= 10);
    for (std::uint64_t s = 0; s < 20; ++s) {
        Rng r(s);
        CHECK(to_bytes(mutate(net, zero_profile(), 1, r)) == to_bytes(net));
    }
}

// ---------------------------------------------------------------- mix rule (spec/03, spec/06)

TEST_CASE("mutation keeps mix rule") {
    // input 0 -> v 2 (identity) -> s 3 / s 4 -> output m 1. Every path: 1 v, 1 s, 1 m.
    auto mix_net = [](std::uint32_t mix) {
        return make_net<std::uint16_t>(1, 1, 3, {{D16, M(AggOp::Max, {3, 4, none})}, {100, V(0, VOp::Mul, 1.0)},
                                                 {200, S(2, AggOp::Mean, 1)}, {200, S(2, AggOp::Mean, 1)}},
                                       Priority::Speed, mix);
    };
    auto structure = [](const N16& n) {
        std::vector<std::pair<std::size_t, std::vector<NodeId>>> out;
        for (auto& x : n.nodes) out.emplace_back(x.neuron ? x.neuron->index() + 1 : 0, connections(x.neuron));
        return out;
    };
    std::vector<std::pair<std::string, Profile>> cases;
    {
        Profile p = zero_profile();
        p.set_overall("removal_max_pct", 100.0);
        p.set_overall("min_age_for_removal", 0.0);
        cases.emplace_back("bypass", p);
    }
    {
        Profile p = zero_profile();
        p.set_overall("edit_max_pct", 100.0);
        p.set_overall("rewire_prob", 1.0);
        cases.emplace_back("rewire", p);
    }
    {
        Profile p = zero_profile();
        p.set_overall("edit_max_pct", 100.0);
        p.set_overall("rewire_prob", 0.0);
        p.set_overall("sigma_slots", 3.0);
        cases.emplace_back("slot_connect", p);
    }
    cases.emplace_back("slot_fill", growth_only("slot_fill"));
    for (auto& [name, p] : cases) {
        INFO(name);
        int changed[2] = {0, 0};
        for (std::uint32_t mix : {0u, 1u}) {
            const N16 net = mix_net(mix);
            net.validate();
            for (std::uint64_t s = 0; s < 80; ++s) {
                Rng r(s);
                const N16 child = mutate(net, p, 10, r);
                child.validate();  // includes the mix rule
                changed[mix] += structure(child) != structure(net);
            }
        }
        CHECK(changed[0] > changed[1]);  // the rule blocked something
    }
}

// ---------------------------------------------------------------- growth_budget, constant_folding

TEST_CASE("exact growth budget adds exactly one") {
    Rng r0(1);
    const N32 net = N32::create(3, 1, 3, Priority::Speed, r0);
    std::set<std::size_t> grew[2];
    for (int exact = 0; exact < 2; ++exact) {
        Profile p = growth_only("v_insert");
        p.set_overall("growth_max_pct", 1e-9);
        p.apply_patch(json{{"overall", {{"growth_budget", exact ? "exact" : "up_to"}}}});
        for (std::uint64_t s = 0; s < 40; ++s) {
            Rng r(s);
            grew[exact].insert(mutate(net, p, 1, r).nodes.size() - net.nodes.size());
        }
    }
    CHECK(grew[1] == std::set<std::size_t>{1});
    CHECK(grew[0] == std::set<std::size_t>{0, 1});
}

TEST_CASE("constant folding") {
    // output m max [2, 3]; 2 = identity of input 0; 3 = (4 + 2) where 4 = input * 0 (constant)
    const auto net = make_net<std::uint16_t>(1, 1, 3, {{D16, M(AggOp::Max, {2, 3, none})}, {200, V(0, VOp::Mul, 1.0)},
                                                       {200, V(4, VOp::Add, 2.0)}, {100, V(0, VOp::Mul, 0.0)}});
    net.validate();
    auto rows = sample_inputs(1);
    auto copy = net;
    const auto ref = outputs_for(copy, rows);
    Profile p = zero_profile();
    p.set_overall("min_age_for_removal", 0.0);
    p.apply_patch(json{{"overall", {{"constant_folding", "on"}}}});
    Rng r(0);
    auto child = mutate(net, p, 1, r);
    REQUIRE(child.nodes.size() == 4);
    const auto& v = std::get<VNeuron>(*child.nodes[3].neuron);
    CHECK((!v.input && v.op == VOp::Add && v.weight == 2.0 && v.inversion == Inversion::None));
    CHECK(outputs_for(child, rows) == ref);
    Profile off = zero_profile();
    off.set_overall("min_age_for_removal", 0.0);
    Rng r2(0);
    CHECK(to_bytes(mutate(net, off, 1, r2)) == to_bytes(net));
    // the only path: folding would cut it, so nothing changes
    const auto only = make_net<std::uint16_t>(1, 1, 3, {{D16, V(3, VOp::Add, 0.0)}, {100, V(0, VOp::Mul, 0.0)},
                                                        {200, S(2, AggOp::Mean, 1)}});
    Rng r3(0);
    CHECK(to_bytes(mutate(only, p, 1, r3)) == to_bytes(only));
}
