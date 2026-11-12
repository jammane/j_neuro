// Spec 0.3.0: population container (spec/11).
#include "helpers.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace th;
using N32 = Net<std::uint32_t>;
using Pop = Population<std::uint32_t>;
static constexpr std::uint32_t D32 = 0xFFFFFFFFu;
static const char* PHASES[] = {"op_change_max_pct", "edit_max_pct", "growth_max_pct",
                               "removal_max_pct", "depth_shift_max_pct"};

/// path_mix 0, so hand-built nets can be added.
static Pop pop(std::optional<std::uint64_t> seed = 11, Priority prio = Priority::Balanced) {
    return Pop(5, 3, 2, prio, seed, 0, 0);
}

static Profile heavy(const std::string& mode = "fresh") {
    auto p = Profile::defaults();
    for (auto f : PHASES) p.set_overall(f, 50.0);
    p.apply_patch(json{{"overall", {{"rng_mode", mode}}}});
    return p;
}

static std::vector<Bytes> genomes(const Pop& p) {
    std::vector<Bytes> out;
    for (auto i : p.ids()) out.push_back(to_bytes(p.get(i)));
    return out;
}

static N32 warm_net() {
    return make_net<std::uint32_t>(3, 2, 5, {{D32, M(AggOp::Mean, {0, 1, 2, none, none})},
                                             {D32, S(5, AggOp::Mean, 2)},
                                             {10, M(AggOp::Mean, {0, 1, 2, none, none})}});
}

TEST_CASE("pop: default rng mode is fresh") { CHECK(Profile::defaults().rng_mode() == "fresh"); }

TEST_CASE("pop: create and ids") {
    auto p = pop();
    CHECK(p.create(4) == std::vector<std::uint64_t>{0, 1, 2, 3});
    CHECK(p.ids() == std::vector<std::uint64_t>{0, 1, 2, 3});
    CHECK(p.size() == 4);
    CHECK(p.generation() == 0);
    for (auto i : p.ids()) p.get(i).validate();
    CHECK(p.create(2) == std::vector<std::uint64_t>{4, 5});
}

TEST_CASE("pop: add rejects mismatch and copies") {
    auto p = pop();
    Rng r(0);
    CHECK_THROWS_AS(p.add(N32::create(2, 2, 5, Priority::Speed, r)), ValidationError);
    CHECK_THROWS_AS(p.add(N32::create(3, 2, 3, Priority::Speed, r)), ValidationError);
    CHECK_THROWS_AS(p.add(N32::create(3, 2, 5, Priority::Speed, r)), ValidationError);  // path_mix 2 vs 0
    auto net = N32::create(3, 2, 5, Priority::Speed, r, 0, 0);
    auto i = p.add(net);
    auto& m = std::get<MNeuron>(*net.nodes[3].neuron);
    m.op = static_cast<AggOp>((static_cast<int>(m.op) + 1) % 4);
    CHECK(to_bytes(p.get(i)) != to_bytes(net));
}

TEST_CASE("pop: ids never reused and unknown ids") {
    auto p = pop();
    p.create(3);
    p.remove(1);
    CHECK(p.ids() == std::vector<std::uint64_t>{0, 2});
    CHECK(p.clone(0) == 3);
    CHECK(to_bytes(p.get(3)) == to_bytes(p.get(0)));
    CHECK_THROWS_AS(p.get(1), std::out_of_range);
    CHECK_THROWS_AS(p.remove(1), std::out_of_range);
    CHECK_THROWS_AS(p.clone(1), std::out_of_range);
    CHECK_THROWS_AS(p.mutate(1, heavy()), std::out_of_range);
    CHECK_THROWS_AS(p.inject_diversity(1, 10.0), std::out_of_range);
}

TEST_CASE("pop: evaluate matches step and skips warmup") {
    auto p = pop();
    p.create(2);
    auto w = p.add(warm_net());
    auto rows = sample_inputs(3, 6);
    auto res = p.evaluate(rows);
    std::vector<std::uint64_t> keys;
    for (auto& [k, v] : res) keys.push_back(k);
    CHECK(keys == p.ids());
    for (auto i : p.ids()) {
        N32 net = p.get(i).copy_genome();
        std::vector<std::vector<double>> full;
        for (auto& r : rows) full.push_back(net.step(r));
        full.erase(full.begin(), full.begin() + static_cast<std::ptrdiff_t>(net.get_warmup_steps()));
        REQUIRE(res[i].size() == full.size());
        for (std::size_t s = 0; s < full.size(); ++s)
            for (std::size_t o = 0; o < full[s].size(); ++o) CHECK(same(res[i][s][o], full[s][o]));
    }
    CHECK(res[w].size() == 4);
    for (auto& [k, v] : p.evaluate(rows, false)) CHECK(v.size() == 6);
}

TEST_CASE("pop: evaluate priority invariance") {
    std::vector<std::map<std::uint64_t, std::vector<std::vector<double>>>> out;
    for (Priority prio : {Priority::Speed, Priority::Memory, Priority::Balanced}) {
        auto p = pop(5, prio);
        p.create(6, "shared");
        p.add(warm_net());
        out.push_back(p.evaluate(sample_inputs(3, 8)));
    }
    auto as_bits = [](const auto& m) {
        std::vector<std::string> s;
        for (auto& [k, rows] : m) for (auto& r : rows) for (double y : r) s.push_back(bits(y));
        return s;
    };
    CHECK(as_bits(out[0]) == as_bits(out[1]));
    CHECK(as_bits(out[1]) == as_bits(out[2]));
}

TEST_CASE("pop: mutate offspring keeps parent") {
    auto p = pop();
    auto a = p.create(1)[0];
    auto before = to_bytes(p.get(a));
    auto b = p.mutate(a, heavy(), 3);
    CHECK(b == 1);
    CHECK(to_bytes(p.get(a)) == before);
    p.get(b).validate();
}

TEST_CASE("pop: shared mutate equals manual") {
    auto prof = heavy("shared");
    Pop p(5, 3, 2, Priority::Balanced, 7);
    auto a = p.create(1, "shared")[0];
    auto b = p.mutate(a, prof, 2);
    Rng r(7);
    auto net = N32::create(3, 2, 5, Priority::Balanced, r);
    auto m1 = mutate(net, prof, 0, r);
    auto manual = mutate(m1, prof, 0, r);
    CHECK(to_bytes(p.get(b)) == to_bytes(manual));
}

TEST_CASE("pop: shared reproducible, fresh varies") {
    auto session = [](const std::string& mode) {
        Pop p(5, 3, 2, Priority::Balanced, 99);
        for (auto i : p.create(4, mode)) p.mutate(i, heavy(mode), 2);
        return genomes(p);
    };
    CHECK(session("shared") == session("shared"));
    CHECK(session("fresh") != session("fresh"));
}

TEST_CASE("pop: inject diversity in place") {
    auto p = pop();
    auto net = make_net<std::uint32_t>(3, 2, 5, {{D32, M(AggOp::Mean, {5, 1, 2, none, none})},
                                                 {D32, M(AggOp::Max, {5, 1, 2, none, none})},
                                                 {10, V(0, VOp::Mul, 1.0)}});
    auto i = p.add(net);
    p.inject_diversity(i, 100.0);
    CHECK(p.ids() == std::vector<std::uint64_t>{i});
    CHECK(std::get<VNeuron>(*p.get(i).nodes[5].neuron).weight != 1.0);
}

TEST_CASE("pop: generation used for new neurons") {
    auto p = pop();
    auto a = p.create(1)[0];
    p.next_generation();
    p.next_generation();
    auto prof = Profile::defaults();
    for (auto f : PHASES) prof.set_overall(f, 0.0);
    prof.set_overall("growth_max_pct", 100.0);
    for (int k = 0; k < 30; ++k) {
        auto b = p.mutate(a, prof);
        bool any = false;
        for (auto& n : p.get(b).nodes)
            if (n.birth >= p.get(a).next_birth) { any = true; CHECK(n.born_gen == 2); }
        if (any) return;
    }
    FAIL("no growth happened");
}

static Pop session_with_everything() {
    Pop p(5, 3, 2, Priority::Speed, std::nullopt, 0, 0);  // adds a hand-built net
    auto ids = p.create(3);
    p.create(2, "shared");
    auto a = p.mutate(ids[0], heavy("fresh"), 2);
    auto b = p.mutate(ids[1], heavy("shared"));
    p.next_generation();
    auto c = p.clone(a);
    p.inject_diversity(c, 50.0, 2.0);
    p.inject_diversity(b, 30.0, 1.0, "shared");
    p.add(warm_net());
    p.remove(ids[2]);
    auto prof = heavy("shared");
    prof.set_overall("sigma_weight", 0.3);
    p.mutate(c, prof, 3);
    return p;
}

TEST_CASE("pop: replay reproduces exactly") {
    auto p = session_with_everything();
    auto log = p.run_log();
    auto q = Pop::replay(log);
    CHECK(q.ids() == p.ids());
    CHECK(q.generation() == p.generation());
    CHECK(genomes(q) == genomes(p));
    CHECK(q.run_log() == log);
    auto x = p.mutate(0, heavy("shared"));
    auto y = q.mutate(0, heavy("shared"));
    CHECK(to_bytes(p.get(x)) == to_bytes(q.get(y)));
}

TEST_CASE("pop: profiles deduplicated in log") {
    auto p = pop();
    auto a = p.create(1)[0];
    for (int k = 0; k < 3; ++k) p.mutate(a, heavy());
    p.mutate(a, heavy("shared"));
    auto log = json::parse(p.run_log());
    CHECK(log["profiles"].size() == 2);
    std::vector<int> idx;
    std::vector<bool> shared;
    for (auto& o : log["ops"])
        if (o["op"] == "mutate") { idx.push_back(o["profile"].get<int>()); shared.push_back(o["seed"].is_null()); }
    CHECK(idx == std::vector<int>{0, 0, 0, 1});
    CHECK(shared == std::vector<bool>{false, false, false, true});
}

template <class D>
static void check_population_case(const json& c) {
    auto q = Population<D>::replay(c["log"].dump());
    const auto& exp = c["expected"];
    CHECK(q.ids() == exp["ids"].get<std::vector<std::uint64_t>>());
    CHECK(q.generation() == exp["generation"].get<std::uint64_t>());
    for (auto i : q.ids()) {
        Bytes want = hex_to_bytes(exp["jnets"][std::to_string(i)].template get<std::string>());
        CHECK(to_bytes(q.get(i)) == want);
    }
    std::vector<std::vector<double>> in;
    for (auto& row : c["evaluate"]["inputs"]) {
        std::vector<double> r;
        for (auto& h : row) r.push_back(from_hex(h));
        in.push_back(r);
    }
    for (auto& [i, rows] : q.evaluate(in)) {
        const auto& want = c["evaluate"]["outputs"][std::to_string(i)];
        REQUIRE(rows.size() == want.size());
        for (std::size_t s = 0; s < rows.size(); ++s)
            for (std::size_t o = 0; o < rows[s].size(); ++o) CHECK(bits(rows[s][o]) == want[s][o].template get<std::string>());
    }
    auto dir = std::filesystem::temp_directory_path() / ("jneuro_pg_" + std::to_string(fresh_seed()));
    std::filesystem::create_directories(dir);
    q.save(dir / "pop.jpnet");
    CHECK(bytes_to_hex(read_file(dir / "pop.jpnet")) == c["save"]["jpnet"].template get<std::string>());
    for (auto i : q.ids())
        CHECK(bytes_to_hex(read_file(dir / "pop_nets" / (std::to_string(i) + ".jnet"))) ==
              exp["jnets"][std::to_string(i)].template get<std::string>());
    std::filesystem::remove_all(dir);
}

TEST_CASE("pop: population golden") {
    auto cases = load_golden("population");
    REQUIRE(cases.size() >= 2);
    for (auto& [name, c] : cases) {
        INFO(name);
        switch (parse_depth_type(c["log"]["population"]["depth_type"].get<std::string>())) {
            case DepthType::U16: check_population_case<std::uint16_t>(c); break;
            case DepthType::U32: check_population_case<std::uint32_t>(c); break;
            case DepthType::U64: check_population_case<std::uint64_t>(c); break;
        }
    }
}

// ------------------------------------------------------------------ .jpnet files

namespace {
struct TempDir {
    std::filesystem::path path;
    TempDir() : path(std::filesystem::temp_directory_path() / ("jneuro_pop_" + std::to_string(fresh_seed()))) {
        std::filesystem::create_directories(path);
    }
    ~TempDir() { std::filesystem::remove_all(path); }
};

Pop seeded_session(std::uint64_t seed) {
    Pop p(5, 3, 2, Priority::Speed, seed, 0, 0);  // adds a hand-built net
    auto ids = p.create(3);
    auto a = p.mutate(ids[0], heavy("fresh"), 2);
    p.next_generation();
    auto c = p.clone(a);
    p.inject_diversity(c, 50.0, 2.0);
    p.add(warm_net());
    p.remove(ids[2]);
    p.mutate(c, heavy("shared"), 3);
    return p;
}
}  // namespace

TEST_CASE("pop: save load roundtrip") {
    TempDir d;
    auto p = seeded_session(21);
    p.save(d.path / "run.jpnet");
    for (auto i : p.ids()) CHECK(std::filesystem::is_regular_file(d.path / "run_nets" / (std::to_string(i) + ".jnet")));
    auto q = Pop::load(d.path / "run.jpnet");
    CHECK(q.ids() == p.ids());
    CHECK(q.generation() == p.generation());
    CHECK(genomes(q) == genomes(p));
    auto x = p.mutate(0, heavy("shared"));
    auto y = q.mutate(0, heavy("shared"));
    CHECK(to_bytes(p.get(x)) == to_bytes(q.get(y)));
    CHECK(p.create(1) == q.create(1));
}

TEST_CASE("pop: save is byte identical") {
    TempDir d;
    auto p = seeded_session(4);
    std::filesystem::create_directories(d.path / "x");
    std::filesystem::create_directories(d.path / "y");
    p.save(d.path / "x" / "pop.jpnet");
    p.save(d.path / "y" / "pop.jpnet");
    CHECK(read_file(d.path / "x" / "pop.jpnet") == read_file(d.path / "y" / "pop.jpnet"));
}

TEST_CASE("pop: load rejects tampering") {
    TempDir d;
    auto p = seeded_session(5);
    p.save(d.path / "run.jpnet");
    auto victim = d.path / "run_nets" / (std::to_string(p.ids()[0]) + ".jnet");
    Bytes good = read_file(victim);
    Bytes bad = good;
    bad.back() ^= 1;
    write_file(victim, bad);
    CHECK_THROWS_AS(Pop::load(d.path / "run.jpnet"), ValidationError);
    std::filesystem::remove(victim);
    CHECK_THROWS_AS(Pop::load(d.path / "run.jpnet"), ValidationError);
    write_file(victim, good);
    Bytes jp = read_file(d.path / "run.jpnet");
    jp.back() ^= 1;
    write_file(d.path / "run.jpnet", jp);
    CHECK_THROWS_AS(Pop::load(d.path / "run.jpnet"), ValidationError);
}

TEST_CASE("pop: embedded log replays from scratch") {
    TempDir d;
    auto p = session_with_everything();
    p.save(d.path / "run.jpnet", true);
    auto q = Pop::load(d.path / "run.jpnet");
    CHECK(q.run_log() == p.run_log());
    q.mutate(0, heavy());
    auto r = Pop::replay(q.run_log());
    CHECK(genomes(r) == genomes(q));
}

TEST_CASE("pop: resumed log needs base") {
    TempDir d;
    auto p = session_with_everything();
    p.save(d.path / "run.jpnet");
    auto q = Pop::load(d.path / "run.jpnet");
    auto log0 = json::parse(q.run_log());
    CHECK(log0["resumed_from"].get<std::string>() == bytes_to_hex(sha256(read_file(d.path / "run.jpnet"))));
    CHECK(log0["ops"].empty());
    q.mutate(0, heavy());
    q.remove(1);
    q.next_generation();
    CHECK_THROWS_AS(Pop::replay(q.run_log()), ValidationError);
    auto base = Pop::load(d.path / "run.jpnet");
    auto r = Pop::replay(q.run_log(), &base);
    CHECK(r.ids() == q.ids());
    CHECK(genomes(r) == genomes(q));
    CHECK(r.run_log() == q.run_log());
    auto other = seeded_session(3);
    other.save(d.path / "other.jpnet");
    auto wrong = Pop::load(d.path / "other.jpnet");
    CHECK_THROWS_AS(Pop::replay(q.run_log(), &wrong), ValidationError);
}

TEST_CASE("pop: default path_mix is two") {
    Population<std::uint32_t> p(3, 9, 1, Priority::Speed, 5);
    CHECK(p.path_mix() == 2);
    for (auto i : p.create(3)) CHECK(p.get(i).path_mix == 2);
    auto child = p.mutate(0, heavy("fresh"), 5);
    p.get(child).validate();  // mix rule preserved
    CHECK(Population<std::uint32_t>::replay(p.run_log()).path_mix() == 2);
}

TEST_CASE("pop: mutation counts cover every operation") {
    Population<std::uint32_t> p(5, 4, 2, Priority::Speed, 3);
    auto ids = p.create(6);
    Profile prof = heavy("fresh");
    prof.set_overall("min_age_for_removal", 1.0);
    prof.set_overall("slot_retarget_prob", 0.5);
    prof.apply_patch(json{{"overall", {{"constant_folding", "on"},
        {"add_method_weights", {{"v_tap", 40}, {"s_tap", 40}, {"m_tap", 40}, {"m_clone", 40}}}}}});
    auto all_used = [&] {
        for (auto& [op, n] : p.mutation_counts()) if (n == 0) return false;
        return true;
    };
    for (int gen = 0; gen < 60 && !all_used(); ++gen) {
        for (auto& i : ids) i = p.mutate(i, prof);
        p.next_generation();
    }
    for (auto& [op, n] : p.mutation_counts()) {
        INFO(op);
        CHECK(n > 0);
    }
    CHECK(p.mutation_counts().size() == mutation_operations().size());
    p.reset_mutation_counts();
    for (auto& [op, n] : p.mutation_counts()) CHECK(n == 0);
}
