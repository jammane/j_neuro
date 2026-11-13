// Population: nets by stable id, evaluation, operations, RNG modes, replay (spec/11).
// Mirrors python/jneuro/population.py.
#pragma once

#include <jneuro/io.hpp>
#include <jneuro/mutate.hpp>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace jneuro {

template <class D>
class Population {
public:
    using Outputs = std::map<std::uint64_t, std::vector<std::vector<double>>>;

    Population(std::uint32_t in_qty_max, std::uint32_t n_inputs, std::uint32_t n_outputs,
               Priority priority = Priority::Balanced, std::optional<std::uint64_t> seed = std::nullopt,
               std::uint32_t path_mix = 2, std::uint32_t max_run = 3)
        : in_qty_max_(in_qty_max), n_inputs_(n_inputs), n_outputs_(n_outputs), path_mix_(path_mix),
          max_run_(max_run), priority_(priority), seed_(seed ? *seed : fresh_seed()), rng_(seed_) {
        detail::check_params(n_inputs, n_outputs, in_qty_max);
        detail::check_path_mix(path_mix);
        detail::check_max_run(max_run);
    }

    // ---------------------------------------------------------------- inspection
    std::uint64_t generation() const { return generation_; }
    std::uint64_t seed() const { return seed_; }
    std::uint32_t path_mix() const { return path_mix_; }
    std::uint32_t max_run() const { return max_run_; }

    /// Successful uses of each mutation operation since creation or the last reset; every
    /// operation of mutation_operations() is listed (0 if unused). Informative; not saved.
    MutationCounts mutation_counts() const {
        MutationCounts out;
        for (const auto& op : mutation_operations()) {
            auto it = mutation_counts_.find(op);
            out[op] = it == mutation_counts_.end() ? 0 : it->second;
        }
        return out;
    }
    void reset_mutation_counts() { mutation_counts_.clear(); }
    std::size_t size() const { return nets_.size(); }
    std::vector<std::uint64_t> ids() const {
        std::vector<std::uint64_t> out;
        for (auto& [k, v] : nets_) out.push_back(k);
        return out;
    }
    const Net<D>& get(std::uint64_t id) const { return nets_.at(id); }
    Net<D>& get(std::uint64_t id) { return nets_.at(id); }

    // ---------------------------------------------------------------- operations
    /// Add `count` minimal nets, all drawn from one operation RNG.
    std::vector<std::uint64_t> create(std::uint64_t count, const std::string& rng_mode = "fresh") {
        return create_impl(count, rng_mode, std::nullopt);
    }

    /// Add a copy of an existing net with the population's parameters.
    std::uint64_t add(const Net<D>& net) {
        if (net.in_qty_max != in_qty_max_ || net.n_inputs != n_inputs_ || net.n_outputs != n_outputs_ ||
            net.path_mix != path_mix_ || net.max_run != max_run_)
            throw ValidationError("net parameters differ from the population's");
        net.validate();
        const std::uint64_t id = store(net.copy_genome());
        ops_.push_back(nlohmann::json{{"op", "add"}, {"new_id", id}, {"jnet", bytes_to_hex(to_bytes(net))}});
        return id;
    }

    /// Offspring of `id` after `times` mutations; the parent stays.
    std::uint64_t mutate(std::uint64_t id, const Profile& profile, std::uint32_t times = 1) {
        return mutate_impl(id, profile, times, std::nullopt);
    }

    std::uint64_t clone(std::uint64_t id) {
        const std::uint64_t nid = store(nets_.at(id).copy_genome());
        ops_.push_back(nlohmann::json{{"op", "clone"}, {"id", id}, {"new_id", nid}});
        return nid;
    }

    /// Replace the net at `id` with a diversity-injected version (id kept).
    void inject_diversity(std::uint64_t id, double pct, double sigma_reset = 1.0,
                          const std::string& rng_mode = "fresh") {
        diversity_impl(id, pct, sigma_reset, rng_mode, std::nullopt);
    }

    void remove(std::uint64_t id) {
        nets_.at(id);
        nets_.erase(id);
        ops_.push_back(nlohmann::json{{"op", "remove"}, {"id", id}});
    }

    void next_generation() {
        ++generation_;
        ops_.push_back(nlohmann::json{{"op", "next_generation"}});
    }

    // ---------------------------------------------------------------- evaluation
    /// Outputs of every net on the same input rows, keyed by id. Concurrency follows the
    /// population priority; results are identical in every case (spec/11).
    Outputs evaluate(const std::vector<std::vector<double>>& inputs, bool skip_warmup = true) {
        std::vector<std::pair<std::uint64_t, Net<D>*>> work;
        for (auto& [k, v] : nets_) work.emplace_back(k, &v);
        std::vector<std::vector<std::vector<double>>> results(work.size());
        auto run = [&](std::size_t first, std::size_t stride) {
            for (std::size_t i = first; i < work.size(); i += stride) {
                Net<D>& net = *work[i].second;
                net.reset();
                auto& rows = results[i];
                for (auto& r : inputs) rows.push_back(net.step(r));
                if (skip_warmup) {
                    const auto w = std::min<std::uint64_t>(net.get_warmup_steps(), rows.size());
                    rows.erase(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(w));
                }
            }
        };
        const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
        std::size_t workers = priority_ == Priority::Memory ? 1
                            : priority_ == Priority::Balanced ? std::max<std::size_t>(1, hw / 2) : hw;
        workers = std::min(workers, std::max<std::size_t>(1, work.size()));
        if (workers <= 1) {
            run(0, 1);
        } else {
            std::vector<std::thread> threads;
            for (std::size_t w = 0; w < workers; ++w) threads.emplace_back(run, w, workers);
            for (auto& t : threads) t.join();
        }
        Outputs out;
        for (std::size_t i = 0; i < work.size(); ++i) out[work[i].first] = std::move(results[i]);
        return out;
    }

    // ---------------------------------------------------------------- run log
    std::string run_log() const {
        nlohmann::json d = nlohmann::json::object();
        d["format"] = "j_neuro.runlog";
        d["spec_version"] = kSpecVersion;
        d["population"] = nlohmann::json{{"depth_type", to_string(Net<D>::depth_type)},
                                          {"in_qty_max", in_qty_max_}, {"n_inputs", n_inputs_},
                                          {"n_outputs", n_outputs_}, {"path_mix", path_mix_}, {"max_run", max_run_},
                                          {"priority", to_string(priority_)},
                                          {"seed", seed_}};
        d["resumed_from"] = resumed_from_ ? nlohmann::json(*resumed_from_) : nlohmann::json(nullptr);
        d["profiles"] = profiles_;
        d["ops"] = ops_;
        return d.dump();
    }

    // ---------------------------------------------------------------- .jpnet files (spec/09)
    /// Write <name>.jpnet plus <name>_nets/<id>.jnet beside it.
    void save(const std::filesystem::path& path, bool include_log = false) const {
        const std::string name = path.stem().string();
        const std::filesystem::path net_dir = path.parent_path() / (name + "_nets");
        std::filesystem::create_directories(net_dir);
        detail::Writer w;
        w.header("JPOP");
        w.u(static_cast<std::uint8_t>(priority_), 1);
        w.u(static_cast<std::uint8_t>(Net<D>::depth_type), 1);
        w.u(in_qty_max_, 4);
        w.u(n_inputs_, 4);
        w.u(n_outputs_, 4);
        w.u(path_mix_, 1);
        w.u(max_run_, 1);
        w.u(seed_, 8);
        for (std::uint64_t s : rng_.get_state()) w.u(s, 8);
        w.u(generation_, 8);
        w.u(next_id_, 8);
        w.u(nets_.size(), 8);
        for (const auto& [id, net] : nets_) {
            const Bytes data = to_bytes(net);
            write_file(net_dir / (std::to_string(id) + ".jnet"), data);
            const std::string rel = name + "_nets/" + std::to_string(id) + ".jnet";
            w.u(id, 8);
            w.u(rel.size(), 2);
            w.raw(rel);
            const Digest h = sha256(data);
            w.raw(h.data(), h.size());
        }
        if (include_log) {
            const std::string log = run_log();
            w.u(1, 1);
            w.u(log.size(), 8);
            w.raw(log);
        } else {
            w.u(0, 1);
        }
        write_file(path, w.finish());
    }

    /// Load a .jpnet, verifying every .jnet hash; continues exactly where it was saved.
    static Population load(const std::filesystem::path& path) {
        const Bytes data = read_file(path);
        detail::Reader r(data, "jpnet");
        const bool legacy = r.header("JPOP");  // 0.5: nets convert, the log is not kept (spec/11)
        const auto priority = static_cast<Priority>(r.enum_byte(3, "priority"));
        if (static_cast<DepthType>(r.enum_byte(3, "depth_type")) != Net<D>::depth_type)
            throw ValidationError("jpnet: depth_type differs from this population type");
        const auto q = static_cast<std::uint32_t>(r.u(4));
        const auto n_in = static_cast<std::uint32_t>(r.u(4));
        const auto n_out = static_cast<std::uint32_t>(r.u(4));
        const auto path_mix = static_cast<std::uint32_t>(r.u(1));
        const auto max_run = static_cast<std::uint32_t>(r.u(1));
        const std::uint64_t seed = r.u(8);
        std::array<std::uint64_t, 4> state;
        for (auto& s : state) s = r.u(8);
        const std::uint64_t generation = r.u(8), next_id = r.u(8), count = r.u(8);
        Population pop(q, n_in, n_out, priority, seed, path_mix, max_run);
        pop.rng_.set_state(state);
        pop.generation_ = generation;
        pop.next_id_ = next_id;
        std::optional<std::uint64_t> last;
        for (std::uint64_t k = 0; k < count; ++k) {
            const std::uint64_t id = r.u(8);
            const auto len = static_cast<std::size_t>(r.u(2));
            const std::uint8_t* p = r.take(len);
            const std::string rel(reinterpret_cast<const char*>(p), len);
            const std::uint8_t* digest = r.take(32);
            if ((last && id <= *last) || id >= next_id)
                throw ValidationError("jpnet: net ids must ascend and be below next_id");
            last = id;
            const std::filesystem::path file = path.parent_path() / rel;
            if (!std::filesystem::is_regular_file(file)) throw ValidationError("jpnet: missing net file " + rel);
            const Bytes raw = read_file(file);
            const Digest h = sha256(raw);
            if (!std::equal(h.begin(), h.end(), digest)) throw ValidationError("jpnet: hash mismatch for " + rel);
            AnyNet any = from_bytes(raw);
            auto* net = std::get_if<Net<D>>(&any);
            if (!net || net->in_qty_max != q || net->n_inputs != n_in || net->n_outputs != n_out ||
                net->path_mix != path_mix || net->max_run != max_run)
                throw ValidationError("jpnet: " + rel + " does not match the population's parameters");
            pop.nets_.emplace(id, std::move(*net));
        }
        const std::uint8_t flag = r.enum_byte(2, "log flag");
        std::optional<nlohmann::json> log;
        if (flag == 1) {
            const auto len = static_cast<std::size_t>(r.u(8));
            const std::uint8_t* p = r.take(len);
            try {
                log = nlohmann::json::parse(std::string(reinterpret_cast<const char*>(p), len));
            } catch (const nlohmann::json::exception& e) {
                throw ValidationError(std::string("jpnet: invalid run log: ") + e.what());
            }
        }
        r.done();
        pop.snapshot_hash_ = bytes_to_hex(sha256(data));
        if (log && !legacy) {
            check_format(*log, "j_neuro.runlog");
            if (log->contains("resumed_from") && !log->at("resumed_from").is_null())
                pop.resumed_from_ = log->at("resumed_from").get<std::string>();
            pop.profiles_ = log->at("profiles");
            for (auto& p : pop.profiles_) pop.profile_keys_.push_back(Profile::from_json(p).to_json().dump());
            pop.ops_ = log->at("ops");
        } else {
            pop.resumed_from_ = pop.snapshot_hash_;
        }
        return pop;
    }

    /// Re-execute a run log; reproduces ids, nets, generation and shared-stream state.
    /// A log that resumed from a .jpnet needs `base`: the population loaded from that file.
    static Population replay(const std::string& text, const Population* base = nullptr) {
        nlohmann::json d;
        try {
            d = nlohmann::json::parse(text);
        } catch (const nlohmann::json::exception& e) {
            throw ValidationError(std::string("invalid JSON: ") + e.what());
        }
        check_format(d, "j_neuro.runlog");
        const auto& pp = d.at("population");
        if (parse_depth_type(pp.at("depth_type").get<std::string>()) != Net<D>::depth_type)
            throw ValidationError("run log depth_type differs from this population type");
        const bool resumed = d.contains("resumed_from") && !d.at("resumed_from").is_null();
        Population pop = resumed ? Population(3, 1, 1) : Population(pp.at("in_qty_max").get<std::uint32_t>(),
                       pp.at("n_inputs").get<std::uint32_t>(), pp.at("n_outputs").get<std::uint32_t>(),
                       parse_priority(pp.at("priority").get<std::string>()), pp.at("seed").get<std::uint64_t>(),
                       pp.at("path_mix").get<std::uint32_t>(), pp.at("max_run").get<std::uint32_t>());
        if (resumed) {
            const std::string h = d.at("resumed_from").get<std::string>();
            if (!base || base->snapshot_hash_ != h)
                throw ValidationError("run log resumes from a .jpnet: pass the population loaded from that file as base");
            pop = base->copy_state();
            pop.resumed_from_ = h;
        }
        std::vector<Profile> profiles;
        for (auto& p : d.at("profiles")) profiles.push_back(Profile::from_json(p));
        auto seed_of = [](const nlohmann::json& op) -> std::optional<std::uint64_t> {
            if (op.at("seed").is_null()) return std::nullopt;
            return op.at("seed").get<std::uint64_t>();
        };
        auto expect = [](std::uint64_t got, const nlohmann::json& want) {
            if (got != want.get<std::uint64_t>()) throw ValidationError("run log: id mismatch during replay");
        };
        for (auto& op : d.at("ops")) {
            const std::string kind = op.at("op").get<std::string>();
            if (kind == "create") {
                auto s = seed_of(op);
                auto got = pop.create_impl(op.at("count").get<std::uint64_t>(), s ? "fresh" : "shared", s);
                if (got != op.at("ids").get<std::vector<std::uint64_t>>())
                    throw ValidationError("run log: id mismatch during replay");
            } else if (kind == "add") {
                AnyNet any = from_bytes(hex_to_bytes(op.at("jnet").get<std::string>()));
                auto* net = std::get_if<Net<D>>(&any);
                if (!net) throw ValidationError("run log: added genome has a different depth type");
                expect(pop.add(*net), op.at("new_id"));
            } else if (kind == "mutate") {
                auto s = seed_of(op);
                const Profile& prof = profiles.at(op.at("profile").get<std::size_t>());
                if ((s ? "fresh" : "shared") != prof.rng_mode())
                    throw ValidationError("run log: mutate seed does not match the profile's rng_mode");
                expect(pop.mutate_impl(op.at("id").get<std::uint64_t>(), prof, op.at("times").get<std::uint32_t>(), s),
                       op.at("new_id"));
            } else if (kind == "clone") {
                expect(pop.clone(op.at("id").get<std::uint64_t>()), op.at("new_id"));
            } else if (kind == "inject_diversity") {
                auto s = seed_of(op);
                pop.diversity_impl(op.at("id").get<std::uint64_t>(), op.at("pct").get<double>(),
                                   op.at("sigma_reset").get<double>(), s ? "fresh" : "shared", s);
            } else if (kind == "remove") {
                pop.remove(op.at("id").get<std::uint64_t>());
            } else if (kind == "next_generation") {
                pop.next_generation();
            } else {
                throw ValidationError("run log: unknown op " + kind);
            }
        }
        return pop;
    }

private:
    std::uint32_t in_qty_max_, n_inputs_, n_outputs_, path_mix_, max_run_;
    Priority priority_;
    std::uint64_t seed_;
    Rng rng_;
    std::map<std::uint64_t, Net<D>> nets_;
    std::uint64_t next_id_ = 0;
    std::uint64_t generation_ = 0;
    nlohmann::json profiles_ = nlohmann::json::array();
    std::vector<std::string> profile_keys_;
    nlohmann::json ops_ = nlohmann::json::array();
    std::optional<std::string> resumed_from_;   // hex SHA-256 of the .jpnet this run resumed from
    std::optional<std::string> snapshot_hash_;  // hex SHA-256 of the .jpnet this object was loaded from
    MutationCounts mutation_counts_;            // informative; not saved

    /// Same nets, ids, generation and shared-stream state; an empty journal.
    Population copy_state() const {
        Population p(in_qty_max_, n_inputs_, n_outputs_, priority_, seed_, path_mix_, max_run_);
        p.rng_.set_state(rng_.get_state());
        p.nets_ = nets_;
        for (auto& [k, v] : p.nets_) v.reset();
        p.next_id_ = next_id_;
        p.generation_ = generation_;
        p.snapshot_hash_ = snapshot_hash_;
        return p;
    }

    struct OpRng {
        Rng own;
        Rng* use;
        std::optional<std::uint64_t> seed;
    };

    OpRng op_rng(const std::string& mode, std::optional<std::uint64_t> recorded) {
        if (mode == "shared") return OpRng{Rng(0), &rng_, std::nullopt};
        if (mode != "fresh") throw ValidationError("unknown rng_mode " + mode);
        const std::uint64_t s = recorded ? *recorded : fresh_seed();
        return OpRng{Rng(s), nullptr, s};
    }
    static Rng& rng_of(OpRng& o) { return o.use ? *o.use : o.own; }
    static nlohmann::json seed_json(const std::optional<std::uint64_t>& s) {
        return s ? nlohmann::json(*s) : nlohmann::json(nullptr);
    }

    std::uint64_t store(Net<D> net) {
        const std::uint64_t id = next_id_++;
        nets_.emplace(id, std::move(net));
        return id;
    }

    std::size_t profile_index(const Profile& p) {
        const nlohmann::json j = p.to_json();
        const std::string key = j.dump();
        auto it = std::find(profile_keys_.begin(), profile_keys_.end(), key);
        if (it != profile_keys_.end()) return static_cast<std::size_t>(it - profile_keys_.begin());
        profile_keys_.push_back(key);
        profiles_.push_back(j);
        return profile_keys_.size() - 1;
    }

    std::vector<std::uint64_t> create_impl(std::uint64_t count, const std::string& mode,
                                           std::optional<std::uint64_t> recorded) {
        OpRng o = op_rng(mode, recorded);
        std::vector<std::uint64_t> ids;
        for (std::uint64_t i = 0; i < count; ++i)
            ids.push_back(store(Net<D>::create(n_inputs_, n_outputs_, in_qty_max_, priority_, rng_of(o), path_mix_, max_run_)));
        ops_.push_back(nlohmann::json{{"op", "create"}, {"count", count}, {"ids", ids}, {"seed", seed_json(o.seed)}});
        return ids;
    }

    std::uint64_t mutate_impl(std::uint64_t id, const Profile& profile, std::uint32_t times,
                              std::optional<std::uint64_t> recorded) {
        const Net<D>& parent = nets_.at(id);
        if (times < 1) throw std::invalid_argument("times must be >= 1");
        OpRng o = op_rng(profile.rng_mode(), recorded);
        const bool grow_gen = is_growth_generation(seed_, generation_, profile);
        Net<D> child = ::jneuro::mutate(parent, profile, generation_, rng_of(o), &mutation_counts_, grow_gen);
        for (std::uint32_t t = 1; t < times; ++t)
            child = ::jneuro::mutate(child, profile, generation_, rng_of(o), &mutation_counts_, grow_gen);
        const std::uint64_t nid = store(std::move(child));
        ops_.push_back(nlohmann::json{{"op", "mutate"}, {"id", id}, {"times", times},
                                      {"profile", profile_index(profile)}, {"new_id", nid},
                                      {"seed", seed_json(o.seed)}});
        return nid;
    }

    void diversity_impl(std::uint64_t id, double pct, double sigma_reset, const std::string& mode,
                        std::optional<std::uint64_t> recorded) {
        Net<D>& net = nets_.at(id);
        OpRng o = op_rng(mode, recorded);
        net = ::jneuro::inject_diversity(net, pct, rng_of(o), sigma_reset);
        ops_.push_back(nlohmann::json{{"op", "inject_diversity"}, {"id", id}, {"pct", pct},
                                      {"sigma_reset", sigma_reset}, {"seed", seed_json(o.seed)}});
    }
};

}  // namespace jneuro
