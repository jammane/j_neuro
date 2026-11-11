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
               Priority priority = Priority::Balanced, std::optional<std::uint64_t> seed = std::nullopt)
        : in_qty_max_(in_qty_max), n_inputs_(n_inputs), n_outputs_(n_outputs), priority_(priority),
          seed_(seed ? *seed : fresh_seed()), rng_(seed_) {
        detail::check_params(n_inputs, n_outputs, in_qty_max);
    }

    // ---------------------------------------------------------------- inspection
    std::uint64_t generation() const { return generation_; }
    std::uint64_t seed() const { return seed_; }
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
        if (net.in_qty_max != in_qty_max_ || net.n_inputs != n_inputs_ || net.n_outputs != n_outputs_)
            throw ValidationError("net parameters differ from the population's");
        net.validate();
        const std::uint64_t id = store(net.copy_genome());
        ops_.push_back(nlohmann::json{{"op", "add"}, {"new_id", id},
                                      {"genome", nlohmann::json::parse(genome_to_json(net))}});
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
                                          {"n_outputs", n_outputs_}, {"priority", to_string(priority_)},
                                          {"seed", seed_}};
        d["profiles"] = profiles_;
        d["ops"] = ops_;
        return d.dump();
    }

    /// Re-execute a run log; reproduces ids, genomes, generation and shared-stream state.
    static Population replay(const std::string& text) {
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
        Population pop(pp.at("in_qty_max").get<std::uint32_t>(), pp.at("n_inputs").get<std::uint32_t>(),
                       pp.at("n_outputs").get<std::uint32_t>(), parse_priority(pp.at("priority").get<std::string>()),
                       pp.at("seed").get<std::uint64_t>());
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
                AnyNet any = genome_from_json(op.at("genome").dump());
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
    std::uint32_t in_qty_max_, n_inputs_, n_outputs_;
    Priority priority_;
    std::uint64_t seed_;
    Rng rng_;
    std::map<std::uint64_t, Net<D>> nets_;
    std::uint64_t next_id_ = 0;
    std::uint64_t generation_ = 0;
    nlohmann::json profiles_ = nlohmann::json::array();
    std::vector<std::string> profile_keys_;
    nlohmann::json ops_ = nlohmann::json::array();

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
            ids.push_back(store(Net<D>::create(n_inputs_, n_outputs_, in_qty_max_, priority_, rng_of(o))));
        ops_.push_back(nlohmann::json{{"op", "create"}, {"count", count}, {"ids", ids}, {"seed", seed_json(o.seed)}});
        return ids;
    }

    std::uint64_t mutate_impl(std::uint64_t id, const Profile& profile, std::uint32_t times,
                              std::optional<std::uint64_t> recorded) {
        const Net<D>& parent = nets_.at(id);
        if (times < 1) throw std::invalid_argument("times must be >= 1");
        OpRng o = op_rng(profile.rng_mode(), recorded);
        Net<D> child = ::jneuro::mutate(parent, profile, generation_, rng_of(o));
        for (std::uint32_t t = 1; t < times; ++t) child = ::jneuro::mutate(child, profile, generation_, rng_of(o));
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
