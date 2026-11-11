// Mutation profile (spec/06): overall values plus per-type overrides.
#pragma once

#include <jneuro/types.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <map>
#include <string>

namespace jneuro {

class Profile {
public:
    static Profile defaults() { return Profile{}; }

    /// Per-type override if present, else the overall value (numeric fields).
    double effective(const std::string& field, NeuronType t) const;
    bool has_override(const std::string& field, NeuronType t) const;
    double overall(const std::string& field) const;
    double override_value(const std::string& field, NeuronType t) const;
    void set_overall(const std::string& field, double v);

    const std::string& depth_shift_mode() const { return mode_; }
    /// Order: slot_fill, v_insert, s_insert, m_after_m.
    const std::array<double, 4>& add_method_weights() const { return amw_; }
    /// Order: v, s, m.
    const std::array<double, 3>& slot_fill_type_weights() const { return sftw_; }

    /// JSON Merge Patch (RFC 7386); null removes a per-type override.
    void apply_patch(const nlohmann::json& patch);
    nlohmann::json to_json() const;
    static Profile from_json(const nlohmann::json& d);

private:
    Profile();
    void merge_overall(const nlohmann::json& body);
    void set_override(std::size_t t, const std::string& k, const nlohmann::json& v);

    std::map<std::string, double> overall_;
    std::string mode_ = "uniform";
    std::array<double, 4> amw_{40, 30, 20, 10};
    std::array<double, 3> sftw_{1, 1, 1};
    std::array<std::map<std::string, double>, 3> per_type_;
};

std::string profile_to_json(const Profile& p);

/// Seed, starting profile, every profile change and every diversity injection (spec/09).
class RunLog {
public:
    struct Diversity {
        std::uint64_t generation;
        double pct;
        double sigma_reset;
    };

    RunLog(std::uint64_t seed, Profile initial) : seed_(seed), initial_(std::move(initial)) {}

    void record(std::uint64_t generation, const nlohmann::json& patch);
    void record_diversity(std::uint64_t generation, double pct, double sigma_reset = 1.0);
    /// Profile in effect for mutations of `generation`.
    Profile profile_at(std::uint64_t generation) const;
    std::vector<Diversity> diversity_events() const;
    std::uint64_t seed() const { return seed_; }

    std::string to_json() const;
    static RunLog from_json(const std::string& text);

private:
    std::uint64_t seed_;
    Profile initial_;
    nlohmann::json entries_ = nlohmann::json::array();
};
Profile profile_from_json(const std::string& text);

}  // namespace jneuro
