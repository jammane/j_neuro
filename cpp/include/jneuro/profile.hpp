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
    /// "fresh" or "shared" (spec/11); used by Population::mutate.
    const std::string& rng_mode() const { return rng_mode_; }
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
    std::string rng_mode_ = "fresh";
    std::array<double, 4> amw_{40, 30, 20, 10};
    std::array<double, 3> sftw_{1, 1, 1};
    std::array<std::map<std::string, double>, 3> per_type_;
};

std::string profile_to_json(const Profile& p);

Profile profile_from_json(const std::string& text);

}  // namespace jneuro
