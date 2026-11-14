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
    /// "up_to", "exact" or "fill" (spec/06): whether the growth budget draws a fraction of its cap,
    /// and whether it counts attempts or neurons added.
    const std::string& growth_budget() const { return growth_budget_; }
    /// "off" or "on" (spec/07): fold constant neurons before garbage collection.
    const std::string& constant_folding() const { return constant_folding_; }
    /// "off" or "on": phase 6 grows as many neurons as the mutation removed (spec/06 replacement).
    const std::string& gc_replace() const { return gc_replace_; }
    /// Order: slot_fill, v_insert, s_insert, m_after_m, v_tap, s_tap, m_tap, m_clone, m_insert, v_link, s_link.
    const std::array<double, 11>& add_method_weights() const { return amw_; }
    /// "uniform" or "midpoint" (spec/06): where new nodes are placed within their depth range.
    const std::string& insert_depth() const { return insert_depth_; }
    /// "none" or "center" (spec/06): whether width-adding growth favors mid-depth receivers.
    const std::string& width_bias() const { return width_bias_; }
    /// "any" or "static" (spec/06): whether edits may cut only static (constant) branches.
    const std::string& prune() const { return prune_; }
    /// "off" or "on" (spec/06): an offspring that misses growth in a growth generation swaps s <-> v.
    const std::string& no_growth_swap() const { return no_growth_swap_; }
    /// "toggle" or "biased" (spec/06): how an m slot edit picks a slot.
    const std::string& slot_edit() const { return slot_edit_; }
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
    std::string growth_budget_ = "up_to";
    std::string constant_folding_ = "off";
    std::string gc_replace_ = "off";
    std::array<double, 11> amw_{0, 10, 5, 5, 10, 10, 5, 60, 20, 24, 24};
    std::string insert_depth_ = "midpoint";
    std::string width_bias_ = "deficit";
    std::string prune_ = "static";
    std::string no_growth_swap_ = "on";
    std::string slot_edit_ = "toggle";
    std::array<double, 3> sftw_{1, 1, 1};
    std::array<std::map<std::string, double>, 3> per_type_;
};

std::string profile_to_json(const Profile& p);

Profile profile_from_json(const std::string& text);

}  // namespace jneuro
