#include <jneuro/profile.hpp>

#include <set>

namespace jneuro {

using json = nlohmann::json;

void check_format(const json& d, const char* fmt, bool accept_05);  // io.cpp

namespace {
const std::map<std::string, double>& numeric_defaults() {
    static const std::map<std::string, double> d = {
        {"op_change_max_pct", 1.0}, {"edit_max_pct", 5.0}, {"growth_max_pct", 1.5},
        {"removal_max_pct", 1.0},   {"depth_shift_max_pct", 5.0}, {"sigma_weight", 0.1},
        {"sign_flip_prob", 0.02},   {"sigma_k", 1.0}, {"sigma_slots", 1.0},
        {"rewire_prob", 0.05},       {"insert_inversion_prob", 0.05}, {"min_age_for_removal", 5.0},
        {"slot_retarget_prob", 0.05}, {"width_peak", 0.5}, {"tap_peak", 0.25}, {"link_max_span", 0.25},
        {"growth_gen_prob", 1.0}, {"growth_net_prob", 1.0}, {"slot_add_prob", 0.75},
        {"open_slot_fill_pct", 0.0}, {"history_read_prob", 0.0}};
    return d;
}
const std::set<std::string>& overridable() {
    static const std::set<std::string> s = {
        "op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct",
        "depth_shift_max_pct", "sigma_weight", "sign_flip_prob", "sigma_k", "sigma_slots",
        "rewire_prob", "insert_inversion_prob", "min_age_for_removal", "slot_retarget_prob"};
    return s;
}
// Key order of the overall section as written by to_json (matches spec/09).
const char* const kOverallOrder[] = {
    "op_change_max_pct", "edit_max_pct", "growth_max_pct", "removal_max_pct", "depth_shift_max_pct",
    "depth_shift_mode", "sigma_weight", "sign_flip_prob", "sigma_k", "sigma_slots", "rewire_prob",
    "insert_inversion_prob", "min_age_for_removal", "add_method_weights", "slot_fill_type_weights",
    "rng_mode", "growth_budget", "constant_folding", "insert_depth", "slot_retarget_prob", "width_bias",
    "width_peak", "tap_peak", "prune", "link_max_span", "growth_gen_prob", "growth_net_prob", "no_growth_swap",
    "slot_edit", "slot_add_prob", "open_slot_fill_pct", "history_read_prob", "gc_replace"};
const char* const kMethods[] = {"slot_fill", "v_insert", "s_insert", "m_after_m", "v_tap", "s_tap", "m_tap",
                                "m_clone",   "m_insert", "v_link",   "s_link"};
constexpr std::array<double, 11> kMethodDefaults{0, 10, 5, 5, 10, 10, 5, 60, 20, 24, 24};
const char* const kTypes[] = {"v", "s", "m"};

double number(const json& v, const std::string& k) {
    if (!v.is_number()) throw ValidationError("profile field " + k + " must be a number");
    return v.get<double>();
}
}  // namespace

Profile::Profile() : overall_(numeric_defaults()) {}

double Profile::overall(const std::string& f) const {
    auto it = overall_.find(f);
    if (it == overall_.end()) throw ValidationError("unknown profile field " + f);
    return it->second;
}

double Profile::effective(const std::string& f, NeuronType t) const {
    const auto& m = per_type_[static_cast<std::size_t>(t)];
    auto it = m.find(f);
    return it != m.end() ? it->second : overall(f);
}

bool Profile::has_override(const std::string& f, NeuronType t) const {
    return per_type_[static_cast<std::size_t>(t)].count(f) > 0;
}

double Profile::override_value(const std::string& f, NeuronType t) const {
    return per_type_[static_cast<std::size_t>(t)].at(f);
}

void Profile::set_overall(const std::string& f, double v) {
    if (!numeric_defaults().count(f)) throw ValidationError("unknown profile field " + f);
    if ((f == "width_peak" || f == "tap_peak") && !(v > 0.0 && v < 1.0)) throw ValidationError(f + " must be in (0, 1)");
    if (f == "link_max_span" && !(v > 0.0 && v <= 1.0)) throw ValidationError("link_max_span must be in (0, 1]");
    if ((f == "growth_gen_prob" || f == "growth_net_prob" || f == "slot_add_prob") && !(v >= 0.0 && v <= 1.0))
        throw ValidationError(f + " must be in [0, 1]");
    if (f == "open_slot_fill_pct" && !(v >= 0.0 && v <= 100.0)) throw ValidationError(f + " must be in [0, 100]");
    if (f == "history_read_prob" && !(v >= 0.0 && v <= 1.0)) throw ValidationError(f + " must be in [0, 1]");
    overall_[f] = v;
}

void Profile::merge_overall(const json& body) {
    if (!body.is_object()) throw ValidationError("overall must be an object");
    for (auto& [k, v] : body.items()) {
        if (numeric_defaults().count(k)) {
            overall_[k] = v.is_null() ? numeric_defaults().at(k) : number(v, k);
            if ((k == "width_peak" || k == "tap_peak") && !(overall_[k] > 0.0 && overall_[k] < 1.0))
                throw ValidationError(k + " must be in (0, 1)");
            if (k == "link_max_span" && !(overall_[k] > 0.0 && overall_[k] <= 1.0))
                throw ValidationError("link_max_span must be in (0, 1]");
            if ((k == "growth_gen_prob" || k == "growth_net_prob" || k == "slot_add_prob") &&
                !(overall_[k] >= 0.0 && overall_[k] <= 1.0))
                throw ValidationError(k + " must be in [0, 1]");
            if (k == "history_read_prob" && !(overall_[k] >= 0.0 && overall_[k] <= 1.0))
                throw ValidationError(k + " must be in [0, 1]");
            if (k == "open_slot_fill_pct" && !(overall_[k] >= 0.0 && overall_[k] <= 100.0))
                throw ValidationError(k + " must be in [0, 100]");
        } else if (k == "depth_shift_mode") {
            if (v.is_null()) { mode_ = "uniform"; continue; }
            if (!v.is_string() || (v != "uniform" && v != "center"))
                throw ValidationError("depth_shift_mode must be uniform or center");
            mode_ = v.get<std::string>();
        } else if (k == "rng_mode") {
            if (v.is_null()) { rng_mode_ = "fresh"; continue; }
            if (!v.is_string() || (v != "fresh" && v != "shared"))
                throw ValidationError("rng_mode must be fresh or shared");
            rng_mode_ = v.get<std::string>();
        } else if (k == "growth_budget") {
            if (v.is_null()) { growth_budget_ = "up_to"; continue; }
            if (!v.is_string() || (v != "up_to" && v != "exact" && v != "fill"))
                throw ValidationError("growth_budget must be up_to, exact or fill");
            growth_budget_ = v.get<std::string>();
        } else if (k == "constant_folding") {
            if (v.is_null()) { constant_folding_ = "off"; continue; }
            if (!v.is_string() || (v != "off" && v != "on"))
                throw ValidationError("constant_folding must be off or on");
            constant_folding_ = v.get<std::string>();
        } else if (k == "gc_replace") {
            if (v.is_null()) { gc_replace_ = "off"; continue; }
            if (!v.is_string() || (v != "off" && v != "on")) throw ValidationError("gc_replace must be off or on");
            gc_replace_ = v.get<std::string>();
        } else if (k == "insert_depth") {
            if (v.is_null()) { insert_depth_ = "midpoint"; continue; }
            if (!v.is_string() || (v != "uniform" && v != "midpoint"))
                throw ValidationError("insert_depth must be uniform or midpoint");
            insert_depth_ = v.get<std::string>();
        } else if (k == "slot_edit") {
            if (v.is_null()) { slot_edit_ = "toggle"; continue; }
            if (!v.is_string() || (v != "toggle" && v != "biased")) throw ValidationError("slot_edit must be toggle or biased");
            slot_edit_ = v.get<std::string>();
        } else if (k == "no_growth_swap") {
            if (v.is_null()) { no_growth_swap_ = "on"; continue; }
            if (!v.is_string() || (v != "off" && v != "on")) throw ValidationError("no_growth_swap must be off or on");
            no_growth_swap_ = v.get<std::string>();
        } else if (k == "prune") {
            if (v.is_null()) { prune_ = "static"; continue; }
            if (!v.is_string() || (v != "any" && v != "static")) throw ValidationError("prune must be any or static");
            prune_ = v.get<std::string>();
        } else if (k == "width_bias") {
            if (v.is_null()) { width_bias_ = "deficit"; continue; }
            if (!v.is_string() || (v != "none" && v != "center" && v != "deficit"))
                throw ValidationError("width_bias must be none, center or deficit");
            width_bias_ = v.get<std::string>();
        } else if (k == "add_method_weights" || k == "slot_fill_type_weights") {
            const bool am = k == "add_method_weights";
            if (v.is_null()) {
                if (am) amw_ = kMethodDefaults; else sftw_ = {1, 1, 1};
                continue;
            }
            if (!v.is_object()) throw ValidationError(k + " must be an object");
            for (auto& [kk, vv] : v.items()) {
                std::size_t idx = 99;
                if (am) { for (std::size_t i = 0; i < 11; ++i) if (kk == kMethods[i]) idx = i; }
                else    { for (std::size_t i = 0; i < 3; ++i) if (kk == kTypes[i]) idx = i; }
                if (idx == 99) throw ValidationError("unknown key " + kk + " in " + k);
                const double def = am ? kMethodDefaults[idx] : 1.0;
                double val = vv.is_null() ? def : number(vv, kk);
                if (am) amw_[idx] = val; else sftw_[idx] = val;
            }
        } else {
            throw ValidationError("unknown profile field " + k);
        }
    }
}

void Profile::set_override(std::size_t t, const std::string& k, const json& v) {
    if (!overridable().count(k)) throw ValidationError(k + " cannot be overridden per type");
    if (v.is_null()) per_type_[t].erase(k);
    else per_type_[t][k] = number(v, k);
}

void Profile::apply_patch(const json& patch) {
    if (!patch.is_object()) throw ValidationError("patch must be an object");
    for (auto& [sec, body] : patch.items()) {
        if (sec == "overall") { merge_overall(body); continue; }
        if (sec == "format" || sec == "spec_version") continue;
        std::size_t t = 99;
        for (std::size_t i = 0; i < 3; ++i) if (sec == kTypes[i]) t = i;
        if (t == 99) throw ValidationError("unknown profile section " + sec);
        if (body.is_null()) continue;
        if (!body.is_object()) throw ValidationError("profile section must be an object");
        for (auto& [k, v] : body.items()) set_override(t, k, v);
    }
}

json Profile::to_json() const {
    json o = json::object();
    json overall = json::object();
    for (const char* k : kOverallOrder) {
        std::string key = k;
        if (key == "depth_shift_mode") overall[key] = mode_;
        else if (key == "rng_mode") overall[key] = rng_mode_;
        else if (key == "growth_budget") overall[key] = growth_budget_;
        else if (key == "constant_folding") overall[key] = constant_folding_;
        else if (key == "gc_replace") overall[key] = gc_replace_;
        else if (key == "insert_depth") overall[key] = insert_depth_;
        else if (key == "width_bias") overall[key] = width_bias_;
        else if (key == "prune") overall[key] = prune_;
        else if (key == "no_growth_swap") overall[key] = no_growth_swap_;
        else if (key == "slot_edit") overall[key] = slot_edit_;
        else if (key == "add_method_weights") {
            json w = json::object();
            for (std::size_t i = 0; i < 11; ++i) w[kMethods[i]] = amw_[i];
            overall[key] = w;
        } else if (key == "slot_fill_type_weights") {
            json w = json::object();
            for (std::size_t i = 0; i < 3; ++i) w[kTypes[i]] = sftw_[i];
            overall[key] = w;
        } else {
            overall[key] = overall_.at(key);
        }
    }
    o["format"] = "j_neuro.profile";
    o["spec_version"] = "0.6.0";
    o["overall"] = overall;
    for (std::size_t t = 0; t < 3; ++t) {
        json sec = json::object();
        for (auto& [k, v] : per_type_[t]) sec[k] = v;
        o[kTypes[t]] = sec;
    }
    return o;
}

Profile Profile::from_json(const json& d) {
    Profile p;
    json patch = json::object();
    if (d.contains("overall")) patch["overall"] = d.at("overall");
    for (const char* t : kTypes) if (d.contains(t)) patch[t] = d.at(t);
    p.apply_patch(patch);
    return p;
}

std::string profile_to_json(const Profile& p) { return p.to_json().dump(); }

Profile profile_from_json(const std::string& text) {
    json d;
    try {
        d = json::parse(text);
    } catch (const json::exception& e) {
        throw ValidationError(std::string("invalid JSON: ") + e.what());
    }
    check_format(d, "j_neuro.profile", true);  // 0.5 profiles lack only history_read_prob
    return Profile::from_json(d);
}

}  // namespace jneuro
