// How a net categorizes an image, and how a categorization is scored (examples/categorize).
//
// The net has one output, read as an unsigned 16-bit category number: floor(clamp(y, 0, 65535)).
// 0 means "uncategorized". Nothing says which categories are right; the score only asks that
// they be usable:
//   consistency   changed copies of an image land where the image does (the main term): a few
//                 pixels flipped; shifted one pixel; and (--placement) the same character in the
//                 same font at another size or alignment, so placement is not what decides
//   enough        a category counts only if it holds between --min-share and --max-share of the
//                 images; fewer than --min-cats counted categories costs 0.25 each
//   not too many  the minimum share: images in categories too small to count are scored as
//                 uncategorized, so one category per image gains nothing
//   not too few   the maximum share: a category holding more than that is not counted either,
//                 so one big category cannot take most of the images
//   uncategorized allowed up to --max-uncat of the images, then costs 2 per unit over
#pragma once

#include "glyphs.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace cat {

using D = std::uint32_t;
using jneuro::Net;
constexpr int kUncat = -1;

struct ScoreParams {
    double min_share = 0.01, max_share = 0.10, max_uncat = 0.20, noise = 0.02;
    int min_cats = 3;
    bool placement = true;
};

/// The category the net gives an image: its output as an unsigned 16-bit number, 0 = kUncat.
inline int categorize(Net<D>& net, const Image& im) {
    net.reset_state();  // each image stands alone
    const double y = net.step(inputs_of(im))[0];
    const int c = static_cast<int>(std::floor(std::clamp(y, 0.0, 65535.0)));
    return c == 0 ? kUncat : c;
}

/// An image and its changed copies.
struct Sample {
    Instance x;
    Image base, flip, shift, place;  // place: the same character and font, another size or alignment
};

inline Sample make_sample(GlyphSet& glyphs, std::uint64_t idx, std::uint64_t seed, double noise) {
    Sample s;
    s.x = glyphs.decode(idx);
    s.base = glyphs.image(s.x);
    Rng r(seed ^ (idx * 0x9E3779B97F4A7C15ULL));  // the same copies for every net
    s.flip = flipped(s.base, noise, r);
    s.shift = shifted(s.base, static_cast<int>(r.below(4)));
    Instance y = s.x;
    const std::uint64_t others = kSizes.size() * kAligns - 1;  // every other (size, alignment)
    std::uint64_t pick = r.below(others);
    if (pick >= static_cast<std::uint64_t>(s.x.size) * kAligns + static_cast<std::uint64_t>(s.x.align)) ++pick;
    y.size = static_cast<int>(pick / kAligns);
    y.align = static_cast<int>(pick % kAligns);
    s.place = glyphs.image(y);
    return s;
}

struct Result {
    double score = 0, consistency = 0, uncat = 0;
    int cats = 0;                 // counted categories
    std::vector<int> category;    // per sample: the effective category (kUncat included)
    std::map<int, int> share;     // per category: images (before the share limits)
};

inline Result score(Net<D>& net, const std::vector<Sample>& samples, const ScoreParams& p) {
    Result r;
    const std::size_t n = samples.size();
    std::vector<int> base(n), flip(n), shift(n), place(n);
    for (std::size_t i = 0; i < n; ++i) {
        base[i] = categorize(net, samples[i].base);
        flip[i] = categorize(net, samples[i].flip);
        shift[i] = categorize(net, samples[i].shift);
        if (p.placement) place[i] = categorize(net, samples[i].place);
    }
    for (int c : base) if (c != kUncat) ++r.share[c];
    const double need = p.min_share * static_cast<double>(n), cap = p.max_share * static_cast<double>(n);
    std::set<int> counted;
    for (const auto& [c, sh] : r.share)
        if (sh >= need && sh <= cap) counted.insert(c);
    r.cats = static_cast<int>(counted.size());
    auto eff = [&](int c) { return c != kUncat && counted.count(c) ? c : kUncat; };
    double agree = 0;
    int uncat = 0;
    r.category.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const int b = eff(base[i]);
        r.category[i] = b;
        uncat += b == kUncat;
        if (p.placement) agree += ((eff(flip[i]) == b) + (eff(shift[i]) == b) + (eff(place[i]) == b)) / 3.0;
        else agree += 0.5 * ((eff(flip[i]) == b) + (eff(shift[i]) == b));
    }
    r.consistency = n ? agree / static_cast<double>(n) : 0.0;
    r.uncat = n ? static_cast<double>(uncat) / static_cast<double>(n) : 0.0;
    r.score = r.consistency - 2.0 * std::max(0.0, r.uncat - p.max_uncat) - 0.25 * std::max(0, p.min_cats - r.cats);
    return r;
}

}  // namespace cat
