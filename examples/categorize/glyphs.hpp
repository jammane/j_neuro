// Test images for the categorization example: printable ASCII and Latin-1 characters rendered from many
// fonts at several sizes and alignments into 64x64 strictly black-and-white bitmaps.
//
// An instance is (character, font, size, alignment). Every instance of the set is visited in one
// seeded random order, so whatever point training stops at, characters, fonts, sizes and
// alignments have been seen in an even mix. Images are rendered on demand (glyph bitmaps cached).
#pragma once

#include <jneuro/jneuro.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cat {

using jneuro::Rng;
namespace fs = std::filesystem;

constexpr int kSide = 64;                     // images are kSide x kSide
constexpr int kPixels = kSide * kSide;
/// The symbols: printable ASCII '!'..'~' (space draws nothing) and printable Latin-1 U+00A1..U+00FF
/// (U+0080..U+009F are control codes, U+00A0 is a no-break space and U+00AD a soft hyphen, which
/// draw nothing in normal text). 188 code points.
inline const std::vector<int>& symbols() {
    static const std::vector<int> s = [] {
        std::vector<int> v;
        for (int c = 0x21; c <= 0x7E; ++c) v.push_back(c);
        for (int c = 0xA1; c <= 0xFF; ++c)
            if (c != 0xAD) v.push_back(c);
        return v;
    }();
    return s;
}
inline int n_chars() { return static_cast<int>(symbols().size()); }

/// A code point as UTF-8, for printing.
inline std::string utf8(int c) {
    std::string s;
    if (c < 0x80) s += static_cast<char>(c);
    else { s += static_cast<char>(0xC0 | (c >> 6)); s += static_cast<char>(0x80 | (c & 0x3F)); }
    return s;
}
constexpr std::array<int, 5> kSizes{12, 20, 28, 40, 56};  // pixel heights (ascent - descent)
constexpr int kAligns = 9;                    // 3 x 3: left / center / right, top / middle / bottom

using Image = std::vector<std::uint8_t>;      // kPixels values, 1 black, 0 white

/// The fonts used when none are given: many families, in several styles.
inline std::vector<fs::path> default_fonts() {
    const fs::path f = "/usr/share/fonts";
    return {f / "liberation-sans-fonts/LiberationSans-Regular.ttf",   f / "liberation-sans-fonts/LiberationSans-Bold.ttf",
            f / "liberation-sans-fonts/LiberationSans-Italic.ttf",    f / "liberation-sans-fonts/LiberationSans-BoldItalic.ttf",
            f / "liberation-serif-fonts/LiberationSerif-Regular.ttf", f / "liberation-serif-fonts/LiberationSerif-Bold.ttf",
            f / "liberation-serif-fonts/LiberationSerif-Italic.ttf",  f / "liberation-serif-fonts/LiberationSerif-BoldItalic.ttf",
            f / "liberation-mono-fonts/LiberationMono-Regular.ttf",   f / "liberation-mono-fonts/LiberationMono-BoldItalic.ttf",
            f / "open-sans/OpenSans-Light.ttf",                       f / "open-sans/OpenSans-Regular.ttf",
            f / "open-sans/OpenSans-ExtraBold.ttf",                   f / "open-sans/OpenSans-Italic.ttf",
            f / "adobe-source-code-pro-fonts/SourceCodePro-ExtraLight.otf",
            f / "adobe-source-code-pro-fonts/SourceCodePro-Regular.otf",
            f / "adobe-source-code-pro-fonts/SourceCodePro-Black.otf", f / "adobe-source-code-pro-fonts/SourceCodePro-It.otf",
            f / "google-crosextra-caladea-fonts/Caladea-Regular.ttf", f / "google-crosextra-caladea-fonts/Caladea-Bold.ttf",
            f / "google-crosextra-caladea-fonts/Caladea-Italic.ttf",  f / "google-carlito-fonts/Carlito-Regular.ttf",
            f / "google-carlito-fonts/Carlito-BoldItalic.ttf",        f / "google-droid-sans-fonts/DroidSans.ttf",
            f / "google-droid-sans-fonts/DroidSans-Bold.ttf",         f / "google-noto-vf/NotoSans[wght].ttf",
            f / "google-noto-vf/NotoSans-Italic[wght].ttf",           f / "google-noto-vf/NotoSerif[wght].ttf",
            f / "google-noto-vf/NotoSerif-Italic[wght].ttf",          f / "gdouros-symbola/Symbola.ttf"};
}

/// A font file list: one path per line (# comments), or the defaults.
inline std::vector<fs::path> read_font_list(const fs::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot read font list " + file.string());
    std::vector<fs::path> out;
    for (std::string line; std::getline(in, line);) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (!line.empty() && line[0] != '#') out.emplace_back(line);
    }
    return out;
}

struct Instance {
    int ch, font, size, align;
};

/// A rendered glyph: its black-and-white bitmap, cropped to its ink.
struct Glyph {
    int w = 0, h = 0;
    std::vector<std::uint8_t> bits;  // w * h
};

/// Renders glyphs with stb_truetype (glyphs.cpp); thresholded at 50% grey, so strictly black and white.
class GlyphSet {
public:
    explicit GlyphSet(const std::vector<fs::path>& fonts);
    ~GlyphSet();
    GlyphSet(const GlyphSet&) = delete;
    GlyphSet& operator=(const GlyphSet&) = delete;

    std::size_t fonts() const { return names_.size(); }
    const std::string& font_name(std::size_t i) const { return names_[i]; }
    std::uint64_t instances() const { return static_cast<std::uint64_t>(n_chars()) * fonts() * kSizes.size() * kAligns; }

    Instance decode(std::uint64_t idx) const {
        Instance x;
        x.ch = symbols()[idx % static_cast<std::uint64_t>(n_chars())];  // a code point
        idx /= static_cast<std::uint64_t>(n_chars());
        x.font = static_cast<int>(idx % fonts());
        idx /= fonts();
        x.size = static_cast<int>(idx % kSizes.size());
        idx /= kSizes.size();
        x.align = static_cast<int>(idx % kAligns);
        return x;
    }

    /// The 64x64 image of an instance.
    Image image(const Instance& x);

private:
    struct Font;
    std::vector<std::unique_ptr<Font>> fonts_;
    std::vector<std::string> names_;
    std::map<std::array<int, 3>, Glyph> cache_;  // (char, font, size) -> glyph
    std::mutex mutex_;
    const Glyph& glyph(int ch, int font, int size);
};

/// Every instance index once, in an order drawn from `seed` (Fisher-Yates with the spec RNG).
inline std::vector<std::uint32_t> shuffled_order(std::uint64_t n, std::uint64_t seed) {
    std::vector<std::uint32_t> v(n);
    for (std::uint64_t i = 0; i < n; ++i) v[i] = static_cast<std::uint32_t>(i);
    Rng r(seed);
    for (std::uint64_t i = n; i > 1; --i) std::swap(v[i - 1], v[r.below(i)]);
    return v;
}

/// Slightly changed copies of an image, for the consistency score: `noise` of the pixels flipped,
/// or the whole image shifted one pixel (direction 0..3: left, right, up, down).
inline Image flipped(const Image& im, double noise, Rng& r) {
    Image out = im;
    for (auto& p : out)
        if (r.u() < noise) p ^= 1;
    return out;
}
inline Image shifted(const Image& im, int dir) {
    Image out(kPixels, 0);
    const int dx = dir == 0 ? -1 : dir == 1 ? 1 : 0, dy = dir == 2 ? -1 : dir == 3 ? 1 : 0;
    for (int y = 0; y < kSide; ++y)
        for (int x = 0; x < kSide; ++x) {
            const int sx = x - dx, sy = y - dy;
            if (sx >= 0 && sx < kSide && sy >= 0 && sy < kSide) out[y * kSide + x] = im[sy * kSide + sx];
        }
    return out;
}

/// The net's inputs: 1.0 for black, 0.0 for white.
inline std::vector<double> inputs_of(const Image& im) {
    std::vector<double> in(kPixels);
    for (int i = 0; i < kPixels; ++i) in[i] = im[i] ? 1.0 : 0.0;
    return in;
}

/// Write a 1-bit black-and-white .bmp (rows bottom-up, padded to 4 bytes).
inline void write_bmp(const fs::path& path, int w, int h, const std::vector<std::uint8_t>& black) {
    const int row = ((w + 31) / 32) * 4;
    const std::uint32_t data = static_cast<std::uint32_t>(row * h), off = 14 + 40 + 8;
    std::vector<std::uint8_t> b;
    auto u16 = [&](std::uint32_t v) { b.push_back(v & 0xFF); b.push_back((v >> 8) & 0xFF); };
    auto u32 = [&](std::uint32_t v) { u16(v & 0xFFFF); u16(v >> 16); };
    b.push_back('B'); b.push_back('M'); u32(off + data); u32(0); u32(off);
    u32(40); u32(static_cast<std::uint32_t>(w)); u32(static_cast<std::uint32_t>(h)); u16(1); u16(1); u32(0); u32(data);
    u32(2835); u32(2835); u32(2); u32(0);
    u32(0x00FFFFFF);  // palette 0: white
    u32(0x00000000);  // palette 1: black
    for (int y = h - 1; y >= 0; --y) {
        std::vector<std::uint8_t> r(static_cast<std::size_t>(row), 0);
        for (int x = 0; x < w; ++x)
            if (black[static_cast<std::size_t>(y) * w + x]) r[static_cast<std::size_t>(x / 8)] |= static_cast<std::uint8_t>(0x80 >> (x % 8));
        b.insert(b.end(), r.begin(), r.end());
    }
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

/// A grid of up to cols x rows images with a 2-pixel black frame around each cell.
inline void write_sheet(const fs::path& path, const std::vector<Image>& ims, int cols, int rows) {
    const int cell = kSide + 4, w = cols * cell, h = rows * cell;
    std::vector<std::uint8_t> px(static_cast<std::size_t>(w) * h, 0);
    for (std::size_t k = 0; k < ims.size() && k < static_cast<std::size_t>(cols * rows); ++k) {
        const int cx = static_cast<int>(k % cols) * cell, cy = static_cast<int>(k / cols) * cell;
        for (int y = 0; y < cell; ++y)
            for (int x = 0; x < cell; ++x) {
                const bool frame = x < 2 || y < 2 || x >= cell - 2 || y >= cell - 2;
                px[static_cast<std::size_t>(cy + y) * w + cx + x] =
                    frame ? 1 : ims[k][static_cast<std::size_t>(y - 2) * kSide + (x - 2)];
            }
    }
    write_bmp(path, w, h, px);
}

}  // namespace cat
