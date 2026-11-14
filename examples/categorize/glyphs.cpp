// stb_truetype rendering for glyphs.hpp. stb_truetype (public domain / MIT, by Sean Barrett) is
// downloaded by CMake at configure time from a pinned commit and checked by SHA-256; it is not
// part of this repository (CREDITS.md).
#include "glyphs.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#endif
#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include <stb_truetype.h>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace cat {

struct GlyphSet::Font {
    std::vector<unsigned char> data;
    stbtt_fontinfo info{};
};

GlyphSet::GlyphSet(const std::vector<fs::path>& fonts) {
    for (const auto& path : fonts) {
        std::ifstream in(path, std::ios::binary);
        auto f = std::make_unique<Font>();
        f->data.assign(std::istreambuf_iterator<char>(in), {});
        if (f->data.empty() || !stbtt_InitFont(&f->info, f->data.data(), stbtt_GetFontOffsetForIndex(f->data.data(), 0))) {
            std::cerr << "skipping font " << path.string() << " (missing or unreadable)\n";
            continue;
        }
        bool all = true;
        for (int c : symbols()) all = all && stbtt_FindGlyphIndex(&f->info, c) != 0;
        if (!all) {
            std::cerr << "skipping font " << path.string() << " (lacks some of the symbols)\n";
            continue;
        }
        names_.push_back(path.stem().string());
        fonts_.push_back(std::move(f));
    }
    if (fonts_.empty()) throw std::runtime_error("no usable fonts");
}

GlyphSet::~GlyphSet() = default;

const Glyph& GlyphSet::glyph(int ch, int font, int size) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::array<int, 3> key{ch, font, size};
    if (auto it = cache_.find(key); it != cache_.end()) return it->second;
    const stbtt_fontinfo& fi = fonts_[static_cast<std::size_t>(font)]->info;
    float scale = stbtt_ScaleForPixelHeight(&fi, static_cast<float>(kSizes[static_cast<std::size_t>(size)]));
    Glyph g;
    for (int tries = 0; tries < 20; ++tries) {  // shrink until the glyph fits the frame
        int w = 0, h = 0, xo = 0, yo = 0;
        unsigned char* bm = stbtt_GetCodepointBitmap(&fi, 0, scale, ch, &w, &h, &xo, &yo);
        if (w <= kSide && h <= kSide) {
            // threshold at 50% grey, then crop to the ink
            int x0 = w, y0 = h, x1 = -1, y1 = -1;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    if (bm[y * w + x] >= 128) { x0 = std::min(x0, x); y0 = std::min(y0, y); x1 = std::max(x1, x); y1 = std::max(y1, y); }
            if (x1 >= 0) {
                g.w = x1 - x0 + 1;
                g.h = y1 - y0 + 1;
                g.bits.assign(static_cast<std::size_t>(g.w) * g.h, 0);
                for (int y = 0; y < g.h; ++y)
                    for (int x = 0; x < g.w; ++x) g.bits[static_cast<std::size_t>(y) * g.w + x] = bm[(y0 + y) * w + (x0 + x)] >= 128;
            }
            stbtt_FreeBitmap(bm, nullptr);
            break;
        }
        stbtt_FreeBitmap(bm, nullptr);
        scale *= 0.9f;
    }
    if (g.bits.empty()) {  // no ink at this size (a tiny mark thresholded away): one black pixel
        g.w = g.h = 1;
        g.bits = {1};
    }
    return cache_.emplace(key, std::move(g)).first->second;
}

Image GlyphSet::image(const Instance& x) {
    const Glyph& g = glyph(x.ch, x.font, x.size);
    const int ax = x.align % 3, ay = x.align / 3;
    const int ox = ax == 0 ? 0 : ax == 1 ? (kSide - g.w) / 2 : kSide - g.w;
    const int oy = ay == 0 ? 0 : ay == 1 ? (kSide - g.h) / 2 : kSide - g.h;
    Image im(kPixels, 0);
    for (int y = 0; y < g.h; ++y)
        for (int xx = 0; xx < g.w; ++xx)
            im[static_cast<std::size_t>(oy + y) * kSide + (ox + xx)] = g.bits[static_cast<std::size_t>(y) * g.w + xx];
    return im;
}

}  // namespace cat
