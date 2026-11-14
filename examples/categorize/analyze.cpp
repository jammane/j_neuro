// What categories did the nets define? Scores every net of a population (or one .jnet) on fresh
// images, then for the best net prints each category's size, its most common characters, and how
// many fonts, sizes and alignments it spans, and writes a .bmp sheet of each category's members.
//
// Usage: categorize_analyze POPULATION.jpnet|NET.jnet [--images N] [--seed N] [--out DIR] [--fonts LIST]
//                           [--min-share F] [--max-share F] [--placement 0|1] [--min-cats N] [--max-uncat F] [--noise F]

#include "score.hpp"

#include <iomanip>
#include <set>

using namespace cat;
using namespace jneuro;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: categorize_analyze POPULATION.jpnet|NET.jnet [--images N] [--seed N] [--out DIR] [--fonts LIST]\n";
        return 2;
    }
    const fs::path src = argv[1];
    int n_images = 2000;
    std::uint64_t seed = 4242;
    fs::path out = src.parent_path() / "categories";
    std::optional<fs::path> font_list;
    ScoreParams sp;
    for (int i = 2; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--images") n_images = std::stoi(v);
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--out") out = v;
        else if (a == "--fonts") font_list = v;
        else if (a == "--min-share") sp.min_share = std::stod(v);
        else if (a == "--max-share") sp.max_share = std::stod(v);
        else if (a == "--placement") sp.placement = v == "1" || v == "on" || v == "yes";
        else if (a == "--min-cats") sp.min_cats = std::stoi(v);
        else if (a == "--max-uncat") sp.max_uncat = std::stod(v);
        else if (a == "--noise") sp.noise = std::stod(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    std::vector<Net<D>> nets;
    std::vector<std::string> names;
    if (src.extension() == ".jpnet") {
        auto pop = Population<D>::load(src);
        for (auto id : pop.ids()) { nets.push_back(pop.get(id).copy_genome()); names.push_back("net " + std::to_string(id)); }
    } else {
        nets.push_back(std::get<Net<D>>(load_net(src)));
        names.push_back(src.filename().string());
    }

    GlyphSet glyphs(font_list ? read_font_list(*font_list) : default_fonts());
    const auto order = shuffled_order(glyphs.instances(), seed);
    std::vector<Sample> samples;
    for (int i = 0; i < n_images && static_cast<std::size_t>(i) < order.size(); ++i)
        samples.push_back(make_sample(glyphs, order[static_cast<std::size_t>(i)], seed, sp.noise));

    std::size_t best = 0;
    std::vector<Result> res;
    for (std::size_t k = 0; k < nets.size(); ++k) {
        res.push_back(score(nets[k], samples, sp));
        if (res[k].score > res[best].score) best = k;
    }
    std::cout << nets.size() << " net(s) from " << src.string() << ", " << samples.size() << " fresh images (seed " << seed
              << "), " << glyphs.fonts() << " fonts\n\n";
    for (std::size_t k = 0; k < nets.size() && k < 10; ++k)
        std::cout << "  " << std::left << std::setw(10) << names[k] << std::right << std::fixed << std::setprecision(3)
                  << "  score " << res[k].score << "  consistency " << res[k].consistency << "  categories " << res[k].cats
                  << "  uncategorized " << std::setprecision(1) << 100 * res[k].uncat << "%\n";
    const Result& r = res[best];
    std::cout << "\nBest: " << names[best] << " (" << nets[best].nodes.size() - kPixels << " neurons)\n";

    fs::create_directories(out);
    std::map<int, std::vector<std::size_t>> members;
    for (std::size_t i = 0; i < samples.size(); ++i) members[r.category[i]].push_back(i);
    for (auto& [c, list] : members) {
        std::map<int, int> chars;
        std::set<int> fonts, sizes, aligns;
        for (std::size_t i : list) {
            ++chars[samples[i].x.ch];
            fonts.insert(samples[i].x.font);
            sizes.insert(samples[i].x.size);
            aligns.insert(samples[i].x.align);
        }
        std::vector<std::pair<int, int>> top(chars.begin(), chars.end());
        std::stable_sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
        const std::string name = c == kUncat ? "uncategorized" : "category " + std::to_string(c);
        std::cout << "\n  " << name << ": " << list.size() << " images (" << std::setprecision(1)
                  << 100.0 * list.size() / samples.size() << "%), " << chars.size() << " characters, " << fonts.size()
                  << " fonts, " << sizes.size() << " sizes, " << aligns.size() << " alignments\n    most common:";
        for (std::size_t t = 0; t < top.size() && t < 12; ++t)
            std::cout << "  '" << utf8(top[t].first) << "' " << top[t].second;
        std::cout << "\n";
        std::vector<Image> sheet;
        for (std::size_t t = 0; t < list.size() && t < 64; ++t) sheet.push_back(samples[list[t]].base);
        const fs::path file = out / (c == kUncat ? std::string("uncategorized.bmp") : "category_" + std::to_string(c) + ".bmp");
        write_sheet(file, sheet, 8, 8);
    }
    std::cout << "\nsheets of up to 64 members per category in " << out.string() << "\n";
    return 0;
}
