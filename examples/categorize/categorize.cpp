// Example trainer: evolve nets that sort images into categories they define themselves.
//
// Images are printable ASCII and Latin-1 characters (188 symbols) from many fonts, at several sizes and alignments, as
// 64x64 black-and-white pixels (glyphs.hpp). Every image of the set is visited once in a seeded
// random order; each generation every net sees the next --batch images. Nets have one output, read
// as an unsigned 16-bit category number (0 = uncategorized); the score (score.hpp) asks only for consistent, usable categories, not particular ones.
// The top 10 nets are kept, the other 30 culled, and each survivor is mutated 3 times.
//
// Usage: categorize_trainer [--out DIR] [--seed N] [--pop-seed N] [--resume FILE.jpnet]
//                           [--profile FILE] [--fonts LIST] [--batch N] [--q N]
//                           [--min-share F] [--max-share F] [--placement 0|1] [--min-cats N] [--max-uncat F] [--noise F]
//                           [--stop-at GENERATIONS] [--console-every SECONDS]
// Runs until interrupted (Ctrl+C) or --stop-at, then saves the population.

#include "score.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <ctime>
#include <iomanip>
#include <optional>
#include <sstream>
#include <thread>

using namespace cat;
using namespace jneuro;
using Pop = Population<D>;

namespace {

constexpr int kPopulation = 40, kKeep = 10, kChildren = 3;
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

std::string timestamp() {
    const std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y%m%d-%H%M%S", std::localtime(&t));
    return buf;
}

Profile load_profile(const fs::path& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot read profile " + file.string());
    std::stringstream text;
    text << in.rdbuf();
    return profile_from_json(text.str());
}

}  // namespace

int main(int argc, char** argv) {
    fs::path out = fs::path("runs") / ("categorize-" + timestamp());
    std::optional<std::uint64_t> seed, pop_seed, stop_at;
    std::optional<fs::path> resume, font_list;
    fs::path profile_file = "profiles/expansion.json";
    int batch = 400;
    std::uint32_t q = 5;
    double console_every = 5.0;
    ScoreParams sp;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i], v = argv[i + 1];
        if (a == "--out") out = v;
        else if (a == "--seed") seed = std::stoull(v);
        else if (a == "--pop-seed") pop_seed = std::stoull(v);
        else if (a == "--resume") resume = v;
        else if (a == "--profile") profile_file = v;
        else if (a == "--fonts") font_list = v;
        else if (a == "--batch") batch = std::stoi(v);
        else if (a == "--q") q = static_cast<std::uint32_t>(std::stoul(v));
        else if (a == "--min-share") sp.min_share = std::stod(v);
        else if (a == "--max-share") sp.max_share = std::stod(v);
        else if (a == "--placement") sp.placement = v == "1" || v == "on" || v == "yes";
        else if (a == "--min-cats") sp.min_cats = std::stoi(v);
        else if (a == "--max-uncat") sp.max_uncat = std::stod(v);
        else if (a == "--noise") sp.noise = std::stod(v);
        else if (a == "--stop-at") stop_at = std::stoull(v);
        else if (a == "--console-every") console_every = std::stod(v);
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    fs::create_directories(out);
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    GlyphSet glyphs(font_list ? read_font_list(*font_list) : default_fonts());
    const std::uint64_t stream_seed = seed ? *seed : fresh_seed();
    const auto order = shuffled_order(glyphs.instances(), stream_seed);
    const Profile profile = load_profile(profile_file);

    Pop pop = resume ? Pop::load(*resume) : Pop(q, kPixels, 1, Priority::Speed, pop_seed);
    if (!resume) pop.create(kPopulation, profile.rng_mode());
    if (pop.get(pop.ids()[0]).n_outputs != 1) {
        std::cerr << "the resumed population's nets must have one output (the category number)\n";
        return 2;
    }
    {
        std::ofstream info(out / "run_info.txt", std::ios::app);
        info << "start " << timestamp() << " stream_seed " << stream_seed << " population_seed " << pop.seed() << " fonts "
             << glyphs.fonts() << " instances " << glyphs.instances() << " batch " << batch
             << " min_share " << sp.min_share << " max_share " << sp.max_share << " placement " << sp.placement << " min_cats " << sp.min_cats << " max_uncat " << sp.max_uncat
             << " noise " << sp.noise << " profile " << profile_file.string()
             << (resume ? " resumed_from " + resume->string() : std::string()) << "\n";
    }
    const bool new_csv = !fs::exists(out / "log.csv");
    std::ofstream csv(out / "log.csv", std::ios::app);
    if (new_csv)
        csv << "generation,best_score,top10_avg_score,best_consistency,best_categories,best_uncategorized,"
               "best_neurons,top10_avg_neurons,images_seen,elapsed_s\n";
    std::cout << "categorize trainer: " << glyphs.fonts() << " fonts, " << glyphs.instances() << " images, batch "
              << batch << "; output in " << out.string() << ". Ctrl+C to stop and save.\n"
              << std::setw(8) << "gen" << std::setw(8) << "best" << std::setw(9) << "top10" << std::setw(8) << "cons"
              << std::setw(6) << "cats" << std::setw(8) << "uncat" << std::setw(9) << "neurons" << std::setw(10) << "seen"
              << "\n";

    const auto t0 = std::chrono::steady_clock::now();
    auto last_print = t0 - std::chrono::hours(1);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    while (!g_stop && !(stop_at && pop.generation() >= *stop_at)) {
        // this generation's images: the next stretch of the shuffled order (wrapping around)
        const std::uint64_t gen = pop.generation();
        std::vector<Sample> samples;
        samples.reserve(static_cast<std::size_t>(batch));
        for (int i = 0; i < batch; ++i) {
            const std::uint64_t pos = (gen * static_cast<std::uint64_t>(batch) + static_cast<std::uint64_t>(i)) % order.size();
            samples.push_back(make_sample(glyphs, order[pos], stream_seed, sp.noise));
        }
        const std::vector<std::uint64_t> ids = pop.ids();
        std::vector<Result> res(ids.size());
        std::atomic<std::size_t> next{0};
        auto worker = [&] {
            for (std::size_t k; (k = next++) < ids.size();) res[k] = score(pop.get(ids[k]), samples, sp);
        };
        std::vector<std::thread> pool;
        for (unsigned t = 0; t < std::min<unsigned>(threads, static_cast<unsigned>(ids.size())); ++t) pool.emplace_back(worker);
        for (auto& t : pool) t.join();

        std::vector<std::size_t> rank(ids.size());
        for (std::size_t k = 0; k < rank.size(); ++k) rank[k] = k;
        std::stable_sort(rank.begin(), rank.end(), [&](std::size_t a, std::size_t b) { return res[a].score > res[b].score; });
        auto neurons = [&](std::size_t k) { return static_cast<double>(pop.get(ids[k]).nodes.size() - kPixels); };
        double top = 0, top_n = 0;
        const std::size_t keep = std::min<std::size_t>(kKeep, rank.size());
        for (std::size_t r = 0; r < keep; ++r) { top += res[rank[r]].score; top_n += neurons(rank[r]); }
        top /= static_cast<double>(keep);
        top_n /= static_cast<double>(keep);
        const Result& b = res[rank[0]];
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        csv << gen << ',' << b.score << ',' << top << ',' << b.consistency << ',' << b.cats << ',' << b.uncat << ','
            << neurons(rank[0]) << ',' << top_n << ',' << (gen + 1) * static_cast<std::uint64_t>(batch) << ',' << elapsed << '\n';
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - last_print).count() >= console_every) {
            last_print = std::chrono::steady_clock::now();
            std::cout << std::fixed << std::setprecision(3) << std::setw(8) << gen << std::setw(8) << b.score << std::setw(9)
                      << top << std::setw(8) << b.consistency << std::setw(6) << b.cats << std::setw(8) << b.uncat
                      << std::setw(9) << std::setprecision(0) << neurons(rank[0]) << std::setw(10)
                      << (gen + 1) * static_cast<std::uint64_t>(batch) << "\n"
                      << std::flush;
            csv.flush();
        }
        if (gen % 50 == 0) save_net(out / "best.jnet", pop.get(ids[rank[0]]));

        for (std::size_t r = keep; r < rank.size(); ++r) pop.remove(ids[rank[r]]);
        for (std::size_t r = 0; r < keep; ++r)
            for (int c = 0; c < kChildren; ++c) pop.mutate(ids[rank[r]], profile);
        pop.next_generation();
        if (pop.generation() % 200 == 0) pop.save(out / "population.jpnet");
    }
    pop.save(out / "population.jpnet");
    std::cout << "stopped at generation " << pop.generation() << "; saved " << (out / "population.jpnet").string() << "\n";
    return 0;
}
