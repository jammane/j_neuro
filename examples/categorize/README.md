# Categorization (C++ example)

Evolves nets that sort images into categories **they define themselves**. Nobody supplies
labels or says which categories are right: the trainer only scores whether a net's categories
are consistent and usable, and selection does the rest. A test case for unsupervised use of the
library, built on `jneuro::Population` like the other examples.

## Build and run

```
cmake -S cpp -B build && cmake --build build --target categorize_trainer categorize_analyze
build/categorize_trainer --stop-at 300                 # or until Ctrl+C, then saves
build/categorize_analyze runs/categorize-…/population.jpnet
```

CMake downloads [stb_truetype](https://github.com/nothings/stb) (public domain / MIT) at configure
time, from a pinned commit checked by SHA-256, into the build directory; it is not part of the
repository ([CREDITS.md](../../CREDITS.md)). Without network access the two programs are skipped.
The default fonts are the ones installed on Fedora (Liberation, Open Sans, Source Code Pro,
Caladea, Carlito, Droid Sans, Noto, Symbola); `--fonts FILE` takes a list of font paths instead,
one per line. A font is skipped if it is missing or lacks any of the symbols.

## Images

- **Symbols (188):** printable ASCII `!` to `~` and printable Latin-1 `¡` to `ÿ`. Space, the
  no-break space U+00A0 and the soft hyphen U+00AD draw nothing in normal text and are left out;
  U+0080 to U+009F are control codes.
- **Fonts (30):** 11 families (sans, serif, monospace, symbol) in regular, bold, italic, light and
  black styles, so the same letter appears in many artistic forms.
- **Sizes (5):** pixel heights 12, 20, 28, 40 and 56. **Alignments (9):** the glyph's ink sits at a
  corner, the middle of an edge, or the center of the frame.
- Every image is **64 × 64 pixels, strictly black or white**: glyphs are rendered by stb_truetype
  and thresholded at 50% grey. The net's 4,096 inputs are 1.0 for black, 0.0 for white.

That is 188 × 30 × 5 × 9 = 253,800 images. They are rendered on demand, never stored: every image
of the set is visited once in an order drawn from `--seed`, and each generation takes the next
`--batch` images (default 400). Symbols, fonts, sizes and alignments are shuffled together, so
wherever training stops, it has seen an even mix.

## Categories and the score

The net has **one output, read as an unsigned 16-bit category number**:
`floor(clamp(output, 0, 65535))`. **0 means "uncategorized"**, for images that do not fit.

The score asks only that the categories be usable (`score.hpp`):

| Term | Rule | Why |
|---|---|---|
| Consistency (the score's main part) | Three changed copies of each image should get its category: 2% of pixels flipped (`--noise`); shifted one pixel; and the same symbol in the same font at another size or alignment (`--placement 1`, the default) | Categories must not hang on single pixels or on where the ink sits |
| Category size | A category counts only if it holds between 1% (`--min-share`) and 10% (`--max-share`) of the images; images in categories outside that range count as uncategorized | Not one category per image, and not one category for everything |
| Enough categories | Fewer than 3 counted categories (`--min-cats`) costs 0.25 each | Categorizing has to be worth something |
| Uncategorized | Up to 20% of the images (`--max-uncat`) is free; beyond that costs 2 per unit | Things that do not fit have a place, but the net cannot send everything there |

With the 10% cap and the 20% allowance, a net needs at least 8 categories; up to about 100 can
count. Which ones it forms is up to the net.

Each generation all 40 nets are scored on the same images and copies; the top 10 are kept and each
is mutated 3 times (profile from `--profile`, default `profiles/expansion.json`).

Options: `--out DIR`, `--seed N` (image order and copies), `--pop-seed N`, `--resume FILE.jpnet`,
`--profile FILE`, `--fonts FILE`, `--batch N`, `--q N` (`in_qty_max` of new nets, default 5),
`--min-share F`, `--max-share F`, `--placement 0|1`, `--min-cats N`, `--max-uncat F`, `--noise F`,
`--stop-at N`, `--console-every SECONDS`.

Output: `log.csv` (per generation: best and top-10 scores, the best net's consistency, categories,
uncategorized share and size), `best.jnet` every 50 generations, `population.jpnet` every 200
generations and on exit, `run_info.txt`.

A new net's output starts below 1 for every image (it begins near the mean of its inputs), so at
first everything is uncategorized; evolution has to spread the output over several whole numbers
before categories appear.

## Analysis

```
build/categorize_analyze POPULATION.jpnet|NET.jnet [--images N] [--seed N] [--out DIR]
```

Scores every net on `--images` fresh images (default 2,000; another `--seed` than training gives
images the nets have not been selected on), then for the best net prints each category's size,
its most common symbols, and how many symbols, fonts, sizes and alignments it spans, and writes
`category_<n>.bmp` (and `uncategorized.bmp`): a 1-bit sheet of up to 64 of its members, to see
what the net grouped together. The score options are the same as the trainer's.
