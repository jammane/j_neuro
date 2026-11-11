# RNG oracle

Independent generator for `tests/golden/rng/*.json`, built on the authors' public-domain
reference code rather than on either j_neuro implementation.

```
curl -O https://prng.di.unimi.it/xoshiro256starstar.c
curl -O https://prng.di.unimi.it/splitmix64.c
head -n 114 xoshiro256starstar.c > xo_trim.c     # drop the arbitrary-jump section (needs f2x.c)
gcc -O1 -ffp-contract=off -std=c11 -I. harness.c -o harness -lm
./harness      # one JSON case per line, seeds 0, 42, 2^64-1
```

The reference sources are not stored in this repository.
