# Credits

Recognition for the people, projects and research j_neuro builds on.

## Contributors

| Name | Role |
|---|---|
| Jeremiah Mane ([@jammane](https://github.com/jammane)) | Creator, designer and maintainer |

Design discussion and documentation drafting were assisted by Claude (Anthropic).

New contributors are added here when their first contribution is merged
(see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Third-party software

j_neuro only depends on software whose license requires nothing beyond attribution
(see [CONTRIBUTING.md](CONTRIBUTING.md#dependency-license-rule)). Dependencies are installed
by the user's package manager or build system and are **not vendored** into this repository.

When a dependency is actually adopted, its full copyright notice and license text are added
to this section in the same change.

### Adopted

| Package | Used by | Purpose | License |
|---|---|---|---|
| [NumPy](https://numpy.org) | Python | float64 arrays, bit-level views of doubles | BSD-3-Clause |
| [pytest](https://pytest.org) | Python (tests only) | test runner | MIT |
| [Catch2](https://github.com/catchorg/Catch2) | C++ (tests only) | test framework | BSL-1.0 |
| [nlohmann/json](https://github.com/nlohmann/json) | C++ | profile and run-log JSON | MIT |

None of these are bundled with j_neuro; they are installed by the user's package manager.
(NumPy's own binary wheels bundle further libraries under their own licenses; j_neuro does not
redistribute them.)

<details><summary>NumPy license</summary>

```
Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

</details>

<details><summary>pytest license</summary>

```
The MIT License (MIT)

Copyright (c) 2004 Holger Krekel and others

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

<details><summary>Catch2 license</summary>

```
Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
```

</details>

<details><summary>nlohmann/json license</summary>

```
MIT License 

Copyright (c) 2013-2025 Niels Lohmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>

## Algorithms and research

| Work | Authors | Used for |
|---|---|---|
| xoshiro256\*\* and splitmix64 ([prng.di.unimi.it](https://prng.di.unimi.it/)) | David Blackman, Sebastiano Vigna (splitmix64 after Guy L. Steele Jr., Doug Lea, Christine H. Flood) | Shared cross-language PRNG ([spec/08](spec/08-rng.md)). Reference code is public domain. |
| Improved Kahan–Babuška summation (1974) | Arnold Neumaier, after William Kahan | Compensated mean ([spec/04](spec/04-neurons.md)) |
| Irwin–Hall distribution | Joseph Oscar Irwin, Philip Hall | Transcendental-free Gaussian approximation ([spec/08](spec/08-rng.md)) |
| Dominator analysis (e.g. "A Simple, Fast Dominance Algorithm", 2001; Lengauer–Tarjan, 1979) | Keith D. Cooper, Timothy J. Harvey, Ken Kennedy; Thomas Lengauer, Robert E. Tarjan | Computing deletion immunity ([spec/07](spec/07-removal.md)) |
| SHA-256, FIPS 180-4 | NIST | `.jnet` / `.jpnet` checksums and hashes ([spec/09](spec/09-serialization.md)); implemented in-house for C++, Python's `hashlib` |
| NEAT — NeuroEvolution of Augmenting Topologies (2002) | Kenneth O. Stanley, Risto Miikkulainen | Inspiration for neutral structural insertion |

## Licenses and standards

- [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — The PolyForm Project.
- IEEE 754-2008/2019 floating-point arithmetic.
