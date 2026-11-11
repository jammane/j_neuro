# j_neuro — Python implementation

Python ≥ 3.11, NumPy only. Mirrors the C++ implementation bit for bit (see the
[specification](../spec/README.md)).

## Set up and test

```
cd python
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest
```

## Modules

| Module | Contents |
|---|---|
| `jneuro.types` | enums, `ValidationError` |
| `jneuro.numerics`, `jneuro.rng`, `jneuro.aggregate` | exact float helpers, shared RNG, aggregation |
| `jneuro.net` | `Net`, `Node`, `VNeuron`, `SNeuron`, `MNeuron`; `Net.create`, `step`, `reset`, `validate` |
| `jneuro.analysis` | path rule, immunity, depth ranges |
| `jneuro.mutate`, `jneuro.removal` | `Profile`, `mutate`; removal and compaction |
| `jneuro.io` | genome / profile / run-log JSON |

Regenerate implementation-produced golden cases (from the repository root):
`python/.venv/bin/python tools/make_impl_golden.py`.
