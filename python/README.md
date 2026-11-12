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
| `jneuro.analysis` | path rule, mix rule, immunity, depth ranges |
| `jneuro.mutate`, `jneuro.removal` | `Profile`, `mutate`; removal and compaction |
| `jneuro.jnet` | `.jnet` files: `net_to_bytes`, `net_from_bytes`, `save_net`, `load_net` (also `net.save`, `Net.load`) |
| `jneuro.population` | `Population`: nets by id, `evaluate`, operations, `save`/`load` (`.jpnet`), `replay` |
| `jneuro.io` | profile and run-log JSON helpers |

Regenerate implementation-produced golden cases (from the repository root):
`python/.venv/bin/python tools/make_impl_golden.py`.
