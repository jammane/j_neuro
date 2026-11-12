"""Write hand-derived golden cases to tests/golden/{handmade,invalid}.

Expected values are worked out by hand from spec/04 and written here as plain Python
arithmetic; this script does not use the jneuro package. Run from the repository root:

    python3 tools/make_handmade_golden.py
"""
import copy
import hashlib
import json
import math
import os
import struct

INF = math.inf
MAX = 1.7976931348623157e308   # largest finite binary64; canon clamps +-inf to +-MAX
NAN = math.nan
D16, D32, D64 = 2**16 - 1, 2**32 - 1, 2**64 - 1
ALL_PRIORITIES = ["SPEED", "MEMORY", "BALANCED"]


def bits(x):
    return struct.pack(">d", x).hex()


def from_bits(b):
    return struct.unpack(">d", b.to_bytes(8, "big"))[0]


def bn(x):  # bitwise NOT of the binary64 pattern
    return from_bits(~int(bits(x), 16) & (2**64 - 1))


def v(inp, op, w, inv="none"):
    return {"type": "v", "input": inp, "inversion": inv, "op": op, "weight": w}


def s(inp, op, k):
    return {"type": "s", "input": inp, "op": op, "k": k}


def m(op, slots):
    return {"type": "m", "op": op, "slots": slots}


def genome(n_in, n_out, q, noninput, depth_type="U16"):
    nodes = [{"birth": i, "born_gen": 0, "depth": 0} for i in range(n_in)]
    for j, (depth, neuron) in enumerate(noninput):
        nodes.append({"birth": n_in + j, "born_gen": 0, "depth": depth, "neuron": neuron})
    return {
        "format": "j_neuro.genome",
        "spec_version": "0.3.0",
        "params": {"priority": "SPEED", "depth_type": depth_type, "in_qty_max": q,
                   "n_inputs": n_in, "n_outputs": n_out, "path_mix": 0, "max_run": 0},
        "next_birth": len(nodes),
        "nodes": nodes,
    }


# ---- independent .jnet encoder, written from spec/09 (deliberately permissive so that
# ---- invalid genomes can be encoded too)
WIDTH = {"U16": 2, "U32": 4, "U64": 8}
PRI = {"SPEED": 0, "MEMORY": 1, "BALANCED": 2}
DT = {"U16": 0, "U32": 1, "U64": 2}
INV = {"none": 0, "bitnot": 1, "recip": 2}
VOPS = {"mul": 0, "div": 1, "add": 2, "sub": 3}
AGG = {"mean": 0, "median": 1, "min": 2, "max": 3, "lag": 4}


def encode(g, magic=b"JNET", fmt=1, ver=(0, 5, 0), dt_byte=None, pri_byte=None,
           cut=0, extra=b"", bad_hash=False):
    p = g["params"]
    w = WIDTH[p["depth_type"]]
    none = (1 << (8 * w)) - 1

    def u(v):
        return (none if v is None else v).to_bytes(w, "little")

    b = magic + struct.pack("<HHHH", fmt, *ver)
    b += bytes([PRI[p["priority"]] if pri_byte is None else pri_byte,
                DT[p["depth_type"]] if dt_byte is None else dt_byte])
    b += struct.pack("<IIIBBQ", p["in_qty_max"], p["n_inputs"], p["n_outputs"], p["path_mix"],
                     p["max_run"], g["next_birth"])
    b += u(len(g["nodes"]))
    for n in g["nodes"]:
        b += struct.pack("<QQ", n["birth"], n["born_gen"]) + n["depth"].to_bytes(w, "little")
        x = n.get("neuron")
        if x is None:
            b += b"\x00"
        elif x["type"] == "v":
            b += b"\x01" + u(x["input"]) + bytes([INV.get(x["inversion"], 3), VOPS.get(x["op"], 9)])
            b += struct.pack("<d", x["weight"])
        elif x["type"] == "s":
            b += b"\x02" + u(x["input"]) + bytes([AGG.get(x["op"], 9)]) + struct.pack("<I", x["k"])
        elif x["type"] == "m":
            b += b"\x03" + bytes([AGG.get(x["op"], 9)]) + b"".join(u(s) for s in x["slots"])
        else:
            b += b"\x04"
    b = b[:len(b) - cut] + extra
    digest = hashlib.sha256(b).digest()
    if bad_hash:
        digest = bytes([digest[0] ^ 1]) + digest[1:]
    return (b + digest).hex()


def eval_case(g, steps):
    return {"kind": "eval", "jnet": encode(g), "priorities": ALL_PRIORITIES,
            "steps": [{"inputs": [bits(x) for x in i], "outputs": [bits(y) for y in o]}
                      for i, o in steps]}


cases = {}

# 1. v ops, division by zero, signed zero / NaN / inf inputs
g = genome(1, 4, 3, [(D16, v(0, "mul", 2.5)), (D16, v(0, "div", 0.0)),
                     (D16, v(0, "add", -1.5)), (D16, v(0, "sub", 2.0))])
cases["eval_v_ops"] = eval_case(g, [
    ([3.0], [7.5, MAX, 1.5, 1.0]),                    # 3/0 = inf -> MAX
    ([0.0], [0.0, 0.0, -1.5, -2.0]),      # 0/0 = NaN -> 0
    ([-0.0], [0.0, 0.0, -1.5, -2.0]),     # input canonicalized
    ([NAN], [0.0, 0.0, -1.5, -2.0]),      # input canonicalized
    ([INF], [MAX, MAX, MAX, MAX]),                    # input clamped to MAX; overflow -> MAX
    ([-2.0], [-5.0, -MAX, -3.5, -4.0]),
])

# 2. inversions
g = genome(1, 4, 3, [(D16, v(0, "mul", 1.0, "bitnot")), (D16, v(0, "mul", 1.0, "recip")),
                     (D16, v(0, "add", 0.0, "bitnot")), (D16, v(0, "div", 2.0, "recip"))])
cases["eval_v_inversions"] = eval_case(g, [
    ([1.0], [bn(1.0), 1.0, bn(1.0), 0.5]),
    ([0.0], [0.0, MAX, 0.0, MAX]),                    # bitnot(+0) is NaN -> 0; 1/0 = inf -> MAX
    ([2.0], [bn(2.0), 0.5, bn(2.0), 0.25]),
    ([-INF], [bn(-MAX), 1.0 / -MAX, bn(-MAX), (1.0 / -MAX) / 2.0]),   # input clamped to -MAX
])
assert bn(1.0) == -3.9999999999999996

# 3. m ops with empty slots (Q=5)
g = genome(2, 4, 5, [(D16, m(op, [0, 1, None, None, None]))
                     for op in ("mean", "median", "min", "max")])
cases["eval_m_empty_slots"] = eval_case(g, [
    ([3.0, 5.0], [8.0 / 5.0, 0.0, 3.0, 5.0]),
    ([-3.0, -5.0], [-8.0 / 5.0, 0.0, -5.0, 0.0]),     # max sees the empty zeros; min skips them
    ([INF, -INF], [0.0, 0.0, -MAX, MAX]),             # inputs clamped; MAX + -MAX = 0
])

# 4. full m (Q=3): compensated mean and overflow
g = genome(3, 4, 3, [(D16, m(op, [0, 1, 2])) for op in ("mean", "median", "min", "max")])
cases["eval_m_full_compensated"] = eval_case(g, [
    ([1e16, 1.0, -1e16], [1.0 / 3.0, 1.0, -1e16, 1e16]),   # naive sum would give 0
    ([1e308, 1e308, -1e308], [MAX, 1e308, -1e308, 1e308]),  # overflow: plain sum inf -> MAX
])

# 5. s warm-up (history excludes current value, zero-filled; min skips unfilled)
g = genome(1, 4, 3, [(D16, s(0, "mean", 3)), (D16, s(0, "median", 3)),
                     (D16, s(0, "min", 3)), (D16, s(0, "max", 1))])
xs = [1.0, 2.0, 3.0, 4.0, -1.0]
mean_o = [1.0, 2.0 - (1.0 / 3.0), 3.0 - 1.0, 4.0 - 2.0, -1.0 - 3.0]
median_o = [1.0, 2.0, 2.0, 2.0, -4.0]
min_o = [1.0, 1.0, 2.0, 3.0, -3.0]
max_o = [1.0, 1.0, 1.0, 1.0, -5.0]
cases["eval_s_warmup"] = eval_case(g, [([xs[i]], [mean_o[i], median_o[i], min_o[i], max_o[i]])
                                       for i in range(5)])

# 6. depth-ordered chain, U32
chain = genome(2, 1, 3, [
    (D32, v(5, "sub", 0.5)),                 # id 2: output
    (100, v(0, "mul", 2.0)),                 # id 3
    (100, v(1, "add", 1.0)),                 # id 4
    (200, m("max", [3, 4, None])),           # id 5
], "U32")
cases["eval_chain_u32"] = eval_case(chain, [
    ([1.0, 1.0], [1.5]),
    ([-3.0, -5.0], [-0.5]),                  # max(-6, -4, 0) = 0
])

# 6b. s lag: the raw input from k steps ago, +0.0 until k steps have passed
g = genome(1, 3, 3, [(D16, s(0, "lag", 1)), (D16, s(0, "lag", 3)), (D16, s(0, "lag", 2))])
xs = [1.0, 2.0, -3.0, 4.5, 5.0]
cases["eval_s_lag"] = eval_case(g, [([xs[i]], [xs[i - 1] if i >= 1 else 0.0, xs[i - 3] if i >= 3 else 0.0,
                                               xs[i - 2] if i >= 2 else 0.0]) for i in range(len(xs))])

# 7. U64 depth above 2^53
g = genome(1, 1, 3, [(D64, v(2, "add", 1.0)), (2**53 + 1, v(0, "mul", 3.0))], "U64")
cases["eval_u64_depth"] = eval_case(g, [([2.0], [7.0])])

# invalid genomes, each derived from the valid chain genome
invalid = {}


def bad(name, reason, fn, **enc):
    g2 = copy.deepcopy(chain)
    fn(g2)
    invalid[name] = {"kind": "invalid", "jnet": encode(g2, **enc), "reason": reason}


def same(g):
    return None


bad("even_in_qty_max", "in_qty_max must be odd",
    lambda g: (g["params"].__setitem__("in_qty_max", 4),
               g["nodes"][5]["neuron"]["slots"].append(None)))
bad("wrong_slot_count", "slot count", lambda g: g["nodes"][5]["neuron"]["slots"].append(None))
bad("deeper_connection", "connection depth",
    lambda g: g["nodes"][3]["neuron"].__setitem__("input", 5))
bad("path_rule", "path rule", lambda g: g["nodes"][5]["neuron"].__setitem__("slots", [3, None, None]))
bad("mix_rule", "mix rule: no s_neuron on any path, path_mix 1",
    lambda g: g["params"].__setitem__("path_mix", 1))
bad("run_rule", "run rule: two v_neurons in a row (0 -> 3 -> 4), max_run 1",
    lambda g: (g["params"].__setitem__("max_run", 1), g["nodes"][4].__setitem__("depth", 150),
               g["nodes"][4]["neuron"].__setitem__("input", 3),
               g["nodes"][5]["neuron"].__setitem__("slots", [3, 4, 1])))
bad("input_with_neuron", "input node has neuron",
    lambda g: g["nodes"][0].__setitem__("neuron", v(None, "add", 1.0)))
bad("output_depth", "output depth", lambda g: g["nodes"][2].__setitem__("depth", 300))
bad("hidden_depth_zero", "hidden depth", lambda g: g["nodes"][3].__setitem__("depth", 0))
bad("duplicate_birth", "birth", lambda g: g["nodes"][4].__setitem__("birth", 3))
bad("birth_too_large", "next_birth", lambda g: g.__setitem__("next_birth", 5))
bad("magic", "magic", same, magic=b"XNET")
bad("format_version", "format version", same, fmt=2)
bad("major_version", "spec_version", same, ver=(1, 0, 0))
bad("minor_version_pre_1", "spec_version 0.x minor", same, ver=(0, 4, 0))
bad("truncated", "truncated", same, cut=1)
bad("trailing_bytes", "trailing bytes", same, extra=b"\x00")
bad("checksum", "checksum", same, bad_hash=True)
bad("depth_type_byte", "enum range", same, dt_byte=3)
bad("priority_byte", "enum range", same, pri_byte=3)
bad("neuron_type_byte", "enum range", lambda g: g["nodes"][3]["neuron"].__setitem__("type", "x"))
bad("inversion_byte", "enum range", lambda g: g["nodes"][3]["neuron"].__setitem__("inversion", "flip"))
bad("median_even_k", "median k",
    lambda g: g["nodes"][4].__setitem__("neuron", s(1, "median", 2)))
bad("k_too_large", "k range", lambda g: g["nodes"][4].__setitem__("neuron", s(1, "mean", 5)))
bad("m_lag_op", "lag is an s_neuron op", lambda g: g["nodes"][5]["neuron"].__setitem__("op", "lag"))
bad("bad_op", "op", lambda g: g["nodes"][3]["neuron"].__setitem__("op", "pow"))
bad("dangling_reference", "reference", lambda g: g["nodes"][3]["neuron"].__setitem__("input", 9))
bad("dead_node", "dead node", lambda g: (
    g["nodes"].append({"birth": 6, "born_gen": 0, "depth": 50, "neuron": v(0, "mul", 1.0)}),
    g.__setitem__("next_birth", 7)))
bad("in_qty_max_too_small", "in_qty_max >= 3",
    lambda g: (g["params"].__setitem__("in_qty_max", 1),
               g["nodes"][5]["neuron"].__setitem__("slots", [3])))

root = os.path.join("tests", "golden")
for sub, group in (("handmade", cases), ("invalid", invalid)):
    os.makedirs(os.path.join(root, sub), exist_ok=True)
    for name, case in group.items():
        with open(os.path.join(root, sub, name + ".json"), "w") as f:
            json.dump(case, f, indent=1)
            f.write("\n")
print(len(cases), "eval cases,", len(invalid), "invalid cases")
