# Checks a pruned MiniMax-H3 checkpoint's timestep mapping: block 0's modulation from the full FL2VA transformer
# (time embedder, SiLU, the 2688-wide adaln_proj) against the pruned file's 8-wide projection fed by its
# adaln_t_table, for three hypotheses of how t indexes the 1025-point table. The right one (t * 1024, linearly
# interpolated; what H3 implements) agrees to about 2e-4; the others are off by 10-25%. Reads only the needed
# tensors: by HTTP range from a URL, or from a local file.
#   .venv/bin/python tools/check_pruned_curve.py CACHE_DIR URL_OR_PATH [FL2VA_TRANSFORMER_DIR]
import json, struct, sys, os, urllib.request, math
import torch
from safetensors import safe_open

URL = sys.argv[2]
cache = sys.argv[1]
os.makedirs(cache, exist_ok=True)

def fetch(rng):
    if os.path.exists(URL):
        with open(URL, "rb") as f:
            f.seek(rng[0])
            return f.read(rng[1] - rng[0] + 1)
    req = urllib.request.Request(URL, headers={"Range": f"bytes={rng[0]}-{rng[1]}", "User-Agent": "curl/8"})
    with urllib.request.urlopen(req) as r:
        return r.read()

hp = os.path.join(cache, "header.json")
if not os.path.exists(hp):
    n = struct.unpack("<Q", fetch((0, 7)))[0]
    open(hp, "w").write(fetch((8, 8 + n - 1)).decode())
    open(os.path.join(cache, "header_len"), "w").write(str(n))
header = json.load(open(hp)); n = int(open(os.path.join(cache, "header_len")).read())
header.pop("__metadata__", None)
base = 8 + n

def tensor(name):
    fp = os.path.join(cache, name + ".bin")
    info = header[name]
    if not os.path.exists(fp):
        a, b = info["data_offsets"]
        open(fp, "wb").write(fetch((base + a, base + b - 1)))
    raw = open(fp, "rb").read()
    dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[info["dtype"]]
    return torch.frombuffer(bytearray(raw), dtype=dt).reshape(info["shape"]).to(torch.float64)

print("pruned tensors:", {k: (header[k]["dtype"], header[k]["shape"]) for k in header if "adaln" in k and ("blocks.0." in k or "final" in k or k == "adaln_t_table")})
table = tensor("adaln_t_table")           # [1025][8]
W8 = tensor("blocks.0.adaln_proj.linear.weight")  # [96768][8]
b8 = tensor("blocks.0.adaln_proj.linear.bias")

# The full model's block 0 path.
tdir = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~/models/minimax-h3/FL2VA/transformer")
idx = json.load(open(os.path.join(tdir, [n for n in os.listdir(tdir) if n.endswith(".index.json")][0])))["weight_map"]
def full(name):
    with safe_open(os.path.join(tdir, idx[name]), "pt") as f:
        return f.get_tensor(name).to(torch.float64)
p_in_w, p_in_b = full("time_embedder.proj_in.weight"), full("time_embedder.proj_in.bias")
p_out_w, p_out_b = full("time_embedder.proj_out.weight"), full("time_embedder.proj_out.bias")
Wf, bf = full("blocks.0.adaln_proj.linear.weight"), full("blocks.0.adaln_proj.linear.bias")

def temb(t):
    i = torch.arange(128, dtype=torch.float64)
    freq = torch.exp(-math.log(10000.0) * i / 128)
    arg = t * freq
    emb = torch.cat([torch.cos(arg), torch.sin(arg)])
    h = torch.nn.functional.silu(p_in_w @ emb + p_in_b)
    return p_out_w @ h + p_out_b

def mods_full(t):
    return Wf @ torch.nn.functional.silu(temb(t)) + bf

def c_of(pos):
    pos = min(max(pos, 0.0), 1024.0)
    i0 = min(int(math.floor(pos)), 1023)
    w = pos - i0
    return (1 - w) * table[i0] + w * table[i0 + 1]

def rel(a, b):
    return float(torch.linalg.norm(a - b) / torch.linalg.norm(b))

for t in [0.0, 0.0769230769, 0.25, 0.5, 0.75, 0.9, 0.999, 1.0]:
    ref = mods_full(t)
    hyps = {"t*1024": t * 1024, "(1-t)*1024": (1 - t) * 1024, "t*1000": t * 1000}
    print(f"t={t:<12.6g} " + "  ".join(f"{k}: {rel(W8 @ c_of(v) + b8, ref):.2e}" for k, v in hyps.items()))
