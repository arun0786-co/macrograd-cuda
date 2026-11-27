# writes tests/rope_macrograd.txt: macrograd's rotary_embedding and its backward on a
# small (B, H, T, d) tensor, starting at position 5, for kernels/rope.cu to compare
# against. run it from the repo root with macrograd next to this repo, or give its path:
#   python3 tests/make_rope_fixture.py [path/to/macrograd]
#
# the file has "rows T d start base" on the first line (rows = B * H * T), then x, dy, out
# and dx, one line each, 17 significant digits. macrograd gets dx by running backward on
# sum(out * dy)
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import rotary_embedding

B, H, T, D, START, BASE = 2, 2, 6, 8, 5, 10000.0
random.seed(0)
shape = (B, H, T, D)
n = B * H * T * D
x = tensor([random.uniform(-1, 1) for _ in range(n)], shape=shape)
dy = tensor([random.uniform(-1, 1) for _ in range(n)], shape=shape)
out = rotary_embedding(x, base=BASE, start=START)
(out * dy).sum().backward()
lines = [f"{B * H * T} {T} {D} {START} {BASE}"]
for values in [x.data, dy.data, out.data, x.grad]:
    lines.append(" ".join(f"{v:.17g}" for v in values))
with open(os.path.join(here, "rope_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote rotary embedding of {shape} from position {START}")
