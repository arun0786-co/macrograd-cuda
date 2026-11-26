# writes tests/linear_macrograd.txt: one forward and backward pass of macrograd's linear
# layer, for kernels/linear.cu to compare against. run it from the repo root with
# macrograd next to this repo, or give its path:
#   python3 tests/make_linear_fixture.py [path/to/macrograd]
#
# the file has the shape "rows in out" on the first line, then one line each for
# x, w, b, dy, y, dx, dw, db, every number written with 17 significant digits.
# dy is the gradient that arrives at y; macrograd gets dx, dw and db by running
# backward on sum(y * dy), whose gradient with respect to y is dy
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import linear

ROWS, IN, OUT = 6, 10, 4
random.seed(0)
layer = linear(IN, OUT)
layer.b.data = [random.uniform(-1, 1) for _ in range(OUT)]  # macrograd starts b at 0
x = tensor([[random.uniform(-1, 1) for _ in range(IN)] for _ in range(ROWS)])
dy = tensor([[random.uniform(-1, 1) for _ in range(OUT)] for _ in range(ROWS)])

y = layer(x)
(y * dy).sum().backward()

lines = [f"{ROWS} {IN} {OUT}"]
for values in [x.data, layer.w.data, layer.b.data, dy.data, y.data, x.grad, layer.w.grad, layer.b.grad]:
    lines.append(" ".join(f"{v:.17g}" for v in values))
with open(os.path.join(here, "linear_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {len(lines) - 1} arrays for a ({ROWS}, {IN}) -> ({ROWS}, {OUT}) linear layer")
