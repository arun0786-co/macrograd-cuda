# writes tests/layer_norm_macrograd.txt: one forward and backward pass of macrograd's
# layer_norm (with gamma and beta) for kernels/layer_norm.cu to compare against. run it
# from the repo root with macrograd next to this repo, or give its path:
#   python3 tests/make_layer_norm_fixture.py [path/to/macrograd]
#
# the file has "rows cols" on the first line, then one line each for x, gamma, beta, dy,
# y, dx, dgamma, dbeta, with 17 significant digits. dy is the gradient that arrives at
# y; macrograd gets the rest by running backward on sum(y * dy)
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import layer_norm

ROWS, COLS = 5, 12
random.seed(0)
norm = layer_norm(COLS)
# gamma and beta start at 1 and 0 in macrograd; other values test that they're used
norm.gamma.data = [random.uniform(0.5, 1.5) for _ in range(COLS)]
norm.beta.data = [random.uniform(-0.5, 0.5) for _ in range(COLS)]
rows = [[random.uniform(-2, 2) for _ in range(COLS)] for _ in range(ROWS)]
rows[1] = [3.0] * COLS  # a constant row: variance 0, so eps sets the scale
x = tensor(rows)
dy = tensor([[random.uniform(-1, 1) for _ in range(COLS)] for _ in range(ROWS)])

y = norm(x)
(y * dy).sum().backward()

lines = [f"{ROWS} {COLS}"]
for values in [x.data, norm.gamma.data, norm.beta.data, dy.data, y.data, x.grad, norm.gamma.grad, norm.beta.grad]:
    lines.append(" ".join(f"{v:.17g}" for v in values))
with open(os.path.join(here, "layer_norm_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote a ({ROWS}, {COLS}) layer norm")
