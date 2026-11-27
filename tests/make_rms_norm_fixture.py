# writes tests/rms_norm_macrograd.txt: one forward and backward pass of macrograd's
# rms_norm (with gamma) for kernels/rms_norm.cu to compare against. run it
# from the repo root with macrograd next to this repo, or give its path:
#   python3 tests/make_rms_norm_fixture.py [path/to/macrograd]
#
# the file has "rows cols" on the first line, then one line each for x, gamma, dy, y,
# dx, dgamma, with 17 significant digits. dy is the gradient that arrives at
# y; macrograd gets the rest by running backward on sum(y * dy)
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import rms_norm

ROWS, COLS = 5, 12
random.seed(0)
norm = rms_norm(COLS)
# gamma starts at 1 in macrograd; other values test that it's used
norm.gamma.data = [random.uniform(0.5, 1.5) for _ in range(COLS)]
rows = [[random.uniform(-2, 2) for _ in range(COLS)] for _ in range(ROWS)]
rows[1] = [0.0] * COLS  # an all-zero row: eps sets the scale
x = tensor(rows)
dy = tensor([[random.uniform(-1, 1) for _ in range(COLS)] for _ in range(ROWS)])

y = norm(x)
(y * dy).sum().backward()

lines = [f"{ROWS} {COLS}"]
for values in [x.data, norm.gamma.data, dy.data, y.data, x.grad, norm.gamma.grad]:
    lines.append(" ".join(f"{v:.17g}" for v in values))
with open(os.path.join(here, "rms_norm_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote a ({ROWS}, {COLS}) rms norm")
