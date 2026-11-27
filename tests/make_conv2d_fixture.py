# writes tests/conv2d_macrograd.txt: macrograd's conv2d on small images, once with stride
# 1 and padding 1 and once with stride 2 and no padding, for kernels/conv2d.cu to compare
# against. run it from the repo root with macrograd next to this repo, or give its path:
#   python3 tests/make_conv2d_fixture.py [path/to/macrograd]
#
# the file has "N C H W O KH KW" on the first line, then x and w, then for each of the two
# settings a line "stride padding" followed by the output, 17 significant digits
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor

N, C, H, W, O, KH, KW = 2, 3, 7, 6, 4, 3, 3
random.seed(0)
x = tensor([random.uniform(-1, 1) for _ in range(N * C * H * W)], shape=(N, C, H, W))
w = tensor([random.uniform(-1, 1) for _ in range(O * C * KH * KW)], shape=(O, C, KH, KW))
fmt = lambda values: " ".join(f"{v:.17g}" for v in values)
lines = [f"{N} {C} {H} {W} {O} {KH} {KW}", fmt(x.data), fmt(w.data)]
for stride, padding in [(1, 1), (2, 0)]:
    out = x.conv2d(w, stride=stride, padding=padding)
    lines += [f"{stride} {padding}", fmt(out.data)]
with open(os.path.join(here, "conv2d_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote conv2d of ({N}, {C}, {H}, {W}) with ({O}, {C}, {KH}, {KW}) filters")
