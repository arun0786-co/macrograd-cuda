# writes tests/optim_macrograd.txt: a few steps of macrograd's SGD, Adam and AdamW on the
# same weights and gradients, for kernels/optim.cu to compare against. run it from the
# repo root with macrograd next to this repo, or give its path:
#   python3 tests/make_optim_fixture.py [path/to/macrograd]
#
# the file has "N STEPS" on the first line, then the N starting weights, then the N
# gradients of each step. then for each optimizer a line with its name and settings,
# followed by the N weights after each step. numbers have 17 significant digits
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from optim import SGD, Adam, AdamW

N, STEPS = 37, 5
random.seed(0)
w0 = [random.uniform(-1, 1) for _ in range(N)]
grads = [[random.uniform(-1, 1) for _ in range(N)] for _ in range(STEPS)]
# the gradient of one weight is tiny, so Adam's eps matters for it
for g in grads:
    g[3] *= 1e-7

# name, settings as the kernel reads them (lr, momentum or beta1, beta2, eps, weight decay), and the macrograd optimizer
configs = [
    ("sgd", (0.1, 0.9, 0.0, 0.0, 0.01), lambda p: SGD([p], lr=0.1, momentum=0.9, weight_decay=0.01)),
    ("adam", (0.01, 0.9, 0.999, 1e-8, 0.0), lambda p: Adam([p], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8)),
    ("adamw", (0.01, 0.9, 0.999, 1e-8, 0.1), lambda p: AdamW([p], lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1)),
]

fmt = lambda values: " ".join(f"{v:.17g}" for v in values)
lines = [f"{N} {STEPS}", fmt(w0)] + [fmt(g) for g in grads]
for name, settings, make in configs:
    p = tensor(w0[:])
    opt = make(p)
    lines.append(f"{name} {fmt(settings)}")
    for g in grads:
        p.grad = g[:]
        opt.step()
        lines.append(fmt(p.data))
with open(os.path.join(here, "optim_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote {STEPS} steps of {len(configs)} optimizers on {N} weights")
