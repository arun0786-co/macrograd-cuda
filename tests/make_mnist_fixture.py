# writes tests/mnist_macrograd.txt: the loss of macrograd's MLP on the first training
# steps of mnist, for mnist/train_mlp.cu --check to compare against. needs
# data/mnist (python3 mnist/get_mnist.py) and macrograd next to this repo, or its path:
#   python3 tests/make_mnist_fixture.py [path/to/macrograd]
#
# the model is linear(784, 128) -> relu -> linear(128, 10), cross entropy, SGD with
# lr 0.1 and momentum 0.9, on batches of 32 images taken in file order. the weights come
# from a hash of their index instead of a random generator, so the CUDA side can make
# exactly the same numbers: every one is a whole number times 2^-24 (first layer) or
# 2^-22 (second), which float and double both hold exactly. the biases start at 0.
# the file has one line per step: the step number and the loss before that step's update
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import linear
from optim import SGD

BATCH, STEPS = 32, 20


def hashed(n, seed, shift):
    # (i * 2654435761 + seed * 40503) mod 2^32, top 20 bits, centred on 0, times 2^-shift
    values = []
    for i in range(n):
        u = ((i * 2654435761 + seed * 40503) % 2**32) >> 12
        values.append((u - 2**19) * 2.0 ** -shift)
    return values


def read_idx(path, offset):
    with open(path, "rb") as f:
        return f.read()[offset:]


data = os.path.join(here, "..", "data", "mnist")
pixels = read_idx(os.path.join(data, "train-images-idx3-ubyte"), 16)
labels = read_idx(os.path.join(data, "train-labels-idx1-ubyte"), 8)

l1, l2 = linear(784, 128), linear(128, 10)
l1.w.data, l2.w.data = hashed(784 * 128, 1, 24), hashed(128 * 10, 2, 22)
params = [l1.w, l1.b, l2.w, l2.b]
opt = SGD(params, lr=0.1, momentum=0.9)

lines = []
for step in range(STEPS):
    start = time.time()
    first = step * BATCH
    x = tensor([[pixels[(first + r) * 784 + j] / 255.0 for j in range(784)] for r in range(BATCH)])
    y = list(labels[first:first + BATCH])
    loss = l2(l1(x).relu()).log_softmax().nll_loss(y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    lines.append(f"{step} {loss.data[0]:.17g}")
    print(f"step {step}: loss {loss.data[0]:.6f} ({time.time() - start:.1f}s)")
with open(os.path.join(here, "mnist_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
