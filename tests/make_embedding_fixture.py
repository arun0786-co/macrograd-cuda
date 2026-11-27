# writes tests/embedding_macrograd.txt: one forward and backward pass of macrograd's
# embedding, with repeated ids, for kernels/embedding.cu to compare against. run it from
# the repo root with macrograd next to this repo, or give its path:
#   python3 tests/make_embedding_fixture.py [path/to/macrograd]
#
# the file has "V D N" on the first line (vocabulary, row size, number of ids), then the
# table, the ids, dout, out and dtable, one line each, 17 significant digits. dout is the
# gradient that arrives at the output; macrograd gets dtable by running backward on
# sum(out * dout)
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import embedding

V, D = 7, 5
random.seed(0)
emb = embedding(V, D)
ids = [3, 0, 3, 6, 3, 1, 0, 6, 2, 3, 5]  # 3 four times, 0 and 6 twice, 4 never
dout = tensor([[random.uniform(-1, 1) for _ in range(D)] for _ in ids])
out = emb(ids)
(out * dout).sum().backward()
lines = [f"{V} {D} {len(ids)}"]
for values in [emb.weight.data, ids, dout.data, out.data, emb.weight.grad]:
    lines.append(" ".join(f"{v:.17g}" if isinstance(v, float) else str(v) for v in values))
with open(os.path.join(here, "embedding_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote an embedding of {V} x {D} with {len(ids)} ids")
