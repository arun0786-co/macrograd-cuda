# writes tests/attention_backward_macrograd.txt: the gradients of macrograd's
# scaled_dot_product_attention for q, k and v, with the causal mask and fewer queries than
# keys, for kernels/attention_backward.cu to compare against. run it from the repo root
# with macrograd next to this repo, or give its path:
#   python3 tests/make_attention_backward_fixture.py [path/to/macrograd]
#
# the file has "BH T_q T_k d d_v" on the first line, then q, k, v, dout, dq, dk, dv, one
# line each, 17 significant digits. dout is the gradient that arrives at the output;
# macrograd gets the rest by running backward on sum(out * dout)
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import scaled_dot_product_attention

B, H, TQ, TK, D, DV = 2, 2, 5, 7, 4, 3
random.seed(1)
rand = lambda *shape: tensor([random.uniform(-1, 1) for _ in range(B * H * shape[0] * shape[1])], shape=(B, H) + shape)
q, k, v, dout = rand(TQ, D), rand(TK, D), rand(TK, DV), rand(TQ, DV)
out = scaled_dot_product_attention(q, k, v, causal=True)
(out * dout).sum().backward()
lines = [f"{B * H} {TQ} {TK} {D} {DV}"]
for values in [q.data, k.data, v.data, dout.data, q.grad, k.grad, v.grad]:
    lines.append(" ".join(f"{x:.17g}" for x in values))
with open(os.path.join(here, "attention_backward_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote attention gradients for {B * H} heads, {TQ} queries, {TK} keys")
