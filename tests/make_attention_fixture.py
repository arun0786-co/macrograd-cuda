# writes tests/attention_macrograd.txt: macrograd's scaled_dot_product_attention on small
# q, k, v, without and with the causal mask, for kernels/attention.cu to compare against.
# there are fewer queries than keys, so the causal mask is the shifted one macrograd uses
# when generating with cached keys. run it from the repo root with macrograd next to this
# repo, or give its path:
#   python3 tests/make_attention_fixture.py [path/to/macrograd]
#
# the file has "BH T_q T_k d d_v" on the first line (BH heads in all), then q, k, v, the
# output without the mask and the output with it, one line each, 17 significant digits
import os
import random
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "..", "macrograd"))
from engine import tensor
from nn import scaled_dot_product_attention

B, H, TQ, TK, D, DV = 2, 2, 5, 7, 4, 3
random.seed(0)
rand = lambda *shape: tensor([random.uniform(-1, 1) for _ in range(B * H * shape[0] * shape[1])], shape=(B, H) + shape)
q, k, v = rand(TQ, D), rand(TK, D), rand(TK, DV)
plain = scaled_dot_product_attention(q, k, v)
causal = scaled_dot_product_attention(q, k, v, causal=True)
lines = [f"{B * H} {TQ} {TK} {D} {DV}"]
for t in [q, k, v, plain, causal]:
    lines.append(" ".join(f"{x:.17g}" for x in t.data))
with open(os.path.join(here, "attention_macrograd.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"wrote attention for {B * H} heads, {TQ} queries, {TK} keys")
