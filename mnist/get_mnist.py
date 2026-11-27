# downloads mnist into data/mnist/ (gitignored) and unpacks it: 60,000 training and
# 10,000 test images of 28 x 28 handwritten digits with their labels, in the idx format
# (a short header, then the bytes). run it from the repo root: python3 mnist/get_mnist.py
import gzip
import os
import urllib.request

URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {"train-images-idx3-ubyte": 47040016, "train-labels-idx1-ubyte": 60008,
         "t10k-images-idx3-ubyte": 7840016, "t10k-labels-idx1-ubyte": 10008}
here = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(here, "..", "data", "mnist")
os.makedirs(out, exist_ok=True)
for name, size in FILES.items():
    path = os.path.join(out, name)
    if not os.path.exists(path):
        print(f"downloading {name}")
        with urllib.request.urlopen(URL + name + ".gz", timeout=60) as response:
            data = gzip.decompress(response.read())
        with open(path, "wb") as f:
            f.write(data)
    if os.path.getsize(path) != size:
        raise SystemExit(f"{path} has {os.path.getsize(path)} bytes, expected {size}")
print(f"mnist is in {os.path.relpath(out)}")
