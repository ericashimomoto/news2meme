import subspace

import torch

X = torch.rand(3, 10, 5)

V, w, dims = subspace.subspace_bases(X, torch.tensor([3,4,5], dtype=torch.uint8), contratio=0.1, return_eigvals=True)

print(V)

print(dims)

print(w)