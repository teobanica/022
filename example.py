"""Exact checks for Section 7 (p = 7): the 72 matrices of Legendre type E_ij = w^{alpha_j (i j + eps_j j^2/2)} are
exactly the finite order, non-Clifford matrices found by legendre.py, and E^7 or E^15 is a scalar (exact arithmetic)."""
import itertools, numpy as np
from cyclo import *
from legendre import family
from seven import strata
p = 7
chi = [0]+[1 if pow(j, 3, p) == 1 else -1 for j in range(1, p)]
half = pow(2, -1, p)
mats = {}
for al, r, (ep, em) in itertools.product(range(1, p), (2, 4), [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a != b]):
    s = tuple(((al if chi[j] >= 0 else r*al)*j) % p for j in range(p))
    e = tuple(((al if chi[j] >= 0 else r*al)*(ep if chi[j] >= 0 else em)*j*j*half) % p for j in range(p))
    mats[(s, e)] = (ep, em)
par, P, E = family(p)
cnt, low, ss = strata(p, P, E)
exo = {(tuple(int(x) for x in P[t]), tuple(int(x) for x in E[t])) for t in ss if par[t][1] or par[t][3]}
print("Legendre type matrices: %d, equal to the exotic matrices of legendre.py: %s" % (len(mats), set(mats) == exo))
orders = {}
for (s, e), eps in mats.items():
    H = fourier_twists(p, np.array([s]), np.array([e])).astype(object)[0]
    M, n = H, 1
    def scalar(M):
        M = normal(M, p)
        return all(not M[i, j].any() for i in range(p) for j in range(p) if i != j) and all((M[i, i] == M[0, 0]).all() for i in range(p))
    while not scalar(M):
        M = matmul(M, H, p); n += 1
        assert n <= 15
    orders.setdefault(n, set()).add(eps)
for n, v in sorted(orders.items()): print("exact projective order %d for (eps+, eps-) in %s" % (n, sorted(v)))
