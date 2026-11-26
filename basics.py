"""Checks for Sections 1 and 5: the functional equation of the characteristic polynomial of a complex Hadamard matrix,
the factorization F_p = L Delta U with U invertible over Z[w], and the formulae for (F_p T_c)^3 from Section 8."""
import numpy as np, cmath, math
from math import comb
rng = np.random.default_rng(0)
for p in (3, 5, 7, 11):
    w = cmath.exp(2j*math.pi/p); pi = w-1
    F = np.array([[w**(i*j % p) for j in range(p)] for i in range(p)])
    # a matrix equivalent to F_p, and the functional equation chi_H(x) = (-x/N)^N det(H) chi_{H*}(N/x)
    H = F[:, rng.permutation(p)] * np.exp(2j*np.pi*rng.integers(0, p, p)/p)
    x = 0.7+0.4j
    lhs = np.linalg.det(x*np.eye(p)-H); rhs = (-x/p)**p*np.linalg.det(H)*np.linalg.det(p/x*np.eye(p)-H.conj().T)
    L = np.array([[comb(i, k) for k in range(p)] for i in range(p)], dtype=complex)
    u = [sum(w**t for t in range(j)) for j in range(p)]
    U = np.array([[u[j]**k for j in range(p)] for k in range(p)])
    D = np.diag([pi**k for k in range(p)])
    # det U is a unit: the product of its Galois conjugates has absolute value 1
    nrm = 1
    for a in range(1, p):
        wa = cmath.exp(2j*math.pi*a/p); ua = [sum(wa**t for t in range(j)) for j in range(p)]
        nrm *= np.linalg.det(np.array([[ua[j]**k for j in range(p)] for k in range(p)]))
    ok3 = []
    for c, kind in ((pow(2, -1, p), "scalar"), ((-pow(2, -1, p)) % p, "F^2")):
        G = sum(w**(c*t*t % p) for t in range(p)); T = np.diag([w**(c*j*j % p) for j in range(p)])
        M = np.linalg.matrix_power(F @ T, 3)
        ok3.append(np.allclose(M, p*G*np.eye(p)) if kind == "scalar" else np.allclose(M, G*(F @ F)))
    print("p=%2d  functional equation: %s  F = L Delta U: %s  |norm det U| = %.6f  (F T_c)^3 formulae: %s"
          % (p, np.isclose(lhs, rhs), np.allclose(L @ D @ U, F), abs(nrm), ok3))
