"""Section 6 checks: the matrix G_p = (w^{i/j}), with the convention 1/0 = 0.
Spectrum = {sqrt p, -sqrt p} and the Gauss sums g(chi), chi nontrivial; trace 1+(p-1)w;
G_p^2 = p + (1 + K(i/j)) with K the Kloosterman sums; exact Newton slopes at pi (Stickelberger)."""
import numpy as np, cmath, math
from cyclo import *
for p in (3, 5, 7, 11, 13):
    w = cmath.exp(2j*math.pi/p)
    inv = [0]+[pow(j, -1, p) for j in range(1, p)]
    G = np.array([[w**(i*inv[j] % p) for j in range(p)] for i in range(p)])
    g = next(g for g in range(2, p+1) if len({pow(g, k, p) for k in range(p-1)}) == p-1) if p > 2 else 1
    log = {pow(g, k, p): k for k in range(p-1)}
    gs = [sum(cmath.exp(2j*math.pi*m*log[x]/(p-1))*w**x for x in range(1, p)) for m in range(1, p-1)] + [math.sqrt(p), -math.sqrt(p)]
    key = lambda z: (round(z.real, 6), round(z.imag, 6))
    spec = np.allclose(sorted(np.linalg.eigvals(G), key=key), sorted(gs, key=key), atol=1e-6)
    K = lambda a: sum(w**((x + a*inv[x]) % p) for x in range(1, p))
    S = np.zeros((p, p), dtype=complex); S[0, 0] = p
    for i in range(1, p):
        for j in range(1, p): S[i, j] = 1 + K(i*inv[j] % p)
    H = fourier_twists(p, np.array([inv]), np.zeros((1, p), dtype=np.int64)).astype(object)
    V = [int(val(x, p)[0]) for x in elementary(H, p)]
    print("p=%2d spectrum: %s  trace: %s  square: %s  slopes x (p-1): %s" % (p, spec, np.isclose(np.trace(G), 1+(p-1)*w), np.allclose(G @ G, S), [str(s) for s in newton(V)]))
