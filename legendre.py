"""The 'two-speed' matrices H_ij = w^{ i j (a + b chi(j)) + j^2 (c + e chi(j)) } in the class of F_p, with chi the Legendre symbol:
which of them have finite order (= are supersingular)?  Clifford means b = e = 0."""
import sys, itertools, numpy as np, cmath, math
from collections import Counter
from fractions import Fraction
from cyclo import *
from seven import strata

def family(p):
    chi = np.array([0]+[1 if pow(j, (p-1)//2, p) == 1 else -1 for j in range(1, p)])
    j = np.arange(p)
    par, P, E = [], [], []
    for a, b, c, e in itertools.product(range(p), repeat=4):
        s = (j*(a+b*chi)) % p
        if len(set(s.tolist())) < p: continue
        par.append((a, b, c, e)); P.append(s); E.append((j*j*(c+e*chi)) % p)
    return par, np.array(P), np.array(E)

def spectrum(p, s, e):
    w = cmath.exp(2j*math.pi/p)
    U = np.array([[w**((i*s[j]+e[j]) % p) for j in range(p)] for i in range(p)])/math.sqrt(p)
    M = np.eye(p)
    for n in range(1, 2000):
        M = M @ U
        if np.allclose(M, M[0, 0]*np.eye(p), atol=1e-6): break
    ev = np.linalg.eigvals(U); r = np.angle(ev/ev[0])/(2*np.pi)
    return n, tuple(sorted(int(round(x*n)) % n for x in r))

if __name__ == "__main__":
    for p in [int(x) for x in sys.argv[1:]]:
        par, P, E = family(p)
        cnt, low, ss = strata(p, P, E, chunk=2000)
        exo = [t for t in ss if par[t][1] or par[t][3]]
        print("p = %d: %d matrices, %d of finite order, %d of them not Clifford" % (p, len(par), len(ss), len(exo)))
        types = Counter()
        for t in exo: types[spectrum(p, P[t], E[t])] += 1
        for (n, sp), c in sorted(types.items()): print("   projective order %d, eigenvalues exp(2 pi i k/%d) with k in %s: %d matrices" % (n, n, list(sp), c))
        if exo: print("   parameters (a,b,c,e):", [par[t] for t in exo][:12], "...")
