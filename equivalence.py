"""exact equivalence test for Butson matrices, given by exponent matrices over Z_l: E' = P(E + r 1^t + 1 c^t)Q"""
import numpy as np, math
from collections import Counter

def exponents(H, l):
    D = H/H[0][None, :]; D = D/D[:, [0]]
    a = np.angle(D)*l/(2*math.pi); E = np.round(a).astype(int) % l
    assert np.allclose(a, np.round(a), atol=1e-6)
    return E

def core(E, i0, j0, l):
    N = (E - E[i0][None, :]) % l
    return (N - N[:, [j0]]) % l

def equivalent(E1, E2, l):
    """True iff E1, E2 are equivalent; backtracking on the rows, matching multisets of column prefixes"""
    n = len(E1); N1 = core(E1, 0, 0, l)
    for i0 in range(n):
        for j0 in range(n):
            N2 = core(E2, i0, j0, l)
            if Counter(tuple(sorted(r)) for r in N1) != Counter(tuple(sorted(r)) for r in N2): continue
            def rec(k, used, pref2, target):
                # pref2: list of column prefix tuples of N2 for the chosen rows; target: Counter of column prefixes of N1[:k]
                if k == n: return True
                t1 = Counter(tuple(N1[:k+1, j]) for j in range(n))
                for r in range(n):
                    if r in used: continue
                    new = [pref2[j] + (N2[r, j],) for j in range(n)]
                    if Counter(new) == t1 and rec(k+1, used | {r}, new, t1): return True
                return False
            if rec(1, {i0}, [(0,)]*n, None) if tuple(N1[0]) == tuple([0]*n) else False: return True
    return False

