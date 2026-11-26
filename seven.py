"""Exhaustive families in the class of F_7 (all permutations with D = 1, and all D for 3 given permutations),
and the Clifford test at p = 5: the supersingular matrices F_5 P D are exactly the Clifford ones."""
import itertools, numpy as np
from collections import Counter
from cyclo import *

def strata(p, P, E, chunk=20000):
    cnt, low, ss = Counter(), None, []
    for a in range(0, len(P), chunk):
        H = fourier_twists(p, P[a:a+chunk], E[a:a+chunk])
        V = np.stack([val(x, p) for x in elementary(H, p)], axis=1)
        low = V.min(axis=0) if low is None else np.minimum(low, V.min(axis=0))
        for t, row in enumerate(V):
            sl = newton([int(x) for x in row]); cnt[sl] += 1
            if len(set(sl)) == 1: ss.append(a+t)
    return cnt, low, ss

def show(title, cnt, low):
    print(title, "| %d matrices | min valuations of e_k:" % sum(cnt.values()), [int(x) for x in low])
    for sl, c in sorted(cnt.items()): print("    slopes", " ".join(str(s) for s in sl), "  count", c)

def is_clifford(p, perm, ex):
    """perm affine and ex a polynomial of degree <= 2 on Z_p"""
    a = (perm[1]-perm[0]) % p
    aff = all((perm[j]-perm[0]-a*j) % p == 0 for j in range(p)) and a != 0
    d3 = np.diff(np.concatenate([ex, ex[:3]]), n=3) % p     # third finite differences vanish iff degree <= 2
    return aff and not d3.any()

if __name__ == "__main__":
    # p = 5: the supersingular matrices are exactly the Clifford ones
    p = 5
    perms = np.array(list(itertools.permutations(range(p))))
    ex = np.array([(0,)+e for e in itertools.product(range(p), repeat=p-1)])
    idx = np.array(list(itertools.product(range(len(perms)), range(len(ex)))))
    P, E = perms[idx[:, 0]], ex[idx[:, 1]]
    cnt, low, ss = strata(p, P, E)
    cl = [t for t in range(len(P)) if is_clifford(p, P[t], E[t])]
    print("p = 5: supersingular %d, Clifford %d, equal sets: %s" % (len(ss), len(cl), set(ss) == set(cl)))

    p = 7
    perms = np.array(list(itertools.permutations(range(p))))
    cnt, low, ss = strata(p, perms, np.zeros_like(perms))
    show("p = 7, all F P (D = 1)", cnt, low)
    print("    supersingular all Clifford:", all(is_clifford(p, perms[t], np.zeros(p, dtype=int)) for t in ss), len(ss))
    ex = np.array([(0,)+e for e in itertools.product(range(p), repeat=p-1)])
    for name, s in (("identity", np.arange(p)), ("inversion", np.array([0]+[pow(j, -1, p) for j in range(1, p)])), ("cube j^5", np.array([pow(j, 5, p) for j in range(p)]))):
        Pm = np.tile(s, (len(ex), 1))
        cnt, low, ss = strata(p, Pm, ex)
        show("p = 7, P = %s, all D" % name, cnt, low)
        print("    supersingular %d, all Clifford: %s" % (len(ss), all(is_clifford(p, Pm[t], ex[t]) for t in ss)))
