# order 2q: classes of the cross-ratios lambda under equivalence, then under equivalence and Galois conjugation, by an exact test
import math
from products import *
from equivalence import exponents, equivalent
def classes(L, same):
    cl = []
    for x in L:
        for c in cl:
            if same(c[0], x): c.append(x); break
        else: cl.append([x])
    return cl
for q in (5, 7, 11):
    for chi, l in ((-1, q), (1, 2*q)):
        L = [lam for lam in range(2, q) if legendre(lam, q) == chi]
        E = {lam: exponents(mw(fourier(2), q, (q, 0), (1, lam)), l) for lam in L}
        units = [a for a in range(1, l) if math.gcd(a, l) == 1]
        c1 = classes(L, lambda x, y: equivalent(E[x], E[y], l))
        c2 = classes(L, lambda x, y: any(equivalent(E[x], (a*E[y]) % l, l) for a in units))
        print("q=%d, %s: equivalence %s, with Galois conjugation %s" % (q, "non-squares" if chi == -1 else "squares", c1, c2), flush=True)
