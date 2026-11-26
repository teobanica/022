# order 2q: K = F_2 and the configuration (infinity, 0; 1, lambda), of cross-ratio lambda
# the defect is 0 when lambda is not a square modulo q, and 2(q-1) otherwise; then the configuration S = {1,3}, T = {0,2}, in the labeling by the matrices D^k F_q
from products import *
for q in (3, 5, 7, 11, 13, 17, 19, 23):
    out = []
    for lam in range(2, q):
        assert cross_ratio(q, 0, 1, lam, q) == lam
        H = mw(fourier(2), q, (q, 0), (1, lam))
        out.append((lam, legendre(lam, q), defect(H), level(H)))
    rule = all((d == 0 and l == q) if chi == -1 else (d == 2*(q-1) and l == 2*q) for (lam, chi, d, l) in out)
    print("q=%d: rule holds: %s" % (q, rule))
    print("    lambda:square or not:defect:level  ", " ".join("%d:%s:%d:%d" % (lam, "S" if chi == 1 else "N", d, l) for (lam, chi, d, l) in out), flush=True)
print("odd and even labels, D^k F_q = T_{k/2} F_q up to a permutation of the columns:")
for q in (5, 7, 11, 13, 17, 19, 23):
    h = pow(2, -1, q); s = (h, 3*h % q); t = (0, 2*h % q)
    lam = cross_ratio(s[0], s[1], t[0], t[1], q)
    print("    q=%d (%d mod 3): cross-ratio %d, square: %s, defect %d" % (q, q % 3, lam, legendre(lam, q) == 1, defect(mw(fourier(2), q, s, t))), flush=True)
