# check of the symmetries: PSL_2(F_q) acting on the lines of F_q^2 preserves the equivalence class (exact test, K = F_2),
# and PGL_2(F_q) preserves the defect (K = F_2, F_3)
import random
from products import *
from equivalence import exponents, equivalent
rng = random.Random(4)
def act(g, c, q):
    u = direction(c, q); v = ((g[0]*u[0] + g[1]*u[1]) % q, (g[2]*u[0] + g[3]*u[1]) % q)
    return q if v[0] == 0 else v[1]*pow(2*v[0], -1, q) % q
def element(q, square):
    while True:
        g = [rng.randrange(q) for _ in range(4)]; d = (g[0]*g[3] - g[1]*g[2]) % q
        if d and (legendre(d, q) == 1) == square: return g
for q in (5, 7):
    res = []
    for _ in range(6):
        s, t = configuration(2, q, rng, distinct=True); g = element(q, True)
        gs, gt = tuple(act(g, c, q) for c in s), tuple(act(g, c, q) for c in t)
        l = level(mw(fourier(2), q, s, t))
        res.append(equivalent(exponents(mw(fourier(2), q, s, t), l), exponents(mw(fourier(2), q, gs, gt), l), l))
    print("K = F_2, q = %d: H(s,t) and H(gs,gt) equivalent, for 6 random configurations and g in PSL_2: %s" % (q, res), flush=True)
for N, q, n in ((2, 5, 20), (2, 7, 20), (2, 11, 10), (3, 5, 20), (3, 7, 20)):
    res = []
    for _ in range(n):
        s, t = configuration(N, q, rng); g = element(q, rng.random() < 0.3)
        gs, gt = tuple(act(g, c, q) for c in s), tuple(act(g, c, q) for c in t)
        res.append(twisted(fourier(N), q, s, t) == twisted(fourier(N), q, gs, gt))
    print("K = F_%d, q = %d: same defect for (s,t) and (gs,gt), for %d random configurations and g in PGL_2: %s" % (N, q, n, all(res)), flush=True)
