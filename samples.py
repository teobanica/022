# K = F_N with N = 3, 5, 7: defects of random configurations, sorted by the numbers of distinct points among the s_i and among the t_j
# computed with the twisted formula, cross-checked against the direct computation at order 25
import random
from collections import Counter, defaultdict
from products import *
rng = random.Random(3)
chk = []
for _ in range(3):
    s, t = configuration(5, 5, rng); chk.append(twisted(fourier(5), 5, s, t) == defect(mw(fourier(5), 5, s, t)))
print("cross-check at order 25:", chk, flush=True)
for N, q, n in ((3, 5, 300), (3, 7, 200), (3, 11, 60), (3, 13, 40), (3, 17, 20), (5, 5, 150), (5, 7, 150), (5, 11, 40), (5, 13, 30), (7, 7, 60), (7, 11, 60), (7, 13, 40), (7, 17, 15)):
    tab = defaultdict(Counter); K = fourier(N)
    for _ in range(n):
        s, t = configuration(N, q, rng); tab[(len(set(s)), len(set(t)))][twisted(K, q, s, t)] += 1
    if q + 1 >= 2*N:
        for _ in range(12):
            s, t = configuration(N, q, rng, distinct=True); tab[(N, N)][twisted(K, q, s, t)] += 1
    print("N=%d q=%d:" % (N, q))
    for k in sorted(tab): print("    %d distinct s, %d distinct t: defects %s" % (k[0], k[1], dict(sorted(tab[k].items()))), flush=True)
