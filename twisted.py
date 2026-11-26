# check of the defect formula: direct computation of the defect against the sum of dim ker(Lambda_K - Theta_x)
# the last 3 matrices K, from the affine family through F_4, are there in order to detect a wrong sign in Theta_x
import random, cmath, numpy as np
from products import *
rng = random.Random(2); F2 = fourier(2)
def family(a): return np.array([[1, 1, 1, 1], [1, 1j*a, -1, -1j*a], [1, -1, 1, -1], [1, -1j*a, -1, 1j*a]])
tests = [("F_2", F2, (3, 5, 7), 6), ("F_3", fourier(3), (3, 5, 7), 6), ("F_2 x F_2", np.kron(F2, F2), (3, 5, 7), 6), ("F_4", fourier(4), (3, 5, 7), 6),
         ("F_5", fourier(5), (5,), 2), ("F_6", fourier(6), (5,), 2), ("F_4(e^{0.3i})", family(cmath.exp(0.3j)), (5, 7), 8),
         ("F_4(e^{2 pi i/5})", family(cmath.exp(2j*cmath.pi/5)), (5, 7), 8), ("F_4(e^{2 pi i/7})", family(cmath.exp(2j*cmath.pi/7)), (5, 7), 8)]
for name, K, Q, n in tests:
    for q in Q:
        res = []
        for _ in range(n):
            s, t = configuration(len(K), q, rng)
            res.append((defect(mw(K, q, s, t)), twisted(K, q, s, t)))
        print("K = %s, q = %d: formula holds on %d random configurations: %s, defects found: %s" % (name, q, n, all(a == b for a, b in res), sorted(set(a for a, b in res))), flush=True)
