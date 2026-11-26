"""Traces of powers of H = (w^{i s(j) + t(j)}), s a polynomial of degree d, t of degree <= d, against the bound d^n p^{n/2}
(Deligne), which applies when p does not divide d(d+1) and (-d)^n is not 1 mod p."""
import numpy as np, cmath, math, itertools
def mat(p, s, t=lambda j: 0):
    w = np.exp(2j*np.pi/p); i = np.arange(p)
    sj = np.array([s(j) % p for j in range(p)]); tj = np.array([t(j) % p for j in range(p)])
    return w**((np.outer(i, sj) + tj[None, :]) % p)
def report(name, d, primes, s, t=lambda j: 0, nmax=6):
    worst = 0
    for p in primes:
        H = mat(p, s, t); U = H/math.sqrt(p)
        perm = len({s(j) % p for j in range(p)}) == p
        M = np.eye(p); row = []
        for n in range(1, nmax+1):
            M = M @ U
            good = (d*(d+1)) % p != 0 and pow(-d, n, p) != 1
            r = abs(np.trace(M))/d**n
            row.append(("%.2f" % r) + ("" if good else "*"))
            if good: worst = max(worst, r)
        print("%s p=%3d perm=%s  |Tr U^n|/d^n, n=1..%d: %s" % (name, p, "y" if perm else "n", nmax, " ".join(row)))
    print("   largest ratio over the cases covered by the theorem: %.3f (must be <= 1); * = excluded case" % worst)
report("s=x^3      ", 3, [5, 7, 11, 13, 17, 23, 29, 41, 47, 53, 101], lambda j: j**3)
report("s=x^3+x,t=x^3", 3, [5, 7, 11, 13, 17, 23, 29, 41, 101], lambda j: j**3+j, lambda j: 2*j**3+j*j)
report("s=x^2      ", 2, [3, 5, 7, 11, 13, 17, 31, 43], lambda j: j*j)
report("s=x^5      ", 5, [7, 11, 13, 17, 23, 31, 521], lambda j: j**5, nmax=4)
# equidistribution: Fourier coefficients of the spectral measure for s = x^3, larger p = 2 mod 3
for p in (101, 503, 1013, 2003):
    U = mat(p, lambda j: pow(j, 3, p))/math.sqrt(p); ev = np.linalg.eigvals(U)
    print("p=%4d  |Tr U^n|/p for n=1..5:" % p, " ".join("%.4f" % (abs((ev**n).sum())/p) for n in range(1, 6)))
# Alltop type: H = F_p diag(w^{c j^3}), top form the Fermat cubic: |Tr U^n| <= 2^n for all n (p > 3)
worst = 0
for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 101, 211):
    w = np.exp(2j*np.pi/p); i = np.arange(p)
    for c in (1, 2, 3):
        U = w**((np.outer(i, i) + c*i**3) % p)/math.sqrt(p)
        M = np.eye(p)
        for n in range(1, 9):
            M = M @ U; worst = max(worst, abs(np.trace(M))/2**n)
print("Alltop type, p<=211, n<=8: largest |Tr U^n|/2^n = %.3f (must be <= 1)" % worst)
