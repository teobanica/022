"""The matrices M = (w^{i j^d}), gcd(d, p-1) = 1: F_p with columns permuted by the power map j -> j^d.
Check: (1) spectrum = sqrt p, -sqrt p, and for each orbit of m -> -m/d on Z_{p-1} - {0}, of length k, the k-th roots of the
product of the Gauss sums along the orbit; (2) finite order iff (-d)^v = -1 mod p-1 for some v; projective order."""
import numpy as np, cmath, math
def test(p, d):
    w = cmath.exp(2j*math.pi/p)
    U = np.array([[w**(i*pow(j, d, p) % p) for j in range(p)] for i in range(p)])/math.sqrt(p)
    order = None; M = np.eye(p)
    for n in range(1, 8*p):
        M = M @ U
        if np.allclose(M, M[0, 0]*np.eye(p), atol=1e-7): order = n; break
    # prediction
    x, pred = (-d) % (p-1), False
    for v in range(1, p):
        if x == p-2: pred = True
        x = (x*(-d)) % (p-1)
    # Gauss sum description of the spectrum
    g = next(g for g in range(2, p) if len({pow(g, k, p) for k in range(p-1)}) == p-1)
    log = {pow(g, k, p): k for k in range(p-1)}
    gs = lambda m: sum(cmath.exp(2j*math.pi*m*log[x]/(p-1))*w**x for x in range(1, p))
    dinv = pow(d, -1, p-1)
    ev = [math.sqrt(p), -math.sqrt(p)]; seen = set()
    for m in range(1, p-1):
        if m in seen: continue
        orb, x = [], m
        while x not in orb: orb.append(x); x = (-dinv*x) % (p-1)
        seen |= set(orb); k = len(orb)
        # H chi_m = g(chi_{d' m}) chi_{-d' m}
        prod = np.prod([gs((dinv*y) % (p-1)) for y in orb])
        r = prod**(1/k)
        ev += [r*cmath.exp(2j*math.pi*t/k) for t in range(k)]
    key = lambda z: (round(z.real, 5), round(z.imag, 5))
    spec = np.allclose(sorted(np.linalg.eigvals(U*math.sqrt(p)), key=key), sorted(ev, key=key), atol=1e-5)
    return order, pred, spec
bad = 0
for p in (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47):
    out = []
    for d in range(2, p-2):
        if math.gcd(d, p-1) != 1: continue
        order, pred, spec = test(p, d)
        bad += (order is not None) != pred or not spec
        if order: out.append("d=%d: order %d" % (d, order))
    print("p=%2d  finite order power maps (1 < d < p-2):" % p, ", ".join(out) if out else "none")
print("mismatches with the criterion or with the Gauss sum formula:", bad)
# the first example, p = 11, d = 3
p, d = 11, 3; w = cmath.exp(2j*math.pi/p)
U = np.array([[w**(i*pow(j, d, p) % p) for j in range(p)] for i in range(p)])/math.sqrt(p)
F = np.array([[w**(i*j % p) for j in range(p)] for i in range(p)])/math.sqrt(p)
U2 = U @ U
print("p=11, d=3: U^4 = 1:", np.allclose(np.linalg.matrix_power(U, 4), np.eye(p)), "| U^2 monomial:", np.allclose(np.sort(np.abs(U2), axis=1)[:, :-1], 0),
      "| |entries of U^2|^2 * p range: %.2f..%.2f" % ((np.abs(U2)**2*p).min(), (np.abs(U2)**2*p).max()))
mult = lambda A: {k: int(round(sum(abs(np.linalg.eigvals(A)-k) < 1e-6))) for k in (1, -1, 1j, -1j)}
print("   eigenvalue multiplicities (1,-1,i,-i): U", mult(U), " F", mult(F))
