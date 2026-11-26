"""Section 10 checks (Ising arrays).
(1) Smith normal form of a dephased real Hadamard matrix H: it is 1 + 2 S, with S the Smith form of the binary core M,
    so the number of invariant factors of H with 2-adic valuation 1 is the rank of M modulo 2.
(2) Bent functions on Z_p^2 as p x p matrices B = (w^{f(i,j)}): f is bent iff F_p B F_p has all its entries of modulus p.
    Count at p = 3, and comparison with the matrices F_p P D and their transposes.
(3) The matrices H = F_p P D are bent, and so is the transpose of F_p H F_p / p, which is again of the form F_p P' D'.
(4) Flat eigenvectors of F_p of the form (w^{c j^2}): they exist iff p = 1 mod 4 (4c^2 = -1)."""
import itertools, numpy as np, cmath, math

def smith(A):
    A = [list(map(int, r)) for r in A]; n = len(A); d = []
    for t in range(n):
        while True:
            piv = min(((abs(A[i][j]), i, j) for i in range(t, n) for j in range(t, n) if A[i][j]), default=None)
            if piv is None: return d + [0]*(n-t)
            _, i, j = piv
            A[t], A[i] = A[i], A[t]
            for r in A: r[t], r[j] = r[j], r[t]
            done = True
            for i in range(t+1, n):
                q = A[i][t]//A[t][t]
                if q: A[i] = [x-q*y for x, y in zip(A[i], A[t])]
                if A[i][t]: done = False
            for j in range(t+1, n):
                q = A[t][j]//A[t][t]
                if q:
                    for r in A: r[j] -= q*r[t]
                if A[t][j]: done = False
            if done:
                bad = [(i, j) for i in range(t+1, n) for j in range(t+1, n) if A[i][j] % A[t][t]]
                if not bad: break
                i, j = bad[0]; A[t] = [x+y for x, y in zip(A[t], A[i])]
        d.append(abs(A[t][t]))
    return d

def rank2(M):
    M = [list(r) for r in M]; r = 0
    for c in range(len(M[0])):
        p = next((i for i in range(r, len(M)) if M[i][c] & 1), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]
        for i in range(len(M)):
            if i != r and M[i][c] & 1: M[i] = [(x ^ y) & 1 for x, y in zip(M[i], M[r])]
        r += 1
    return r

W2 = np.array([[1, 1], [1, -1]])
def walsh(n):
    W = np.array([[1]])
    for _ in range(n): W = np.kron(W, W2)
    return W
def paley(q):
    chi = lambda x: 0 if x % q == 0 else (1 if pow(x, (q-1)//2, q) == 1 else -1)
    Q = np.array([[chi(i-j) for j in range(q)] for i in range(q)])
    H = np.ones((q+1, q+1), dtype=int); H[1:, 1:] = Q + np.eye(q, dtype=int); H[1:, 0] = -1
    assert (H @ H.T == (q+1)*np.eye(q+1)).all()
    return H
def dephase(H):
    H = H*H[0][None, :]; return H*H[:, 0][:, None]
for name, H in (("W_8", walsh(3)), ("W_16", walsh(4)), ("W_32", walsh(5)), ("Paley 12", paley(11)), ("Paley 20", paley(19)), ("Paley 24", paley(23))):
    H = dephase(H); M = (1-H[1:, 1:])//2
    sH, sM = smith(H), smith(M)
    v1 = sum(1 for x in sH if x % 2 == 0 and x % 4 != 0)
    print("%-9s Smith(H) = 1 + 2 Smith(M): %s | factors of valuation 1: %d = rank_2(M): %d | Smith form: %s"
          % (name, sorted(sH) == sorted([1]+[2*x for x in sM]), v1, rank2(M), {x: sH.count(x) for x in sorted(set(sH))}))

p = 3; w = cmath.exp(2j*math.pi/p)
F = np.array([[w**(i*j % p) for j in range(p)] for i in range(p)])
pw = np.array([w**k for k in range(p)])
mm = set()
for s in itertools.permutations(range(p)):
    for t in itertools.product(range(p), repeat=p):
        mm.add(tuple((i*s[j]+t[j]) % p for i in range(p) for j in range(p)))
mmt = {tuple(np.array(f).reshape(p, p).T.flatten()) for f in mm}
bent = set()
for f in itertools.product(range(p), repeat=p*p):
    B = pw[np.array(f)].reshape(p, p)
    if np.allclose(np.abs(F @ B @ F), p): bent.add(f)
def quadratic(f):
    f = np.array(f).reshape(p, p); g = f
    for _ in range(3):                      # all the third differences vanish iff the degree is at most 2
        g = np.stack([np.roll(g, (-a, -b), axis=(-2, -1)) - g for a in range(p) for b in range(p)]) % p
    return not g.any()
print("p=3: bent functions on Z_3^2: %d, all of degree <= 2: %s | of type x s(y)+t(y): %d | of type y s(x)+t(x): %d | union: %d"
      % (len(bent), all(quadratic(f) for f in bent), len(mm), len(mmt), len(mm | mmt)))

rng = np.random.default_rng(1); ok = True
for p in (3, 5, 7, 11):
    w = cmath.exp(2j*math.pi/p); F = np.array([[w**(i*j % p) for j in range(p)] for i in range(p)])
    for _ in range(20):
        s = rng.permutation(p); t = rng.integers(0, p, p)
        H = np.array([[w**((i*s[j]+t[j]) % p) for j in range(p)] for i in range(p)])
        K = (F @ H @ F).T/p
        Fi = F.conj().T/p; Mo = Fi @ K                      # K = F (P' D') ?
        ok &= np.allclose(np.abs(F @ H @ F), p) and np.allclose(np.sort(np.abs(Mo), axis=1)[:, :-1], 0) and np.allclose(np.abs(Mo).max(axis=1), 1)
print("F_p P D is bent, and its dual (F H F)^t / p is again of the form F_p P' D':", bool(ok))
for p in (3, 5, 7, 11, 13, 17, 19, 29):
    w = cmath.exp(2j*math.pi/p); F = np.array([[w**(i*j % p) for j in range(p)] for i in range(p)])
    cs = []
    for c in range(1, p):
        b = np.array([w**(c*j*j % p) for j in range(p)]); v = F @ b
        if np.allclose(v, (v[0]/b[0])*b): cs.append((c, np.round(v[0]/b[0]/math.sqrt(p), 6)))
    print("p=%2d (p mod 4 = %d): chirp eigenvectors of F_p, with eigenvalue/sqrt p: %s" % (p, p % 4, cs))
