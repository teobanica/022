"""McNulty-Weigert matrices: H_{ia,jb} = sqrt(q) K_ij (B_{s_i}^* B_{t_j})_ab, with K complex Hadamard of order N, q an odd prime,
and B_c = T_c F_q / sqrt(q), T_c = diag(w^{c x^2}), for c in F_q, B_infinity = 1 (infinity is coded by the integer q).
These q+1 bases are labeled by the lines of F_q^2, the line of c being spanned by u_c = (1,2c), and u_infinity = (0,1)."""
import numpy as np, cmath, math, itertools

def fourier(n):
    w = cmath.exp(2j*math.pi/n)
    return np.array([[w**(i*j % n) for j in range(n)] for i in range(n)])

def basis(q, c):
    if c == q: return np.eye(q, dtype=complex)
    w = cmath.exp(2j*math.pi/q)
    return np.diag([w**(c*x*x % q) for x in range(q)]) @ fourier(q)/math.sqrt(q)

def direction(c, q): return (0, 1) if c == q else (1, 2*c % q)

def mw(K, q, s, t):
    N = len(K); H = np.zeros((N*q, N*q), dtype=complex)
    B = {c: basis(q, c) for c in set(s) | set(t)}
    for i in range(N):
        for j in range(N):
            H[i*q:(i+1)*q, j*q:(j+1)*q] = math.sqrt(q)*K[i, j]*(B[s[i]].conj().T @ B[t[j]])
    assert np.allclose(np.abs(H), 1), "not flat: some s_i equals some t_j"
    return H

def defect(H, tol=1e-7):
    """dim of {A real : sum_k H_ik conj(H_jk) (A_ik - A_jk) = 0 for all i<j}, minus 2n-1"""
    n = len(H)
    P = [(i, j) for i in range(n) for j in range(i+1, n)]
    R = np.zeros((2*len(P), n, n))
    for m, (i, j) in enumerate(P):
        c = H[i]*H[j].conj()
        R[2*m, i] = c.real; R[2*m, j] = -c.real; R[2*m+1, i] = c.imag; R[2*m+1, j] = -c.imag
    S = np.linalg.svd(R.reshape(2*len(P), n*n), compute_uv=False)
    return n*n - int((S > tol*S[0]).sum()) - (2*n - 1)

def exponent(x, s, t, q):
    """c(x) = n m det(u_s, u_t), where x = n u_s + m u_t"""
    u, v = direction(s, q), direction(t, q)
    det = (u[0]*v[1] - u[1]*v[0]) % q; inv = pow(det, -1, q)
    n = (x[0]*v[1] - x[1]*v[0])*inv % q; m = (u[0]*x[1] - u[1]*x[0])*inv % q
    return n*m*det % q

def twisted(K, q, s, t):
    """the defect, computed as the sum over x in F_q^2 of dim ker(Lambda_K - Theta_x), minus 2Nq-1"""
    N = len(K); w = cmath.exp(2j*math.pi/q); R = range(N)
    L = np.array([[K[i, j]*K[i, l].conjugate()*K[k, j].conjugate()*K[k, l]/N for k in R for l in R] for i in R for j in R])
    tot = 0
    for x in itertools.product(range(q), repeat=2):
        Th = np.diag([w**exponent(x, s[i], t[j], q) for i in R for j in R])
        tot += int((np.linalg.svd(L - Th, compute_uv=False) < 1e-8).sum())
    return tot - (2*N*q - 1)

def legendre(a, q):
    a %= q
    return 0 if a == 0 else (1 if pow(a, (q-1)//2, q) == 1 else -1)

def cross_ratio(a, b, c, d, q):
    """(a,b;c,d) = (a-c)(b-d)/((a-d)(b-c)) on the projective line over F_q"""
    num, den = 1, 1
    for (x, y, up) in ((a, c, True), (b, d, True), (a, d, False), (b, c, False)):
        if x == q or y == q: continue
        if up: num = num*(x-y) % q
        else: den = den*(x-y) % q
    return num*pow(den, -1, q) % q

def level(H, maxl=400):
    """smallest l such that the dephased form of H has l-th roots of unity as entries"""
    D = H/H[0][None, :]; D = D/D[:, [0]]
    a = np.angle(D)/(2*math.pi)
    for l in range(1, maxl):
        if np.allclose(a*l, np.round(a*l), atol=1e-7): return l

def configuration(N, q, rng, distinct=False):
    if distinct:
        pts = rng.sample(range(q+1), 2*N); return tuple(pts[:N]), tuple(pts[N:])
    while True:
        s = tuple(rng.randrange(q+1) for _ in range(N)); t = tuple(rng.randrange(q+1) for _ in range(N))
        if not set(s) & set(t): return s, t
