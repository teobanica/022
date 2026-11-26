"""Exact arithmetic in Z[w], w = exp(2 pi i/p), p prime, vectorized with numpy.
Elements are integer arrays of length p (coefficients of 1, w, ..., w^{p-1}), defined modulo 1+w+...+w^{p-1}=0."""
import numpy as np
from math import comb

def mul(a, b, p):
    """product in Z[x]/(x^p-1), batched over leading axes"""
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=object if a.dtype == object or b.dtype == object else np.int64)
    for s in range(p):
        out += a[..., s:s+1] * np.roll(b, s, axis=-1)
    return out

def matmul(A, B, p):
    """product of matrices with entries in Z[x]/(x^p-1); shapes (...,N,N,p)"""
    N = A.shape[-2]
    out = np.zeros(A.shape, dtype=A.dtype)
    for s in range(p):
        # A[...,i,j,s] * roll(B[...,j,k,:], s)
        out += np.einsum('...ij,...jkt->...ikt', A[..., s], np.roll(B, s, axis=-1))
    return out

def normal(a, p):
    """unique representative with zero coefficient on w^{p-1}"""
    return a - a[..., p-1:p]

def trace(A):
    return np.einsum('...iit->...t', A)

def elementary(H, p):
    """elementary symmetric functions e_0..e_N of the eigenvalues of H (shape (...,N,N,p)), via Newton identities.
    charpoly = sum_k (-1)^k e_k x^{N-k}"""
    N = H.shape[-2]
    pw, M = [], H
    for k in range(1, N+1):
        pw.append(normal(trace(M), p))
        if k < N: M = matmul(M, H, p)
    one = np.zeros(H.shape[:-3]+(p,), dtype=H.dtype); one[..., 0] = 1
    e = [one]
    for k in range(1, N+1):
        s = np.zeros_like(one)
        for i in range(1, k+1):
            s += (-1)**(i-1) * mul(e[k-i], pw[i-1], p)
        s = normal(s, p)
        assert not (s % k).any(), "non exact division"
        e.append(s // k)
    return e

def val(a, p, big=10**6):
    """valuation at pi = 1-w of elements a (shape (...,p)); the valuation of p is p-1; returns big for 0"""
    a = normal(a, p)[..., :p-1]
    # a = sum_i a_i (1-pi)^i = sum_j b_j pi^j, b_j = (-1)^j sum_i a_i C(i,j)
    C = np.array([[(-1)**j * comb(i, j) for j in range(p-1)] for i in range(p-1)], dtype=a.dtype)
    b = a @ C
    v = np.full(b.shape[:-1], big, dtype=np.int64)
    for j in range(p-1):
        bj = b[..., j].astype(np.int64) if b.dtype != object else b[..., j]
        vp = np.zeros(bj.shape, dtype=np.int64)
        x = np.array(bj, dtype=object) if b.dtype == object else bj.copy()
        nz = x != 0
        cur = np.where(nz, x, 1)
        for _ in range(60):
            d = nz & (cur % p == 0)
            if not d.any(): break
            vp += d
            cur = np.where(d, cur // p, cur)
        v = np.minimum(v, np.where(nz, j + (p-1)*vp, big))
    return v

def newton(vals):
    """lower convex hull of the points (k, vals[k]); returns the tuple of N slopes (as numerators over 1), increasing"""
    from fractions import Fraction
    N = len(vals)-1
    pts = [(k, vals[k]) for k in range(N+1)]
    hull = []
    for q in pts:
        while len(hull) >= 2 and (hull[-1][1]-hull[-2][1])*(q[0]-hull[-2][0]) >= (q[1]-hull[-2][1])*(hull[-1][0]-hull[-2][0]):
            hull.pop()
        hull.append(q)
    sl = []
    for (x0, y0), (x1, y1) in zip(hull, hull[1:]):
        sl += [Fraction(y1-y0, x1-x0)]*(x1-x0)
    return tuple(sl)

def fourier_twists(p, perms, exps, signs=None):
    """the matrices H_ij = w^{i*perm(j) + exps(j)} (times signs_j); perms (B,p), exps (B,p) -> (B,p,p,p) int64"""
    B = perms.shape[0]
    i = np.arange(p)[None, :, None]
    e = (i*perms[:, None, :] + exps[:, None, :]) % p
    H = np.zeros((B, p, p, p), dtype=np.int64)
    bi, ii, jj = np.meshgrid(np.arange(B), np.arange(p), np.arange(p), indexing='ij')
    H[bi, ii, jj, e] = 1
    if signs is not None: H *= signs[:, None, :, None]
    return H
