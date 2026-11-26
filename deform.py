# is the matrix of order 2q with cross-ratio a residue isolated? push it along a nontrivial first order direction and project back on Hadamard matrices
import sys, numpy as np, math
from products import *
try:
    from scipy.optimize import least_squares
except Exception as e:
    print("no scipy", e); sys.exit()
def tangent(H, tol=1e-7):
    M, N = H.shape
    P = [(i, j) for i in range(M) for j in range(i+1, M)]
    R = np.zeros((2*len(P), M, N))
    for n, (i, j) in enumerate(P):
        c = H[i]*H[j].conj()
        R[2*n, i] = c.real; R[2*n, j] = -c.real; R[2*n+1, i] = c.imag; R[2*n+1, j] = -c.imag
    u, s, vt = np.linalg.svd(R.reshape(2*len(P), M*N))
    r = int((s > tol*s[0]).sum())
    return vt[r:].reshape(-1, M, N)
def dephase(A): return (A - A[0][None, :] - A[:, [0]] + A[0, 0])[1:, 1:]
def resid(x, H0, n):
    A = np.zeros((n, n)); A[1:, 1:] = x.reshape(n-1, n-1)
    H = H0*np.exp(1j*A); G = H @ H.conj().T - n*np.eye(n)
    iu = np.triu_indices(n, 1)
    return np.concatenate([G[iu].real, G[iu].imag])
rng = np.random.default_rng(0)
for q, lam in ((5, 4), (7, 2), (5, 2)):
    H = mw(fourier(2), q, (q, 0), (1, lam)); n = 2*q
    H0 = H/H[0][None, :]; H0 = H0/H0[:, [0]]
    T = [dephase(A) for A in tangent(H0)]
    B = np.array([t.ravel() for t in T]); u, s, vt = np.linalg.svd(B, full_matrices=False)
    dirs = vt[s > 1e-8]
    print("q=%d lambda=%d (%s): %d nontrivial first order directions" % (q, lam, "square" if legendre(lam, q) == 1 else "non-square", len(dirs)))
    for trial in range(4):
        if len(dirs) == 0: v = rng.normal(size=(n-1)*(n-1)); v /= np.linalg.norm(v)
        else: v = rng.normal(size=len(dirs)) @ dirs; v /= np.linalg.norm(v)
        for eps in (0.05, 0.2):
            sol = least_squares(resid, eps*v, args=(H0, n), xtol=1e-15, ftol=1e-15, gtol=1e-15)
            print("    start at distance %.2f: residual %.1e, final distance from H %.4f" % (eps, np.abs(sol.fun).max(), np.linalg.norm(sol.x)))
