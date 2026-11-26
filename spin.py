"""Section 8 checks: the cyclic spin models W = t (w^{c(i-j)^2}) of prime order p.
Star-triangle relation sum_x W(a,x) W(b,x) conj W(c,x) = D W(a,b) conj W(a,c) conj W(b,c), with D = sqrt p,
holds exactly when t^2 = conj G(c)/sqrt p, with G(c) = sum_x w^{c x^2} the quadratic Gauss sum.
Also: (F_p T)^3 is a scalar, or a scalar times F_p^2, for T = diag(w^{c j^2}) with 2c = +1, -1 mod p."""
import numpy as np, cmath, math
for p in (3, 5, 7, 11, 13):
    w = cmath.exp(2j*math.pi/p); D = math.sqrt(p)
    F = np.array([[w**(i*j % p) for j in range(p)] for i in range(p)])
    for c in range(1, p):
        G = sum(w**(c*x*x % p) for x in range(p))
        E = np.array([[w**(c*(i-j)**2 % p) for j in range(p)] for i in range(p)])
        ok = []
        for t in (cmath.sqrt(G/D), -cmath.sqrt(G/D), cmath.sqrt(G/D)*1j, 1):
            W = t*E
            L = np.einsum('ax,bx,cx->abc', W, W, W.conj())
            R = D*np.einsum('ab,ac,bc->abc', W, W.conj(), W.conj())
            ok.append(bool(np.allclose(L, R)))
        ev = np.linalg.eigvals(E*cmath.sqrt(G/D)/D)
        fin = np.allclose(ev**(8*p), 1)
        T = np.diag([w**(c*j*j % p) for j in range(p)]); H = F @ T; H3 = H @ H @ H
        kind = "scalar" if np.allclose(H3, H3[0, 0]*np.eye(p)) else ("scalar x F^2" if np.allclose(H3, (H3[0, 0]/p)*(F @ F)) else "-")
        if c <= 2 or kind != "-":
            print("p=%2d c=%2d  G/sqrt p = %s  star-triangle for t=+s,-s,is,1: %s  W/sqrt p of finite order: %s  (F T)^3: %s"
                  % (p, c, np.round(G/D, 6), ok, fin, kind))
