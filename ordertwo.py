"""Section 3 checks: H_z = [[1,z],[1,-z]], z a primitive n-th root of unity.
The Weil polynomial W of H_z satisfies W(1) = 1, and its real Weil polynomial is the
Madan-Pal polynomial P_n(3-t), where P_n(x) = prod (x^2 - (4 + 2cos(2 pi k/n))x + 1)."""
import numpy as np, cmath, math
for n in (3, 4, 5, 7, 8, 9, 12, 15, 16, 30):
    ks = [k for k in range(1, n+1) if math.gcd(k, n) == 1]
    W, ts = np.array([1.0+0j]), []
    for k in ks:
        z = cmath.exp(2j*math.pi*k/n)
        ev = np.linalg.eigvals(np.array([[1, z], [1, -z]]))
        W = np.convolve(W, np.poly(ev)); ts += [2*e.real for e in ev]      # t = pi + 2/pi = 2 Re(pi)
    W = np.round(W.real).astype(int)
    MP = np.array([1.0])
    for k in ks:
        if 2*k <= n:
            s = 4+2*math.cos(2*math.pi*k/n)
            MP = np.convolve(MP, [1, -(6-s), 10-3*s])                      # (3-t)^2 - s(3-t) + 1
    ok = np.allclose(np.convolve(MP, MP), np.poly(ts), atol=1e-6)          # each value of t occurs for k and -k
    print("n=%2d  deg W = %2d  W(1) = %d  real Weil polynomial = P_n(3-t): %s" % (n, len(W)-1, W.sum(), ok))
