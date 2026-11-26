"""Newton polygons at pi = 1-w of all the matrices F_p P D equivalent to the Fourier matrix F_p,
with P a permutation and D diagonal with entries in mu_p (or mu_2p), normalized by D_00 = 1.
Slopes are printed in units where the valuation of p is p-1 (so 'supersingular' = all slopes (p-1)/2)."""
import sys, itertools, numpy as np
from collections import Counter
from cyclo import *

def run(p, signs=False, sample=None, seed=1, chunk=20000):
    rng = np.random.default_rng(seed)
    if sample is None:
        perms = np.array(list(itertools.permutations(range(p))), dtype=np.int64)
        ex = np.array(list(itertools.product(range(p), repeat=p-1)), dtype=np.int64)
        ex = np.concatenate([np.zeros((len(ex), 1), dtype=np.int64), ex], axis=1)
        sg = np.array(list(itertools.product([1, -1], repeat=p-1)), dtype=np.int64) if signs else np.ones((1, p-1), dtype=np.int64)
        sg = np.concatenate([np.ones((len(sg), 1), dtype=np.int64), sg], axis=1)
        idx = np.array(list(itertools.product(range(len(perms)), range(len(ex)), range(len(sg)))), dtype=np.int64)
        P, E, S = perms[idx[:, 0]], ex[idx[:, 1]], sg[idx[:, 2]]
    else:
        P = np.array([rng.permutation(p) for _ in range(sample)], dtype=np.int64)
        E = rng.integers(0, p, size=(sample, p)); E[:, 0] = 0
        S = rng.choice([1, -1], size=(sample, p)) if signs else np.ones((sample, p), dtype=np.int64); S[:, 0] = 1
    cnt, low = Counter(), None
    for a in range(0, len(P), chunk):
        H = fourier_twists(p, P[a:a+chunk], E[a:a+chunk], S[a:a+chunk])
        e = elementary(H, p)
        V = np.stack([val(x, p) for x in e], axis=1)     # (B, N+1)
        low = V.min(axis=0) if low is None else np.minimum(low, V.min(axis=0))
        for row in V:
            cnt[newton([int(x) for x in row])] += 1
    return cnt, low

if __name__ == "__main__":
    p = int(sys.argv[1]); signs = len(sys.argv) > 2 and sys.argv[2] == "signs"
    sample = int(sys.argv[3]) if len(sys.argv) > 3 else None
    cnt, low = run(p, signs, sample)
    print("p = %d, twists by mu_%d, %d matrices%s" % (p, 2*p if signs else p, sum(cnt.values()), " (random sample)" if sample else ""))
    print("Hodge slopes:", list(range(p)), " minimal valuations of e_k over the class:", [int(x) for x in low])
    for sl, c in sorted(cnt.items(), key=lambda t: t[0]):
        print("  slopes", [str(s) for s in sl], " count", c)
