"""All the supersingular (= finite order) matrices F_7 P D with entries in mu_7, normalized by P(0) = 0, D_00 = 1.
Prefilter: supersingularity needs the trace to vanish modulo pi^3, that is, sum a_j = sum a_j^2 = 0 mod 7
for the exponents a_j = j P(j) + d_j of the diagonal entries.  Output: the list, with the Clifford test."""
import itertools, numpy as np
from cyclo import *
from seven import is_clifford, strata

p = 7
ex = np.array([(0,)+e for e in itertools.product(range(p), repeat=p-1)], dtype=np.int64)
found = []
perms = [np.array((0,)+s) for s in itertools.permutations(range(1, p))]
kept = 0
for n, s in enumerate(perms):
    a = (np.arange(p)*s + ex) % p
    ok = (a.sum(axis=1) % p == 0) & ((a*a).sum(axis=1) % p == 0)
    E = ex[ok]; kept += len(E)
    cnt, low, ss = strata(p, np.tile(s, (len(E), 1)), E)
    for t in ss: found.append((tuple(int(x) for x in s), tuple(int(x) for x in E[t]), is_clifford(p, s, E[t])))
    if n % 60 == 59: print("progress %d/720, kept %d, found %d" % (n+1, kept, len(found)), flush=True)
with open("exotic.txt", "w") as f:
    for s, e, c in found: f.write("%s %s %s\n" % (s, e, "Clifford" if c else "exotic"))
print("supersingular with P(0)=0: %d, of which Clifford %d, exotic %d" % (len(found), sum(c for *_, c in found), sum(not c for *_, c in found)))
