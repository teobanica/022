"""Power maps: by the Gauss sum description and Stickelberger, M_d = (w^{i j^d}) is supersingular (= of finite order) iff every orbit O
of m -> -m/d on Z_{p-1} - {0} is balanced: sum of O = |O|(p-1)/2.  Compare with the criterion (-d)^v = -1 mod p-1."""
import math
def primes(n):
    s = bytearray([1])*(n+1); s[:2] = b"\0\0"
    for i in range(2, int(n**.5)+1):
        if s[i]: s[i*i::i] = bytearray(len(s[i*i::i]))
    return [i for i in range(n+1) if s[i]]
mism, exo, firsts = [], 0, []
for p in primes(1500)[2:]:
    q = p-1; found = []
    for d in range(2, q-1):
        if math.gcd(d, q) != 1: continue
        u = (-pow(d, -1, q)) % q
        x, crit = u, False
        while x != 1:
            if x == q-1: crit = True; break
            x = x*u % q
        seen = bytearray(q); bal = True
        for m in range(1, q):
            if seen[m]: continue
            s = k = 0; y = m
            while not seen[y]:
                seen[y] = 1; s += y; k += 1; y = y*u % q
            if 2*s != k*q: bal = False; break
        if bal != crit: mism.append((p, d))
        if bal: found.append(d)
    exo += bool(found)
    if found and len(firsts) < 12: firsts.append((p, found[0]))
ps = primes(1500)[2:]
print("primes 5..1500: %d, with a finite order power map 1<d<p-2: %d" % (len(ps), exo))
print("first cases (p, smallest d):", firsts)
print("balanced but criterion fails, or conversely:", mism[:10], len(mism))
