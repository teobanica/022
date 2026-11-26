# exotic power maps: a number d != 1, prime to p-1, with -1 among the powers of -d modulo p-1,
# exists precisely when phi(p-1) is not a power of 2, or -1 is a square modulo p-1; check for 5 <= p <= 1500
from math import gcd
def primes(n): return [p for p in range(2, n+1) if all(p % k for k in range(2, int(p**0.5)+1))]
def search(p):
    n = p-1
    for d in range(2, n):
        if gcd(d, n) != 1: continue
        e = (-d) % n; x = e
        for _ in range(n):
            if x == n-1: return d
            x = x*e % n
def criterion(p):
    n = p-1; f = sum(1 for a in range(1, n+1) if gcd(a, n) == 1)
    while f % 2 == 0: f //= 2
    return f > 1 or any(b*b % n == n-1 for b in range(1, n))
P = [p for p in primes(1500) if p >= 5]
print("criterion agrees with the search for all %d primes: %s" % (len(P), all((search(p) is not None) == criterion(p) for p in P)))
print("primes having exotic power maps: %d, exceptions: %s" % (sum(1 for p in P if search(p)), [p for p in P if not search(p)]))
print("smallest d for the first primes:", [(p, search(p)) for p in P[:12] if search(p)])
