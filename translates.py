# the conjugates of the 72 matrices of Legendre type at p = 7 by the matrices X^a Z^b, normalized by D_00 = 1:
# these are 3528 distinct matrices, and those with sigma(0) = 0 are the 504 exotic matrices listed in exotic.txt
# with H_ij = w^{i sigma(j) + tau(j)}, the conjugation by X^a Z^b maps (sigma(j), tau(j)) to (sigma(j-a) + b, tau(j-a) - a sigma(j-a) - b j)
import re
p = 7; half = pow(2, -1, p)
def legendre(j): return 0 if j % p == 0 else (1 if pow(j, (p-1)//2, p) == 1 else -1)
seeds = []
for alpha in range(1, p):
    for r in (2, 4):
        for ep in (-1, 0, 1):
            for em in (-1, 0, 1):
                if ep == em: continue
                sigma, tau = [], []
                for j in range(p):
                    a, e = (alpha, ep) if legendre(j) != -1 else (r*alpha % p, em)
                    sigma.append(a*j % p); tau.append(a*e*j*j*half % p)
                seeds.append((tuple(sigma), tuple(tau)))
assert len(set(seeds)) == 72
def conjugate(sigma, tau, a, b):
    s = tuple((sigma[(j-a) % p] + b) % p for j in range(p))
    v = [(tau[(j-a) % p] - a*sigma[(j-a) % p] - b*j) % p for j in range(p)]
    return s, tuple((x - v[0]) % p for x in v)
orbit = set(conjugate(s, t, a, b) for (s, t) in seeds for a in range(p) for b in range(p))
print("matrices of Legendre type: %d, conjugates by X^a Z^b: %d" % (len(seeds), len(orbit)))
slice0 = set(x for x in orbit if x[0][0] == 0)
exotic = set()
for line in open("exotic.txt"):
    m = re.match(r"\((.*?)\) \((.*?)\) (\w+)", line)
    if m and m.group(3) != "Clifford":
        exotic.add((tuple(int(x) for x in m.group(1).split(",")), tuple(int(x) for x in m.group(2).split(","))))
print("conjugates with sigma(0) = 0: %d, exotic matrices in exotic.txt: %d, the 2 sets coincide: %s" % (len(slice0), len(exotic), slice0 == exotic))
