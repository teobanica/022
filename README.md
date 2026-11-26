# Arithmetic aspects of complex Hadamard matrices

**Claude F and Teo B**, November 2026

The eigenvalues of a complex Hadamard matrix having roots of unity as entries are Weil numbers. In prime order p this produces an abelian variety over the field with p elements, whose Frobenius map and Verschiebung are given by the matrix and by its adjoint. The paper studies the arithmetic invariants coming from this, mainly for the matrices which are equivalent to the Fourier matrix F_p, and then applies the same methods to the isolation questions.

- **Order 2.** The matrices of order 2 produce the abelian varieties over the field with 2 elements having a single point, classified by Madan-Pal and Robinson.
- **Polygons.** The Hodge polygon of F_p has slopes k/(p-1), and the Newton polygons of the matrices equivalent to F_p are bounded from below. A matrix has finite order precisely when its Newton polygons are straight lines.
- **Finite order.** At p = 3, 5 the finite order matrices F_p P D are the Clifford ones. At p = 7 there are 5586 of them, namely 2058 Clifford ones and 3528 exotic ones, found by an exhaustive search, and described by an explicit formula.
- **Power maps.** The matrix (w^{i j^d}) has finite order precisely when -1 is a power of -d modulo p-1.
- **Traces.** For the matrices (w^{i s(j) + t(j)}) with s, t polynomials, the traces of the powers are bounded by using the Deligne estimates, and the spectrum becomes equidistributed.
- **Isolation.** The defect of a McNulty-Weigert matrix is computed by a formula involving q^2 twisted matrices. In order 2q, with 4 distinct points having cross-ratio c, the matrix is isolated precisely when c is not a square modulo q, and otherwise it belongs to a family of dimension q-1.
- **Physics.** Spin models, modular invariance, and the relation with the Ising arrays of Hadamard matrices.

---

## Contents

- `022.pdf`, `022.tex`: the paper.
- `*.py`: the programs, listed below.
- `*.log`: the outputs of the programs.
- `exotic.txt`: the 798 finite order matrices F_7 P D with P(0) = 0 and D_00 = 1, produced by `exotic.py`.

---

## Programs

| File | Content | Section |
|------|---------|---------|
| `cyclo.py` | Exact arithmetic in Z[w], characteristic polynomials, valuations at w-1, Newton polygons | all |
| `basics.py` | Functional equation, the factorization F_p = L D U, the cube of F_p T_c | 1, 5, 11 |
| `ordertwo.py` | The matrices of order 2 and the Madan-Pal polynomials | 3 |
| `arrays.py` | Smith forms of real Hadamard matrices and 2-ranks, bent functions on Z_3^2, flat eigenvectors of F_p | 4, 11 |
| `strata.py` | Newton polygons of all the matrices F_p P D at p = 3, 5 | 5, 8 |
| `gauss.py` | The matrix G_p, Gauss sums, Kloosterman sums, Stickelberger slopes | 6 |
| `powermaps.py` | The matrices (w^{i j^d}): spectrum via Gauss sums over orbits, finite order criterion, for p up to 47 | 7 |
| `fermat.py` | The balancing condition against the criterion on the powers of -d, for all primes p up to 1500 | 7 |
| `existence.py` | Criterion for the existence of exotic power maps, for all primes p up to 1500 | 7 |
| `seven.py` | Exhaustive families at p = 7, and the Clifford test at p = 5 | 8 |
| `exotic.py` | All the finite order matrices F_7 P D with P(0) = 0, D_00 = 1, written in `exotic.txt` | 8 |
| `legendre.py` | Matrices with parameters depending on the Legendre symbol, at p = 5, 7, 11, 13 | 8 |
| `example.py` | Exact check of the 72 matrices of Legendre type at p = 7, and of their projective orders | 8 |
| `translates.py` | The conjugates of the 72 matrices of Legendre type by X^a Z^b, against the exotic matrices in `exotic.txt` | 8 |
| `traces.py` | Traces of powers of polynomial maps against the Deligne bounds, including the Alltop case | 9 |
| `products.py` | McNulty-Weigert matrices from the mutually unbiased bases labeled by the projective line, defect, twisted defect formula | 10 |
| `twisted.py` | Check of the defect formula against direct computations | 10 |
| `symmetry.py` | Check of the invariance under PSL_2 and PGL_2 | 10 |
| `samples.py` | Defects of random configurations for K = F_N with N = 3, 5, 7 | 10 |
| `crossratio.py` | Order 2q, defect against the cross-ratio, for q up to 23 | 10 |
| `equivalence.py` | Exact equivalence test for Butson matrices | 10 |
| `classes.py` | Classes of the matrices of order 2q, for q = 5, 7, 11 | 10 |
| `deform.py` | Numerical illustration of the deformations in order 10 and 14, when the cross-ratio is a square | 10 |
| `spin.py` | Cyclic spin models, star-triangle equation, modulus | 11 |

---

## Running the code

The programs need Python 3 with numpy, and `deform.py` needs scipy as well. The files `cyclo.py`, `products.py` and `equivalence.py` are libraries, used by the other programs. Each of the other programs runs on its own, with `python3 gauss.py` and so on, and prints its results, which are those in the corresponding `.log` file. Two of them take arguments, `python3 strata.py 5` and `python3 legendre.py 5 7 11 13`. The program `translates.py` reads `exotic.txt`, which is produced by `exotic.py`. All the computations of Newton polygons are exact, done in the ring Z[w].
