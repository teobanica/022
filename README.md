# Code for "Arithmetic aspects of complex Hadamard matrices"

Python 3 with numpy. All the computations of Newton polygons are exact, done in the ring Z[w], w = exp(2 pi i/p).

- `cyclo.py`: exact arithmetic in Z[w], characteristic polynomials, valuations at w-1, Newton polygons.
- `ordertwo.py`: the matrices of order 2 and the Madan-Pal polynomials (Section 3).
- `basics.py`: functional equation, the factorization F_p = L Delta U, the cube of F_p T_c (Sections 1, 5, 10).
- `strata.py`: Newton polygons of all the matrices F_p P D at p = 3, 5 (Sections 5, 8). Usage: `python3 strata.py 5`, or `python3 strata.py 5 signs`.
- `gauss.py`: the matrix G_p, Gauss sums, Kloosterman sums, Stickelberger slopes (Section 6).
- `powermaps.py`: the matrices (w^{i j^d}): spectrum via Gauss sums over orbits, finite order criterion, p <= 47 (Section 7).
- `fermat.py`: the balancing condition against the criterion (-d)^v = -1 mod p-1, for all primes p <= 1500 (Section 7).
- `seven.py`: exhaustive families at p = 7, and the Clifford test at p = 5 (Section 8).
- `exotic.py`: all the finite order matrices F_7 P D with P(0) = 0, D_00 = 1; output in `exotic.txt` (Section 8).
- `legendre.py`: matrices with parameters depending on the Legendre symbol. Usage: `python3 legendre.py 5 7 11 13`.
- `example.py`: exact check of the 72 matrices of Legendre type at p = 7, and of their projective orders.
- `traces.py`: traces of powers of polynomial maps against the Deligne bounds, including the Alltop case (Section 9).
- `arrays.py`: Smith forms of real Hadamard matrices and 2-ranks, bent functions on Z_3^2, duals, flat eigenvectors of F_p (Sections 4, 11).
- `spin.py`: cyclic spin models, star-triangle equation, modulus (Section 11).
- `existence.py`: criterion for the existence of exotic power maps, checked for p up to 1500 (Section 7).
- `translates.py`: the conjugates of the 72 matrices of Legendre type by X^a Z^b against the exotic matrices in `exotic.txt` (Section 8).
- `products.py`: McNulty-Weigert matrices from the mutually unbiased bases labeled by the projective line, defect, twisted defect formula (Section 10).
- `twisted.py`: check of the defect formula against direct computations.
- `symmetry.py`: check of the invariance under PSL_2 and PGL_2.
- `samples.py`: defects of random configurations for K = F_N with N = 3, 5, 7.
- `crossratio.py`: order 2q, defect against the cross-ratio, for q up to 23.
- `equivalence.py`, `classes.py`: exact equivalence test for Butson matrices, and classes of the matrices of order 2q for q = 5, 7, 11.
- `deform.py`: numerical illustration of the deformations of the matrices of order 10 and 14 whose cross-ratio is a square (needs scipy).

The files `*.log` contain the outputs of the corresponding programs.
