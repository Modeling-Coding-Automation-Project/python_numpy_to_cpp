# Eigen threshold sweep notes

Baseline: `8e22c7b327fa4d783a0db33527cc25b5d941b8ba`. All five eigen epsilon uses and the GMRES breakdown epsilon use are changed together for the main sweep. The independent-role experiments change only one role.

Each type/configuration has 22 cases: general real 7, complex 9, symmetric Jacobi 6. All matrices are 3 x 3. The general QR budget is 128 slots, inverse iteration 30 slots, Jacobi 10 sweeps. `std` is a host diagnostic configuration; `default` uses the target math library.

Errors use exact stored inputs exported with `%a`, NumPy eigenvalues as reference, and binary64 residual arithmetic. Eigenvalue error is the minimum matched maximum absolute error divided by the matrix Frobenius norm. Vector residual is max_j ||A*v_j-lambda_j*v_j||/(||A||_F*||v_j||). NumPy is a binary64 reference, not an arbitrary-precision oracle.

Vector convergence counts concern 16 general-solver cases. Jacobi exposes no independent vector convergence flag. Maximum residual in CSV includes unconverged outputs and must not be interpreted as the accuracy of an accepted solution. `-1` means an unavailable diagnostic.

## default

| Fixed base | float max value error | double max value error | float value conv /22 | double value conv /22 | float vector conv /16 | double vector conv /16 |
|---|---:|---:|---:|---:|---:|---:|
| machine | 6.672e-07 | 3.010e-07 | 22 | 22 | 16 | 2 |
| 1e-6 | 5.114e-06 | 5.192e-06 | 22 | 22 | 15 | 15 |
| 1e-7 | 6.672e-07 | 5.703e-07 | 22 | 22 | 16 | 16 |
| 1e-8 | 6.672e-07 | 3.010e-07 | 22 | 22 | 8 | 13 |
| 1e-9 | 6.672e-07 | 3.010e-07 | 22 | 22 | 4 | 6 |
| 1e-10 | 6.672e-07 | 3.010e-07 | 22 | 22 | 2 | 5 |
| 1e-11 | 6.672e-07 | 3.010e-07 | 22 | 22 | 4 | 4 |
| 1e-12 | 6.673e-07 | 3.010e-07 | 20 | 22 | 2 | 2 |
| 1e-13 | 6.673e-07 | 3.010e-07 | 20 | 22 | 2 | 2 |
| 1e-14 | 6.673e-07 | 3.010e-07 | 20 | 22 | 2 | 2 |
| 1e-15 | 6.673e-07 | 3.010e-07 | 20 | 22 | 2 | 2 |
| 1e-16 | 6.673e-07 | 3.010e-07 | 20 | 22 | 2 | 2 |

## std

| Fixed base | float max value error | double max value error | float value conv /22 | double value conv /22 | float vector conv /16 | double vector conv /16 |
|---|---:|---:|---:|---:|---:|---:|
| machine | 1.524e-06 | 4.385e-15 | 22 | 22 | 16 | 4 |
| 1e-6 | 5.089e-06 | 4.896e-06 | 22 | 22 | 15 | 15 |
| 1e-7 | 1.524e-06 | 5.506e-07 | 22 | 22 | 16 | 16 |
| 1e-8 | 1.018e-06 | 1.819e-10 | 22 | 22 | 8 | 14 |
| 1e-9 | 1.018e-06 | 1.819e-10 | 22 | 22 | 4 | 14 |
| 1e-10 | 1.018e-06 | 1.819e-10 | 22 | 22 | 4 | 14 |
| 1e-11 | 1.018e-06 | 1.819e-10 | 22 | 22 | 4 | 14 |
| 1e-12 | 1.018e-06 | 2.302e-13 | 20 | 22 | 2 | 12 |
| 1e-13 | 1.018e-06 | 2.302e-13 | 20 | 22 | 2 | 4 |
| 1e-14 | 1.018e-06 | 2.302e-13 | 20 | 22 | 2 | 2 |
| 1e-15 | 1.048e-06 | 4.385e-15 | 20 | 22 | 2 | 2 |
| 1e-16 | 1.048e-06 | 4.385e-15 | 20 | 22 | 2 | 3 |

## Interpretation

- A machine epsilon is not a mathematical requirement for these user-policy thresholds. A finite fixed value can be used; it changes stopping, shift size, and breakdown decisions.
- `1e-7` gives all 22 value and all 16 general-vector convergence flags for both types in these fixtures, but it accepts larger errors than strict double settings. It is not a universal recommendation.
- With default math, double eigenvalue error plateaus near `3.01e-7` at `1e-8` and smaller. Standard math double continues improving to about `4.39e-15`.
- At `1e-12` and smaller, the float general real/complex solvers exhaust 128 QR slots on `clustered_1e-10`. Their eigenvalue error is only about `4.08e-11`, but they cannot satisfy the requested deflation test. The output vectors are not converged; the maximum residual `0.236` comes from these unsuccessful cases.
- In the role-only default/double experiment, replacing just breakdown epsilon with `1e-7` raises inner-failure counts from 4/16 to 14/16. Changing all roles to `1e-7` also relaxes the inner GMRES tolerance to about `0.00506`, and then no inner-failure flag is raised. These are different acceptance rules, not proof of improved numerical accuracy.
- Changing just the double eigenvector-tolerance base to `1e-7` increases vector convergence from 2/16 to 14/16 while leaving eigenvalues unchanged. It accepts a maximum residual around `1.12e-6` rather than the baseline all-output maximum around `8.16e-7`.
- No solver-object size changed across the sweep; the threshold is a compile-time constant.

## Reproduction

Both runners archive the fixed commit and its math/utility gitlinks to temporary directories. They do not edit production headers. NumPy is required only by the host comparison.

```sh
python3 test_vs/algorithm_improvements/run_epsilon_eigen_sweep.py
python3 test_vs/algorithm_improvements/run_epsilon_eigen_roles.py
```

The archived original headers and dependencies were byte-compared with the measured snapshot (68 files). After adding revision pinning, one `1e-7` full case and one breakdown-only `1e-7` case reproduced the original raw CSV byte for byte.
