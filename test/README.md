# SAM regression tests

From the SAM root, after `make config`:

```sh
make test
```

This builds `lib/libsam.a` if needed, compiles every `test_*.cpp` into
`bin/test/`, and runs each one as a single MPI process. On a cluster, choose
the launcher with `make test MPIRUN="srun -n 1"`. The tests need no data
files: each program runs with `SAM_DATA_DIR` set to a fresh temporary
directory, removed afterwards, so your data directory is never touched.

Each case prints `[TEST]` and then `[PASS]` or `[FAIL]` with what went wrong,
and each program ends with a summary. At the end, `make test` prints a table
with every program's passed, failed and skipped cases, run time and result
(`ok`, `FAIL` or `CRASHED`), and the totals. A case is skipped (`[SKIP]`) when
it needs released data that is not in the data directory. `make test` fails if any case fails or
any program crashes.

## Coverage

Each test program is a copy of an example program in `app/`, using the same
examples, with the numbers they produce added as checks.

Single-shell tensors:

- `test_singleshelltensors_decomposition_phys.cpp` (from `app/U3xU4SingleShellTensorsDecomposition_Phys.cpp`):
  five examples expanded with `SingleShellTensor_phys::expand()`. Checks the
  number of terms, the normalization (sum of squared coefficients = 1 per
  multiplicity) and a signed checksum over all terms (`expansion_checksum.h`).
  Example 3 is also checked exactly: all 12 coefficients of the scalar ad . a
  in shell 1 are 1/sqrt(12).
- `test_singleshelltensors_decomposition_gelfand.cpp` (from `app/U3xU4SingleShellTensorsDecomposition_Gelfand.cpp`):
  three examples expanded with `SingleShellTensor_gelfand::expand()`; the same
  checks, with Example 2 exact.
- `test_singleshelltensors_rme.cpp` (from `app/U3xU4SingleShellTensorsRME.cpp`):
  six examples of single-shell triple-reduced matrix elements, compared value
  by value. Examples 3-6 are exact and are checked for every shell
  n = 0..8: with N = 2(n+1)(n+2) single-particle states, < ad > = 1,
  < at > = sqrt(N), < [ad x at]^0 > = 1/sqrt(N), and the identity gives a
  unit matrix.

Multi-shell tensors:

- `test_multishelltensors_decomposition.cpp` (from `app/U3xU4MultiShellTensorsDecomposition.cpp`):
  Examples 1-7 expanded with `MultiShellTensor_phys::expand()` (Example 8 of
  the app is marked "should not run"). The same checks as for single-shell
  tensors, plus Example 1 expanded again from the single-shell pieces that the
  decomposition database wrote to disk.
- `test_multishelltensors_rme.cpp` (from `app/U3xU4MultiShellTensorsRME.cpp`):
  seven examples of multi-shell RMEs with `MultiShellTensor::RME()`, compared
  value by value.
- `test_multishelltensors_sharegammas_rme.cpp` (from `app/U3xU4MultiShellTensors_ShareGammas_RME.cpp`):
  RMEs of `MultiShellTensor_ShareGammas_phys` through the RME database, as in
  the matrix-element code. They are also cross-checked against
  `MultiShellTensor::RME()`: Example 1 (first coupling) equals Example 7 of
  `test_multishelltensors_rme.cpp`, and Example 2 equals its Example 1.

Operator expansions (Nmax = maxShell = 4, hw = 20 MeV, the cheap operators
computed on the fly; no nuclear potentials, which need interaction files):

- `test_onebodyoperator_expansion.cpp`: `Kinetic` and `CoMNumberOp` with
  `OneBodyOperator::expand_new()`. Number of tensors, a checksum over all
  strengths, and every strength exactly: with N_n = 2(n+1)(n+2),
  `CoMNumberOp` gives n sqrt(N_n) in shell n, and `Kinetic` gives
  (hw/2)(n + 3/2) sqrt(N_n) in shell n and -(hw/2) sqrt((n+1)(n+2)(n+3)(n+4))/2
  between shells n and n+2. Also the file round trip: `expand()` stops on a
  missing file with the `make data` message and reads the file back after
  `writeBinary()`; the maxShell-4 file serves maxShell 3 (equal to
  `expand_new()` there) but not maxShell 5 (the "only reaches" error); read at
  hw = 25, `Kinetic` strengths scale by 25/20 and `CoMNumberOp` stays.
- `test_twobodyoperator_expansion.cpp`: the same for the two-body parts with
  `TwoBodyOperator::expand_new()`, which runs on OpenMP threads, and both
  `expand()` (MPI reader) and `readBinary()`.

Operator matrices (`JT::Operator::matrix_CSR`):

- `test_operator_ll_ls.cpp`: `L.L` and `L.S` for 4He and 6He (J = 0,
  Nmax = 2) and 6Li (J = 1, Nmax = 0 and 2). The basis has good L, S and J,
  so both matrices must be diagonal with L(L+1) and
  [J(J+1) - L(L+1) - S(S+1)]/2 exactly. The test makes its own expansions of
  `L.L` and `L.S`.
- `test_operator_radius_e2.cpp`: the proton- and neutron-only operators
  `Radius_p`, `Radius_n` and `E2` keep both their isoscalar and isovector
  parts. At Nmax = 0 every state of 4He, 6Li and 6He has exact radii
  (b^2 times the sum of N + 3/2 over the protons or neutrons); in 6He at
  Nmax = 2, `Radius_p` + (b_p/b_n)^2 `Radius_n` = `Radius` element by element
  while the two differ; the E2 from 0+ to 2+ in 6He at Nmax = 0 vanishes
  (the protons fill the s shell), the 1+ to 3+ in 6Li does not.
- `test_operator_hamiltonian.cpp`: "Hamiltonian with N2LO_opt_bare" at
  hw = 20 MeV and Nmax = 4 for 4He (J = 0), 6Li (J = 1) and 6He (J = 0): the
  five lowest eigenvalues and the ground-state U(3) and U(4) probabilities in
  each hw subspace. It needs the released expansions
  (`make data OPS="N2LO_opt_bare Kinetic CoMNumberOp"`), which it links from
  the configured data directory; without them its cases are skipped, not
  failed.

## Tolerances

- Operator matrices: exact values to absolute 1e-10. Eigenvalues: absolute
  1e-7 MeV plus relative 1e-9; probabilities: absolute 1e-6 % plus relative
  1e-7.

- Normalizations and exact values: absolute 1e-10 (1e-9 for multi-shell
  normalizations, 1e-12 for single coefficients).
- Expansion checksums: absolute 1e-10 plus relative 1e-8. A checksum sums
  millions of terms, and the summation order changes with the compiler and
  the number of threads.
- RMEs: absolute 1e-12 plus relative 1e-11. Where highest-weight states come
  from the iterative null-space solver (KraNS LOBPCG, tolerance 1e-14, random
  start), the last digits vary slightly from run to run, about 1e-14 relative.

## Reference values

The expected numbers are written in the test sources next to each example.
They were produced by this code on Linux (GCC 13, Open MPI) in November 2026.
If a deliberate change in conventions alters them, update them there and say
why in the commit message.

## Adding a test

Create `test_<topic>.cpp` using `test_helpers.h` (`require`, `near`,
`Suite`); `make test` picks it up automatically.
