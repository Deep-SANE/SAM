# SAM tutorial programs

Example programs shipped with SAM, built into `bin/tutorial/` by
`make -j 8 apps`. Each starts with a comment saying what it computes, and most
have a SETTINGS block at the top (nucleus, Nmax, operator, ...).

**Do not edit them here.** Copy the one you need into `app/work/` (your own
folder, ignored by git), change the copy, and run it from `bin/work/`:

```sh
cp app/tutorial/MatrixOperatorJT.cpp app/work/
# edit the SETTINGS block of app/work/MatrixOperatorJT.cpp
make -j 8 apps
mpirun -np 1 bin/work/MatrixOperatorJT
```

The programs use MPI; run them with `mpirun` (more ranks, e.g. `-np 4`, speed
up the matrix elements and the two-body expansions).

## A first calculation

These three use the released data (`make data`) and run in under a minute at
their default settings: 6Li, J = 1, Nmax = 4, hw = 20 MeV, with the NNLOopt
interaction.

| Program | What it does |
|---|---|
| `MatrixOperatorJT` | Builds the basis (full or truncated), computes the Hamiltonian matrix, prints the lowest eigenvalues and the U(3)/U(4) content of the lowest states, and saves the eigenpairs. |
| `WavefunctionAnalyzer` | Reads the saved eigenpairs and prints the U(3)/U(4) probabilities again, for more states, without recomputing. |
| `ObservablesJT` | Reads the saved eigenpairs: radii, moments and B(E2), B(M1) in one nucleus; Gamow-Teller or Fermi transitions between two nuclei (e.g. 6He -> 6Li, after running `MatrixOperatorJT` for 6He too). |

`WavefunctionAnalyzer` and `ObservablesJT` must use the same nucleus, Nmax,
hw, Hamiltonian and truncation as the `MatrixOperatorJT` run that saved the
eigenpairs.

## The basis

| Program | What it does |
|---|---|
| `U3xU4ManyNucleonBasis` | Asks for Z, N, Nmax and 2J, builds the basis and prints its dimension, optionally with every irrep and its L, S, T content. |

## Operator expansions

SAM writes every operator as a sum of U(3)xU(4) tensors. The common ones are
released (`make data-list`); these programs make the others, or extend a
released one to a larger maxShell. SAM itself never expands during a
calculation, and stops with a message pointing here when a file is missing.

| Program | What it does |
|---|---|
| `OneBodyOperatorJT` | Expands a one-body operator (radii, E2, M1, Fermi, Gamow-Teller, ...) and writes its file. Seconds. |
| `TwoBodyOperatorJT` | Expands a two-body operator or interaction, or extends an existing file to a larger maxShell. Expensive for interactions: run on a cluster. |
| `TensorAnalyzer` | Groups the tensors of an expansion by U(3), U(4), L and S, with statistics of their strengths. |

## Tensors and reduced matrix elements

The building blocks of the expansions and matrix elements, one example at a
time (switch examples by swapping comments in a copy). The tests in `test/`
run the same examples and check their numbers.

| Program | What it does |
|---|---|
| `U3xU4SingleShellTensorsDecomposition_Phys` | A tensor of one HO shell in the physical basis |
| `U3xU4SingleShellTensorsDecomposition_Gelfand` | The same in the Gelfand basis |
| `U3xU4SingleShellTensorsRME` | Reduced matrix elements of a single-shell tensor |
| `U3xU4MultiShellTensorsDecomposition` | A tensor over several shells in the physical basis |
| `U3xU4MultiShellTensorsRME` | Reduced matrix elements between many-nucleon basis irreps |
| `U3xU4MultiShellTensors_ShareGammas_RME` | The same for tensors that share their single-shell parts, through the RME database |
