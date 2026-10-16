# SAM: Symmetry-Adapted Many-Body Framework

**SAM** is an **AI-enabled computational framework for Symmetry-Adapted Many-body nuclear structure calculations** based on the $\mathrm{SU(3)}\times\mathrm{SU(4)}$ algebraic formulation of the nuclear shell model.

SAM constructs symmetry-adapted many-nucleon bases, computes and expands single- and two-body nuclear interactions and electroweak transition operators in terms of SU(3)×SU(4) spherical tensors, computes matrix elements across multiple harmonic oscillator shells, and calculates observables using high-performance parallel eigensolvers (Spectra, OpenMP, MPI).

- **Prerequisites / Sibling libraries:** [u3lib](https://github.com/Deep-SANE/u3lib) (SU(3) algebra), [u4lib](https://github.com/Deep-SANE/u4lib) (SU(4) algebra), and [untou3](https://github.com/Deep-SANE/untou3) (U(N) to U(3) reductions).

---

## Key Capabilities

- **Many-Nucleon Basis Construction** — Construct multi-shell symmetry-adapted bases labelled by proton/neutron numbers, $N_{\max}$, total angular momentum $J$, isospin $T$, and $\mathrm{SU(3)}\times\mathrm{SU(4)}$ irreps $(\lambda,\mu)\times(\zeta, S, T)$.
- **Tensor Expansion of Interactions** — Exact decomposition of one- and two-body interactions (chiral EFT potentials such as $\text{N}^2\text{LO}$, $\text{N}^3\text{LO}$, $\text{N}^4\text{LO}$, Coulomb, spin-orbit, and center-of-mass) into $\mathrm{SU(3)}\times\mathrm{SU(4)}$ tensor components.
- **Electroweak & Transition Operators** — Evaluation of electromagnetic ($E2$, $M1$, radius) and weak transition operators (Fermi, Gamow-Teller $\beta$-decay).
- **Hybrid Parallel Evaluation** — Hybrid OpenMP and MPI multi-node evaluation of multi-shell tensor matrix elements with dynamic work distribution.
- **Eigenvalues & Observables** — Large-scale sparse symmetric diagonalization powered by Spectra, CSR storage, and deterministic parallel Lanczos.

---

## Build System

SAM uses a self-contained configuration system matching `u3lib` and `u4lib`. No environment variables are required:

```bash
make config   # Auto-detects u3lib, u4lib, untou3, MPI, OpenMP, Spectra, Boost, Eigen, fmt
make          # Compiles lib/libsam.a and all bin/* applications
make lib      # Compiles only lib/libsam.a
make apps     # Compiles only the applications in app/ -> bin/
make clean    # Cleans build artifacts
make distclean# Resets to pristine state
```

### Configuration Options
Run `./config/configure.sh --help` for available options:
- `--u3lib=PATH`: Path to `u3lib` root [default: `../u3lib`]
- `--u4lib=PATH`: Path to `u4lib` root [default: `../u4lib`]
- `--untou3=PATH`: Path to `untou3` root [default: `../untou3`]
- `--spectra=PATH`: Path to Spectra headers
- `--no-mpi`: Disable MPI (serial/OpenMP-only build)
- `--data-dir=PATH`: Path to SAM data files [default: `./data`]

---

## Repository Layout

- `include/` — Public headers defining the SAM API (`sam.h`, basis, tensor expansion, operators).
- `src/` — Implementation files building `lib/libsam.a`.
- `app/` — Driver applications and physics calculation programs (`bin/*`).
- `utils/` — Self-contained numerical utilities (CSRMatrix, Spectra eigensolver, compatibility forwarding to `u3lib/utils`).
- `docs/` — Technical documentation, recoupling formulations, and derivations.
- `data/` — Nuclear interaction matrix elements, tensor decompositions, and precomputed databases.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
