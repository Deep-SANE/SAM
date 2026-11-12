# SAM: Symmetry Adapted Model

**SAM** is a computational framework for symmetry-adapted many-body nuclear structure calculations based on the $\mathrm{U(3)}\times\mathrm{U(4)}$ algebraic formulation of the nuclear shell model.

SAM constructs symmetry-adapted many-nucleon bases, computes and expands single- and two-body nuclear interactions and electroweak transition operators in terms of U(3)×U(4) tensors, computes matrix elements across multiple harmonic oscillator shells, and calculates observables using high-performance parallel eigensolvers (Spectra, OpenMP, MPI).

- **Prerequisites / Sibling libraries:** [u3lib](https://github.com/Deep-SANE/u3lib) (U(3) algebra), [u4lib](https://github.com/Deep-SANE/u4lib) (U(4) algebra), and [untou3](https://github.com/Deep-SANE/untou3) (U(N) to U(3) reductions).

---

## Key Capabilities

- **Many-Nucleon Basis Construction** — Construct multi-shell symmetry-adapted bases labelled by proton/neutron numbers, $N_{\max}$, $\mathrm{U(3)}\times\mathrm{U(4)}$ irreps, angular momentum $L$, spin $S$, total angular momentum $J$, and isospin $T$.
- **Tensor Expansion of Operators** — Exact decomposition of one- and two-body operators (e.g., kinetic energy and chiral EFT potentials) into $\mathrm{U(3)}\times\mathrm{U(4)}$ tensor components.
- **Electroweak & Transition Operators** — Evaluation of electromagnetic ($E2$, $M1$, radius) and weak transition operators (Fermi, Gamow-Teller $\beta$-decay).
- **Hybrid Parallel Evaluation** — Hybrid OpenMP and MPI multi-node evaluation of multi-shell tensor matrix elements with dynamic work distribution.
- **Eigenvalues & Observables** — Large-scale sparse symmetric diagonalization powered by Spectra, CSR storage, and deterministic parallel Lanczos.

---

## Build System

SAM uses a self-contained configuration system matching `u3lib` and `u4lib`. No environment variables are required:

```bash
make config    # Auto-detects u3lib, u4lib, untou3, MPI (required), OpenMP, Spectra, Boost, Eigen, fmt
make           # Compiles lib/libsam.a and all bin/* applications
make lib       # Compiles only lib/libsam.a
make apps      # Compiles only the applications in app/ -> bin/
make test      # Builds and runs the regression tests in test/
make clean     # Cleans build artifacts
make distclean # Resets to pristine state
```

### Data Directory and Released Data
`make config` asks where SAM should keep its data (operator expansions, rmes, hws, decompositions). The default is `SAM/data`. On a cluster, give a path on a work file system instead (e.g. `/work/$USER/sam-data`): the data then lives there and `SAM/data` becomes a link to it. The chosen path is built into the binaries; setting the `SAM_DATA_DIR` environment variable overrides it at run time.

Expanding operators is the most expensive step, so precomputed expansions are released with SAM:

```bash
make data          # download the released expansions into the data directory
make data-verify   # check them against the release checksums
make data-list     # show what is available
```

See [config/data_release/README.md](config/data_release/README.md) for the list of files and their details; `make data` also puts a copy in `<data dir>/data_release/`.

### Configuration Options
Run `./config/configure.sh --help` for available options:
- `--u3lib=PATH`: Path to `u3lib` root [default: `../u3lib`]
- `--u4lib=PATH`: Path to `u4lib` root [default: `../u4lib`]
- `--untou3=PATH`: Path to `untou3` root [default: `../untou3`]
- `--spectra=PATH`: Path to Spectra headers
- `--mpicxx=PATH`: MPI compiler wrapper [default: `mpicxx` on the path]
- `--cxx=PATH`: Compiler that the MPI wrapper calls [default: the one u3lib/u4lib were built with]
- `--data-dir=PATH`: Where SAM keeps its data [default: `SAM/data`]; another path makes `SAM/data` a link to it
- `--yes`, `-y`: Non-interactive mode (accept all defaults)

---

## Repository Layout

- `include/` — Public headers defining the SAM API (`sam.h`, basis, tensor expansion, operators).
- `src/` — Implementation files building `lib/libsam.a`.
- `app/` — Driver applications and physics calculation programs (`bin/*`).
- `config/` — Configuration script and the data release definition (`config/data_release/`: README, checksums, download script).
- `test/` — Regression tests run by `make test` (see [test/README.md](test/README.md)).
- `utils/` — Self-contained numerical utilities (CSRMatrix, Spectra eigensolver, compatibility forwarding to `u3lib/utils`).
- `data/` — Data directory (or a link to it): operator expansions, interaction matrix elements, tensor decompositions and precomputed databases. Not tracked by git.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
