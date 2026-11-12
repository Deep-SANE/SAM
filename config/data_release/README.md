# SAM data release

Expanding an operator into U(3)xU(4) tensors is the most expensive step before a
SAM calculation. This release provides precomputed expansions of the one- and
two-body operators below, so they don't need to be recomputed. The files are
published as assets of the GitHub release named in `manifest.tsv`, and `make
data` downloads them.

## Getting the data

```
make config          # choose the data directory (default SAM/data)
make data            # download every released expansion
make data OPS="Kinetic N3LO_EMN500_bare"   # or only some operators
make data-list       # show what the release contains
```

Each file lands at `<data dir>/<Body>/<Operator>/Expansion/<Operator>_hw20`,
which is where SAM looks for it. A copy of this README and of `manifest.tsv`
goes into `<data dir>/data_release/`, so the list sits next to the data. Files that are already present and correct are
skipped. To put the data somewhere else for one run, use
`make data SAM_DATA_DIR=/path`.

## Checking the files

The expansions are binary files. Any change, even one byte, makes them garbage,
and SAM can't detect it while reading. For that reason:

- `make data` checks every download against the SHA-256 checksum in
  `manifest.tsv` and makes the file read-only.
- `make data-verify` re-checks all downloaded files at any time. If it reports
  a modified file, restore it with `make data FORCE=1`.
- A file you extended yourself to a larger maxShell is reported as extended,
  not modified, and `make data` never replaces it, even with `FORCE=1`.

## What is available

<!-- table:begin -->
| Body | Operator | Description | maxShell | Tensors | Size |
|---|---|---|---:|---:|---:|
| One | `CoMNumberOp` | one-body part of the CoM HO number operator | 14 | 14 | 1 kB |
| One | `E2` | electric quadrupole | 14 | 40 | 3 kB |
| One | `FermiMinus` | Fermi (isospin lowering) | 8 | 9 | 1 kB |
| One | `FermiPlus` | Fermi (isospin raising) | 8 | 9 | 1 kB |
| One | `GamowTellerMinus` | Gamow-Teller (isospin lowering) | 14 | 15 | 1 kB |
| One | `GamowTellerPlus` | Gamow-Teller (isospin raising) | 14 | 15 | 1 kB |
| One | `Kinetic` | one-body part of the relative kinetic energy | 14 | 41 | 3 kB |
| One | `L.L` | one-body part of L.L | 4 | 7 | 0 kB |
| One | `L.S` | one-body part of L.S | 2 | 2 | 0 kB |
| One | `M1` | magnetic dipole | 14 | 43 | 3 kB |
| One | `Radius` | point-nucleon radius squared | 14 | 41 | 3 kB |
| One | `Radius_n` | point-neutron radius squared | 14 | 41 | 3 kB |
| One | `Radius_p` | point-proton radius squared | 14 | 41 | 3 kB |
| One | `Rrel` | one-body part of the intrinsic radius squared | 14 | 41 | 3 kB |
| Two | `CoMNumberOp` | two-body part of the CoM HO number operator | 14 | 196 | 21 kB |
| Two | `Kinetic` | two-body part of the relative kinetic energy | 14 | 609 | 65 kB |
| Two | `L.L` | two-body part of L.L | 2 | 3 | 0 kB |
| Two | `L.S` | two-body part of L.S | 2 | 6 | 0 kB |
| Two | `N2LO_opt_bare` | NNLOopt chiral NN, bare | 12 | 256,854 | 35.3 MB |
| Two | `N2LO_opt_srg2.0` | NNLOopt chiral NN, SRG 2.0 fm^-1 | 8 | 34,273 | 4.5 MB |
| Two | `N2LO_sat_bare` | NNLOsat chiral NN part, bare | 8 | 34,273 | 4.5 MB |
| Two | `N3LO_EM500_srg2.0` | N3LO Entem-Machleidt 500 MeV, SRG 2.0 fm^-1 | 12 | 256,774 | 35.3 MB |
| Two | `N3LO_EMN500_bare` | N3LO Entem-Machleidt-Nosyk 500 MeV, bare | 12 | 256,830 | 35.2 MB |
| Two | `N4LO_EMN500_bare` | N4LO Entem-Machleidt-Nosyk 500 MeV, bare | 12 | 256,834 | 35.2 MB |
| Two | `Rrel` | two-body part of the intrinsic radius squared | 14 | 608 | 65 kB |
<!-- table:end -->

**maxShell** is the highest harmonic-oscillator shell the expansion covers (it
was made with Nmax = maxShell and no valence shell). A one-body file serves a
calculation whose maxShell (valence shell v + Nmax) is not larger. A two-body
file holds the pairs of up to maxShell quanta, and a calculation needs pairs of
up to 2v + Nmax quanta: the maxShell-12 interactions serve 4He up to Nmax = 12
and p-shell nuclei up to Nmax = 10. Otherwise,
or when a file is missing, SAM stops with a message: expanding is very costly,
so SAM never does it automatically. To expand or extend an operator yourself,
set the operator name, hw and maxShell in `app/OneBodyOperatorJT.cpp` or
`app/TwoBodyOperatorJT.cpp`, rebuild with `make apps`, and run the program with
`mpirun`. A released file is read-only, so unlock it first with `chmod u+w`.

**hw.** Operators computed on the fly (all one-body operators, and the two-body
`L.L`, `L.S`, `Kinetic`, `CoMNumberOp` and `Rrel`) are stored once at hw = 20
MeV and used for any hw; SAM rescales `Kinetic` and `Rrel`. The NN interactions
depend on hw, and these files serve hw = 20 MeV only.

## File format

Each file holds `maxShell` (int8), the number of tensors (64-bit unsigned,
little-endian), then the tensors. The files were written on a little-endian
64-bit machine and read correctly on x86-64 and ARM64.

## Not included

Log files, expansions to smaller Nmax, other hw values, the rmes, hws and
decomp_phys caches (SAM regenerates them), and the interaction input files.

## For maintainers

```
python3 config/data_release/make_release.py --data-dir data --out ../SAM-data-v1 --release data-v1
gh release create data-v1 --repo Deep-SANE/SAM --title "SAM data v1" \
   --notes "Operator expansions; see config/data_release/README.md" ../SAM-data-v1/*.gz
```

The release definition lives in the repository at `config/data_release/`
(`README.md`, `manifest.tsv`, `fetch.sh`, `make_release.py`). `make_release.py`
writes one gzip file per expansion, regenerates `manifest.tsv` and the table
above. For a new release, bump the tag (`data-v2`, ...). Users get
the new files with `make data` once the updated manifest is committed.
