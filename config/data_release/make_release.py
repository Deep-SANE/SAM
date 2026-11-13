#!/usr/bin/env python3
"""Build the SAM data release (maintainer tool; users never run this).

Selects the operator expansions to publish from a SAM data directory, writes
one gzip asset per file for the GitHub release, and regenerates
config/data_release/manifest.tsv and the file table in its README.md.

Selection: only the canonical expansion files <Operator>_hw20 under
<Body>/<Operator>/Expansion/. Log files and variants with extra suffixes
(e.g. _Nmax10, _Nmax8_PD) are skipped; the canonical file is the one that
reaches the largest maxShell.

Usage:
  python3 config/data_release/make_release.py --data-dir data --out ../release-assets \\
                                   --release data-v1
"""

import argparse
import gzip
import hashlib
import re
import shutil
import struct
from pathlib import Path

BODIES = ("OneBodyOperatorJT", "TwoBodyOperatorJT")
HW = 20

# short descriptions shown in the README table:
DESCRIPTIONS = {
  ("OneBodyOperatorJT", "CoMNumberOp"): "one-body part of the CoM HO number operator",
  ("OneBodyOperatorJT", "E2"): "electric quadrupole",
  ("OneBodyOperatorJT", "FermiMinus"): "Fermi (isospin lowering)",
  ("OneBodyOperatorJT", "FermiPlus"): "Fermi (isospin raising)",
  ("OneBodyOperatorJT", "GamowTellerMinus"): "Gamow-Teller (isospin lowering)",
  ("OneBodyOperatorJT", "GamowTellerPlus"): "Gamow-Teller (isospin raising)",
  ("OneBodyOperatorJT", "Kinetic"): "one-body part of the relative kinetic energy",
  ("OneBodyOperatorJT", "L.L"): "one-body part of L.L",
  ("OneBodyOperatorJT", "L.S"): "one-body part of L.S",
  ("OneBodyOperatorJT", "M1"): "magnetic dipole",
  ("OneBodyOperatorJT", "NumberOp"): "HO number operator (total quanta)",
  ("OneBodyOperatorJT", "Radius"): "point-nucleon radius squared",
  ("OneBodyOperatorJT", "Radius_n"): "point-neutron radius squared",
  ("OneBodyOperatorJT", "Radius_p"): "point-proton radius squared",
  ("OneBodyOperatorJT", "Rrel"): "one-body part of the intrinsic radius squared",
  ("TwoBodyOperatorJT", "CoMNumberOp"): "two-body part of the CoM HO number operator",
  ("TwoBodyOperatorJT", "Kinetic"): "two-body part of the relative kinetic energy",
  ("TwoBodyOperatorJT", "L.L"): "two-body part of L.L",
  ("TwoBodyOperatorJT", "L.S"): "two-body part of L.S",
  ("TwoBodyOperatorJT", "Rrel"): "two-body part of the intrinsic radius squared",
  ("TwoBodyOperatorJT", "N2LO_opt_bare"): "NNLOopt chiral NN, bare",
  ("TwoBodyOperatorJT", "N2LO_opt_srg2.0"): "NNLOopt chiral NN, SRG 2.0 fm^-1",
  ("TwoBodyOperatorJT", "N2LO_sat_bare"): "NNLOsat chiral NN part, bare",
  ("TwoBodyOperatorJT", "N3LO_EM500_srg2.0"): "N3LO Entem-Machleidt 500 MeV, SRG 2.0 fm^-1",
  ("TwoBodyOperatorJT", "N3LO_EMN500_bare"): "N3LO Entem-Machleidt-Nosyk 500 MeV, bare",
  ("TwoBodyOperatorJT", "N4LO_EMN500_bare"): "N4LO Entem-Machleidt-Nosyk 500 MeV, bare",
}


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      h.update(block)
  return h.hexdigest()


def header(path):
  # file layout: int8 maxShell, then size_t (8 bytes, little-endian) n_tensors
  with open(path, "rb") as f:
    raw = f.read(9)
  max_shell = struct.unpack("<b", raw[:1])[0]
  n_tensors = struct.unpack("<Q", raw[1:9])[0]
  return max_shell, n_tensors


def select(data_dir):
  rows = []
  for body in BODIES:
    for exp_dir in sorted((data_dir / body).glob("*/Expansion")):
      op = exp_dir.parent.name
      path = exp_dir / f"{op}_hw{HW}"
      if path.is_file():
        rows.append((body, op, path))
  return rows


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--data-dir", required=True, type=Path)
  ap.add_argument("--out", required=True, type=Path)
  ap.add_argument("--release", required=True, help="release tag, e.g. data-v1")
  args = ap.parse_args()

  here = Path(__file__).resolve().parent
  args.out.mkdir(parents=True, exist_ok=True)

  lines = [f"# release: {args.release}",
           "# sha256 and bytes refer to the uncompressed file on disk",
           "asset\tpath\tbytes\tsha256\tbody\toperator\thw\tmaxShell\tn_tensors"]
  table = ["| Body | Operator | Description | maxShell | Tensors | Size |",
           "|---|---|---|---:|---:|---:|"]

  for body, op, path in select(args.data_dir):
    rel = path.relative_to(args.data_dir).as_posix()
    asset = f"{body}.{path.name}.gz"
    with open(path, "rb") as src, gzip.open(args.out / asset, "wb", 9) as dst:
      shutil.copyfileobj(src, dst)
    nbytes = path.stat().st_size
    max_shell, n_tensors = header(path)
    lines.append("\t".join(map(str, (asset, rel, nbytes, sha256(path), body,
                                     op, HW, max_shell, n_tensors))))
    size = f"{nbytes / 1e6:.1f} MB" if nbytes >= 1e5 else f"{nbytes / 1e3:.0f} kB"
    table.append(f"| {body[:3]} | `{op}` | {DESCRIPTIONS.get((body, op), '')} "
                 f"| {max_shell} | {n_tensors:,} | {size} |")
    print(f"{asset:55s} maxShell={max_shell:2d} tensors={n_tensors}")

  (here / "manifest.tsv").write_text("\n".join(lines) + "\n")

  readme = here / "README.md"
  text = readme.read_text()
  text = re.sub(r"(<!-- table:begin -->\n).*?(<!-- table:end -->)",
                lambda m: m.group(1) + "\n".join(table) + "\n" + m.group(2),
                text, flags=re.S)
  readme.write_text(text)
  print(f"wrote {len(lines) - 3} assets to {args.out}, manifest.tsv, README table")


if __name__ == "__main__":
  main()
